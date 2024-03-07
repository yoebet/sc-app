import torch
import torchvision.transforms.functional as F
import logging
import random
import yaml
from tqdm import tqdm
from PIL import Image
from inference.utils import to_pil_images, calculate_latent_sizes
from train import WurstCoreC, WurstCoreB, ControlNetCore
from .runner_base import RunnerBase, MAX_SEED
from .common import prepare_image_tensor


class RunnerSc(RunnerBase):
    def __init__(self, device, app_config, logger: logging.Logger = None):
        super().__init__(device, app_config, logger)
        self.core = None
        self.core_b = None
        self.extras = None
        self.extras_b = None
        self.models = None
        self.models_b = None
        self.ip_models_loaded = False
        self.cn_core = None
        self.cn_models = None
        self.cn_extras = None

    def load_models(self):
        # SETUP STAGE C
        config_file = 'configs/inference/stage_c_3b.yaml'
        with open(config_file, "r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file)

        core = WurstCoreC(config_dict=loaded_config, device=self.device, training=False)

        # SETUP STAGE B
        config_file_b = 'configs/inference/stage_b_3b.yaml'
        with open(config_file_b, "r", encoding="utf-8") as file:
            config_file_b = yaml.safe_load(file)

        core_b = WurstCoreB(config_dict=config_file_b, device=self.device, training=False)

        # SETUP MODELS & DATA
        extras = core.setup_extras_pre()
        models = core.setup_models(extras)
        models.generator.eval().requires_grad_(False)
        print("STAGE C READY")

        extras_b = core_b.setup_extras_pre()
        models_b = core_b.setup_models(extras_b, skip_clip=True)
        models_b = WurstCoreB.Models(
            **{**models_b.to_dict(), 'tokenizer': models.tokenizer, 'text_model': models.text_model}
        )
        models_b.generator.bfloat16().eval().requires_grad_(False)
        print("STAGE B READY")

        self.core = core
        self.core_b = core_b
        self.extras = extras
        self.extras_b = extras_b
        self.models = models
        self.models_b = models_b

        self.models_loaded = True

    def ensure_ip_models(self):
        if self.cn_core is not None:
            return

        if not self.models_loaded:
            self.load_models()

        config_file = 'configs/inference/controlnet_c_3b_inpainting.yaml'
        with open(config_file, "r", encoding="utf-8") as file:
            loaded_config = yaml.safe_load(file)

        core = ControlNetCore(config_dict=loaded_config, device=self.device, training=False)

        extras = core.setup_extras_pre()

        models = core.setup_models(extras, shared=vars(self.models))
        models.generator.eval().requires_grad_(False)
        print("CONTROLNET READY")

        self.cn_core = core
        self.cn_models = models
        self.cn_extras = extras

    def _inference(self,
                   task_id: str,
                   prompt: str,
                   negative_prompt: str = "",
                   seed: int = 0,
                   width: int = 1024,
                   height: int = 1024,
                   batch_size: int = 1,
                   image=None,
                   mask=None,
                   mask_invert=False,
                   auto_mask_threshold=0.2,
                   outpaint_ext=None,  # [top, right, bottom, left]
                   prior_num_inference_steps: int = 20,
                   prior_guidance_scale: float = 4.0,
                   decoder_num_inference_steps: int = 10,
                   decoder_guidance_scale: float = 1.1,
                   task_type='txt2img',  # img2img, img_variate, inpaint, outpaint
                   return_images_format: str = 'base64',  # pil
                   sub_dir: str = None,
                   ):
        if width is None:
            width = 1024
        if height is None:
            height = 1024
        caption = prompt
        neg_caption = negative_prompt if negative_prompt is not None else ""
        if decoder_guidance_scale < 0.5:
            decoder_guidance_scale = 0.5
        if decoder_guidance_scale == 7:
            decoder_guidance_scale = 4
        generated_seed = False
        if seed <= 0:
            seed = random.randint(0, MAX_SEED)
            generated_seed = True
            print("seed:", seed)

        if task_type in ['inpaint', 'outpaint']:
            self.ensure_ip_models()
            core = self.cn_core
            models = self.cn_models
            extras = self.cn_extras
            use_cnet = True
        else:
            core = self.core
            models = self.models
            extras = self.extras
            use_cnet = False
        core_b = self.core_b
        extras_b = self.extras_b
        models_b = self.models_b

        # PREPARE CONDITIONS
        batch = {'captions': [caption] * batch_size, 'neg_captions': [neg_caption] * batch_size}

        image0 = None
        padded_image = None
        if task_type in ['img2img', 'img_variate', 'inpaint', 'outpaint']:
            tensor_image = prepare_image_tensor(image)
            # _, h, w = tensor_image.shape
            # size = 1024 if h >= 1024 or w >= 1024 else 768
            # tensor_image = F.resize(tensor_image, size, antialias=True)
            image0 = tensor_image.to(self.device)

            if task_type in ['inpaint'] and mask is not None:
                mask = prepare_image_tensor(mask).to(self.device)

        if task_type == 'outpaint':
            img_height = image0.size(1)
            img_width = image0.size(2)
            if outpaint_ext is None:
                outpaint_ext = [64]
            elif type(outpaint_ext) == int:
                outpaint_ext = [outpaint_ext]
            if len(outpaint_ext) == 1:
                outpaint_ext = outpaint_ext * 4
            elif len(outpaint_ext) == 2:
                outpaint_ext = outpaint_ext * 2
            elif len(outpaint_ext) == 3:
                outpaint_ext = outpaint_ext + [outpaint_ext[1]]
            elif len(outpaint_ext) > 4:
                outpaint_ext = outpaint_ext[0: 4]
            ext_top, ext_right, ext_bottom, ext_left = outpaint_ext

            full_height = img_height + ext_top + ext_bottom
            full_width = img_width + ext_left + ext_right

            mask = torch.ones(batch_size, 1, full_height, full_width).bool()
            mask_keep = torch.zeros(batch_size, 1, img_height, img_width).bool()
            mask[..., ext_top:ext_top + img_height, ext_left:ext_left + img_width] = mask_keep
            mask.to(self.device)

            paddings = (ext_left, ext_right, ext_top, ext_bottom)
            padded_image = torch.nn.ReflectionPad2d(paddings)(image0)
            # padded_image = torch.randn((3, full_height, full_width))
            # padded_image[..., ext_top:ext_top + img_height, ext_left:ext_left + img_width] = image0

            height, width = full_height, full_width

        if image0 is not None:
            images = image0.expand(batch_size, -1, -1, -1)
            batch['images'] = images.to(self.device)
            if padded_image is not None:
                batch['images_ori'] = batch['images']
                padded_images = padded_image.expand(batch_size, -1, -1, -1)
                batch['images'] = padded_images.to(self.device)

        noise_level = 1
        noised = None
        if task_type in ['img2img', 'inpaint', 'outpaint']:
            if task_type == 'img2img':
                noise_level = 0.8
            effnet_latents = core.encode_latents(batch, models, extras)
            t = torch.ones(effnet_latents.size(0), device=self.device) * noise_level
            noised = extras.gdf.diffuse(effnet_latents, t=t)[0]

        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        # Stage C Parameters
        extras.sampling_configs['cfg'] = prior_guidance_scale
        extras.sampling_configs['shift'] = 2
        extras.sampling_configs['timesteps'] = int(prior_num_inference_steps * noise_level)
        extras.sampling_configs['t_start'] = noise_level
        extras.sampling_configs['x_init'] = noised

        # Stage B Parameters
        extras_b.sampling_configs['cfg'] = decoder_guidance_scale
        extras_b.sampling_configs['shift'] = 1
        extras_b.sampling_configs['timesteps'] = decoder_num_inference_steps
        extras_b.sampling_configs['t_start'] = 1.0

        eval_image_embeds = task_type == 'img_variate' or (image is not None and (prompt is None or prompt == ''))
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False,
                                         eval_image_embeds=eval_image_embeds)
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True,
                                           eval_image_embeds=False)
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        if use_cnet:
            outpaint = task_type == 'outpaint' or (type == 'inpaint' and mask_invert)
            cnet_multiplier = 1.0  # 0.8, 0.3
            if auto_mask_threshold is None:
                auto_mask_threshold = 0.2  # 0.0 ~ 0.4

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                cnet, cnet_input = core.get_cnet(batch, models, extras, mask=mask, outpaint=outpaint,
                                                 threshold=auto_mask_threshold)
                cnet_uncond = cnet
                conditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet]
                unconditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet_uncond]

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            torch.manual_seed(seed)

            sampling_c = extras.gdf.sample(
                models.generator, conditions, stage_c_latent_shape,
                unconditions, device=self.device, **extras.sampling_configs,
            )
            # st = 0
            for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
                sampled_c = sampled_c
                # st += 1
                # preview_c = models.previewer(sampled_c).float()
                # pi = show_images(preview_c)
                # pi.save(f'{task_type}_preview_{st}.png')

            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)

            sampling_b = extras_b.gdf.sample(
                models_b.generator, conditions_b, stage_b_latent_shape,
                unconditions_b, device=self.device, **extras_b.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
                sampled_b = sampled_b
            sampled = models_b.stage_a.decode(sampled_b).float()

        images = to_pil_images(sampled)

        result = self.build_results(task_type,
                                    images,
                                    task_id,
                                    sub_dir=sub_dir,
                                    file_name_part=f'{width}x{height}',
                                    return_images_format=return_images_format)
        if generated_seed:
            result['seed'] = seed
        return result

    def _txt2img(self, **params):
        return self._inference(**params)

    def _img2img(self, **params):
        return self._inference(**params)

    def _img_variate(self, **params):
        return self._inference(**params)

    def _img_gen(self, **params):
        return self._inference(**params)
