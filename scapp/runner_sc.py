import os
import random
from threading import Thread, Lock
import yaml
import torch
from tqdm import tqdm
import base64
from inference.utils import *
from core.utils import load_or_fail
from train import WurstCoreC, WurstCoreB
from .runner_base import RunnerBase, MAX_SEED


class RunnerSc(RunnerBase):
    def __init__(self, device):
        super().__init__(device)
        self.core = None
        self.core_b = None
        self.extras = None
        self.extras_b = None
        self.models = None
        self.models_b = None

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

        # models = WurstCoreC.Models(
        #    **{**models.to_dict(), 'generator': torch.compile(models.generator, mode="reduce-overhead", fullgraph=True)}
        # )
        #
        # models_b = WurstCoreB.Models(
        #    **{**models_b.to_dict(), 'generator': torch.compile(models_b.generator, mode="reduce-overhead", fullgraph=True)}
        # )

        self.core = core
        self.core_b = core_b
        self.extras = extras
        self.extras_b = extras_b
        self.models = models
        self.models_b = models_b

    def _txt2img(self,
                 prompt: str,
                 negative_prompt: str = "",
                 seed: int = 0,
                 width: int = 1024,
                 height: int = 1024,
                 batch_size: int = 2,
                 prior_num_inference_steps: int = 20,
                 prior_guidance_scale: float = 4.0,
                 decoder_num_inference_steps: int = 10,
                 decoder_guidance_scale: float = 1.1,
                 ):

        caption = prompt
        stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

        core = self.core
        core_b = self.core_b
        extras = self.extras
        extras_b = self.extras_b
        models = self.models
        models_b = self.models_b

        # Stage C Parameters
        extras.sampling_configs['cfg'] = prior_guidance_scale
        extras.sampling_configs['shift'] = 2
        extras.sampling_configs['timesteps'] = prior_num_inference_steps
        extras.sampling_configs['t_start'] = 1.0

        # Stage B Parameters
        extras_b.sampling_configs['cfg'] = decoder_guidance_scale
        extras_b.sampling_configs['shift'] = 1
        extras_b.sampling_configs['timesteps'] = decoder_num_inference_steps
        extras_b.sampling_configs['t_start'] = 1.0
        if seed == 0:
            seed = random.randint(0, MAX_SEED)

        # PREPARE CONDITIONS
        batch = {'captions': [caption] * batch_size}
        conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False,
                                         eval_image_embeds=False)
        unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True,
                                           eval_image_embeds=False)
        conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
        unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            torch.manual_seed(seed)

            sampling_c = extras.gdf.sample(
                models.generator, conditions, stage_c_latent_shape,
                unconditions, device=self.device, **extras.sampling_configs,
            )
            for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
                sampled_c = sampled_c

            # preview_c = models.previewer(sampled_c).float()
            # show_images(preview_c)

            conditions_b['effnet'] = sampled_c
            unconditions_b['effnet'] = torch.zeros_like(sampled_c)

            sampling_b = extras_b.gdf.sample(
                models_b.generator, conditions_b, stage_b_latent_shape,
                unconditions_b, device=self.device, **extras_b.sampling_configs
            )
            for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
                sampled_b = sampled_b
            sampled = models_b.stage_a.decode(sampled_b).float()

        # show_images(sampled)
        encoded_images = to_base64_images(sampled)
        return {
            'success': True,
            'encoded_images': encoded_images
        }
