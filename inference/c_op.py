import os
import yaml
import torch
from tqdm import tqdm
from PIL import Image

from inference.utils import *
from train import ControlNetCore, WurstCoreB

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

# SETUP STAGE C
config_file = 'configs/inference/controlnet_c_3b_inpainting.yaml'
with open(config_file, "r", encoding="utf-8") as file:
    loaded_config = yaml.safe_load(file)

core = ControlNetCore(config_dict=loaded_config, device=device, training=False)

# SETUP STAGE B
config_file_b = 'configs/inference/stage_b_3b.yaml'
with open(config_file_b, "r", encoding="utf-8") as file:
    config_file_b = yaml.safe_load(file)

core_b = WurstCoreB(config_dict=config_file_b, device=device, training=False)

# SETUP MODELS & DATA
extras = core.setup_extras_pre()
models = core.setup_models(extras)
models.generator.eval().requires_grad_(False)
print("CONTROLNET READY")

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

batch_size = 1
url = "https://oss-dev.tishi.top/sds/hJXUclrhEfxarc6m/KzObvl9DxBhm0IDb/1709099669-1.png"
image_ori = resize_image(download_image(url), 1024)

img_height = image_ori.size(1)
img_width = image_ori.size(2)

ext_top = 64
ext_bottom = 128
ext_left = 128
ext_right = 64

full_height = img_height + ext_top + ext_bottom
full_width = img_width + ext_left + ext_right

pil_image = PIL.Image.new('RGB', size=(full_width, full_height))

pil_image.paste(F.to_pil_image(image_ori.clamp(0, 1)), box=(ext_left, ext_top))

image = pil_image.convert('RGB')
image = F.to_tensor(image)
images = image.unsqueeze(0).to(device)

show_images(images, return_images=True).save('image-op-0.png')

batch = {'images': images}

# show_images(batch['images'])


# mask = None
mask_keep = torch.zeros(batch_size, 1, img_height, img_width).bool()

mask_img_left = torch.ones(batch_size, 1, img_height, ext_left).bool()
mask_img_right = torch.ones(batch_size, 1, img_height, ext_right).bool()

mask_m = torch.cat([mask_img_left, mask_keep, mask_img_right], 3)

mask_top = torch.ones(batch_size, 1, ext_top, full_width).bool()
mask_bottom = torch.ones(batch_size, 1, ext_bottom, full_width).bool()

mask = torch.cat([mask_top, mask_m, mask_bottom], 2)

outpaint = True
threshold = 0.2

# caption = "a person riding a rodent"
caption = 'a green pole on the side of a dirt road, a sign on top of it, with red Word "danger"'

cnet_multiplier = 1.0  # 0.8 # 0.3

noise_level = 1
height, width = full_height, full_width
stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

effnet_latents = core.encode_latents(batch, models, extras)
t = torch.ones(effnet_latents.size(0), device=device) * noise_level
noised = extras.gdf.diffuse(effnet_latents, t=t)[0]

# Stage C Parameters
extras.sampling_configs['cfg'] = 4
extras.sampling_configs['shift'] = 2
extras.sampling_configs['timesteps'] = int(20 * noise_level)
extras.sampling_configs['t_start'] = noise_level
extras.sampling_configs['x_init'] = noised

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    # torch.manual_seed(42)

    conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False,
                                     eval_image_embeds=False)
    unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True,
                                       eval_image_embeds=False)
    conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
    unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

    cnet, cnet_input = core.get_cnet(batch, models, extras, mask=mask, outpaint=outpaint, threshold=threshold)
    cnet_uncond = cnet

    conditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet]
    unconditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet_uncond]

    sampling_c = extras.gdf.sample(
        models.generator, conditions, stage_c_latent_shape,
        unconditions, device=device, **extras.sampling_configs,
    )
    for (sampled_c, _, _) in tqdm(sampling_c, total=extras.sampling_configs['timesteps']):
        sampled_c = sampled_c

    # preview_c = models.previewer(sampled_c).float()
    # show_images(preview_c)

    conditions_b['effnet'] = sampled_c
    unconditions_b['effnet'] = torch.zeros_like(sampled_c)

    sampling_b = extras_b.gdf.sample(
        models_b.generator, conditions_b, stage_b_latent_shape,
        unconditions_b, device=device, **extras_b.sampling_configs
    )
    for (sampled_b, _, _) in tqdm(sampling_b, total=extras_b.sampling_configs['timesteps']):
        sampled_b = sampled_b
    sampled = models_b.stage_a.decode(sampled_b).float()

show_images(sampled, return_images=True).save('sampled-op-4.png')
