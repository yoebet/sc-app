import os
import yaml
import torch
from tqdm import tqdm

from inference.utils import *
from core.utils import load_or_fail
from train import ControlNetCore, WurstCoreB

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
print(device)

# SETUP STAGE C
config_file = 'configs/inference/controlnet_c_3b_sr.yaml'
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
images = F.to_tensor(download_image(url)).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)

batch = {'images': images}

# show_images(batch['images'])

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
    effnet_latents = core.encode_latents(batch, models, extras)
    effnet_latents_up = torch.nn.functional.interpolate(effnet_latents, scale_factor=2, mode="nearest")
    cnet = models.controlnet(effnet_latents_up)
    cnet_uncond = cnet
    cnet_input = torch.nn.functional.interpolate(images, scale_factor=2, mode="nearest")

# caption = "a person riding a rodent"
caption = 'a green pole on the side of a dirt road, a sign on top of it, with red Word "Blue"'

cnet_multiplier = 1.0  # 0.8 # 0.3

if "controlnet_c_3b_sr" in config_file:
    # height, width = int(cnet[0].size(-2) * 32 * 4 / 3), int(cnet[0].size(-1) * 32 * 4 / 3)
    height, width = 2048, 2048
else:
    height, width = 1024, 1024

stage_c_latent_shape, stage_b_latent_shape = calculate_latent_sizes(height, width, batch_size=batch_size)

# Stage C Parameters
extras.sampling_configs['cfg'] = 1
extras.sampling_configs['shift'] = 2
extras.sampling_configs['timesteps'] = 20
extras.sampling_configs['t_start'] = 1.0

# Stage B Parameters
extras_b.sampling_configs['cfg'] = 1.1
extras_b.sampling_configs['shift'] = 1
extras_b.sampling_configs['timesteps'] = 10
extras_b.sampling_configs['t_start'] = 1.0

# PREPARE CONDITIONS
batch['captions'] = [caption] * batch_size

conditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=False,
                                 eval_image_embeds=False)
unconditions = core.get_conditions(batch, models, extras, is_eval=True, is_unconditional=True,
                                   eval_image_embeds=False)
conditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=False)
unconditions_b = core_b.get_conditions(batch, models_b, extras_b, is_eval=True, is_unconditional=True)

conditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet]
unconditions['cnet'] = [c.clone() * cnet_multiplier if c is not None else c for c in cnet_uncond]

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
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

show_images(sampled, return_images=True).save('sampled-sr-3.png')
