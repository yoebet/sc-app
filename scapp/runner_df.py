import logging
import sys

import torch
import random
import gc
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from .runner_base import RunnerBase, MAX_SEED
from inference.utils import download_image
from .common import decode_to_pil_image


class RunnerDf(RunnerBase):
    def __init__(self, device, app_config, logger: logging.Logger = None):
        super().__init__(device, app_config, logger)
        self.prior = None
        self.decoder = None

    def load_models(self):
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                           torch_dtype=torch.bfloat16)
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.half)
        self.prior = prior.to(self.device)
        self.decoder = decoder.to(self.device)
        self.models_loaded = True

    def _generate_prior(self, prompt, negative_prompt, image,
                        generator, width, height, num_inference_steps, guidance_scale, num_images_per_prompt):
        prior_output = self.prior(
            prompt=prompt,
            image=image,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps
        )
        # prior.to('cpu')
        # torch.cuda.empty_cache()
        # gc.collect()
        return prior_output.image_embeddings

    def _generate_decoder(self, prior_embeds, prompt, negative_prompt, generator, num_inference_steps, guidance_scale):
        decoder_output = self.decoder(
            image_embeddings=prior_embeds.to(device=self.device, dtype=self.decoder.dtype),
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            output_type="pil",
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images
        # decoder.to('cpu')
        # torch.cuda.empty_cache()
        # gc.collect()
        return decoder_output

    @torch.inference_mode()
    def _txt2img(self, **params):
        return self._img2img(**params)

    @torch.inference_mode()
    def _img2img(
            self,
            task_id: str,
            prompt: str,
            negative_prompt: str = "",
            image=None,
            seed: int = 0,
            width: int = 1024,
            height: int = 1024,
            batch_size: int = 2,
            prior_num_inference_steps: int = 20,
            prior_guidance_scale: float = 4.0,
            decoder_num_inference_steps: int = 10,
            decoder_guidance_scale: float = 0.0,  # ignore
            task_type='txt2img',
            return_images_format: str = 'base64',  # pil
            sub_dir: str = None,
            **extra_params,
    ):
        if width is None:
            width = 1024
        if height is None:
            height = 1024
        if seed <= 0:
            seed = random.randint(0, MAX_SEED)
            print("seed:", seed)
        if image is not None:
            if isinstance(image, str):
                if image.startswith('http'):
                    image = download_image(image)
                else:
                    image = decode_to_pil_image(image)
            task_type = 'img2img'
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prior_embeds = self._generate_prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=prior_num_inference_steps,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=batch_size,
        )

        images = self._generate_decoder(
            prior_embeds=prior_embeds,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=decoder_num_inference_steps,
            guidance_scale=0.0,
        )

        return self.build_results(task_type,
                                  images,
                                  task_id,
                                  sub_dir=sub_dir,
                                  file_name_part=f'{width}x{height}',
                                  return_images_format=return_images_format)
