import torch
import random
import gc
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
from .runner_base import RunnerBase,MAX_SEED
from inference.utils import pils_to_base64


class RunnerDf(RunnerBase):
    def __init__(self, device):
        super().__init__(device)
        self.prior = None
        self.decoder = None

    def load_models(self):
        prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior",
                                                           torch_dtype=torch.bfloat16)
        decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.half)
        self.prior = prior.to(self.device)
        self.decoder = decoder.to(self.device)

    def generate_prior(self, prompt, negative_prompt,
                       generator, width, height, num_inference_steps, guidance_scale, num_images_per_prompt):
        prior_output = self.prior(
            prompt=prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps
        )
        # prior.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        return prior_output.image_embeddings

    def generate_decoder(self, prior_embeds, prompt, negative_prompt, generator, num_inference_steps, guidance_scale):
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
        torch.cuda.empty_cache()
        gc.collect()
        return decoder_output

    @torch.inference_mode()
    def _txt2img(
            self,
            prompt: str,
            negative_prompt: str = "",
            seed: int = 0,
            width: int = 1024,
            height: int = 1024,
            batch_size: int = 2,
            prior_num_inference_steps: int = 20,
            prior_guidance_scale: float = 4.0,
            decoder_num_inference_steps: int = 10,
            decoder_guidance_scale: float = 0.0,
    ):
        """Generate images using Stable Cascade."""
        if seed == 0:
            seed = random.randint(0, MAX_SEED)
        print("seed:", seed)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        prior_embeds = self.generate_prior(
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            width=width,
            height=height,
            num_inference_steps=prior_num_inference_steps,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=batch_size,
        )

        decoder_output = self.generate_decoder(
            prior_embeds=prior_embeds,
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            num_inference_steps=decoder_num_inference_steps,
            guidance_scale=decoder_guidance_scale,
        )

        encoded_images = pils_to_base64(decoder_output)
        return {
            'success': True,
            'encoded_images': encoded_images
        }
