import base64

import PIL
import torch
import requests
import torchvision
from math import ceil
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import display, Image
import torchvision.transforms.functional as F


def download_image(url):
    return PIL.Image.open(requests.get(url, stream=True).raw).convert("RGB")


def resize_image(image, size=768):
    tensor_image = F.to_tensor(image)
    resized_image = F.resize(tensor_image, size, antialias=True)
    return resized_image


def downscale_images(images, factor=3 / 4):
    scaled_height, scaled_width = int(((images.size(-2) * factor) // 32) * 32), int(
        ((images.size(-1) * factor) // 32) * 32)
    scaled_image = F.resize(images, (scaled_height, scaled_width),
                            interpolation=torchvision.transforms.InterpolationMode.NEAREST)
    return scaled_image


def show_images(images, rows=None, cols=None, return_images=True, **kwargs):
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]

    if rows is None:
        rows = 1
    if cols is None:
        cols = images.size(0) // rows

    _, _, h, w = images.shape
    grid = PIL.Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(images):
        img = F.to_pil_image(img.clamp(0, 1))
        grid.paste(img, box=(i % cols * w, i // cols * h))

    bio = BytesIO()
    grid.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

    if return_images:
        return grid


def to_pil_images(images):
    if images.size(1) == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.size(1) > 3:
        images = images[:, :3]

    return [F.to_pil_image(img.clamp(0, 1)) for img in images]


def to_base64_images(images):
    pil_imgs = to_pil_images(images)
    return pils_to_base64(pil_imgs)


def pils_to_base64(images):
    encoded_imgs = []
    for pil in images:
        bio = BytesIO()
        pil.save(bio, format='png')
        b64 = base64.b64encode(bio.getvalue()).decode('utf-8')
        encoded_imgs.append(f'data:image/png;base64,{b64}')
    return encoded_imgs


def calculate_latent_sizes(height=1024, width=1024, batch_size=4, compression_factor_b=42.67, compression_factor_a=4.0):
    latent_height = ceil(height / compression_factor_b)
    latent_width = ceil(width / compression_factor_b)

    def even(n):
        return n - n % 2

    stage_c_latent_shape = (batch_size, 16, even(latent_height), even(latent_width))

    latent_height = ceil(height / compression_factor_a)
    latent_width = ceil(width / compression_factor_a)
    stage_b_latent_shape = (batch_size, 4, even(latent_height), even(latent_width))

    return stage_c_latent_shape, stage_b_latent_shape
