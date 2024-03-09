import os
import io
import shutil
import base64
import ipaddress
import requests
from PIL import Image
from urllib.parse import urlparse
import torch
import torchvision.transforms.functional as F


def trans_unit(bytes, unit):
    if unit is None:
        return bytes
    k = 1024
    m = k * k
    div = {'B': 1, 'K': k, 'M': m, 'G': k * m}.get(unit.upper())
    return bytes / div


def get_device_name(launch_params):
    if torch.cuda.is_available():
        device_index = launch_params.get('device_index')
        return f'cuda:{device_index}' if device_index is not None else 'cuda'
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return 'mps'
    return 'cpu'


def get_task_dir(tasks_dir: str, task_id: str, sub_dir: str = None):
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        task_dir = f'{tasks_dir}/{sub_dir}/t_{task_id}'
    else:
        task_dir = f'{tasks_dir}/t_{task_id}'

    return task_dir


def get_tf_logging_dir(tf_logs_dir: str, task_id: str, sub_dir: str = None):
    if sub_dir == '' or sub_dir == '_':
        sub_dir = None
    if sub_dir is not None:
        logging_dir = f'{tf_logs_dir}/{sub_dir}/t_{task_id}'
    else:
        logging_dir = f'{tf_logs_dir}/t_{task_id}'
    return logging_dir


def verify_url(url):
    import socket
    from urllib.parse import urlparse
    try:
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        host = socket.gethostbyname_ex(domain_name)
        for ip in host[2]:
            ip_addr = ipaddress.ip_address(ip)
            if not ip_addr.is_global:
                return False
    except Exception:
        return False

    return True


def download(resource_url, target_dir, filename, default_ext):
    if not resource_url.startswith('http'):
        raise Exception(f'must be url: {resource_url}')
    # if not verify_url(resource_url):
    #     raise Exception(f'local resource not allowed')
    resource_path = urlparse(resource_url).path
    resource_name = os.path.basename(resource_path)
    base_name, ext = os.path.splitext(resource_name)
    if filename is None:
        filename = base_name
    if ext is None:
        ext = default_ext
    elif ext == '.jfif':
        ext = '.jpg'
    if ext is not None:
        filename = f'{filename}{ext}'

    full_path = f'{target_dir}/{filename}'
    with requests.get(resource_url, stream=True) as res:
        with open(full_path, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
    return full_path


def decode_to_pil_image(encoded):
    if encoded.startswith("data:image/"):
        encoded = encoded.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def decode_to_pil_images(encoded_images):
    images = []
    for idx, encoded in enumerate(encoded_images):
        image = decode_to_pil_image(encoded)
        images.append(image)
    return images


def prepare_image_tensor(image):
    if isinstance(image, str):
        if image.startswith('http'):
            image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
        else:
            image = decode_to_pil_image(image)

    return F.to_tensor(image)


def img_tensor_paste(a, b, x1, y1, channels=None):
    assert len(a.shape) >= 3, f"Expected at least 3 dimensions for tensor 'a', got {len(a.shape)}"
    assert len(b.shape) == 3 or len(b.shape) == 2, f"Expected [2 or 3] dimensions for tensor 'b', got {len(a.shape)}"

    channels = channels if not channels is None else list(range(a.shape[-3]))
    if len(b.shape) == 3:
        assert a.shape[-3] == b.shape[-3] or len(channels) == b.shape[-3] or b.shape[
            -3] == 1, "tensors a and b must have the same number of channels or 'b' 1."

    _h, _w = b.shape[-2], b.shape[-1]
    h = _h if y1 + _h < a.shape[-2] else a.shape[-2] - y1
    w = _w if x1 + _w < a.shape[-1] else a.shape[-1] - x1

    if len(b.shape) == 3 and a.shape[-3] == b.shape[-3]:
        a[..., channels, y1:y1 + h, x1:x1 + w] = b[channels, :h, :w]
    else:
        a[..., channels, y1:y1 + h, x1:x1 + w] = b[..., :h, :w]

    return a
