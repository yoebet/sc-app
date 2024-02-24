import os
from threading import Thread, Lock
import yaml
import torch
from tqdm import tqdm
import base64
import numpy as np
from inference.utils import *
from .fifo_lock import FIFOLock

MAX_WAITING = 1

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1536


class RunnerBase:
    def __init__(self, device):
        self.device = device
        self.models_loaded = False
        self.queue_lock = FIFOLock()

    def wrap_queued_call(self, func):
        def f(*args, **kwargs):
            with self.queue_lock:
                if not self.models_loaded:
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info(self.device)
                        if free < 20 * (1024 ** 3):
                            return {
                                'success': False,
                                'error_message': 'device occupied',
                            }
                    self.load_models()
                    self.models_loaded = True
                res = func(*args, **kwargs)
            return res

        return f

    def load_models(self):
        ...

    def _txt2img(self, **params):
        ...

    def txt2img(self, **params):
        if self.queue_lock.pending_count >= MAX_WAITING:
            return {
                'success': False,
                'error_message': f"busy"
            }
        result = self.wrap_queued_call(self._txt2img)(**params)
        # print(result)
        return result
