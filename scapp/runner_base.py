import logging
import os
import gc
import numpy as np
from inference.utils import *
from .fifo_lock import FIFOLock
from .common import get_task_dir, trans_unit

MAX_WAITING = 1

MAX_SEED = np.iinfo(np.int32).max
# MAX_IMAGE_SIZE = 1536

MIN_FREE_GPU_MEM_G = 20


class RunnerBase:
    def __init__(self, device, app_config, logger: logging.Logger):
        self.device = device
        self.app_config = app_config
        self.logger = logger
        self.models_loaded = False
        self.queue_lock = FIFOLock()

    def build_results(self,
                      task_type: str,
                      pil_images,
                      task_id: str,
                      sub_dir: str = None,
                      file_name_part=None,
                      return_images_format: str = 'base64'):
        if pil_images is None:
            return {
                'success': False
            }

        if sub_dir is None:
            sub_dir = task_type
        else:
            sub_dir = f'{sub_dir}/{task_type}'
        task_dir = get_task_dir(self.app_config['TASKS_DIR'], task_id, sub_dir)
        os.makedirs(task_dir, exist_ok=True)
        for idx, img in enumerate(pil_images):
            file_base = f'output_{idx + 1}'
            if file_name_part is not None:
                file_base = f'{file_base}-{file_name_part}'
            img.save(os.path.join(task_dir, f'{file_base}.png'))

        if return_images_format == 'pil':
            images = pil_images
        else:
            images = pils_to_base64(pil_images)

        return {
            'success': True,
            'images': images
        }

    def load_models(self):
        ...

    def _txt2img(self, **params):
        ...

    def _img2img(self, **params):
        ...

    def _img_variate(self, **params):
        ...

    def wrap_queued_call(self, func):
        def f(*args, **kwargs):
            with self.queue_lock:
                if not self.models_loaded:
                    if torch.cuda.is_available():
                        free, total = torch.cuda.mem_get_info(self.device)
                        if free < MIN_FREE_GPU_MEM_G * (1024 ** 3):
                            free_t, total_t = trans_unit(free, 'G'), trans_unit(total, 'G')
                            self.logger.error(f'device occupied: free {free_t:.02f} G, total {total_t:.02f} G')
                            return {
                                'success': False,
                                'error_message': 'device occupied',
                            }
                    self.load_models()
                    self.models_loaded = True
                res = func(*args, **kwargs)
                gc.collect()
                torch.cuda.empty_cache()
            return res

        return f

    def _run(self, fn, params):
        if self.queue_lock.pending_count >= MAX_WAITING:
            return {
                'success': False,
                'error_message': f"busy"
            }

        target = self.wrap_queued_call(fn)

        result = target(**params)
        if result is None:
            task_id = params.get('task_id', '?')
            self.logger.error(f'task {task_id} failed.')
            raise Exception('task failed.')

        task_id = params.get('task_id', '?')
        self.logger.info(f'task {task_id} finished.')
        return result

    def txt2img(self, **params):
        return self._run(self._txt2img, params=params)

    def img2img(self, **params):
        return self._run(self._img2img, params=params)

    def img_variate(self, **params):
        return self._run(self._img_variate, params=params)
