from threading import Thread
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
            rh = kwargs.pop('result_holder')
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
            if rh is not None:
                rh['res'] = res
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

        fn = self.wrap_queued_call(self._txt2img)

        result_holder = {}
        thread = Thread(target=fn, args=[], kwargs={**params, 'result_holder': result_holder})
        thread.start()
        thread.join()
        result = result_holder.get('res')
        # print(result)
        return result
