import torch
import logging
from scapp.common import get_device_name
from scapp.runner_sc import RunnerSc
from scapp.runner_df import RunnerDf


class TaskExecutor:
    def __init__(self, app_config, logger: logging.Logger = None):
        self.app_config = app_config
        if logger is None:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
            )
            self.logger = logging.getLogger('launch')
        else:
            self.logger = logger
        self.runners = {}
        self.runners_df = {}

    def get_runner(self, launch_params):
        device_index = self.app_config['device_index']
        if device_index is not None:
            device_name = f'cuda:{device_index}'
        else:
            device_name = get_device_name(launch_params)
        self.logger.info(f'device: {device_name}')
        type = launch_params.get('runner', 'sc')
        runners = self.runners if type == 'sc' else self.runners_df
        runner = runners.get(device_name)
        if runner is None:
            device = torch.device(device_name)
            if type == 'df':
                runner = RunnerDf(device, self.app_config, logger=self.logger)
            else:
                runner = RunnerSc(device, self.app_config, logger=self.logger)
            runners[device_name] = runner
        return runner

    def txt2img(self, task_params, launch_params):
        runner = self.get_runner(launch_params)
        return runner.txt2img(**task_params)

    def img2img(self, task_params, launch_params):
        runner = self.get_runner(launch_params)
        return runner.img2img(**task_params)

    def img_variate(self, task_params, launch_params):
        runner = self.get_runner(launch_params)
        return runner.img_variate(**task_params)
