import torch
from scapp.common import get_device_name
from scapp.runner_sc import RunnerSc
from scapp.runner_df import RunnerDf


class TaskExecutor:
    def __init__(self, app_config):
        self.app_config = app_config
        self.runners = {}
        self.runners_df = {}

    def get_runner(self, launch_params):
        device_name = get_device_name(launch_params)
        type = launch_params.get('runner', 'sc')
        runners = self.runners if type == 'sc' else self.runners_df
        runner = runners.get(device_name)
        if runner is None:
            device = torch.device(device_name)
            if type == 'df':
                runner = RunnerDf(self.app_config, device)
            else:
                runner = RunnerSc(self.app_config, device)
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
