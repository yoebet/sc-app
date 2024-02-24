import torch
from scapp.common import get_device_name
from scapp.runner_sc import RunnerSc


class TaskExecutor:
    def __init__(self):
        self.runners = {}

    def txt2img(self, task_params, launch_params):
        device_name = get_device_name(launch_params)
        device = torch.device(device_name)
        runner = self.runners.get(device_name)
        if runner is None:
            runner = RunnerSc(device)
            self.runners[device_name] = runner
        return runner.txt2img(**task_params)
