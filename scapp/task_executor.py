import torch
from scapp.common import get_device_name
from scapp.runner_sc import RunnerSc
from scapp.runner_df import RunnerDf


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

    def txt2img_df(self, task_params, launch_params):
        device_name = get_device_name(launch_params)
        device = torch.device(device_name)
        runner = self.runners.get(device_name)
        if runner is None:
            runner = RunnerDf(device)
            self.runners[device_name] = runner
        return runner.txt2img(**task_params)


if __name__ == '__main__':
    task_params = {
        'prompt': 'The sprirt of a Tamagotchi wandering in the city of Los Angeles',
        'negative_prompt': '',
        'seed': 0,
        'width': 1024,
        'height': 1024,
        'batch_size': 2,
        'prior_num_inference_steps': 20,
        'prior_guidance_scale': 4.0,
        'decoder_num_inference_steps': 10,
        'decoder_guidance_scale': 1.1,
        'return_images_format': 'pil',
    }
    launch_params = {
        'device_index': 6
    }

    executor = TaskExecutor()
    result = executor.txt2img(task_params, launch_params)
    if not result.get('success'):
        print(result.get('error_message'))
    else:
        images = result.get('images')
        for idx, image in enumerate(images):
            image.save(f't2i-{idx}.png')
