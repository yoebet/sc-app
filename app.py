import os
import time
import traceback
from pprint import pformat
import torch
from flask import Flask, jsonify, request, abort
from dotenv import dotenv_values
from scapp.task_executor import TaskExecutor
from scapp.common import trans_unit

app = Flask(__name__)

app.config.from_mapping(dotenv_values())
app.config.from_mapping(dotenv_values('.env.local'))
device_index = os.environ.get('DEVICE_INDEX')
if device_index is not None and device_index != '':
    app.config.update(device_index=int(device_index))

logger = app.logger

task_executor = TaskExecutor(app.config, logger=logger)


@app.route('/', methods=('GET',))
def index():
    return 'ok'


@app.before_request
def before_request_callback():
    path = request.path
    if path != '/':
        auth = request.headers.get('AUTHORIZATION')
        if not auth == app.config['AUTHORIZATION']:
            abort(400)


@app.route('/check_mem_all/available', methods=('GET',))
def check_mem_all():
    unit = request.args.get('unit')
    import accelerate
    d = accelerate.utils.get_max_memory()
    pairs = [(i, trans_unit(n, unit)) for i, n in d.items()]
    return jsonify(pairs)


@app.route('/check_mem/<device_index>', methods=('GET',))
def check_device_mem(device_index):
    device_index = int(device_index)
    unit = request.args.get('unit')
    free, total = torch.cuda.mem_get_info(device_index)
    return jsonify({
        'free': trans_unit(free, unit),
        'total': trans_unit(total, unit),
    })


def _gen_images(task_type=None):
    req = request.get_json()
    logger.info(pformat(req))
    params = req.get('task')
    launch_params = req.get('launch')
    if launch_params is None:
        launch_params = {}

    fn_name = task_type
    if task_type is None:
        task_type = params.get('task_type')
        fn_name = 'img_gen'
    else:
        params['task_type'] = task_type

    generated_task_id = False
    task_id = params.get('task_id')
    if task_id is None:
        task_id = str(int(time.time()))
        generated_task_id = True
    task_params = {
        'task_type': task_type,
        'task_id': task_id,
        'sub_dir': params.get('sub_dir', None),
        'prompt': params.get('prompt', None),
        'negative_prompt': params.get('negative_prompt', ''),
        'seed': params.get('seed', 0),
        'width': params.get('width', 1024),
        'height': params.get('height', 1024),
        'batch_size': params.get('batch_size', 1),
        'prior_num_inference_steps': params.get('steps', 20),
        'prior_guidance_scale': params.get('guidance_scale', 4.0),
        'decoder_num_inference_steps': params.get('steps2', 10),
        'decoder_guidance_scale': params.get('guidance_scale2', 1.1),
        # 'return_images_format': 'base64',
    }
    if task_type != 'txt2img':
        task_params['image'] = params.get('image', None)
    if task_type == 'inpaint':
        task_params = {
            **task_params,
            'mask': params.get('mask', None),
            'mask_invert': params.get('mask_invert', False),
            'auto_mask_threshold': params.get('auto_mask_threshold', 0.2),
        }
    elif task_type == 'outpaint':
        task_params['outpaint_ext'] = params.get('outpaint_ext', [64])

    try:
        fn = getattr(task_executor, fn_name)
        result = fn(task_params, launch_params)
    except Exception as e:
        traceback.print_exc()
        task_id = task_params.get('task_id')
        logger.error(f"task {task_id} ({task_type}) failed: {type(e)}: {e}")
        return jsonify({
            'success': False,
            'error_message': str(e)
        })

    if generated_task_id:
        result['task_id'] = task_id
    return jsonify(result)


@app.route('/task/txt2img', methods=('POST',))
def txt2img():
    return _gen_images('txt2img')


@app.route('/task/img2img', methods=('POST',))
def img2img():
    return _gen_images('img2img')


@app.route('/task/img_variate', methods=('POST',))
def img_variate():
    return _gen_images('img_variate')


@app.route('/task/img_gen', methods=('POST',))
def img_gen():
    return _gen_images()


def get():
    return app


if __name__ == '__main__':
    app.run()
