import os
import re
from pprint import pformat
import base64
import shutil
from io import BytesIO
import json
import torch
from flask import Flask, jsonify, request, Response, abort, send_file
from dotenv import dotenv_values
from launch import launch
from scapp.common import get_task_dir

app = Flask(__name__)

app.config.from_mapping(dotenv_values())
app.config.from_mapping(dotenv_values('.env.local'))

logger = app.logger


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


def trans_unit(bytes, unit):
    if unit is None:
        return bytes
    k = 1024
    m = k * k
    div = {'B': 1, 'K': k, 'M': m, 'G': k * m}.get(unit.upper())
    return bytes / div


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


@app.route('/launch/txt2img', methods=('POST',))
def launch_task():
    req = request.get_json()
    logger.info(pformat(req))
    task_params = req.get('params')
    launch_params = req.get('launch')
    if launch_params is None:
        launch_params = {}
    # launch_params['task_type'] = 'txt2img'

    try:
        launch_result = launch(app.config, task_params, launch_params, logger=logger)
    except Exception as e:
        logger.error(e)
        return jsonify({
            'success': False,
            'error_message': f"[launch] {type(e)}: {e}"
        })

    return jsonify(launch_result)


@app.route('/task/<task_id>/<sub_dir>/result', methods=('GET',))
def get_result_info(task_id, sub_dir):
    TASKS_DIR = app.config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task_id, sub_dir)

    result_file = os.path.join(task_dir, 'result.json')
    if not os.path.exists(result_file):
        return jsonify({
            'success': False,
            'error_message': 'no result file'
        })

    return send_file(result_file, mimetype='application/json', as_attachment=False)


@app.route('/task/<task_id>/<sub_dir>/result_file/<filename>', methods=('GET',))
def get_result_file(task_id, sub_dir, filename: str):
    TASKS_DIR = app.config['TASKS_DIR']
    task_dir = get_task_dir(TASKS_DIR, task_id, sub_dir)

    if filename.startswith('/') or '..' in filename:
        abort(404)

    path = os.path.join(task_dir, filename)
    if not os.path.exists(path):
        abort(404)

    return send_file(path)


def get():
    return app


if __name__ == '__main__':
    app.run()
