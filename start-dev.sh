#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
export HF_HOME=/data/huggingface
export HF_HUB_OFFLINE=1
source /data/projects/sc/venv/bin/activate
export SCRIPT_NAME=/sc
export FLASK_DEBUG=1
export DEVICE_INDEX=5
gunicorn -w 1 --threads=3 --log-level debug --timeout 300 -b 0.0.0.0:8008 "app:get()"
