#!/bin/bash
TURBO=/data/turbo-on
if [ -f "$TURBO" ]; then
    source $TURBO
fi
source /data/cascade/venv/bin/activate
export SCRIPT_NAME=/sc
nohup gunicorn -w 6 --log-level debug --timeout 120 -b 0.0.0.0:9005 "app:get()" &
