#!/bin/bash
# Batch-of-3 run on the REAL arm. Order: alarm check -> "Please FIX THE ARM" + Enter -> QR scanner (MODE_1 / MODE_2)
# -> 30 s countdown to place the blocks -> photo -> belt + arm -> 7 s belt roll-on -> next countdown.
# No MOVE prompt: after the photo is decided and the email is triggered the arm starts straight away. Show the other QR during a countdown to switch mode. In the camera window:
# SPACE = photo now, r = new QR, q = quit. Extra options pass through, e.g. ./run_batch.sh --no-email
cd "$(dirname "$0")" && exec .venv/bin/python -u -I kitting_batch.py --real "$@"
