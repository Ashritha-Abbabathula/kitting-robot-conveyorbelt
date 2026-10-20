#!/bin/bash
# One-click real run: asks you to home the arm by hand + Enter, then opens the QR scanner and runs.
cd "$(dirname "$0")" && exec .venv/bin/python -I kitting_run.py --real "$@"
