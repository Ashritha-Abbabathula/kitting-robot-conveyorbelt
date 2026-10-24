"""Run the conveyor on the Dobot for a few seconds so you can measure its speed (SIMULATED unless --real).

    .venv/bin/python -I tools/belt_test.py --seconds 3 --speed 5000           # prints only
    .venv/bin/python -I tools/belt_test.py --real --stepper 0 --seconds 3 --speed 5000

Put a block on the belt, mark its start, run it, measure how far it moved: mm_per_s = distance_mm / seconds.
Put that and the speed into config/checklists.yaml (batch.belt). Try a negative --speed if it runs the wrong way.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kitting_logic.dobot_logic import RealDobot, SimulatedDobot

ap = argparse.ArgumentParser()
ap.add_argument("--real", action="store_true")
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--stepper", type=int, default=0, choices=[0, 1])
ap.add_argument("--speed", type=int, default=5000, help="pulses/s, negative reverses")
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()

arm = RealDobot(a.port) if a.real else SimulatedDobot()
arm.connect()
try:
    arm.set_conveyor(True, a.speed, a.stepper)
    time.sleep(a.seconds)
finally:
    arm.set_conveyor(False, 0, a.stepper)
    arm.disconnect()
print(f"ran {a.seconds} s at {a.speed} pulses/s - measure the distance the block moved")
