"""
Motion-only test for the Dobot — no camera, no ROS, no YOLO.

Picks one block from the shared pickup spot and puts it on the LEFT side,
waits (default 5 s) so you can put the next block at the pickup spot, then
picks it again and puts it on the RIGHT side. Does NOT re-home the arm.

    .venv/bin/python -I tools/dobot_side_test.py --dry-run        # print moves only
    .venv/bin/python -I tools/dobot_side_test.py                  # move the real arm
"""
import argparse
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from kitting_logic.dobot_logic import RealDobot, build_sequence  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--wait", type=float, default=5.0)
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

with open(os.path.join(HERE, "..", "config", "coordinates.yaml")) as f:
    coords = yaml.safe_load(f)


def steps_for(item):
    # build_sequence brackets the run with home actions; drop them (no re-homing).
    return [s for s in build_sequence([item], coords, "mode_1") if s["action"] != "home"]


plan = [("red_block -> LEFT", steps_for("red_block")), ("blue_block -> RIGHT", steps_for("blue_block"))]

if args.dry_run:
    for title, steps in plan:
        print(f"== {title}")
        for s in steps:
            print("  ", s)
    sys.exit(0)

arm = RealDobot(args.port)
arm.connect()
try:
    arm.set_suction(False)
    x, y, z, r = arm._dobot.pose()[:4]
    print(f"start pose: x={x:.1f} y={y:.1f} z={z:.1f}")
    safe_z = max(z, 60.0)
    print(f"lifting straight up to z={safe_z:.1f}")
    arm.move_to(x, y, safe_z, r)

    for n, (title, steps) in enumerate(plan):
        print(f"== {title}")
        for s in steps:
            if s["action"] == "move":
                print(f"   move  x={s['x']} y={s['y']} z={s['z']}")
                arm.move_to(s["x"], s["y"], s["z"], s.get("r", 0.0))
            elif s["action"] == "suck":
                print(f"   suction {'ON' if s['on'] else 'OFF'}")
                arm.set_suction(s["on"])
                time.sleep(1.0 if s["on"] else 0.5)
        if n == 0:
            print(f"-- waiting {args.wait:.0f} s: put the next block at the pickup spot --")
            time.sleep(args.wait)
    print("done")
finally:
    arm.set_suction(False)
    arm.disconnect()
