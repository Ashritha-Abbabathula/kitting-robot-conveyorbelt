"""
Record real arm positions by hand-guiding, one label at a time.

Usage:  python tools/record_poses.py [--port /dev/dobot] [--out config/measured_poses.json]

For each label: hold the arm's unlock button, guide the cup to the spot,
let go, press Enter. The live pose is read from the arm and saved. This
never moves the arm and never edits coordinates.yaml - review the JSON,
then copy the numbers over.
"""
import argparse
import json
import os
import time

import pydobot
from pydobot.message import Message

LABELS = [
    ("pickup_block_top",  "PICKUP spot: cup just touching the TOP of a red/blue block"),
    ("pickup_cube_top",   "PICKUP spot: cup just touching the TOP of the Rubik's cube"),
    ("place_left",        "LEFT drop (MODE_1 red): cup holding the block at the height it should release"),
    ("place_right",       "RIGHT drop (MODE_1 blue): cup holding the block at the height it should release"),
    ("stack_cube",        "MODE_2 STACK spot: cup holding the cube at the height it should release"),
    ("stack_red_on_cube", "MODE_2 STACK spot: cup holding the red block ON TOP of the cube, release height"),
]


def alarms(d):
    m = Message()
    m.id, m.ctrl, m.params = 20, 0, bytearray()
    b = bytes(d._send_command(m).params)
    return [i * 8 + k for i, x in enumerate(b) for k in range(8) if x >> k & 1]


ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/dobot")
ap.add_argument("--out", default="config/measured_poses.json")
args = ap.parse_args()

d = pydobot.Dobot(port=args.port, verbose=False)
d.suck(False)
saved = {}
if os.path.exists(args.out):
    saved = json.load(open(args.out))
try:
    print("Active alarms:", alarms(d) or "none")
    for key, text in LABELS:
        old = saved.get(key)
        ans = input(f"\n{text}\n  [{key}] guide the arm there, then Enter "
                    f"(s = skip{', keep ' + str(old) if old else ''}, q = quit): ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        x, y, z, r = d.pose()[:4]
        saved[key] = {"x": round(x, 1), "y": round(y, 1), "z": round(z, 1),
                      "radius": round((x * x + y * y) ** 0.5, 1)}
        print(f"  saved {key}: {saved[key]}")
        json.dump(saved, open(args.out, "w"), indent=2)
finally:
    d.close()
print("\nDone ->", args.out)
