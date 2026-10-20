"""MODE_1 / MODE_2 movement, RELATIVE to the pickup spot.

Call it once the arm is parked at the pickup position and the QR told you the mode:

    from kitting_moves import run_mode, move_item
    run_mode(arm, "mode_1", pickup_xy=(x, y))      # whole kit, red -> left, blue -> right
    move_item(arm, "mode_2", "rubiks_cube", pickup_xy=(x, y))   # one item

`pickup_xy` is the x/y the arm REPORTS while parked at the pickup spot. Every place spot is
the pickup spot swung +-30 deg around the arm base (left / right), same distance from the base.
MODE_2 cube and red block use the SAME x/y; only z differs. Heights: table level from the saved pickup z
(cube/blocks released CLEARANCE above the surface, red on cube = cube z + block height); hover from coordinates.yaml.
(r is always 0.) Tune SWING_DEG at the top of the tables.

Stand-alone test (arm must be parked at the pickup spot; SIMULATED unless --real):
    .venv/bin/python -I kitting_moves.py mode_1
    .venv/bin/python -I kitting_moves.py mode_2 --real      # asks for MOVE first
"""
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import yaml

from kitting_logic.dobot_logic import RealDobot, SimulatedDobot, run_pick_and_place

# Where each item goes: swing around the arm BASE by this many degrees from the pickup spot (same
# distance from the base, so reach never gets worse). +ve = left, -ve = right. MODE_2 base and top share one spot.
SWING_DEG = {
    "mode_1": {"red_block": +30.0, "blue_block": -30.0},
    "mode_2": {"rubiks_cube": -30.0, "red_block": -30.0},
}
MODES = SWING_DEG
BLOCK_HEIGHT = 23.0   # mm, red/blue block (measured: red-on-cube release was 22.9 above cube release)
CLEARANCE = 3.0       # mm, release this far above the surface
CUBE_LOWER = 43.7     # mm, release the Rubik's cube this much lower than the other items
CUBE_PICK_LOWER = 3.0         # mm, MODE_2: pick the Rubik's cube this much lower (pick step only; releases unchanged)
RED_ON_CUBE_LOWER = 46.7      # mm (to z 17.0), MODE_2: release the red block on the cube this much lower


def place_xy(mode, item, pickup_xy):
    a = math.radians(SWING_DEG[mode][item])
    px, py = pickup_xy
    return px * math.cos(a) - py * math.sin(a), px * math.sin(a) + py * math.cos(a)


def place_z(mode, item, c):
    """Table-level release heights from the saved pickup heights (everything sits on the pickup table)."""
    pk = c["pickup"]
    if mode == "mode_2" and item == "red_block":      # on top of the cube
        return pk["rubiks_cube"]["z"] + BLOCK_HEIGHT - RED_ON_CUBE_LOWER
    z = pk.get(item, pk["default"])["z"] + CLEARANCE
    return z - CUBE_LOWER if item == "rubiks_cube" else z


def _coords():
    with open(os.path.join(HERE, "config", "coordinates.yaml")) as f:
        return yaml.safe_load(f)


def saved_pickup_xy(coords=None):
    pk = (coords or _coords())["pickup"]["default"]
    return pk["x"], pk["y"]


def item_order(mode):
    return list(MODES[mode])


def item_steps(mode, item, pickup_xy, coords=None):
    """Move/suck steps for one item. x/y relative to pickup_xy, z as saved in coordinates.yaml."""
    c = coords or _coords()
    px, py = pickup_xy
    hover = c.get("hover_height_offset", 40)
    pick_z = c["pickup"].get(item, c["pickup"]["default"])["z"]
    if mode == "mode_2" and item == "rubiks_cube":
        pick_z -= CUBE_PICK_LOWER      # the release heights below still use the saved pickup z
    pz = place_z(mode, item, c)
    tx, ty = place_xy(mode, item, pickup_xy)
    return [
        {"action": "move", "x": px, "y": py, "z": pick_z + hover},
        {"action": "move", "x": px, "y": py, "z": pick_z},
        {"action": "suck", "on": True},
        {"action": "move", "x": px, "y": py, "z": pick_z + hover},
        {"action": "move", "x": tx, "y": ty, "z": pz + hover},
        {"action": "move", "x": tx, "y": ty, "z": pz},
        {"action": "suck", "on": False},
        {"action": "move", "x": tx, "y": ty, "z": pz + hover},
    ]


def max_radius(steps):
    """Farthest horizontal distance (mm) from the arm base, for a reach warning (arm frame = reported x/y)."""
    return max(math.hypot(s["x"], s["y"]) for s in steps if s["action"] == "move")


def move_item(arm, mode, item, pickup_xy, coords=None):
    run_pick_and_place(arm, item_steps(mode, item, pickup_xy, coords))


def return_to_pickup(arm, pickup_xy, coords=None):
    """Park the arm back above the pickup spot (hover height, suction off) once the kit is finished."""
    c = coords or _coords()
    hover = c.get("hover_height_offset", 40)
    px, py = pickup_xy
    arm.set_suction(False)
    arm.move_to(px, py, c["pickup"]["default"]["z"] + hover, 0.0)


def run_mode(arm, mode, pickup_xy, wait=5.0, coords=None):
    """Move every item of `mode`, waiting `wait` s between items so the next one can be put on the pickup spot."""
    items = item_order(mode)
    for i, item in enumerate(items):
        if i and wait:
            print(f"Put {item} on the pickup spot ... {wait:.0f} s")
            time.sleep(wait)
        print(f"[{mode}] {i + 1}/{len(items)}: {item}")
        move_item(arm, mode, item, pickup_xy, coords)
    return_to_pickup(arm, pickup_xy, coords)
    print(f"[{mode}] done")


def mode_1(arm, pickup_xy, **kw):
    run_mode(arm, "mode_1", pickup_xy, **kw)


def mode_2(arm, pickup_xy, **kw):
    run_mode(arm, "mode_2", pickup_xy, **kw)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=list(MODES))
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--wait", type=float, default=5.0)
    a = ap.parse_args()
    c = _coords()
    if a.real:
        if not c.get("calibrated"):
            sys.exit("Refusing: `calibrated: false` in config/coordinates.yaml.")
        arm = RealDobot(a.port)
        arm.connect()
        arm.set_suction(False)
        if any(arm.alarms()):
            sys.exit("Arm has active alarms; power-cycle it first.")
        px, py, pz, _ = arm.pose_xyzr()
        pickup = (px, py)
        print(f"pickup spot = arm pose x={px:.1f} y={py:.1f} (z={pz:.1f} ignored)")
        if input(f"Type MOVE to run {a.mode} on the real arm: ").strip() != "MOVE":
            sys.exit("Aborted - not moving.")
    else:
        arm = SimulatedDobot()
        arm.connect()
        pickup = saved_pickup_xy(c)
    try:
        run_mode(arm, a.mode, pickup, wait=a.wait, coords=c)
    finally:
        arm.set_suction(False)
