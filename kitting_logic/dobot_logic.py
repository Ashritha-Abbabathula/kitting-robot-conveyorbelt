import math
"""
Arm control logic — no ROS dependency, and no hardware required tonight.

Two interchangeable controllers, same interface:
  - SimulatedDobot: just logs what it would do. Use this tonight to prove
    the whole pipeline reacts correctly once a kit is complete.
  - RealDobot: talks to the actual arm over USB using pydobot. Use this
    tomorrow once you're at the lab and have jogged the arm to find your
    real coordinates (fill them into config/coordinates.yaml).

Both are used the exact same way by the ROS node, so swapping one for the
other tomorrow is a one-line config change, not a code change.
"""
from typing import List, Dict


class SimulatedDobot:
    """Logs intended actions instead of moving anything. Safe to run anywhere, anytime."""

    def __init__(self, port: str = "SIMULATED"):
        self.port = port
        self.connected = False

    def connect(self):
        self.connected = True
        print(f"[SIM] Connected to Dobot on '{self.port}' (simulated)")

    def home(self):
        print("[SIM] Homing arm")

    def move_to(self, x: float, y: float, z: float, r: float = 0.0):
        print(f"[SIM] Move to x={x}, y={y}, z={z}, r={r}")

    def set_suction(self, on: bool):
        print(f"[SIM] Suction {'ON' if on else 'OFF'}")

    def set_conveyor(self, on: bool, speed: float = 0.0, index: int = 0):
        print(f"[SIM] Conveyor {'ON at speed ' + str(speed) if on else 'OFF'}")

    def disconnect(self):
        self.connected = False
        print("[SIM] Disconnected")


# Pose the arm reports right after its home command (measured 2026-10-13).
# Used ONLY to verify a home actually happened; never as a saved "home".
HARDWARE_HOME_XYZ = (265.6, -3.6, -4.3)
HOME_TOLERANCE_MM = 10.0


class FrameOffsetArm:
    """Wraps an arm and makes every target RELATIVE to the start spot.

    The user parks the arm at the start spot; the pose it reports there (start_now) is compared with the
    start-spot pose saved in coordinates.yaml (start_saved). Every target is then sent shifted by
    (start_now - start_saved), so the arm works relative to wherever its own frame puts the start spot.
    Pure translation: exact if the arm's frame is only shifted, not rotated."""

    def __init__(self, arm):
        self._arm = arm
        self.dx = self.dy = self.dz = 0.0

    def set_offset(self, dx: float, dy: float, dz: float):
        self.dx, self.dy, self.dz = dx, dy, dz

    def move_to(self, x: float, y: float, z: float, r: float = 0.0):
        self._arm.move_to(x + self.dx, y + self.dy, z + self.dz, r)

    def __getattr__(self, name):  # alarms, set_suction, pose_xyzr, disconnect ...
        return getattr(self._arm, name)


class RealDobot:
    """
    Thin wrapper around the `pydobot` library. Talks to the arm directly
    over its USB serial connection — this deliberately does NOT depend on
    Dobot's official ROS driver package, since that requires you to build
    and match its exact ROS service names, which is unverified in this
    setup. pydobot is a small, stable, well-documented library that
    controls the exact same arm over the exact same USB cable, so this is
    the lower-risk path for a one-day build. (Nothing stops you from
    switching to the official ROS driver's services later if you want to
    — that would only mean changing THIS file, nothing else.)
    """

    def __init__(self, port: str):
        self.port = port
        self._dobot = None

    def connect(self):
        import pydobot  # imported here so this file loads fine even before
        # pydobot is installed, e.g. while testing SimulatedDobot tonight.

        self._dobot = pydobot.Dobot(port=self.port, verbose=False)
        self._patch_slow_replies()
        self._dobot.speed(70, 70)  # raised from 50 after re-measuring coordinates (2026-10-13)

    def _patch_slow_replies(self):
        """pydobot reads the arm's reply once, 0.1 s after sending, and returns None if nothing arrived yet -
        which later crashes with "'NoneType' object has no attribute 'params'". Keep reading for up to ~2 s."""
        import time
        d = self._dobot
        original = d._read_message

        def read_message():
            for _ in range(20):
                msg = original()
                if msg is not None:
                    return msg
            return None

        d._read_message = read_message

    def home(self):
        # pydobot 1.3.x has no home() helper, but its protocol table has
        # SET_HOME_CMD (31). Send it queued and wait for it to finish.
        if hasattr(self._dobot, "home"):
            self._dobot.home()
            return
        import struct
        from pydobot.message import Message
        from pydobot.enums.CommunicationProtocolIDs import CommunicationProtocolIDs
        msg = Message()
        msg.id = CommunicationProtocolIDs.SET_HOME_CMD
        msg.ctrl = 0x03  # write + queued
        msg.params = bytearray(struct.pack("I", 0))
        self._dobot._send_command(msg, wait=True)

    def alarms(self) -> bytes:
        """Raw 16-byte alarm bitfield from the arm (all zero = no alarm)."""
        from pydobot.message import Message
        msg = Message()
        msg.id = 20  # GET_ALARMS_STATE
        msg.ctrl = 0x00
        return bytes(self._dobot._send_command(msg, wait=False).params)

    def pose_xyzr(self):
        return tuple(self._dobot.pose()[:4])

    def move_to(self, x: float, y: float, z: float, r: float = 0.0):
        self._dobot.move_to(x, y, z, r, wait=True)

    def set_suction(self, on: bool):
        self._dobot.suck(on)

    def set_conveyor(self, on: bool, speed: float = 0.0, index: int = 0):
        """Run the belt on the arm's stepper port `index` (0 = Stepper 1, 1 = Stepper 2).
        speed is in pulses/s (sign = direction). pydobot 1.3.x has no conveyor call, so send
        SetEMotor (id 135: uint8 index, uint8 enable, int32 speed) directly, immediate (not queued)."""
        import struct
        from pydobot.message import Message
        msg = Message()
        msg.id = 135  # SET_E_MOTOR
        msg.ctrl = 0x01  # write, immediate
        msg.params = bytearray(struct.pack("<BBi", index, 1 if on else 0, int(speed) if on else 0))
        self._dobot._send_command(msg, wait=False)

    def disconnect(self):
        if self._dobot is not None:
            self._dobot.close()


def run_pick_and_place(dobot, sequence: List[Dict]):
    """
    Executes a taught pick-and-place sequence, e.g.:
        [
          {"action": "move", "x": 200, "y": 0, "z": 50, "r": 0},
          {"action": "suck", "on": true},
          {"action": "move", "x": 200, "y": 100, "z": 80, "r": 0},
          {"action": "suck", "on": false},
        ]
    Same function, same sequence format, works against SimulatedDobot
    tonight and RealDobot tomorrow — only the coordinates in
    config/coordinates.yaml need to change.
    """
    for step in sequence:
        action = step["action"]
        if action == "move":
            dobot.move_to(step["x"], step["y"], step["z"], step.get("r", 0.0))
        elif action == "suck":
            dobot.set_suction(step["on"])
        elif action == "wait":
            import time
            time.sleep(step["seconds"])
        elif action == "conveyor":
            dobot.set_conveyor(step["on"], step.get("speed", 0.0))
        elif action == "home":
            dobot.home()
        else:
            raise ValueError(f"Unknown action in sequence: {action}")


def build_sequence(required_items, coords, mode=None):
    """
    Builds a move/suck/move/suck sequence for every required item and
    returns home at the end. Uses config/coordinates.yaml.

    If `coords["place"][mode]` exists, it is an ordered list of
    {item, x, y, z, r} targets (e.g. red left / blue right, or a stack with
    the base item first). Items are picked from `pickup[item]`, falling back
    to `pickup["default"]`. Without a `place` entry for the mode, it falls
    back to the old behaviour: one bin per item class from `bins`.

    Shared by dobot_node.py and demo_integration.py so both run the exact
    same moves.
    """
    home = coords["home"]
    pickups = coords["pickup"]
    hover = coords.get("hover_height_offset", 40)

    placements = coords.get("place", {}).get(mode)
    if placements is None:
        bins = coords["bins"]
        placements = [dict(bins.get(item, home), item=item) for item in required_items]
    else:
        placements = [p for p in placements if p["item"] in required_items]

    seq = [{"action": "home"}]
    for target in placements:
        pickup = pickups.get(target["item"], pickups["default"])
        seq += [
            {"action": "move", "x": pickup["x"], "y": pickup["y"], "z": pickup["z"] + hover, "r": pickup["r"]},
            {"action": "move", "x": pickup["x"], "y": pickup["y"], "z": pickup["z"], "r": pickup["r"]},
            {"action": "suck", "on": True},
            {"action": "move", "x": pickup["x"], "y": pickup["y"], "z": pickup["z"] + hover, "r": pickup["r"]},
            {"action": "move", "x": target["x"], "y": target["y"], "z": target["z"] + hover, "r": target["r"]},
            {"action": "move", "x": target["x"], "y": target["y"], "z": target["z"], "r": target["r"]},
            {"action": "suck", "on": False},
            {"action": "move", "x": target["x"], "y": target["y"], "z": target["z"] + hover, "r": target["r"]},
        ]
    seq.append({"action": "home"})
    return seq
