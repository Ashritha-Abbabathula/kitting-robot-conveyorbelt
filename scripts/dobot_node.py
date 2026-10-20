#!/usr/bin/env python3
"""
ROS1 node: watches /verification_complete and /task_mode. When the camera has
seen every item of the mode held still, it waits ~start_delay s (put the first
item on the pickup spot), moves each item (waiting ~wait s between items), then
parks the arm back above the pickup spot.

Same moves as kitting_run.py (it uses kitting_moves.py): every spot is RELATIVE
to the pickup spot. On the real arm the pickup spot is the pose the arm reports
when this node starts -> park the arm BY HAND above the pickup spot BEFORE
launching. No home command is ever sent.

Defaults to SIMULATION (simulate:=true). Real arm: simulate:=false.
Services (std_srvs/Trigger):
  /dobot_node/reset   re-read the pickup pose from the arm and allow the kit to run again
"""
import json
import os
import sys
import threading
import time

import rospy
import yaml
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger, TriggerResponse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import kitting_moves
from kitting_logic.dobot_logic import SimulatedDobot, RealDobot

_state = {"mode": None, "already_ran_for": None, "running": False, "arm": None, "coords": None,
          "pickup": None, "real": False, "start_delay": 10.0, "wait": 5.0}
_lock = threading.Lock()


def read_pickup():
    """Real arm: the pose it reports now IS the pickup spot. Simulation: the saved one."""
    arm = _state["arm"]
    if not _state["real"]:
        _state["pickup"] = kitting_moves.saved_pickup_xy(_state["coords"])
        return
    al = arm.alarms()
    if any(al):
        arm.set_suction(False)
        raise RuntimeError(f"arm has active alarms ({al.hex()}); power-cycle it first")
    px, py, pz, _ = arm.pose_xyzr()
    _state["pickup"] = (px, py)
    rospy.loginfo(f"dobot_node: pickup spot taken from the arm: x={px:.1f} y={py:.1f} (z={pz:.1f} not used)")


def publish_status(phase, text, done, current=None):
    """Latest kit state for the camera window (perception_node draws it)."""
    _state["status_pub"].publish(String(data=json.dumps(
        {"phase": phase, "text": text, "done": list(done), "current": current})))


def countdown(seconds, item, done, phase, label):
    """Wall-clock countdown, published every 0.5 s so the camera window can show it."""
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0 or rospy.is_shutdown():
            return
        publish_status(phase, f"{label} - arm moves in {left:.0f} s. Put {item} at the pickup point",
                       done, item)
        time.sleep(min(0.5, left))


def mode_callback(msg):
    if msg.data != _state["mode"]:
        _state["already_ran_for"] = None   # a new QR code -> allow a new kit
        publish_status("detect", "", [])
    _state["mode"] = msg.data


def complete_callback(msg):
    if not msg.data:
        return
    with _lock:
        mode = _state["mode"]
        if mode is None or mode == _state["already_ran_for"] or _state["running"]:
            return
        if mode not in kitting_moves.MODES:
            rospy.logwarn(f"dobot_node: unknown mode {mode}")
            return
        _state["running"] = True
    done = []
    try:
        items = kitting_moves.item_order(mode)
        rospy.loginfo(f"dobot_node: all items for {mode} seen - arm starts in {_state['start_delay']:.0f} s")
        for i, item in enumerate(items):
            t0 = time.time()
            if i == 0:
                countdown(_state["start_delay"], item, done, "countdown", "ALL COMPONENTS FOUND")
            else:
                countdown(_state["wait"], item, done, "waiting", "Next item")
            rospy.loginfo(f"dobot_node: waited {time.time() - t0:.1f} s, picking {item}")
            publish_status("moving", f"MOVING: {item}", done, item)
            kitting_moves.move_item(_state["arm"], mode, item, _state["pickup"], _state["coords"])
            done.append(item)
        kitting_moves.return_to_pickup(_state["arm"], _state["pickup"], _state["coords"])
        publish_status("done", "KIT COMPLETE - arm back at the pickup point", done)
        rospy.loginfo(f"dobot_node: kit complete for {mode} - arm back at the pickup point")
    except Exception as e:  # never leave suction on
        rospy.logerr(f"dobot_node: run failed: {e}")
        _state["arm"].set_suction(False)
        publish_status("error", f"RUN FAILED: {e}", done)
    finally:
        _state["already_ran_for"] = mode
        _state["running"] = False


def reset_service(_req):
    if _state["running"]:
        return TriggerResponse(success=False, message="arm is running")
    try:
        read_pickup()
    except RuntimeError as e:
        return TriggerResponse(success=False, message=str(e))
    _state["already_ran_for"] = None
    return TriggerResponse(success=True, message=f"pickup = {_state['pickup']}")


def main():
    rospy.init_node("dobot_node")

    simulate = rospy.get_param("~simulate", True)
    port = rospy.get_param("~port", "/dev/ttyUSB0")
    _state["start_delay"] = rospy.get_param("~start_delay", 10.0)
    _state["wait"] = rospy.get_param("~wait", 5.0)
    coords_path = rospy.get_param("~coords_config", os.path.join(HERE, "..", "config", "coordinates.yaml"))
    with open(coords_path) as f:
        _state["coords"] = yaml.safe_load(f)

    if simulate:
        rospy.logwarn("dobot_node: SIMULATION mode - no real arm movement. Use simulate:=false for the real arm.")
        _state["arm"] = SimulatedDobot()
    else:
        if not _state["coords"].get("calibrated", False):
            rospy.logfatal("dobot_node: config/coordinates.yaml has calibrated: false - refusing to drive the real arm.")
            sys.exit(1)
        _state["real"] = True
        _state["arm"] = RealDobot(port=port)

    _state["arm"].connect()
    _state["arm"].set_suction(False)
    try:
        read_pickup()   # no home command: the arm must already be parked by hand at the pickup spot
    except RuntimeError as e:
        rospy.logfatal(f"dobot_node: {e}")
        _state["arm"].disconnect()
        sys.exit(1)

    _state["status_pub"] = rospy.Publisher("/kit_status", String, queue_size=1, latch=True)
    rospy.Subscriber("/task_mode", String, mode_callback, queue_size=1)
    rospy.Subscriber("/verification_complete", Bool, complete_callback, queue_size=1)
    rospy.Service("~reset", Trigger, reset_service)
    rospy.on_shutdown(lambda: (_state["arm"].set_suction(False), _state["arm"].disconnect()))

    rospy.loginfo("dobot_node: ready")
    rospy.spin()


if __name__ == "__main__":
    main()
