#!/usr/bin/env python3
"""
ROS1 node: watches /missing_items and /task_mode, and sends an email
whenever the missing-items list changes (rate-limited so it doesn't spam
every camera frame).

Uses a FAKE notifier by default (just prints) unless config/secrets.yaml
exists with real email credentials — so this is safe to launch tonight
without any risk of accidentally trying to send with placeholder creds.
"""
import json
import os
import sys

import rospy
from std_msgs.msg import String, Bool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kitting_logic.notify_logic import format_missing_message, format_completion_message, build_notifier

_state = {"mode": None, "notifier": None, "done_for": None}


def complete_callback(msg):
    """One 'got all components' email per mode (like kitting_run.py)."""
    mode = _state["mode"]
    if msg.data and mode and _state["done_for"] != mode:
        _state["done_for"] = mode
        _state["notifier"].maybe_send(format_completion_message(mode), now=rospy.get_time())


def missing_callback(msg):
    if _state["done_for"] is not None and _state["done_for"] == _state["mode"]:
        return  # kit already complete: items leaving the view while the arm moves are not "missing"
    missing = json.loads(msg.data)
    mode = _state["mode"] or "unknown_mode"
    body = format_missing_message(mode, missing)
    _state["notifier"].maybe_send(body, now=rospy.get_time())


def mode_callback(msg):
    _state["mode"] = msg.data
    if _state["done_for"] != msg.data:
        _state["done_for"] = None


def main():
    rospy.init_node("notify_node")
    secrets_path = rospy.get_param(
        "~secrets_path",
        os.path.join(os.path.dirname(__file__), "..", "config", "secrets.yaml"),
    )
    min_interval = rospy.get_param("~min_interval_seconds", 8.0)
    _state["notifier"] = build_notifier(secrets_path, min_interval=min_interval)

    rospy.Subscriber("/missing_items", String, missing_callback)
    rospy.Subscriber("/task_mode", String, mode_callback)
    rospy.Subscriber("/verification_complete", Bool, complete_callback)

    rospy.loginfo("notify_node: ready")
    rospy.spin()


if __name__ == "__main__":
    main()
