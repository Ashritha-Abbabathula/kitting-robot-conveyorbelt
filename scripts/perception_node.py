#!/usr/bin/env python3
"""
ROS1 perception node for the kitting robot.

Features:
- Captures live video from the camera.
- Displays the camera feed in an OpenCV window.
- Detects QR codes and switches task modes.
- Detects required objects using the configured YOLO detector.
- Publishes task mode, required items, missing items, and completion status.
- Press Q in the camera window to stop the node.
"""

import json
import os
import sys

import cv2
import rospy

from std_msgs.msg import String, Bool


# Import the existing project logic.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

from kitting_logic.flow_logic import StableTracker
from kitting_logic.qr_logic import decode_qr, normalize_mode
from kitting_logic.checklist_logic import (
    load_checklists,
    get_required_items,
)
from kitting_logic.vision_logic import (
    find_missing_items,
    build_detector,
    DetectorNotConfigured,
)


def main():
    rospy.init_node("perception_node")

    # ROS parameters.
    camera_index = rospy.get_param("~camera_index", 0)

    config_path = rospy.get_param(
        "~checklist_config",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "config",
            "checklists.yaml",
        ),
    )

    rate_hz = rospy.get_param("~rate_hz", 5.0)

    # All items must stay in view this long before /verification_complete goes True.
    tracker = StableTracker(rospy.get_param("~hold_seconds", 2.0))

    # Load checklist configuration.
    checklists = load_checklists(config_path)

    import yaml

    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file) or {}

    item_classes = config.get("item_classes", {})

    # Locate YOLO weights.
    weights_path = os.path.abspath(
        os.path.join(
            os.path.dirname(config_path),
            "..",
            config.get("yolo", {}).get(
                "weights",
                "models/best.pt",
            ),
        )
    )

    secrets_path = os.path.join(
        os.path.dirname(config_path),
        "secrets.yaml",
    )

    # Initialize the detector.
    try:
        detector = build_detector(
            config,
            secrets_path,
            weights_path,
        )

    except DetectorNotConfigured as error:
        rospy.logerr(
            "perception_node: detector configuration error: %s",
            error,
        )
        return

    # ROS publishers.
    mode_pub = rospy.Publisher(
        "/task_mode",
        String,
        queue_size=1,
        latch=True,
    )

    required_pub = rospy.Publisher(
        "/required_items",
        String,
        queue_size=1,
        latch=True,
    )

    missing_pub = rospy.Publisher(
        "/missing_items",
        String,
        queue_size=1,
        latch=True,
    )

    complete_pub = rospy.Publisher(
        "/verification_complete",
        Bool,
        queue_size=1,
        latch=True,
    )

    # Open the VirtualBox webcam using the Linux V4L2 backend.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if not cap.isOpened():
        rospy.logerr(
            "perception_node: could not open camera index %s",
            camera_index,
        )
        cap.release()
        return

    # The VirtualBox webcam supports MJPG.
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Create the camera display window.
    window_name = "Kitting Robot Camera"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        window_name,
        960,
        540,
    )

    kit = {"phase": "detect", "text": "", "done": [], "current": None}

    def status_callback(msg):
        kit.update(json.loads(msg.data))

    rospy.Subscriber("/kit_status", String, status_callback, queue_size=1)

    current_mode = None
    required_items = []
    missing_items = []

    rate = rospy.Rate(rate_hz)

    rospy.loginfo(
        "perception_node: camera opened successfully."
    )

    rospy.loginfo(
        "perception_node: watching for a QR code..."
    )

    try:
        while not rospy.is_shutdown():

            # Read a frame from the camera.
            ok, frame = cap.read()

            if not ok:
                rospy.logwarn_throttle(
                    5,
                    "perception_node: camera frame read failed",
                )

                rate.sleep()
                continue

            # Work on a copy so the original image is used
            # for QR detection and object detection.
            display_frame = frame.copy()

            # Always look for a QR code.
            raw = decode_qr(frame)
            mode = normalize_mode(raw)

            # Switch task modes when a valid QR code is found.
            if (
                mode
                and mode != current_mode
                and mode in checklists
            ):
                current_mode = mode

                required_items = get_required_items(
                    checklists,
                    current_mode,
                )

                # Initially consider every required item missing.
                missing_items = list(required_items)

                rospy.loginfo(
                    "perception_node: mode switched to %s, "
                    "requires %s",
                    current_mode,
                    required_items,
                )

                mode_pub.publish(
                    String(data=current_mode)
                )

                required_pub.publish(
                    String(
                        data=json.dumps(required_items)
                    )
                )

                tracker.reset()

                complete_pub.publish(
                    Bool(data=False)
                )

            # Run object detection once a mode is selected.
            if current_mode is not None:

                missing_items = find_missing_items(
                    detector,
                    frame,
                    required_items,
                    item_classes,
                )

                missing_pub.publish(
                    String(
                        data=json.dumps(missing_items)
                    )
                )

                is_complete = tracker.update(len(missing_items) == 0, time.time())

                complete_pub.publish(
                    Bool(data=is_complete)
                )

            # Display task information on the camera image.
            if current_mode is None:
                status_text = "Scan a task QR code"
            else:
                status_text = "Task: " + current_mode

            cv2.putText(
                display_frame,
                status_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if current_mode is not None:
                busy = kit["phase"] != "detect"
                for i, item in enumerate(required_items):
                    if item in kit["done"]:
                        mark, col = "[x]", (0, 255, 0)
                    elif item not in missing_items and not busy:
                        mark, col = "[+]", (0, 255, 0)
                    elif item == kit["current"] and busy:
                        mark, col = "[>]", (0, 200, 255)
                    else:
                        mark, col = "[ ]", (0, 0, 255)
                    cv2.putText(display_frame, f"{mark} {item}", (15, 65 + 28 * i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
                row_y = 65 + 28 * len(required_items) + 8
                if busy:
                    col = (0, 0, 255) if kit["phase"] == "moving" else (0, 255, 0)
                    text = kit["text"]
                elif not missing_items:
                    col = (0, 255, 0)
                    text = f"all items seen - hold still ({tracker.progress(time.time()) * 100:.0f}%)"
                else:
                    col = (0, 200, 255)
                    text = "Waiting for: " + ", ".join(missing_items)
                cv2.putText(display_frame, text[:80], (15, row_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

            cv2.putText(
                display_frame,
                "Press Q to quit",
                (15, display_frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Show the live camera feed.
            cv2.imshow(
                window_name,
                display_frame,
            )

            # Required for the OpenCV window to refresh.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                rospy.loginfo(
                    "perception_node: Q pressed; stopping."
                )
                break

            rate.sleep()

    except rospy.ROSInterruptException:
        pass

    finally:
        # Release the camera and close OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()

        rospy.loginfo(
            "perception_node: camera released and windows closed."
        )


if __name__ == "__main__":
    main()
