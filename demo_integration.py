#!/usr/bin/env python3
"""
Runs the WHOLE pipeline end-to-end, right now, with no ROS and no
hardware — so you can actually watch it work tonight.

This is NOT how it runs tomorrow (tomorrow, five separate ROS nodes pass
messages to each other). This script just calls the same functions those
nodes will call, one after another in a single process, so you can see
the full chain of decisions with your own eyes before any ROS wiring
exists.

Two ways to run it:

    python3 demo_integration.py            # fully synthetic — no camera needed, runs anywhere
    python3 demo_integration.py --webcam   # REAL webcam for both the QR code AND item checking

In --webcam mode, nothing is simulated except the arm: the QR code is
read from actual camera frames, and item presence is decided by running
a YOLOv8 model (config/checklists.yaml's item_classes) against the camera
feed. There is no timeout and no fallback — it displays "No QR code"
and does nothing until a real one is shown, then "No objects are seen"
and does nothing until real items actually appear on camera. Exactly one
email goes out for the whole run, the moment everything's present.
Press 'q' in a video window any time to cancel the run yourself.

Watch the printed output — it walks through exactly what each "node"
would be thinking at each moment.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from kitting_logic.qr_logic import decode_qr, normalize_mode
from kitting_logic.checklist_logic import load_checklists, get_required_items
from kitting_logic.vision_logic import find_missing_items, build_detector, DetectorNotConfigured
from kitting_logic.notify_logic import format_missing_message, format_completion_message, build_notifier
from kitting_logic.dobot_logic import SimulatedDobot, run_pick_and_place, build_sequence

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKLIST_PATH = os.path.join(HERE, "config", "checklists.yaml")
COORDS_PATH = os.path.join(HERE, "config", "coordinates.yaml")

CAMERA_INDEX = 0
# No timeouts, no automatic fallback: both waits below run forever until
# the real condition is actually met. Press 'q' in the video window if
# you want to cancel a run yourself — that's a manual quit, not a fallback.


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def make_colored_frame(h, w, bgr):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


class _ScriptedDetector:
    """
    Stands in for a real detector in the no-camera demo. A trained
    detector needs an actual photo of an actual item to say anything useful
    — a solid-colour rectangle won't do — so instead this just reports
    whatever items `currently_present` says are on the table, each with
    confidence 1.0, translated to their configured class_name (the same
    translation item_classes does for a real detector) so find_missing_items'
    class-name lookup matches. That's enough to walk through the same
    logic the real webcam/ROS path uses, without needing a camera, real
    items, or a trained model.
    """

    def __init__(self, item_classes):
        self._item_classes = item_classes
        self.currently_present = []

    def detect(self, frame):
        return {
            self._item_classes.get(name, {}).get("class_name", name): 1.0
            for name in self.currently_present
        }


def _mode_display_text(mode):
    """'mode_1' -> 'Mode 1 QR code detected' — derived from whatever mode
    string comes back, nothing hardcoded to a specific mode number."""
    label = mode.replace("_", " ").strip().title()  # "mode_1" -> "Mode 1"
    return f"{label} QR code detected"


def get_qr_from_webcam(cap):
    """
    Reads REAL frames from an already-open camera and does nothing —
    literally nothing else happens — until a QR code is actually decoded
    from one of them. No timeout, no fallback, no guessed mode. Press 'q'
    if you want to cancel the run yourself; that returns None, and the
    caller stops rather than inventing a mode.
    """
    print("Waiting for a QR code... (press 'q' to cancel)")
    last_status = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        raw = decode_qr(frame)
        display = frame.copy()

        if raw:
            mode = normalize_mode(raw)
            status = _mode_display_text(mode) if mode else f"Unrecognized QR: {raw}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Show your QR code here", display)
            if status != last_status:
                print(f"  -> {status}")
            cv2.waitKey(400)  # brief pause so you can see it locked on
            return raw

        status = "No QR code"
        if status != last_status:
            print(f"  -> {status}")
            last_status = status
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Show your QR code here", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None


def check_items_from_webcam(cap, mode, required_items, item_classes, detector, notifier):
    """
    Reads REAL frames from an already-open camera and repeatedly checks
    them against config/checklists.yaml's YOLO classes via the same
    find_missing_items() used elsewhere in the project — nothing here is
    simulated or pre-decided, and there's no timeout: it waits as long as
    it takes for the real items to actually appear. Exactly ONE email goes
    out for this whole wait — the moment every required item is confirmed
    present, saying the kit is complete and execution is starting. Nothing
    is emailed while items are still missing.
    Returns True once every required item is present, or False if you
    cancel with 'q'.
    """
    print("Waiting for objects... (press 'q' to cancel)")
    last_status = None
    completion_email_sent = False

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        missing = find_missing_items(detector, frame, required_items, item_classes)
        present = [item for item in required_items if item not in missing]

        if not present:
            status = "No objects are seen"
        elif missing:
            status = "Objects detected"
        else:
            status = "Objects detected — all items present"

        if status != last_status:
            print(f"  -> {status}  (present: {present or '(none)'}, missing: {missing or '(none)'})")
            last_status = status

        # Exactly one email for the whole run: fires the moment nothing is
        # missing anymore. No emails at all while items are still missing.
        if not missing and not completion_email_sent:
            notifier.maybe_send(format_completion_message(mode), now=time.time())
            completion_email_sent = True
            print("  -> all items present — sent the one completion email")

        display = frame.copy()
        color = (0, 255, 0) if present else (0, 0, 255)
        cv2.putText(display, f"Mode: {mode}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display, f"Missing: {', '.join(missing) or '-'}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Item check — real camera feed", display)

        if not missing:
            cv2.waitKey(500)  # brief pause so you can see "all items present" before moving on
            return True
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False


def synthetic_qr_frame():
    """Loads the bundled MODE_1.png — a real, scannable QR code encoding
    MODE_1. (Generating one with cv2.QRCodeEncoder needs OpenCV >= 4.5.3,
    which ROS Noetic's system OpenCV 4.2 doesn't have.)"""
    frame = cv2.imread(os.path.join(HERE, "MODE_1.png"))
    if frame is None:
        raise FileNotFoundError(f"Could not read {os.path.join(HERE, 'MODE_1.png')}")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam", action="store_true", help="use a real webcam for the QR step")
    args = parser.parse_args()

    checklists = load_checklists(CHECKLIST_PATH)
    with open(CHECKLIST_PATH) as f:
        config = yaml.safe_load(f) or {}
    item_classes = config.get("item_classes", {})
    with open(COORDS_PATH) as f:
        coords = yaml.safe_load(f)

    secrets_path = os.path.join(HERE, "config", "secrets.yaml")
    notifier = build_notifier(secrets_path, min_interval=10)

    if args.webcam:
        # ---- Fully real path: real QR, real item checking, nothing pretended ----
        weights_path = os.path.join(HERE, config.get("yolo", {}).get("weights", "models/best.pt"))
        try:
            detector = build_detector(config, secrets_path, weights_path)
        except DetectorNotConfigured as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_V4L2
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if not cap.isOpened():
            print(f"ERROR: could not open camera index {CAMERA_INDEX}. Exiting.")
            sys.exit(1)

        try:
            banner("STEP 1 — perception_node: reading the QR code (REAL webcam)")
            raw_text = get_qr_from_webcam(cap)
            if raw_text is None:
                print("Cancelled — no QR code was shown before you pressed 'q'.")
                sys.exit(1)

            mode = normalize_mode(raw_text)
            print(f"  -> decoded text: {raw_text!r}")
            print(f"  -> normalized mode: {mode!r}")

            if mode not in checklists:
                print(f"ERROR: QR decoded to {raw_text!r} (normalized: {mode!r}), which "
                      f"isn't a mode in config/checklists.yaml (available: "
                      f"{list(checklists.keys())}). Exiting.")
                sys.exit(1)

            banner("STEP 2 — perception_node: looking up the checklist for this mode")
            required_items = get_required_items(checklists, mode)
            print(f"  -> required items for {mode}: {required_items}")

            banner("STEP 3/4 — perception_node checks the REAL camera feed, notify_node reports gaps")
            complete = check_items_from_webcam(cap, mode, required_items, item_classes, detector, notifier)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if not complete:
            print("\nCancelled before every item was detected — NOT running the arm.")
            sys.exit(1)

    else:
        # ---- Fully synthetic path: no camera needed, runs anywhere ----
        banner("STEP 1 — perception_node: reading the QR code (synthetic)")
        frame = synthetic_qr_frame()
        raw_text = decode_qr(frame)
        mode = normalize_mode(raw_text)
        print(f"  -> decoded text: {raw_text!r}")
        print(f"  -> normalized mode: {mode!r}")
        if mode not in checklists:
            print(f"ERROR: synthetic QR decoded to {raw_text!r}, which isn't a mode in "
                  f"config/checklists.yaml (available: {list(checklists.keys())}). If it "
                  f"decoded to None, this Python has no working QR decoder — install "
                  f"pyzbar, or run inside .venv. Exiting.")
            sys.exit(1)

        banner("STEP 2 — perception_node: looking up the checklist for this mode")
        required_items = get_required_items(checklists, mode)
        print(f"  -> required items for {mode}: {required_items}")

        banner("STEP 3/4 — perception_node checks the table (synthetic), notify_node reports gaps")
        complete = False
        detector = _ScriptedDetector(item_classes)
        frame = make_colored_frame(200, 400, (40, 40, 40))  # ignored by _ScriptedDetector — just a
        # placeholder frame, since a real YOLO model needs an actual photo of an actual item, not a
        # solid-colour rectangle, to detect anything.
        for round_num in range(len(required_items) + 1):
            present_so_far = required_items[:round_num]
            print(f"\n  -- check #{round_num + 1}: pretend these are currently on the table: "
                  f"{present_so_far or '(nothing yet)'}")

            detector.currently_present = present_so_far
            missing = find_missing_items(detector, frame, required_items, item_classes)
            print(f"     -> [would publish to /missing_items]: {json.dumps(missing)}")
            complete = len(missing) == 0
            if complete:
                notifier.maybe_send(format_completion_message(mode), now=time.time())
            print(f"     -> [would publish to /verification_complete]: {complete}")
            if complete:
                break

        if not complete:
            print("\nVerification did not complete — NOT running the arm.")
            sys.exit(1)

    # ---- Step 5: dobot_node's job — execute only now that everything is confirmed present ----
    banner("STEP 5 — dobot_node: checklist complete, executing pick-and-place (SIMULATED)")
    dobot = SimulatedDobot()
    dobot.connect()
    dobot.home()
    sequence = build_sequence(required_items, coords, mode)
    run_pick_and_place(dobot, sequence)
    dobot.disconnect()

    banner("DONE")


if __name__ == "__main__":
    main()