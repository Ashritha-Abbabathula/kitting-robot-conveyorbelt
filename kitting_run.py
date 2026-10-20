"""Camera-driven kitting run: webcam QR picks MODE_1 / MODE_2, YOLO checks the items,
and the Dobot moves them to their place targets.

Default flow (--flow all), e.g. MODE_1:
    1. Camera looks until ALL required items (red + blue) are in view.
    2. Real email "Got all components ... executing now".
    3. 10 s countdown (--start-delay): put the first block on the pickup point.
    4. Arm picks it and places it (red -> LEFT).
    5. 5 s wait (--wait): put the next block on the pickup point.
    6. Arm picks it and places it (blue -> RIGHT).
Order comes from config/coordinates.yaml `place[mode]` (MODE_2: cube first, then red on top).

Alternative (--flow each): no countdown/timers; the arm moves each item as soon as the camera
sees that one item held still at the pickup point.

    .venv/bin/python -I kitting_run.py --mode mode_1            # SIMULATED arm, real email
    .venv/bin/python -I kitting_run.py --mode mode_1 --no-email # simulated, no email
    ./run_kitting.sh                                            # REAL arm, QR scanner

Real arm: at start-up and at the start of every new QR scan cycle (r key, or a new QR after a finished kit) it asks you
to HOME the robot: move the arm BY HAND above the PICKUP spot (any height) and press Enter. Only then does the QR scanner open; the
code reads the pose, compares it with the pickup saved in coordinates.yaml and shifts every target by the difference
(all actions are relative to the pickup position). No home command is sent and the arm
does not move before that. Nothing about "homed" is remembered.

Keys: q quit (suction off) | r forget mode, re-home and start over.
The arm blocks while it moves, so q is not read mid-move — keep a hand near the arm's power switch.
"""
import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)  # config paths below are relative to the repo

import threading

import cv2
import yaml
from ultralytics import YOLO

from kitting_logic.dobot_logic import RealDobot, SimulatedDobot
from kitting_logic.flow_logic import StableTracker
import kitting_moves
from kitting_logic.notify_logic import build_notifier, format_completion_message
from kitting_logic.qr_logic import decode_qr, normalize_mode

REACH_WARN_MM = 300.0  # Dobot Magician max reach is ~320 mm; measured MODE_2 spot is ~309 mm

ap = argparse.ArgumentParser()
ap.add_argument("camera", nargs="?", type=int, default=0)
ap.add_argument("--real", action="store_true", help="move the REAL arm (default: simulated)")
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--mode", choices=["mode_1", "mode_2"], help="skip the QR scan and force a mode")
ap.add_argument("--flow", choices=["all", "each"], default="all",
                help="all: detect all items -> email -> countdown -> timed pick/place (default); "
                     "each: move each item as soon as it is seen at the pickup point")
ap.add_argument("--hold", type=float, default=1.0, help="seconds items must stay in view before it counts")
ap.add_argument("--start-delay", type=float, default=5.0, help="--flow all: seconds from email to first pick")
ap.add_argument("--wait", type=float, default=2.0, help="--flow all: seconds between items")
ap.add_argument("--no-email", action="store_true", help="do NOT send the real completion email")
args = ap.parse_args()

cfg = yaml.safe_load(open("config/checklists.yaml"))
coords = yaml.safe_load(open("config/coordinates.yaml"))
modes = cfg["modes"]
conf_for = {k: v["confidence"] for k, v in cfg["item_classes"].items()}
model = YOLO(cfg["yolo"]["weights"])

# ---- camera first, so a missing webcam never leaves the arm half-started ----
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    sys.exit(f"Cannot open camera {args.camera}")

# ---- arm ----
if args.real:
    if not coords.get("calibrated"):
        sys.exit("Refusing to move the real arm: `calibrated: false` in config/coordinates.yaml. "
                 "Verify every coordinate on the arm, then set it to true.")
    arm = RealDobot(args.port)
    arm.connect()
    arm.set_suction(False)
else:
    arm = SimulatedDobot()
    arm.connect()


PICKUP_XY = kitting_moves.saved_pickup_xy(coords)  # replaced by the arm's own reading on the real arm


def ensure_homed():
    """REAL arm only. Never sends a home command and never moves the arm before you confirm.
    Asks you to park the arm at the PICKUP spot and reads the x/y it reports there. That reading IS the pickup
    spot; left/right/stack spots are pickup + fixed (dx, dy) from kitting_moves.py. Heights stay as saved.
    Runs at start-up and at the start of every new QR scan cycle; nothing is remembered."""
    global PICKUP_XY
    if not args.real:
        return
    al = arm.alarms()
    if any(al):
        arm.set_suction(False)
        sys.exit(f"Arm has active alarms ({al.hex()}). Power-cycle it and re-run before moving.")
    input("\nPlease set the Dobot to the PICKUP position: move the arm BY HAND to the pickup spot, "
          "keep it still, then press Enter (the QR scanner opens after this): ")
    px, py, pz, _ = arm.pose_xyzr()
    PICKUP_XY = (px, py)
    print(f"Pickup spot taken from the arm: x={px:.1f} y={py:.1f} (z={pz:.1f} not used; heights stay as saved)")


ensure_homed()  # before the camera loop / QR scan starts

notifier = None if args.no_email else build_notifier("config/secrets.yaml", min_interval=10)


def start_mode(m):
    queue = kitting_moves.item_order(m)
    print(f"Mode set: {m}  flow: {args.flow}  order: {' -> '.join(queue)}")
    for it in queue:
        rad = kitting_moves.max_radius(kitting_moves.item_steps(m, it, PICKUP_XY, coords))
        print(f"  {it}: farthest move {rad:.0f} mm from base" + ("  ** NEAR REACH LIMIT **" if rad > REACH_WARN_MM else ""))
    return queue


def put(frame, text, row, color=(255, 255, 0), scale=0.7):
    cv2.putText(frame, text, (10, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def detect(frame, wanted):
    """Run YOLO once; return {class: best_conf} for `wanted` classes above their threshold, and draw boxes."""
    res = model.predict(frame, conf=0.25, imgsz=320, verbose=False, device="cpu")[0]
    seen = {}
    for b in res.boxes:
        name, c = model.names[int(b.cls)], float(b.conf)
        if name in wanted and c >= conf_for.get(name, 0.5):
            seen[name] = max(seen.get(name, 0.0), c)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {c:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return seen


def execute(item, mode, frame, row):
    print(f"Picking {item}")
    put(frame, f"MOVING: {item}", row, (0, 0, 255), 0.9)
    cv2.imshow(WINDOW, frame)
    cv2.waitKey(1)
    kitting_moves.move_item(arm, mode, item, PICKUP_XY, coords)
    for _ in range(5):  # drop frames buffered while the arm was moving
        cap.grab()


def send_email(mode, now):
    if notifier is None:
        print("(email disabled)")
        return
    print("All components found - sending email (in background)")
    # SMTP can block for many seconds; keep it off the camera/arm loop.
    threading.Thread(target=notifier.maybe_send, args=(format_completion_message(mode), now),
                     daemon=True).start()


WINDOW = "kitting run (q quit, r reset)"
mode = args.mode
queue = start_mode(mode) if mode else []
done = []
phase = "detect"            # detect -> countdown -> running <-> waiting -> done   (flow all)
phase_until = 0.0           # deadline for countdown / waiting
tracker = StableTracker(args.hold)
mail_sent = False
frame_n = 0


def reset(new_mode=None, rehome=False):
    global mode, queue, done, phase, mail_sent
    if rehome:
        ensure_homed()  # a new QR scan cycle: home again
    mode = new_mode
    queue = start_mode(mode) if mode else []
    done, phase, mail_sent = [], "detect", False
    tracker.reset()


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        frame_n += 1
        now = time.time()

        # QR may (re)select the mode only while the arm is idle (before the run starts, or after it).
        idle = (not done and phase == "detect") or phase == "done"
        if idle and not args.mode and frame_n % 3 == 0:
            found = normalize_mode(decode_qr(frame))
            if found in modes and found != mode:
                reset(found, rehome=(phase == "done"))

        if mode is None:
            put(frame, "Show QR code (MODE_1 / MODE_2)...", 0, (0, 200, 255))
        else:
            required = kitting_moves.item_order(mode)
            put(frame, f"{mode.upper()}  [{'REAL' if args.real else 'SIM'}]  flow={args.flow}", 0)
            current = queue[0] if queue else None
            seen = {}
            if args.flow == "all" and phase == "detect":
                seen = detect(frame, required)
            for i, it in enumerate(required):
                if it in done:
                    mark, col = "[x]", (0, 255, 0)
                elif it in seen:
                    mark, col = "[+]", (0, 255, 0)      # seen by the camera, not moved yet
                elif it == current and phase != "detect":
                    mark, col = "[>]", (0, 200, 255)
                else:
                    mark, col = "[ ]", (0, 0, 255)
                put(frame, f"{mark} {it}", 1 + i, col)
            row = 2 + len(required)

            if args.flow == "all":
                if phase == "detect":
                    all_in = all(it in seen for it in required)
                    if tracker.update(all_in, now):
                        send_email(mode, now)
                        phase, phase_until = "countdown", now + args.start_delay
                        print(f"Arm starts in {args.start_delay:.0f} s - put {queue[0]} at the pickup point")
                    elif all_in:
                        put(frame, f"all items seen - hold still ({tracker.progress(now) * 100:.0f}%)", row, (0, 255, 0))
                    else:
                        missing = [it for it in required if it not in seen]
                        put(frame, f"Waiting for: {', '.join(missing)}", row, (0, 200, 255))
                elif phase == "countdown":
                    put(frame, f"ALL COMPONENTS FOUND - arm starts in {max(0, phase_until - now):.0f} s", row, (0, 255, 0), 0.8)
                    put(frame, f"Put {queue[0]} at the pickup point", row + 1, (0, 200, 255))
                    if now >= phase_until:
                        phase = "running"
                elif phase == "waiting":
                    put(frame, f"Put {queue[0]} at the pickup point - arm moves in {max(0, phase_until - now):.0f} s",
                        row, (0, 200, 255))
                    if now >= phase_until:
                        phase = "running"
                if phase == "running":
                    item = queue[0]
                    execute(item, mode, frame, row)
                    done.append(queue.pop(0))
                    if queue:
                        phase, phase_until = "waiting", time.time() + args.wait
                        print(f"Waiting {args.wait:.0f} s - put {queue[0]} at the pickup point")
                    else:
                        kitting_moves.return_to_pickup(arm, PICKUP_XY, coords)
                        phase = "done"
                        print(f"Kit complete for {mode} - arm back at pickup point")
                if phase == "done":
                    put(frame, "KIT COMPLETE (r to start over)", row, (0, 255, 0), 0.9)

            else:  # --flow each: move an item as soon as the camera sees that one item held still
                if current is None:
                    put(frame, "KIT COMPLETE (r to start over)", row, (0, 255, 0), 0.9)
                    if not mail_sent:
                        mail_sent = True
                        print(f"Kit complete for {mode}")
                        send_email(mode, now)
                else:
                    seen = detect(frame, [current])
                    present = current in seen
                    if not present:
                        put(frame, f"Place {current} at the pickup spot", row, (0, 200, 255))
                    else:
                        put(frame, f"{current} seen - hold still ({tracker.progress(now) * 100:.0f}%)", row, (0, 255, 0))
                    if tracker.update(present, now):
                        execute(current, mode, frame, row)
                        done.append(queue.pop(0))
                        tracker.reset()
                        if not queue:
                            kitting_moves.return_to_pickup(arm, PICKUP_XY, coords)

        cv2.imshow(WINDOW, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            reset(args.mode, rehome=True)
finally:
    try:
        arm.set_suction(False)
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
