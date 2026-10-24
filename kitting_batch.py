"""Batch-of-3 kitting run: QR picks the mode, ONE photo of the belt shows 2 required blocks + 1 reject,
and the Dobot (arm + belt) works through the batch. The reject is never picked; the belt keeps running
and it rolls off the end.

    .venv/bin/python -I kitting_batch.py --mode mode_1      # SIMULATED arm and belt, QR skipped
    .venv/bin/python -I kitting_batch.py                    # SIMULATED, QR scanner
    .venv/bin/python -I kitting_batch.py --real             # REAL arm: asks you to fix the arm + Enter, then QR scanner

Flow: (real arm: 'Please fix the arm' + Enter; its position is then fixed for the run. --park-seconds N = timed window instead)
QR -> --countdown s (30) to place the blocks, live camera -> photo -> plan (leading block first, see kitting_logic/batch_logic.py) ->
for each block: belt advances it onto the pickup point; required block = belt stops, arm picks and places it
(MODE_1: red -> left, blue -> right); reject = belt keeps running. Photo not showing exactly the required items
plus rejects -> nothing moves. Belt settings (stepper port, speed, mm/s) live in config/checklists.yaml `batch`.
An email goes out for every valid batch (\"got all components - executing now\") and when a batch is refused for
being out of order. MODE_2 (cube + red on top): the cube must come BEFORE the red block (leading), otherwise nothing moves and an
email "not in order of height" is sent (--no-email to disable). Show the other mode's QR during a countdown to switch mode. Keys: SPACE take the photo now, r new QR, q quit. After each batch the countdown starts again.
"""
import argparse
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

import kitting_moves
from kitting_logic import batch_logic
from kitting_logic.batch_logic import BatchError, OrderError
from kitting_logic.blob_logic import find_unknown_blocks
from kitting_logic.notify_logic import build_notifier, format_batch_message
from kitting_logic.dobot_logic import RealDobot, SimulatedDobot
from kitting_logic.qr_logic import decode_qr, normalize_mode

SUPPORTED = ["mode_1", "mode_2"]

ap = argparse.ArgumentParser()
ap.add_argument("camera", nargs="?", type=int, default=0)
ap.add_argument("--real", action="store_true", help="move the REAL arm and belt (default: simulated)")
ap.add_argument("--port", default="/dev/ttyUSB0")
ap.add_argument("--mode", choices=SUPPORTED, help="skip the QR scan and force a mode")
ap.add_argument("--park-seconds", type=float, default=0.0,
                help="REAL: 0 (default) = ask 'please fix the arm' and wait for Enter; >0 = a timed window of this many "
                     "seconds instead (no Enter, for running unattended). Either way the arm's position is then FIXED "
                     "as the pickup spot for the whole run")
ap.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)   # old option, no longer needed (there is no MOVE prompt)
ap.add_argument("--toward", choices=["left", "right", "up", "down"], help="direction the belt carries blocks in the camera image (default: config)")
ap.add_argument("--no-email", action="store_true", help="do not email when a batch is refused for being out of order")
ap.add_argument("--slots", help="skip the photo: batch order, leading block first, e.g. red_block,reject,blue_block "
                                "(for testing the belt and arm moves)")
ap.add_argument("--image", help="use this photo instead of the camera (testing; runs once)")
ap.add_argument("--countdown", type=float, default=30.0, help="seconds to place the blocks before each photo (SPACE = photo now)")
args = ap.parse_args()

cfg = yaml.safe_load(open("config/checklists.yaml"))
coords = yaml.safe_load(open("config/coordinates.yaml"))
bcfg = cfg["batch"]
belt = bcfg["belt"]
conf_for = {k: v["confidence"] for k, v in cfg["item_classes"].items()}
model = YOLO(cfg["yolo"]["weights"])

cap = cv2.VideoCapture(args.camera)
if not (args.image or args.slots) and not cap.isOpened():
    sys.exit(f"Cannot open camera {args.camera}")

if args.real:
    if not coords.get("calibrated"):
        sys.exit("Refusing to move the real arm: `calibrated: false` in config/coordinates.yaml.")
    arm = RealDobot(args.port)
    arm.connect()
    arm.set_suction(False)
else:
    arm = SimulatedDobot()
    arm.connect()

PICKUP_XY = kitting_moves.saved_pickup_xy(coords)
notifier = None if args.no_email else build_notifier("config/secrets.yaml", min_interval=10)


def park_at_pickup():
    """REAL arm only: --park-seconds to move the arm BY HAND onto the pickup point (live camera + countdown), then the
    arm's own reading becomes the pickup x/y for the rest of this run (same idea as kitting_run.py)."""
    global PICKUP_XY
    if not args.real:
        return
    al = arm.alarms()
    if any(al):
        arm.set_suction(False)
        sys.exit(f"Arm has active alarms ({al.hex()}). Power-cycle it and re-run before moving.")
    if args.park_seconds <= 0:
        input("\nPlease FIX THE ARM: move it by hand onto the pickup point, hold it still, then press Enter "
              "(the QR scanner opens after this): ")
    else:
        print(f"Move the arm BY HAND onto the PICKUP point now - {args.park_seconds:.0f} s, then it is fixed for this run", flush=True)
        end = time.time() + args.park_seconds
        pose, last_read = (0.0, 0.0, 0.0), 0.0
        while True:
            now = time.time()
            if now - last_read >= 0.3:                       # live reading, shown on screen so you can see what will be fixed
                pose, last_read = arm.pose_xyzr()[:3], now
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok:
                left = max(0.0, end - now)
                put(frame, f"SET THE ARM on the pickup point: {left:4.1f} s", 0, (0, 200, 255))
                put(frame, f"arm now: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}  (x,y get fixed)", 1, (0, 255, 0))
                cv2.imshow(WINDOW, frame)
                cv2.waitKey(1)
            else:
                time.sleep(0.1)
            if now >= end:
                break
    # wait (up to 5 s extra) until the arm has stopped moving, then fix the average of a short burst of readings
    for attempt in range(6):
        reads = []
        for _ in range(5):
            reads.append(arm.pose_xyzr()[:3])
            time.sleep(0.1)
        spread = max(max(r[i] for r in reads) - min(r[i] for r in reads) for i in range(2))
        if spread <= 2.0:
            break
        print(f"arm still moving ({spread:.1f} mm) - hold it still...", flush=True)
        time.sleep(1.0)
    px = sum(r[0] for r in reads) / len(reads)
    py = sum(r[1] for r in reads) / len(reads)
    pz = sum(r[2] for r in reads) / len(reads)
    PICKUP_XY = (px, py)
    print(f"Pickup spot FIXED for this run: x={px:.1f} y={py:.1f} (z={pz:.1f} not used; heights stay as saved)", flush=True)


def put(frame, text, row, color=(255, 255, 0)):
    cv2.putText(frame, text, (10, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["cx"] - a["w"] / 2, a["cy"] - a["h"] / 2, a["cx"] + a["w"] / 2, a["cy"] + a["h"] / 2
    bx1, by1, bx2, by2 = b["cx"] - b["w"] / 2, b["cy"] - b["h"] / 2, b["cx"] + b["w"] / 2, b["cy"] + b["h"] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a["w"] * a["h"] + b["w"] * b["h"] - inter + 1e-9)


def detect(frame):
    """All blocks the model knows, above their confidence threshold. The model is inconsistent between input sizes
    (a small block or the cube is sometimes found at 320 and missed at 640, and vice versa), so run both and merge:
    overlapping boxes (IoU > 0.4) are one object - keep the more confident."""
    thresh = dict(conf_for)
    thresh.update(bcfg.get("min_conf", {}))
    dets = []
    for size in (320, 640):
        res = model.predict(frame, conf=0.25, imgsz=size, verbose=False, device="cpu")[0]
        for b in res.boxes:
            name, c = model.names[int(b.cls)], float(b.conf)
            if c < thresh.get(name, 0.5):
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0])
            dets.append({"name": name, "conf": c, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "w": x2 - x1, "h": y2 - y1})
    dets.sort(key=lambda d: -d["conf"])
    kept = []
    for d in dets:
        if all(iou(d, k) <= 0.4 for k in kept):
            kept.append(d)
    return kept


def take_photo(mode, n=5):
    """Best of n frames (most blocks seen), so one blurry frame doesn't decide the batch."""
    best, best_dets = None, []
    frames = [cv2.imread(args.image)] if args.image else [cap.read()[1] for _ in range(n)]
    for frame in frames:
        if frame is None:
            continue
        dets = batch_logic.select_batch(detect(frame), kitting_moves.item_order(mode))
        dets += find_unknown_blocks(frame, dets)    # coloured blocks the model doesn't know = reject
        if best is None or len(dets) > len(best_dets):
            best, best_dets = frame, dets
    return best, best_dets


def annotate(frame, dets, slots=None):
    for d in dets:
        x1, y1 = int(d["cx"] - d["w"] / 2), int(d["cy"] - d["h"] / 2)
        cv2.rectangle(frame, (x1, y1), (int(d["cx"] + d["w"] / 2), int(d["cy"] + d["h"] / 2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{d['name'].upper() if d['name'] == 'reject' else d['name']} {d['conf']:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def notify_order(mode, err):
    """Out-of-order batch: say so on the console and by email (in a background thread so SMTP can't freeze the loop)."""
    label = mode.replace("_", " ").title()
    body = f"{label}: batch NOT run - {err} [{time.strftime('%H:%M:%S')}]"
    print(f"[NOTIFY] {body}", flush=True)
    if notifier is not None:
        threading.Thread(target=notifier.maybe_send, args=(body, time.time()), daemon=True).start()


def notify_batch(mode, slots):
    """Valid batch: email 'got all components - executing now' (background thread; never blocks or stops the run)."""
    body = format_batch_message(mode, slots, time.strftime("%H:%M:%S"))
    print(f"[NOTIFY] {body}", flush=True)
    if notifier is not None:
        threading.Thread(target=notifier.maybe_send, args=(body, time.time()), daemon=True).start()


def run_batch(mode, frame_for_display):
    required = kitting_moves.item_order(mode)
    if args.slots:
        slots = [None if x == "reject" else x for x in args.slots.split(",")]
        if len(slots) != batch_logic.BATCH_SIZE or sorted(x for x in slots if x) != sorted(required):
            print(f"--slots must be {batch_logic.BATCH_SIZE} entries: {required} plus reject")
            return
        frame = None
        try:    # same order rule as the photo path
            batch_logic.check_order(slots, bcfg.get("ordered", {}).get(mode))
        except OrderError as e:
            print(f"[{mode}] NOT RUNNING: {e}", flush=True)
            notify_order(mode, e)
            time.sleep(1)
            return
        print(f"[{mode}] batch from --slots (leading block first): {[x or 'REJECT' for x in slots]}")
        notify_batch(mode, slots)
        return execute(mode, slots)
    frame, dets = take_photo(mode)
    if frame is None:
        print("Camera gave no frame")
        return
    stamp = time.strftime("%H%M%S")
    os.makedirs("captures/batch/runs", exist_ok=True)
    cv2.imwrite(f"captures/batch/runs/{stamp}_raw.jpg", frame)
    annotate(frame, dets)
    try:
        slots = batch_logic.plan_slots(dets, required, args.toward or bcfg["toward"], bcfg.get("ordered", {}).get(mode))
    except BatchError as e:
        print(f"[{mode}] NOT RUNNING: {e}", flush=True)
        print(f"  saw: {[(d['name'], round(d['cx']), round(d['cy'])) for d in dets]}", flush=True)
        put(frame, f"NOT RUNNING: {e}"[:70], 0, (0, 0, 255))
        if isinstance(e, OrderError):
            notify_order(mode, e)
        cv2.imwrite(f"captures/batch/runs/{stamp}_refused.jpg", frame)
        cv2.imshow(WINDOW, frame)
        cv2.waitKey(2000)
        return
    print(f"[{mode}] batch (leading block first): {[s or 'REJECT' for s in slots]}", flush=True)
    cv2.imwrite(f"captures/batch/runs/{stamp}_ok.jpg", frame)
    put(frame, "batch: " + " | ".join(s or "REJECT" for s in slots), 0, (0, 255, 0))
    cv2.imshow(WINDOW, frame)
    cv2.waitKey(1000)

    notify_batch(mode, slots)
    execute(mode, slots)


def execute(mode, slots):
    steps = batch_logic.batch_steps(slots, belt["mm_per_s"], bcfg.get("sizes_mm"), bcfg.get("reject_mm", 25.0))
    running = False
    try:
        for s in steps:
            if s["action"] == "belt":
                if not running:
                    arm.set_conveyor(True, belt["speed"], belt["stepper"])
                    running = True
                time.sleep(s["seconds"])
            elif s["action"] == "pick":
                arm.set_conveyor(False, 0, belt["stepper"])
                running = False
                print(f"Picking {s['item']}")
                kitting_moves.move_item(arm, mode, s["item"], PICKUP_XY, coords)
            else:
                print("Reject at the pickup point - belt keeps running")
        kitting_moves.return_to_pickup(arm, PICKUP_XY, coords)     # arm back above the pickup point first
        if not running:
            arm.set_conveyor(True, belt["speed"], belt["stepper"])
            running = True
        print(f"Belt rolling on for {belt['flush_seconds']:.0f} s to clear any blocks left")
        time.sleep(belt["flush_seconds"])
        print(f"[{mode}] batch done")
    finally:
        arm.set_conveyor(False, 0, belt["stepper"])
        arm.set_suction(False)


WINDOW = "kitting batch (SPACE photo now, r new QR, q quit)"
park_at_pickup()
once = bool(args.image or args.slots)           # testing options: one batch, then exit
mode = args.mode or ("mode_1" if once else None)
deadline = None      # when the next photo is taken; set once a mode is known, re-armed after every batch
n = 0
try:
    while True:
        if once:
            run_batch(mode, None)
            break
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break
        n += 1
        now = time.time()
        take = False
        if mode is None:
            put(frame, "Show the MODE QR code (MODE_1 or MODE_2)...", 0, (0, 200, 255))
            if n % 3 == 0:
                found = normalize_mode(decode_qr(frame))
                if found in SUPPORTED:
                    mode = found
                    print(f"Mode set: {mode}")
                elif found:
                    print(f"{found} is not built for batches yet")
        else:
            if not args.mode and n % 3 == 0:                 # showing the OTHER mode's QR during a countdown switches mode
                found = normalize_mode(decode_qr(frame))
                if found in SUPPORTED and found != mode:
                    print(f"Mode changed: {mode} -> {found}", flush=True)
                    mode, deadline = found, None
            if deadline is None:
                deadline = now + args.countdown
                print(f"[{mode}] place the 3 blocks - photo in {args.countdown:.0f} s")
            left = deadline - now
            put(frame, f"{mode.upper()} [{'REAL' if args.real else 'SIM'}]: place your blocks", 0, (0, 255, 0))
            put(frame, f"photo in {max(0, left):4.1f} s   (SPACE = now, r = new QR, q = quit)", 1, (0, 200, 255))
            take = left <= 0
        cv2.imshow(WINDOW, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            mode, deadline = args.mode, None
            continue
        if mode is not None and (take or k == ord(" ")):
            run_batch(mode, frame)
            deadline = None             # the next batch gets its own countdown
            for _ in range(5):
                cap.grab()
finally:
    try:
        arm.set_conveyor(False, 0, belt["stepper"])
        arm.set_suction(False)
    finally:
        arm.disconnect()
        cap.release()
        cv2.destroyAllWindows()
