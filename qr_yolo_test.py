"""Webcam test: scan a QR code (MODE_1 / MODE_2), then detect only that mode's
items with YOLO and show a live checklist. No Dobot involved.

    .venv/bin/python -I qr_yolo_test.py [camera_index]
Keys: q = quit, r = forget mode and scan again.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import yaml
from ultralytics import YOLO
import time
from kitting_logic.qr_logic import decode_qr, normalize_mode
from kitting_logic.notify_logic import build_notifier, format_completion_message

cfg = yaml.safe_load(open("config/checklists.yaml"))
modes = cfg["modes"]
conf = {k: v["confidence"] for k, v in cfg["item_classes"].items()}
model = YOLO(cfg["yolo"]["weights"])

cap = cv2.VideoCapture(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
notifier = build_notifier("config/secrets.yaml", min_interval=10)  # real email if secrets.yaml is filled in
mode, frame_n = None, 0
complete_frames, mail_sent = 0, False  # need 5 frames in a row before "all found" (stops flicker)
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame_n += 1
    if frame_n % 3 == 0:  # QR scan every few frames
        found = normalize_mode(decode_qr(frame))
        if found in modes and found != mode:
            mode = found
            complete_frames, mail_sent = 0, False
            print("Mode set:", mode)

    if mode is None:
        cv2.putText(frame, "Show QR code (MODE_1 / MODE_2)...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    else:
        required = modes[mode]
        r = model.predict(frame, conf=0.25, imgsz=320, verbose=False, device="cpu")[0]
        seen = {}
        for b in r.boxes:
            name, c = model.names[int(b.cls)], float(b.conf)
            if name in required and c >= conf.get(name, 0.5):
                seen[name] = max(seen.get(name, 0), c)
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {c:.2f}", (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{mode.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        for i, item in enumerate(required):
            ok_item = item in seen
            cv2.putText(frame, f"{'[x]' if ok_item else '[ ]'} {item}", (10, 60 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if ok_item else (0, 0, 255), 2)
        complete_frames = complete_frames + 1 if all(i in seen for i in required) else 0
        if complete_frames >= 5:
            cv2.putText(frame, "ALL ITEMS FOUND", (10, 60 + 28 * len(required) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
            if not mail_sent:
                mail_sent = True
                print("All items found for", mode, "- sending email")
                notifier.maybe_send(format_completion_message(mode), now=time.time())

    cv2.imshow("QR + YOLO test (q quit, r reset)", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("r"):
        mode = None
        complete_frames, mail_sent = 0, False
cap.release()
cv2.destroyAllWindows()
