"""Save raw webcam frames for building a training set.
Usage: .venv/bin/python -I tools/capture_frames.py <out_dir> [seconds=45] [every=0.35] [camera=0]
"""
import os, sys, time
import cv2
out = sys.argv[1]; secs = float(sys.argv[2]) if len(sys.argv) > 2 else 45
every = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
cap = cv2.VideoCapture(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
os.makedirs(out, exist_ok=True)
for _ in range(10): cap.read()
t0 = last = time.time(); n = 0
while time.time() - t0 < secs:
    ok, f = cap.read()
    if ok and time.time() - last >= every:
        last = time.time(); n += 1
        cv2.imwrite(f"{out}/{time.strftime('%H%M%S')}_{n:03d}.jpg", f)
cap.release()
print("saved", n, "frames to", out)
