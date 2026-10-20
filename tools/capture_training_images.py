#!/usr/bin/env python3
"""
Point your webcam at an item and press SPACE to save training photos for
it — the first step toward a working YOLOv8 model (see the comment at the
top of kitting_logic/vision_logic.py for the full training steps).

Move the item around a bit between captures (angle, position, background,
lighting) — 30-50 varied shots per item is enough for a fixed-tabletop
demo. Aim for the same class names you'll use in config/checklists.yaml's
item_classes (e.g. "red_block").

Usage:
    python3 tools/capture_training_images.py red_block blue_block rubiks_cube [camera_index]

Controls:
    SPACE       save the current frame for the active item
    n           move to the next item in the list
    q           quit

Saves to training_images/<item_name>/<NNN>.jpg — hand this whole folder
to Roboflow or LabelImg to label, then train per vision_logic.py's steps.
"""
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "training_images")


def main():
    args = [a for a in sys.argv[1:] if not a.isdigit()]
    cam_args = [a for a in sys.argv[1:] if a.isdigit()]
    if not args:
        print("Usage: python3 tools/capture_training_images.py <item_name> [item_name ...] [camera_index]")
        return

    items = args
    cam_index = int(cam_args[0]) if cam_args else 0

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Could not open camera index {cam_index}. Try a different index (0, 1, 2...).")
        return

    counts = {item: 0 for item in items}
    for item in items:
        item_dir = os.path.join(OUT_DIR, item)
        os.makedirs(item_dir, exist_ok=True)
        counts[item] = len([f for f in os.listdir(item_dir) if f.lower().endswith(".jpg")])

    current = 0
    print(f"Capturing for: {items[current]}  |  SPACE=save  n=next item  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        item = items[current]
        display = frame.copy()
        cv2.putText(display, f"Item: {item}  ({counts[item]} saved)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save  n=next item  q=quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("capture_training_images", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            item_dir = os.path.join(OUT_DIR, item)
            path = os.path.join(item_dir, f"{counts[item]:03d}.jpg")
            cv2.imwrite(path, frame)
            counts[item] += 1
            print(f"  saved {path}")
        elif key == ord("n"):
            current = (current + 1) % len(items)
            print(f"Now capturing for: {items[current]}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone. Totals:")
    for item, count in counts.items():
        print(f"  {item}: {count} images in training_images/{item}/")


if __name__ == "__main__":
    main()
