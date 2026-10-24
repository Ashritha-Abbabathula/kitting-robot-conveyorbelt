"""
Find the reject (a coloured block the YOLO model does not know) — no ROS, no hardware, no fixed camera view.

The batch is 3 blocks in a line. YOLO finds the required ones; the reject is searched for relative to THEM, not
at a fixed place in the image: a narrow band along the line through the found blocks, reaching about one block
spacing beyond each end. A saturated blob there counts only if it is about the size and shape of the found blocks
(so the table, a hand, the belt's yellow sensor etc. are rejected). Yellow counts as a reject.
Dark or white rejects are not found by colour.
"""
from typing import Dict, List

import cv2
import numpy as np

SAT_MIN, VAL_MIN = 60, 60      # table/belt measure S <= 30; glare can pull a green block down to ~85
BAND_HALF = 0.6          # band half-width, in block sizes, either side of the line through the known blocks
REACH = 2.2              # how far past the outermost known block to look, in block sizes (~ one pitch)
AREA_RATIO = (0.4, 2.2)  # accepted blob area vs a known block's box area
ASPECT = (0.55, 1.8)     # accepted blob width/height
MIN_EXTENT = 0.55        # blob area / its bounding-box area (blocks fill their box; irregular things don't)


def belt_axis(dets: List[Dict]) -> np.ndarray:
    """Unit vector along the line of blocks (sign arbitrary). Needs >= 2 detections."""
    pts = np.array([[d["cx"], d["cy"]] for d in dets], float)
    if len(pts) < 2:
        raise ValueError("need at least two blocks to know the belt direction")
    u, s, vt = np.linalg.svd(pts - pts.mean(axis=0))
    return vt[0]


def find_unknown_blocks(frame: np.ndarray, known: List[Dict]) -> List[Dict]:
    """Detections shaped like YOLO's ({"name": "reject", conf, cx, cy, w, h}) for block-like coloured blobs on the
    line of `known` blocks that none of them covers (at most one: the one closest to the known blocks)."""
    if len(known) < 2:
        return []
    pts = np.array([[d["cx"], d["cy"]] for d in known], float)
    size = float(np.mean([np.sqrt(d["w"] * d["h"]) for d in known]))
    areas = [d["w"] * d["h"] for d in known]
    min_area, max_area = min(areas), max(areas)   # items differ in size (MODE_2: cube vs block), so use the range, not the mean
    axis = belt_axis(known)
    normal = np.array([-axis[1], axis[0]])
    centre = pts.mean(axis=0)
    along = (pts - centre) @ axis
    lo, hi = along.min() - REACH * size, along.max() + REACH * size

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    mask = ((s >= SAT_MIN) & (v >= VAL_MIN)).astype(np.uint8) * 255
    ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    rel = np.stack([xs - centre[0], ys - centre[1]], axis=-1)
    a, n = rel @ axis, rel @ normal
    band = (np.abs(n) <= BAND_HALF * size) & (a >= lo) & (a <= hi)
    mask[~band] = 0
    for d in known:                                    # blank out blocks YOLO already labelled
        x1, y1 = int(d["cx"] - d["w"] / 2) - 4, int(d["cy"] - d["h"] / 2) - 4
        x2, y2 = int(d["cx"] + d["w"] / 2) + 4, int(d["cy"] + d["h"] / 2) + 4
        mask[max(y1, 0):y2, max(x1, 0):x2] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if not (AREA_RATIO[0] * min_area <= area <= AREA_RATIO[1] * max_area):
            continue
        if not (ASPECT[0] <= w / h <= ASPECT[1]) or area / (w * h) < MIN_EXTENT:
            continue
        out.append({"name": "reject", "conf": 1.0, "cx": x + w / 2, "cy": y + h / 2, "w": float(w), "h": float(h)})
    # A batch holds ONE reject, and it sits right beside the known blocks; a stray coloured thing further out
    # (skin, clothing) must not become a second reject. Keep the candidate with the smallest gap to a known block.
    def gap(c):
        return min(np.hypot(c["cx"] - d["cx"], c["cy"] - d["cy"]) - (np.sqrt(c["w"] * c["h"]) + np.sqrt(d["w"] * d["h"])) / 2
                   for d in known)
    return sorted(out, key=gap)[:1]
