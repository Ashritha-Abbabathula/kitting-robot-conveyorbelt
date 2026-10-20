"""Auto-label cube-only frames (cube held in front of a static background) -> YOLO txt, class 2 (rubiks_cube).
Moving pixels (vs. median background) that are bright-white or strongly coloured = cube; hand/skin is excluded by colour.
Writes <out>/images + <out>/labels and a contact sheet <out>/review.jpg for eyeballing.
Usage: .venv/bin/python -I tools/autolabel_cube.py captures/cube_alone captures/cube_labeled
"""
import glob, os, sys, shutil
import cv2, numpy as np
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{src}/*.jpg"))
imgs = [cv2.imread(f) for f in files]
bg = np.median(np.stack(imgs[::2]), axis=0).astype(np.uint8)
os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/labels", exist_ok=True)
thumbs, kept = [], 0
for f, im in zip(files, imgs):
    h, w = im.shape[:2]
    diff = cv2.absdiff(cv2.GaussianBlur(im, (7, 7), 0), cv2.GaussianBlur(bg, (7, 7), 0)).max(axis=2) > 40
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (vv > 190) & (ss < 70)
    sticker = (ss > 110) & (vv > 90)
    skin = (hh >= 3) & (hh <= 22) & (ss > 40) & (ss < 150)
    m = diff & (white | (sticker & ~skin))
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    vis = im.copy(); ok = False
    if n > 1:
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = stats[i]
        if area > 0.01 * w * h and 0.4 < bw / bh < 2.5 and area / (bw * bh) > 0.35:
            ok = True
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 3)
            name = os.path.splitext(os.path.basename(f))[0]
            shutil.copy(f, f"{out}/images/cube_{name}.jpg")
            open(f"{out}/labels/cube_{name}.txt", "w").write(
                f"2 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")
            kept += 1
    if not ok: cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
    thumbs.append(cv2.resize(vis, (240, 180)))
cols = 6
while len(thumbs) % cols: thumbs.append(np.zeros_like(thumbs[0]))
rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
cv2.imwrite(f"{out}/review.jpg", np.vstack(rows[:8]))
cv2.imwrite(f"{out}/review2.jpg", np.vstack(rows[8:16]) if len(rows) > 8 else np.zeros((10, 10, 3), np.uint8))
print(f"labeled {kept}/{len(files)}")
