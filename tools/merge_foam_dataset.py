"""Merge the Roboflow 'red-green-blue-foam-objects' export (YOLOv8 format) into dataset_v5 -> dataset_v6.

Keeps only 'blue block' (-> 0) and 'red block' (-> 1); balls and green blocks are dropped, and foam
images with no kept labels are skipped. Existing dataset_v5 images/labels (incl. rubiks_cube = 2) are copied as-is.

Usage: .venv/bin/python -I tools/merge_foam_dataset.py <extracted_foam_export_dir>
The export zip comes from the Roboflow API key in config/secrets.yaml (roboflow: api_key), version 1, format yolov8.
"""
import glob
import os
import shutil
import sys

FOAM_NAMES = ['blue ball', 'blue block', 'green ball', 'green block', 'red ball', 'red block']
REMAP = {FOAM_NAMES.index('blue block'): 0, FOAM_NAMES.index('red block'): 1}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src, dst = f'{root}/dataset_v5', f'{root}/dataset_v6'
foam = sys.argv[1]

kept = {s: 0 for s in ('train', 'valid', 'test')}
skipped = 0
for s in kept:
    for sub in ('images', 'labels'):
        os.makedirs(f'{dst}/{s}/{sub}', exist_ok=True)
        for f in glob.glob(f'{src}/{s}/{sub}/*'):
            shutil.copy(f, f'{dst}/{s}/{sub}/')
    for lf in glob.glob(f'{foam}/{s}/labels/*.txt'):
        out = []
        for line in open(lf):
            p = line.split()
            if p and int(p[0]) in REMAP:
                out.append(' '.join([str(REMAP[int(p[0])])] + p[1:]))
        if not out:
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(lf))[0]
        img = glob.glob(f'{foam}/{s}/images/{stem}.*')
        if not img:
            continue
        shutil.copy(img[0], f'{dst}/{s}/images/foam_{stem}{os.path.splitext(img[0])[1]}')
        open(f'{dst}/{s}/labels/foam_{stem}.txt', 'w').write('\n'.join(out) + '\n')
        kept[s] += 1

open(f'{dst}/data.yaml', 'w').write(
    f"path: {dst}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    "nc: 3\nnames: {0: blue_block, 1: red_block, 2: rubiks_cube}\n")
print('foam images kept', kept, '| skipped (no block labels)', skipped)
