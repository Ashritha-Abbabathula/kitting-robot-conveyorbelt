# Vision-Verified Kitting & Sorting Robot

A robot scans a QR code to pick a task mode, checks with computer vision that
the right items are physically present, emails an alert for anything missing,
and then runs the pick-and-place with a Dobot Magician once the kit is
complete.

See [`docs/Dobot_Kitting_Sorting_Robot_Presentation.pdf`](docs/Dobot_Kitting_Sorting_Robot_Presentation.pdf)
for the original project proposal (problem statement, architecture, timeline,
team). Note the proposal specs WhatsApp/Twilio for notifications — the build
below uses plain email instead (see "What's already done"), since it needs no
Twilio account, sandbox, or phone verification.

## What's already done (built and tested tonight, no ROS or hardware needed)

Everything under `kitting_logic/` is plain Python — no ROS, no Dobot, no
lab camera required. It's tested: run `python3 test/test_logic.py` and you
should see 13/13 passed. This is the actual brains of the project:

- `qr_logic.py` — reads a QR code from a camera frame
- `checklist_logic.py` — looks up what's required for a given mode
- `vision_logic.py` — checks whether an item is visible in a frame, using a
  YOLOv8 object detector (see "Training the YOLOv8 model" below)
- `notify_logic.py` — sends (or fakes sending) an email alert when items are missing
- `dobot_logic.py` — moves the arm, with a **simulated** version that just
  logs what it would do, so you can run the whole thing without the arm

Notifications go by email, not SMS/WhatsApp — no Twilio account, sandbox,
or phone verification needed. Just your own email address and a Gmail
"app password" (see `config/secrets.example.yaml` for the exact steps).

## What's left for tomorrow

1. Get this onto the lab's Ubuntu machine (or your own laptop if it runs
   Ubuntu) — see "Getting this onto the lab machine" below.
2. Install ROS 1 Noetic if it isn't already there.
3. `pip install -r requirements.txt` on that machine too (this now
   includes `ultralytics`, which pulls in PyTorch — a much bigger install
   than before, budget a few extra minutes).
4. Jog the real arm to find your actual coordinates, and put them in
   `config/coordinates.yaml` (it currently has placeholder numbers).
5. Train a YOLOv8 model on your actual sorting items under the lab's
   actual lighting — see "Training the YOLOv8 model" below (do this even
   if you trained one tonight at home — lighting changes what the model
   needs to see).
6. Copy `config/secrets.example.yaml` to `config/secrets.yaml` and fill in
   your real email + app password (see comments in that file).
7. Run it:
   ```
   roslaunch kitting_robot kitting.launch                        # simulated arm, safe default
   roslaunch kitting_robot kitting.launch simulate:=false port:=/dev/ttyUSB0   # real arm
   ```

Nothing else changes between simulation and the real run — same code,
same launch file, one flag.

## Which Python to use

Run everything from the project's `.venv` (it has `ultralytics`, `pyzbar`,
`pydobot` and a newer OpenCV; it also sees ROS's system packages). ROS
Noetic's system OpenCV 4.2 has no working QR decoder, so with plain
`/usr/bin/python3` and no `pyzbar` no QR code will ever be read.

```
source ~/catkin_ws/src/kitting_robot/.venv/bin/activate
cd ~/catkin_ws && catkin_make && source devel/setup.bash   # node wrappers pick up the venv's python
```

Run `catkin_make` with the venv active: the node wrappers in
`devel/lib/kitting_robot/` hard-code whichever Python ran it.

## Running the logic tonight, without ROS

```
pip install -r requirements.txt
python3 test/test_logic.py                 # confirm everything passes (no model needed)
python3 demo_integration.py                # walk the pipeline with no camera or model needed
python3 demo_integration.py --webcam       # real webcam + real YOLO model, end to end
```

`demo_integration.py --webcam` and the ROS `perception_node` both build
their detector via `build_detector()` (see
`kitting_logic/vision_logic.py`), which needs *at least one* of the two
backends below configured. Without either, it fails immediately with a
clear error rather than silently detecting nothing. `test/test_logic.py`
and the no-camera `demo_integration.py` don't need either — they use a
fake detector.

## Training the YOLOv8 model

The color-blob approach (matching a fixed HSV range) was swapped for a
real object detector, since colour matching broke down under lighting
changes and couldn't tell two same-coloured items apart. There are two
detector backends, and `build_detector()` combines whichever you set up
(`CompositeDetector`) — different items can come from different backends:

**red_block / blue_block — hosted, no local training at all.** The
[colored-blocks](https://universe.roboflow.com/autonomous-object-picking-robot/colored-blocks)
model (CC BY 4.0, 98.9% mAP@50, classes `red`/`green`/`blue`) is called
live over Roboflow's serverless inference API (`RoboflowAPIDetector`),
instead of downloading its images and training a local copy — one less
thing to train, at the cost of needing network access at detect() time.
Setup:
1. Get a free API key at
   [app.roboflow.com/settings/api](https://app.roboflow.com/settings/api).
2. Copy `config/secrets.example.yaml` to `config/secrets.yaml` if you
   haven't already, and fill in `roboflow.api_key`.
3. `config/checklists.yaml`'s `roboflow.model_id: colored-blocks/2` is
   already set — nothing else to do.

**rubiks_cube — trained locally** on the
[rubiks_cube_segmentation](https://huggingface.co/datasets/seandavidreed/rubiks_cube_segmentation)
dataset from HuggingFace (Apache-2.0, ~200 images, YOLO format with segmentation
polygons — convert each polygon to its bounding box before training).
Blue/red block images come from the Roboflow colored-blocks dataset (classes
`blue`/`red` only; other colours are dropped) plus your own labelled photos.
Class order in the trained model: `{0: blue_block, 1: red_block, 2: rubiks_cube}`.
Steps:
1. Merge the three sources into one `data.yaml` with that class order
   (train/valid/test splits; keep a few images with no target objects as
   negatives so "missing" is learnable).
2. Train on CPU with a small image size:
   `yolo detect train data=data.yaml model=yolov8n.pt epochs=25 imgsz=320 batch=16 cache=ram device=cpu`
   — on a 4-core VM this runs roughly 3 min/epoch for ~430 images; budget an hour or more.
3. Copy the resulting `runs/detect/<name>/weights/best.pt` to
   `models/kitting_best.pt` (what `config/checklists.yaml`'s `yolo.weights` points at).
   `item_classes[*].class_name` in `checklists.yaml` must match the model's
   class names exactly (`blue_block`, `red_block`, `rubiks_cube`).

No accuracy numbers are recorded here yet — measure on the held-out test split
with `find_missing_items()` after training and fill them in.

Only need one backend? `build_detector()` is fine with just the local
model, or just the Roboflow one — it only combines both when both are
configured. If your actual objects don't look enough like either dataset,
capture and label your own instead:
`tools/capture_training_images.py red_block blue_block rubiks_cube`, then
train locally per the rubiks_cube steps above and point `item_classes` in
`config/checklists.yaml` at your own class names.

Both datasets are CC BY 4.0 — attribution is required if you use them.
Roboflow's project page has a ready-made citation under "Cite This
Project"; the Kaggle page has an equivalent under its metadata section.
Include both in your final report/slides.

Neither trained weights, the Roboflow API key, nor any captured/downloaded
photos are committed to git (see `.gitignore`) — they're per-lab secrets
or build artifacts, not source.

If `pyzbar` fails to install (it needs a system library called `libzbar`
that pip can't install by itself on Linux — Windows usually just works),
the QR code will still work — the code automatically falls back to
OpenCV's built-in QR reader (see `kitting_logic/qr_logic.py`).

One QR-printing tip that actually mattered when testing this: print your
QR codes at a reasonable size (a few cm across) with a plain white margin
around them — a QR code with no white border around it, or a screenshot
cropped too tightly, is much harder for either decoder to read.

Ready-made test codes are included: [`MODE_1.png`](MODE_1.png) and
[`MODE_2.png`](MODE_2.png) encode `MODE_1`/`MODE_2` directly — print or
display either one to trigger that mode without generating your own.

## Getting this onto the lab machine

Easiest: push this to a GitHub repo tonight, then on the lab machine:

```
git clone <your-repo-url> ~/catkin_ws/src/kitting_robot
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

(If you don't already have a catkin workspace on the lab machine, `mkdir
-p ~/catkin_ws/src` first, then clone into it as above.)

## Project structure

```
kitting_robot/
  kitting_logic/        <- pure Python, no ROS — the actual logic, tested tonight
  scripts/               <- thin ROS node wrappers around kitting_logic/
  config/                <- checklists, coordinates, YOLO/email config (fill in for real)
  launch/kitting.launch   <- starts all three nodes together
  tools/capture_training_images.py  <- webcam capture helper for YOLO training photos
  models/                 <- trained YOLO weights go here (gitignored — train your own)
  training_images/        <- captured training photos go here (gitignored)
  test/test_logic.py      <- run tonight, no ROS or model needed
  demo_integration.py     <- run the whole pipeline end-to-end right now, no ROS needed
  MODE_1.png, MODE_2.png  <- printable QR codes for testing mode switching
  docs/                   <- original project proposal (PDF)
```

## Keeping secrets out of git

`config/secrets.yaml` holds your real Gmail address and app password — it's
listed in `.gitignore` and must never be committed. Only
`config/secrets.example.yaml` (a template with placeholder values) is
tracked. If you ever regenerate the Gmail app password in `secrets.yaml`
(e.g. because it was accidentally shared), revoke the old one at
[myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).
