# Changes — 2026-10-12 (model retraining + webcam tests)

## Model
- Trained **v6**: YOLOv8n fine-tuned on `dataset_v6` = previous data + Roboflow *red-green-blue-foam-objects* v1
  (blue/red blocks only; balls and green dropped). Test mAP50: blue 0.77, red 0.98, cube 0.995.
- Trained **v7**: fine-tuned from v6 on `dataset_v7` = v6 data + 59 webcam frames of the real (white-body) Rubik's cube
  + 9 empty-room negatives. Blocks unchanged vs v6; cube on webcam frames mAP50 0.62 -> 0.81 and now detected live.
- `models/kitting_best.pt` is **v7** (committed on purpose; `.gitignore` now allows only that file in `models/`).

## Code / tools
- `qr_yolo_test.py`: shows an **ALL ITEMS FOUND** banner (5 consecutive frames) and sends **one** email per mode completion.
  `r` or a new QR resets it.
- `tools/merge_foam_dataset.py`: merges the Roboflow foam export into dataset_v5 -> dataset_v6.
- `tools/capture_frames.py`: saves raw webcam frames for training sets.
- `tools/autolabel_cube.py`: auto-labels cube-only frames (motion + colour) and writes a review contact sheet.
- `.gitignore`: also ignores `config/secrets.yaml.bak`; allows `models/kitting_best.pt`.

## Verified
- Dry run (`tools/dobot_side_test.py --dry-run`, `demo_integration.py`) prints the expected move sequences.
- Webcam MODE_1 (red + blue) and MODE_2 (red + cube) tests pass with the v7 model, and the email is sent.
- Unit tests 14/14.

## Not changed / still open
- `calibrated: false` still blocks the real arm; one-item-at-a-time flow; MODE_2 stack spot near reach limit;
  camera-guided pickup; real `roslaunch` run. See `PROGRESS.md`.
- Blue-block recall is the weakest class; needs frames of the real blocks on the real table.

# Changes — 2026-10-13 (camera + arm integration)
- Added `kitting_run.py` (camera QR + YOLO driving the arm, one item at a time; simulated by default, `--real` gated by
  `calibrated: true` + typed confirmation, `--email` opt-in).
- Added `kitting_logic/flow_logic.py` and two tests in `test/test_logic.py` (16/16).
- Added `tools/setup_dobot_permanent.sh` (serial permission/udev; user must run with sudo).
- `kitting_run.py`: new default `--flow all` (detect all -> email -> 10 s -> item 1 -> 5 s -> item 2); `--flow each` = old flow;
  email now on by default (`--no-email`); `--start-delay`, `--wait` options.

# Changes — 2026-10-20
- `kitting_run.py`: defaults `--hold 1.0`, `--start-delay 5.0`, `--wait 2.0` (were 2 / 10 / 5); the completion email is sent in a
  background thread so SMTP can't freeze the camera/arm loop.
- `kitting_logic/notify_logic.py`: SMTP connection timeout 15 s.
- `kitting_moves.py` (MODE_2 only): Rubik's cube is PICKED 5 mm lower (`CUBE_PICK_LOWER`, pick/hover z 40.7 -> 35.7);
  red block released on the cube 25 mm lower (`RED_ON_CUBE_LOWER`, z 63.7 -> 38.7). Cube release height unchanged (38.7).
  `coordinates.yaml` NOT edited. Backups were kept in the session scratchpad only.
- Suction code (`kitting_logic/dobot_logic.py`) is unchanged: a temporary edit was reverted, file identical to the original.
- 2026-10-20 (MODE_2 retune, `kitting_moves.py`): cube pick z 37.7 (`CUBE_PICK_LOWER`=3); cube release 5 cm lower than the
  previous 38.7 -> -11.3 (`CUBE_LOWER`=55); red-on-cube release another 2.5 cm lower -> 13.7 (`RED_ON_CUBE_LOWER`=50). Not tried on the arm yet.
- 2026-10-20 (final MODE_2 z, set by user): cube pick 37.7, cube release 20.0 (CUBE_LOWER=23.7), red-on-cube 27.0 (RED_ON_CUBE_LOWER=36.7). Supersedes the entry above. Not tried on the arm.
- 2026-10-20 (latest MODE_2 z, set by user): cube pick 37.7, cube release 10.0 (CUBE_LOWER=33.7), red-on-cube 23.0 (RED_ON_CUBE_LOWER=40.7). Supersedes the entry above. Not tried on the arm.
- 2026-10-20 (latest MODE_2 z, set by user): cube pick 37.7, cube release 0.0 (CUBE_LOWER=43.7), red-on-cube 17.0 (RED_ON_CUBE_LOWER=46.7). Supersedes the entry above. Not tried on the arm.

# Changes — 2026-10-22 (batch-of-3 version, copied folder)
- NEW `kitting_batch.py`: QR -> arm-fix (Enter) -> 30 s countdown -> photo -> batch decision -> belt + arm, repeating per batch. See PROGRESS.md.
- NEW `kitting_logic/batch_logic.py` (select_batch, plan_slots, advance distances, batch_steps) and `kitting_logic/blob_logic.py`
  (colour-blob reject finder, view-independent). 6 new tests in `test/test_logic.py` (22 total, up from 16).
- `kitting_logic/dobot_logic.py`: `set_conveyor(on, speed, index)` now really drives the belt via SetEMotor (id 135) on the real arm.
- `config/checklists.yaml`: new `batch:` section (toward, belt stepper/speed/mm_per_s/flush_seconds=7).
- `kitting_moves.py`: MODE_1 swing red +45 / blue -60 deg (was +30/-30).
- NEW `tools/belt_test.py` (run the belt N s to measure speed). Existing kitting_run.py / MODE_2 flow untouched.
