# Kitting robot — progress notes (written 2026-10-12)

## Working
- **YOLO model** `models/kitting_best.pt` (YOLOv8n, run `runs/detect/kitting_v5`, 320 px, 25 epochs).
  Classes: `blue_block`, `red_block`, `rubiks_cube`. Test split: P 0.99 / R 0.97 / mAP50 0.98.
  **Caveat:** trained on red/blue *foam dice with dots* on a wood floor + Rubik's cube photos. It does
  not recognise cartoon/phone pictures or plain coloured blocks. Retrain on your real blocks later.
- **QR + YOLO webcam test** (no ROS, no arm):
  `cd ~/catkin_ws/src/kitting_robot && .venv/bin/python -I qr_yolo_test.py`  (q quit, r reset mode)
  QR images: `MODE_1.png`, `MODE_2.png`. Plain camera view: `live_view.py`.
- **Dobot motion test** (no camera): `.venv/bin/python -I tools/dobot_side_test.py`
  (`--dry-run` prints moves only). Picks red -> left, waits 5 s, picks blue -> right. Tested OK.
- Unit tests: `.venv/bin/python -I test/test_logic.py` (14/14 pass).

## Measured Dobot coordinates (config/coordinates.yaml)
| Step | x, y, z |
|---|---|
| Pickup red/blue (shared spot) | 278.0, 94.8, 12.3 |
| Pickup Rubik's cube | 276.1, 94.5, 63.8 (cube top) |
| MODE_1 red -> left | 122.2, 258.7, 24.3 |
| MODE_1 blue -> right | 207.1, -193.6, 24.3 |
| MODE_2 cube -> stack base | 231.2, -204.8, 66.8 |
| MODE_2 red -> on cube | 231.2, -204.8, 91.8 (block is 25 mm) |
Hardware home (after homing): 265.6, -3.6, -4.3.

## Open items
1. `calibrated: false` in coordinates.yaml — dobot_node refuses the real arm until set true.
2. **MODE_2 stack spot is ~309 mm from the base** (near the ~320 mm reach limit); the default 40 mm
   hover at z=131.8 may be unreachable. Move the spot ~30 mm inward or lower the hover for it. Untested.
3. **Flow mismatch:** perception_node only starts the arm once *all* required items are visible together;
   you want items placed one after another at the shared pickup spot. Needs a one-item-at-a-time flow.
4. Camera-guided pickup (pixel -> robot calibration) not done; pickup is a fixed spot.
5. Retrain the model on the real blocks/table. `config/secrets.yaml` (email alerts) not verified.
6. The `roslaunch kitting_robot kitting.launch` real-arm run has not been tried.

## Every session (not persistent)
- Attach Dobot (and webcam) in VirtualBox: Devices -> USB / Webcams (or add USB filters in VM Settings).
- Serial permission: `sudo usermod -aG dialout sreehitha` then re-login (or `sudo chmod a+rw /dev/ttyUSB0`).
- After powering the arm off/on it must be re-homed (RealDobot.home() sends the home command; moves the arm).
- Use the project venv: `source ~/catkin_ws/src/kitting_robot/.venv/bin/activate`.
- This folder is not a git repo — no version history.

## Session log — 2026-10-12 (evening): retraining on foam blocks
- Dobot not available this session, so work was model-only.
- Downloaded Roboflow `lances-workspace-gotgs/red-green-blue-foam-objects` v1 (772 imgs, public domain)
  using `roboflow.api_key` from `config/secrets.yaml` (the same key/email details also live in
  `~/Downloads/secrets.yaml`; the repo copy differs — sync before testing email).
- Merged into **`dataset_v6`** (= dataset_v5 + foam blue/red blocks only; balls & green dropped,
  316 foam imgs without block labels skipped): 777 train / 142 val / 88 test.
  Script: `tools/merge_foam_dataset.py`. Classes unchanged: blue_block, red_block, rubiks_cube.
- Training: `yolo detect train model=yolov8n.pt data=dataset_v6/data.yaml epochs=25 imgsz=320 batch=16
  device=cpu name=kitting_v6 project=runs/detect` (~1.6 s/step, ~35-40 min on the VM).
  Output: `runs/detect/kitting_v6`. **Status: DONE.** Val (own split) mAP50 0.92.
  Test-split comparison (mAP50, all / blue / red / cube):
  | model | on v5 test (39 imgs) | on v6 test (88 imgs, incl. foam) |
  |---|---|---|
  | v5 | 0.984 / 0.995 / 0.962 / 0.995 | 0.744 / 0.459 / 0.779 / 0.995 |
  | v6 | 0.995 / 0.995 / 0.995 / 0.995 | 0.914 / 0.766 / 0.981 / 0.995 |
  **v6 wins everywhere -> now `models/kitting_best.pt`** (old model kept as `models/kitting_best_v5.pt`).
  Weakest spot: blue_block recall on foam images (0.745) — add real-block photos when retraining on the actual blocks.
- `config/secrets.yaml` now = the `~/Downloads/secrets.yaml` (email + roboflow key); old one kept as `config/secrets.yaml.bak` (gitignored).
- Still to do this round: compare v6 vs v5 on test split -> copy winner to `models/kitting_best.pt`;
  test MODE_1/MODE_2 with webcam+QR; dry-run Dobot text output; email check; git push
  (use `~/kitting_robot_push`, needs remote URL; keep `config/secrets.yaml`, `.venv`, `runs/` out of git).
- Dry run done (no arm): `tools/dobot_side_test.py --dry-run` and `demo_integration.py` (synthetic, MODE_1) both print the
  expected move sequences. NOTE: demo_integration.py sends a REAL "all items present" email when secrets.yaml is
  filled in — it ran once at ~21:30 with no send error printed (so Gmail login/send worked). Receiver: see secrets.yaml.
- `qr_yolo_test.py` now shows an "ALL ITEMS FOUND" banner (5 consecutive frames) and sends ONE real email per
  mode completion (resets on `r` or new QR). Webcam test: red + blue detected live with v6 model.
- Webcam test PASSED for MODE_1 (2026-10-12 ~21:50): QR read, red+blue detected, "ALL ITEMS FOUND", email sent with no
  error. MODE_2 (red + cube) webcam test still to do.
- Cube not detected live (v6 saw the white-body cube w/ blue stickers as blue_block 0.55; rubiks_cube max conf 0.14).
  Fix: captured 116 webcam frames of the real cube (`tools/capture_frames.py` -> `captures/cube_alone/`),
  auto-labelled by motion+colour (`tools/autolabel_cube.py` -> `captures/cube_labeled/`), merged into **dataset_v7**
  (v6 + cube frames, contiguous 80/10/10 split, 9 empty-room negatives). Fine-tuning from v6: `kitting_v7`, 15 epochs.
  Caveat: all cube frames share one background/person, so val/test for the cube are optimistic; no real-block frames yet.
- **v7 trained** (fine-tune of v6 on dataset_v7, 15 ep) -> `models/kitting_best.pt` (v6 kept as `models/kitting_best_v6.pt`).
  Old test split (dataset_v6 test): blue 0.772 / red 0.979 / cube 0.995 (v6: 0.766 / 0.981 / 0.995 — blocks unchanged).
  Webcam-cube test frames (dataset_v7 test, 15 inst., same background as training): v7 cube mAP50 0.807 (R 0.79).
  Needs the live MODE_2 test to confirm; more varied cube frames + real-block frames would help.
- Webcam MODE_2 test PASSED with v7 (2026-10-12 ~22:40): QR read, red block + cube detected, "ALL ITEMS FOUND", email sent
  (an earlier attempt failed once with a transient DNS error, `gaierror -3`; retry succeeded). Next: git push; with the
  arm: calibrated flag, one-item-at-a-time flow, MODE_2 reach, roslaunch.
- Webcam MODE_1 re-test PASSED with v7 (2026-10-12 late evening).

## Session log — 2026-10-13: permanent Dobot setup
- Arm seen in VM as Silicon Labs CP210x (10c4:ea60) -> /dev/ttyUSB0 (root:dialout). User was NOT in `dialout`; no udev rule; brltty installed.
- Added `tools/setup_dobot_permanent.sh` (needs sudo, run by user): dialout membership, `/etc/udev/rules.d/99-dobot.rules`
  (GROUP=dialout, `/dev/dobot` symlink), masks brltty. After running: re-login, then check `id | grep dialout` and `ls -l /dev/dobot`.
- Added `kitting-env` alias in ~/.bashrc (activates venv + cd to repo).
- NOT automatable from inside the VM: VirtualBox USB filter on the host (VM Settings -> USB -> add filter for 10c4:ea60,
  and the webcam) so the arm auto-attaches at VM start. Do this with the VM powered off.
- Not done yet: `calibrated: true`, one-item-at-a-time flow, MODE_2 reach check, roslaunch with real arm.

## Session log — 2026-10-13: camera + arm integration (kitting_run.py)
- New `kitting_run.py`: webcam QR -> mode, YOLO confirms the *next expected* item at the shared pickup spot (held `--hold` 2 s),
  then the arm picks that one item and places it (one-item-at-a-time flow = open item 3). Order from `place[mode]`:
  MODE_1 red->left, blue->right; MODE_2 cube->stack base, then red on the cube. Simulated arm by default.
  `--real` needs `calibrated: true` AND typing MOVE; `--email` opts in to the real completion email; `--mode` skips QR.
- New `kitting_logic/flow_logic.py` (StableTracker, ordered_items, item_steps, max_radius) + 2 tests (16/16 pass).
- Reach check printed at mode start: MODE_1 spots 294 mm, MODE_2 stack spot 309 mm (warns > 300). Still untested on the arm.
- NOT yet run end-to-end: webcam was not attached to the VM this session (no /dev/video0), so only syntax/tests/startup checked.
- Next: attach webcam, run `kitting_run.py --mode mode_1` simulated with real blocks, then `--real` with go-ahead.
- 2026-10-13: simulated `kitting_run.py` PASSED with the live webcam for MODE_1 (red->left, blue->right) and MODE_2
  (cube->stack base, red on cube). Moves match coordinates.yaml. Arm not moved, no email sent.
  Still open before `--real` on MODE_2: the red-block hover at (231.2, -204.8, z=131.8) is 309 mm out and may be unreachable —
  jog to it by hand, or lower the place hover / move the stack spot inward. Then set `calibrated: true`.
- 2026-10-13 (later): `kitting_run.py` default flow is now `--flow all` (the user's demo flow): camera sees ALL required items ->
  real email -> 10 s countdown (`--start-delay`) -> pick/place item 1 -> 5 s wait (`--wait`) -> pick/place item 2.
  `--flow each` keeps the earlier one-item-at-a-time flow. Email is ON by default now (`--no-email` to disable).
  Simulated MODE_1 run with live webcam passed (red->left 122.2/258.7, blue->right 207.1/-193.6); email send printed no error.
  Real arm not used yet. MODE_2 under the new flow not run yet.

## Session log — 2026-10-13 (evening): homing is now part of every real run
- Real run on an un-homed arm (after a power cycle) put everything ~off: the arm forgets its zero on power loss, and the
  saved coordinates (re-measured 13:25, correct) only hold on a HOMED arm. kitting_run.py --real silently ran un-homed and still
  printed "Kit complete" + sent the email (arm had alarms 0x22/0x49 at the folded pose and ignored the moves).
- `kitting_run.py --real` now, before the camera loop / QR scan: reads alarms (refuses if any) -> asks you to press Enter ->
  sends the arm's home command -> verifies the pose is within 10 mm of the hardware home (265.6, -3.6, -4.3), else aborts.
  Nothing is remembered between runs: homing is asked every time. `RealDobot.alarms()` / `pose_xyzr()` added.
- Not yet run on the arm since this change.
- Homing now repeats at the start of EVERY QR scan cycle (start-up, `r`, or a new QR after a finished kit), via `ensure_homed()`.
- Revised: no home command is sent. The user parks the arm at home by hand + Enter; kitting_run.py reads the pose, derives rotation/z offset vs (265.6,-3.6,-4.3) and applies it to every target (FrameOffsetArm). Alarm check kept (read-only).
- 2026-10-13 late: `kitting_run.py` was accidentally emptied by a bad edit and REBUILT from the session transcript (not byte-identical
  to before; compiles, 16/16 tests pass, simulated start-up OK). One-click real run: `./run_kitting.sh` (asks you to hand-home the
  arm + Enter, then QR scanner, detection, email, moves). Real-arm run with this version NOT yet done; last real attempt failed
  because the arm dropped off USB during the (now removed) home command.
- Next session: attach webcam + Dobot (VirtualBox USB), power the arm on from its folded rest pose, `./run_kitting.sh`.
- Homing is now RELATIVE TO THE START SPOT: user parks arm by hand at the start spot + Enter; pose is compared with `start:` in coordinates.yaml (first run asks to SAVE it) and every target is shifted by the difference (translation only). Not run on the arm yet.
- Reference is now the saved PICKUP position (pickup.default in coordinates.yaml): park arm at pickup by hand + Enter, targets shift by (reading - saved pickup). No separate start spot.
- Reference = saved pickup HOVER (pickup z + hover_height_offset = 52.5): user parks ~4 cm above the pickup spot; xy+z shift applied; warns if dz>15.
- Reference = saved pickup x/y only; park at ANY height above the pickup; z never shifted (heights stay as saved).

## Session log — 2026-10-20: suction not working after teacher reassembled the arm
- Symptom: pump blew air out / did nothing instead of sucking. Cause was physical (pump cable / tube connections after the arm was taken apart); fixed by hand, now sucks and kitting works.
- Arm was also found with the gripper channel ENABLED. NOTE: pydobot `grip(False)` sends enable=1 (opens the gripper, it does NOT turn it off). To disable the gripper send SET_GET_END_EFFECTOR_GRIPPER with params [0, 0]. Read state: ctrl 0x00 on the same id (0000 = off).
- pydobot `suck(False)` = enable 1, suck 0 -> pump runs in the release direction; fully stop the pump with params [0, 0] on SET_GET_END_EFFECTOR_SUCTION_CUP.
- Project code was NOT changed (a temporary edit to `RealDobot.set_suction()` was reverted, file identical to the original). Temporary helper scripts were removed.
- 2026-10-20 (later): email did not arrive because the network currently blocks outgoing mail ports (465/587/993 time out over IPv4,
  web 443 works). It worked at 15:34 on another network. Fix = change network (e.g. phone hotspot) or send over HTTPS. Code is fine.
- 2026-10-20: delays shortened (hold 1 s, start-delay 5 s, wait 2 s), email sends in background. MODE_2: cube pick 5 mm lower,
  red-on-cube release 25 mm lower (see CHANGED.md). NOT yet tried on the arm — red release z now equals the cube release z
  (38.7); watch for the cup pressing the block into the cube.
- 2026-10-20: MODE_2 retuned again (cube pick 37.7, cube release -11.3, red-on-cube 13.7). See CHANGED.md. Not tried on the arm.
- 2026-10-20 (final MODE_2 z, set by user): cube pick 37.7, cube release 20.0 (CUBE_LOWER=23.7), red-on-cube 27.0 (RED_ON_CUBE_LOWER=36.7). Supersedes the entry above. Not tried on the arm.
- 2026-10-20 (latest MODE_2 z, set by user): cube pick 37.7, cube release 10.0 (CUBE_LOWER=33.7), red-on-cube 23.0 (RED_ON_CUBE_LOWER=40.7). Supersedes the entry above. Not tried on the arm.
- 2026-10-20 (latest MODE_2 z, set by user): cube pick 37.7, cube release 0.0 (CUBE_LOWER=43.7), red-on-cube 17.0 (RED_ON_CUBE_LOWER=46.7). Supersedes the entry above. Not tried on the arm.

## Session log — 2026-10-22: BATCH-OF-3 VERSION (this folder is a copy of ~/catkin_ws/src/kitting_robot; the original is untouched)
`.venv` here is a SYMLINK to the original's venv (disk was nearly full) - installing packages here changes the original's env.
**Idea:** QR picks the mode, then ONE photo of the belt shows a batch of 3 blocks (2 required for the mode + 1 reject = anything not on
the mode's checklist). The arm picks only the required ones; the reject is left and the belt carries it off. MODE_1 only so far
(red -> left, blue -> right). MODE_2 is not built for batches.

**Run it (real arm):** `./run_batch.sh` (= `.venv/bin/python -u -I kitting_batch.py --real`) -> "Please FIX THE ARM ... press Enter" (pickup x/y is read
from the arm, held still, then FIXED for the whole run) -> QR scanner -> 30 s countdown with live camera to place blocks -> photo ->
decision -> belt + arm -> 7 s belt roll-on -> next countdown. Keys: SPACE photo now, r new QR, q quit. There is NO MOVE prompt any more (removed 2026-10-22): once a batch is decided the arm starts straight away. `--yes` is an ignored old option. `--mode mode_1` skips the QR; `--park-seconds N` = timed arm-setting window instead of Enter (unattended); `--toward`
sets the belt direction in the image; `--slots red_block,reject,blue_block` skips vision (tests belt+arm only); `--image f.jpg` = decision on a photo.
Simulated by default (no `--real`). Every photo + verdict is saved in `captures/batch/runs/` (`_raw`, `_ok`, `_refused`).

**How a batch is decided (kitting_logic/batch_logic.py, blob_logic.py):** YOLO (imgsz 640 - 320 missed small blocks) finds red/blue;
`select_batch` keeps the tightest group (similar size, close together) and ignores strays on the table; `find_unknown_blocks` finds the
reject as a saturated colour blob on the line of the found blocks (any colour incl. green/yellow; not black/white), size/shape-matched
to them - NOT a fixed image region, so the camera can move. Order along the belt is taken along that line, pointed the way `toward`
says the belt moves in the image. Not exactly 1 red + 1 blue + 1 other -> nothing moves.
**`toward` (config batch.toward, now `right`) MUST be set to the belt's direction in the camera view** - nothing in the picture says which
end is the pickup end. Camera view the belt runs left->right.

**Belt (config/checklists.yaml `batch.belt`):** driven by the Dobot, stepper port 0 (= Stepper 1), speed 5000 pulses/s, measured 140 mm in 4 s
= 35 mm/s (one measurement; short runs may be a bit off). `RealDobot.set_conveyor` now sends SetEMotor (protocol id 135) itself -
pydobot 1.3.x has no conveyor call. `tools/belt_test.py` runs it for N s to measure.
Geometry used: block 25 mm, gap 24 mm edge to edge, leading block's near edge 25 mm before the pickup point -> belt stops 37.5 mm, then
49 mm apart. After the last block the arm returns to the pickup point and the belt ALWAYS rolls 7 s (`flush_seconds`) to clear blocks.

**Arm moves:** same relative scheme as before (place spots = pickup x/y swung around the arm base, same reach). MODE_1 swing now
red +45 deg (left), blue -60 deg (right) (`SWING_DEG` in kitting_moves.py). Heights come from coordinates.yaml (pickup z 12.5), NOT from
the z the arm was held at when parked. A new pickup spot is taken on every run (by design).

**Gotchas hit this session:** (1) a run reads whatever pose the arm is in at the fix step - if it was not parked on the belt pickup point
the whole run is shifted (happened twice); watch the printed `Pickup spot FIXED` line. (2) The first version read the pose the instant a
timed window ended, mid-hand-movement - now waits until the arm is still. (3) `pkill -f`/`pgrep -f` with the script name matches the
calling shell - use `pgrep -f "[.]venv/bin/python -u -I kitting_batch"` in its own command. (4) Python buffers stdout when not on a
terminal - use `-u`.
**Tested:** 22/22 unit tests (`.venv/bin/python -I test/test_logic.py`); many real-arm batches ran correctly (red/blue picked to the
left/right drops, reject skipped, belt cleared). **Not done / open:** MODE_2 batches; no email in the batch flow; belt mm/s only measured once;
drop spots at 45/60 deg only checked by eye; detection weakest for blue at the frame edge and for dark/white rejects.

- 2026-10-22 (late): MODE_2 batches work on the real arm (cube -> stack base, red on top; stack at -60 deg, cube pick z 33.7 = CUBE_PICK_LOWER 7, 1.5 s suction pause). QR scanner switches mode (show the other QR during a countdown). Emails: one per valid batch ('got all components - executing now') and one when MODE_2 is out of order ('not in order of height'); both sent in a background thread. USB tip: if the VM loses camera+Dobot, `dmesg` says 'ohci-pci ... HC died' -> `echo 0000:00:06.0 | sudo tee /sys/bus/pci/drivers/ohci-pci/unbind` then `.../bind`, then re-attach in VirtualBox. Cube width 57 mm is an assumption (config batch.sizes_mm).
