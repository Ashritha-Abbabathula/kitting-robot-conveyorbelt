# 🏭 Kitting Robot — Conveyor Belt Edition

![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111F68?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Dobot](https://img.shields.io/badge/Dobot-Magician_+_Conveyor-E4002B?style=for-the-badge)

An extension of my **[Vision-Verified Kitting Robot](https://github.com/Ashritha-Abbabathula/kitting-robot)**
that adds a **conveyor belt**. Instead of placing items one at a time, a whole batch of 3 blocks rides the
belt: from **one photo** the system works out which blocks the kit needs, which one is a reject, and in what
order they will arrive. The belt stops under the arm for each required block; the reject is never picked and
simply rolls off the end.

---

## 🧠 How It Works

```
QR code ──► mode ──► 30 s countdown to place 3 blocks on the belt ──► one photo
                                                                        │
         YOLOv8 finds the required blocks ◄─────────────────────────────┤
         colour-blob search finds the reject (any unknown block)  ◄─────┘
                                │
         order along the belt + belt travel per block (batch_logic)
                                │
   for each block:  belt advances it onto the pickup point
                    ├─ required → belt stops, arm picks & places it
                    └─ reject   → belt keeps running, it rolls off
                                │
         belt flushes, next countdown starts
```

| Mode | Batch | What happens |
|------|-------|--------------|
| `MODE_1` | red + blue + 1 reject | red → left, blue → right, reject skipped |
| `MODE_2` | Rubik's cube + red + 1 reject | cube → stack base, red on top. The cube must arrive **first**; otherwise nothing moves and an email says the batch is *not in order of height* |

If the photo doesn't show exactly the required items plus a reject, **nothing moves**. Show the other
mode's QR during a countdown to switch modes. An email goes out for every accepted batch.

---

## 🔍 What's New vs the Original

| File | What it does |
|------|--------------|
| `kitting_batch.py` | Batch runner: QR → countdown → photo → belt + arm. Simulated by default, `--real` for the arm |
| `kitting_logic/batch_logic.py` | Orders the blocks along the belt and computes how far the belt travels before each stop (block 25 mm, gap 24 mm) |
| `kitting_logic/blob_logic.py` | Finds the reject **relative to the detected blocks** — a size/shape-matched colour blob on the line through them — so the camera can be moved without re-calibrating a fixed region |
| `tools/belt_test.py` | Runs the belt for N seconds to measure its speed (measured 35 mm/s at 5000 pulses/s) |
| `kitting_logic/dobot_logic.py` | Adds conveyor control by sending the Dobot `SetEMotor` command directly (pydobot has no conveyor call) |

Belt settings (stepper port, speed, mm/s, belt direction in the image, item sizes) live in
`config/checklists.yaml` under `batch`.

---

## 🚀 Getting Started

```bash
pip install -r requirements.txt
python3 test/test_logic.py                                       # 26/26 pass, no hardware
python3 kitting_batch.py --mode mode_1                           # simulated arm + belt
python3 kitting_batch.py --slots red_block,reject,blue_block     # skip vision, test belt + arm moves
python3 kitting_batch.py --image photo.jpg                       # decide a batch from a saved photo
./run_batch.sh                                                   # REAL arm + belt, QR scanner
```

In the camera window: **SPACE** take the photo now · **r** new QR · **q** quit.
Every photo and its verdict is saved under `captures/batch/runs/`.

> `batch.toward` in `config/checklists.yaml` **must** match the direction the belt moves in the camera
> image — nothing in the picture says which end is the pickup end.

---

## ✅ Tested

- 26/26 unit tests (`test/test_logic.py`).
- Many real-arm MODE_1 and MODE_2 batches: required blocks picked to the right spots, reject skipped,
  belt cleared.

**Known limits:** dark or white rejects aren't found by colour; blue detection is weakest at the frame
edge; belt speed was measured once; the cube size along the belt is an assumption.

Session notes: [`PROGRESS.md`](PROGRESS.md) · [`CHANGED.md`](CHANGED.md)

---

## 👩‍💻 Author

**Ashritha Abbabathula**
[![GitHub](https://img.shields.io/badge/GitHub-Ashritha--Abbabathula-181717?style=flat&logo=github)](https://github.com/Ashritha-Abbabathula)
