# Fruit Tracking with YOLOv7 and Step-Compensated SORT

A complete pipeline for **multi-object tracking of agricultural fruits** (flower, immature fruit, mature fruit) from a mobile robot platform. The system covers three stages: detection training, tracking, and MOT evaluation.

---

## Pipeline Overview

```
detection/          →         tracking/          →        evaluation/
YOLOv7 (multi-head)     Step-Compensated SORT          py-motmetrics
   ↓                           ↓                              ↑
det.txt per frame      output/*.txt (with IDs)     MOTA / MOTP / IDF1 ...
```

---

## Repository Structure

```
fruit-tracking/
├── detection/               # YOLOv7-based fruit detector
│   ├── cfg/training/        # Model configs (full / per-head ablation)
│   ├── models/              # YOLOv7 model definitions
│   ├── utils/               # Training utilities
│   ├── data/SLdata.yaml     # Dataset config (3 classes)
│   ├── detect.py            # Run inference
│   ├── trainyolov7.py       # Train full model
│   ├── trainyolov7h[1-3].py # Train single-head variants
│   ├── trainyolov7h[12|13|23].py  # Train dual-head variants
│   └── run.py               # Run all ablation trainings sequentially
│
├── tracking/                # Step-compensated SORT tracker
│   ├── sortwithstep.py      # Main tracker (Kalman + step compensation)
│   ├── Vsort.py             # Location-score assisted tracker
│   ├── stepcal.py           # Compute step value from GT annotations
│   ├── ioucal.py            # Standalone IoU utility
│   ├── dataTransform.py     # Format conversion utilities
│   ├── data000/             # Example detection input (MOT format)
│   ├── gt_data/             # Ground-truth annotations (per category)
│   │   ├── flowerGT/
│   │   ├── immaturefruitGT/
│   │   └── maturefruitGT/
│   └── output/              # Tracking results (generated)
│
└── evaluation/              # MOT metrics evaluation
    ├── motmetrics/          # Modified py-motmetrics library
    │   └── apps/
    │       └── eval_motchallenge.py   # Main evaluation script
    └── yourdata/            # Example GT + tracker result pairs
```

---

## Modules

### 1. Detection — `detection/`

YOLOv7 fine-tuned for three fruit classes: **flower**, **immature fruit**, **mature fruit**.

A key contribution is the **multi-head ablation training system**: the detection head is decomposed into three scale-specific sub-heads (h1: large, h2: medium, h3: small). Seven training configurations let you study which detection scales matter most for your target objects.

**Quick start:**
```bash
cd detection

# Train full model
python trainyolov7.py --cfg cfg/training/yolov7/yolov7.yaml \
                      --data data/SLdata.yaml \
                      --weights '' --epochs 150

# Run inference
python detect.py --weights runs/train/exp/weights/best.pt \
                 --source path/to/video.mp4 \
                 --save-txt   # saves det.txt for tracker input

# Run all ablation experiments sequentially
python run.py
```

**Model configs** live in `cfg/training/`. Each YAML specifies which detection heads are active. The `data/SLdata.yaml` file points to your image directories and defines the 3 class names.

---

### 2. Tracking — `tracking/`

SORT-based multi-object tracker with two domain-specific extensions for robot-mounted cameras.

#### Key innovation: Step Compensation

Fruits are static; the camera moves. This means fruit positions shift by a roughly constant pixel offset (`step`) every frame. `sortwithstep.py` shifts each tracker's predicted bounding box by `step` pixels along the y-axis before IoU matching, effectively compensating for robot motion.

**Compute the step value from your own data:**
```bash
python stepcal.py --input gt_data/flowerGT/4.txt
# Output: Average step: 314.57 pixels/frame
```

**Run the tracker:**
```bash
python sortwithstep.py --seq_path data000 --step 314.57 --max_age 2 --min_hits 0
# Results saved to output/<sequence>.txt
```

**Run the location-score assisted tracker (Vsort):**
```bash
python Vsort.py --input gt_data/flowerGT/4.txt \
                --output stepmatchoutput/flower4.txt \
                --step 315
```

**Convert formats:**
```bash
# GT → evaluation format
python dataTransform.py --mode gt2eval \
    --input gt_data/flowerGT/4.txt --output Fgt4.txt

# SORT result → DarkLabel visualisation
python dataTransform.py --mode sort2darklabel \
    --input output/flower4.txt --output flower4_vis.txt --label "mature fruit"
```

#### Tracking input format (MOT15-2D)

Detection files under `data000/train/<sequence>/det/det.txt`:
```
frame, -1, x, y, w, h, score, -1, -1, -1
```

#### Tracking output format
```
frame, id, x, y, w, h, 1, -1, -1, -1
```

---

### 3. Evaluation — `evaluation/`

Modified [py-motmetrics](https://github.com/cheind/py-motmetrics) library for computing standard MOT metrics.

**Key modifications from the original library** (all in `motmetrics/apps/eval_motchallenge.py`):
- Data paths are now **relative and configurable** via `--groundtruths` / `--tests` arguments (default: `yourdata/`)
- `min_confidence=-1` to accept all tracks (trackers that don't output confidence)
- Matching uses **Euclidean centre-point distance** (threshold: 5000 px) instead of IoU, which works better for dense/overlapping fruits

**Organize your data:**
```
evaluation/yourdata/
├── video1/gt/gt.txt     # ground truth for video1
├── video2/gt/gt.txt
├── video1.txt           # tracker result for video1
└── video2.txt
```

Each file uses MOT15-2D format:
```
frame, id, x, y, w, h, confidence, -1, -1, -1
```
Set `confidence=-1` if your tracker does not output a score.

**Run evaluation:**
```bash
cd evaluation
python motmetrics/apps/eval_motchallenge.py
# or with custom paths:
python motmetrics/apps/eval_motchallenge.py \
    --groundtruths yourdata --tests yourdata
```

**Output metrics:** MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, FM, and more.

---

## Installation

### Detection
```bash
cd detection
pip install -r requirements.txt
```

### Tracking
```bash
cd tracking
pip install -r requirements.txt
# filterpy==1.4.5  scikit-image==0.17.2  lap==0.4.0
```

### Evaluation
```bash
cd evaluation
pip install -r requirements.txt
```

---

## Dataset

The ground-truth annotations in `tracking/gt_data/` cover three fruit categories across 44 video sequences captured from a mobile agricultural robot. File naming convention: `<date>_<id>_<camera>.txt` (field recordings) or `<number>.txt` (lab sequences).

GT file format (DarkLabel export → converted via `dataTransform.py`):
```
frame, id, x, y, w, h, 1, -1, -1, -1
```

---

## Citation

If you use this code, please cite the original SORT paper:

```bibtex
@inproceedings{Bewley2016_sort,
  author    = {Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
  booktitle = {2016 IEEE International Conference on Image Processing (ICIP)},
  title     = {Simple online and realtime tracking},
  year      = {2016},
  pages     = {3464--3468},
  doi       = {10.1109/ICIP.2016.7533003}
}
```

And the YOLOv7 paper:

```bibtex
@inproceedings{wang2023yolov7,
  title     = {YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle = {CVPR},
  year      = {2023}
}
```

---

## License

- Tracking code (`tracking/sortwithstep.py`): GPL-3.0 (inherited from SORT)
- Evaluation code (`evaluation/`): MIT (inherited from py-motmetrics)
- Detection code (`detection/`): GPL-3.0 (inherited from YOLOv7)
