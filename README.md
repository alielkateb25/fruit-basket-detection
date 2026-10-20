# Fresh Fruit Basket Detection

**MLOps pipeline for detecting fruit type and freshness in whole-basket photos using YOLO object detection.**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Framework](https://img.shields.io/badge/Framework-Ultralytics%20YOLOv8-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [API](#api)
- [Dashboard](#dashboard)
- [EDA Summary](#eda-summary)
- [Model Selection](#model-selection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project implements a complete MLOps pipeline for **post-harvest quality control**: given a photo of a fruit basket, the model detects each item with a bounding box and classifies it as `fresh` or `rotten` per fruit type (e.g. `fresh apple`, `rotten tomato`). The output drives an automated sorting decision — `REMOVE_ROTTEN` or `OK`.

**Problem:** 20–30% of fruits and vegetables are lost post-harvest. One rotten item contaminates the entire crate. Manual sorting is slow, inconsistent, and unavailable to smallholder farmers.

**Solution:** A YOLO-based object detector that processes a basket photo in ~3ms (nano) to ~6ms (small), deployable as a REST API accessible from any phone browser.

**Pipeline:** Data acquisition → EDA → Preprocessing → Training → Evaluation → Deployment → Monitoring

---

## Features

- **End-to-end MLOps pipeline** — from raw dataset to deployed API
- **YOLO object detection** — 18 combined `fruit-freshness` classes
- **Interactive Streamlit dashboard** — EDA explorer, live detector, config viewer
- **FastAPI REST service** — phone-accessible inference endpoint
- **MLflow experiment tracking** — params, metrics, artifacts
- **DVC-ready data versioning** — `.dvc` pointer files for reproducibility
- **Comprehensive EDA** — class balance, box statistics, leakage detection, occlusion analysis, color separation
- **Manual annotation workflow** — Roboflow Annotate → export → merge pipeline
- **Config-driven training** — single `configs/base.yaml` with CLI overrides

---

## Project Structure

```
fresh-fruit-mlops/
├── configs/
│   └── base.yaml              # Model, training, evaluation, API settings
├── data/
│   ├── raw/                   # YOLO-format dataset (git-ignored, DVC-tracked)
│   │   ├── train/{images,labels}/
│   │   ├── valid/{images,labels}/
│   │   ├── test/{images,labels}/
│   │   ├── data.yaml          # 18 class definitions
│   │   ├── kaggle_cls/        # 30k Kaggle single-item images (source)
│   │   └── kaggle_yolo/       # 24k symlinked YOLO samples from Kaggle
│   └── processed/             # EDA plots + eda_summary.json
├── scripts/
│   ├── download_dataset.py    # Roboflow → YOLOv8 format
│   ├── prepare_data.py        # Validate splits, write data.yaml
│   ├── eda.py                 # Full EDA: stats, plots, leakage, occlusion
│   ├── augment_from_kaggle.py # Kaggle images → full-frame YOLO boxes
│   ├── merge_export.py        # Merge Roboflow manual annotations
│   └── upload_to_roboflow.py  # Upload dataset to Roboflow
├── src/
│   ├── config.py              # Shared config loader (yaml + CLI overrides)
│   ├── train.py               # YOLO training with MLflow logging
│   ├── evaluate.py            # mAP50, mAP50-95, per-class AP
│   └── predict.py             # Basket inference → labeled image
├── api/
│   └── app.py                 # FastAPI POST /predict → detections + action
├── dashboard/
│   └── app.py                 # Streamlit: Overview, EDA, Detector, Config
├── pyproject.toml             # Project metadata + dependencies
├── requirements.txt           # Pinned dependencies
├── .gitignore                 # Excludes data/, .env, runs/, tabular_analysis/
├── README.md                  # This file
├── STATUS.md                  # Detailed project status + findings
└── STRUCTURE.txt              # Quick file tree reference
```

---

## Dataset

### Primary Source

**Roboflow Universe:** `college-74jj5/freshness-fruits-and-vegetables` (v7)
- **9,283 images** with bounding box annotations
- **17 combined classes** (`fresh apple` … `rotten tomato`)
- **License:** CC BY 4.0
- **URL:** https://universe.roboflow.com/college-74jj5/freshness-fruits-and-vegetables
- **Published baseline:** YOLOv8s mAP@50 = 57.9%

### Augmentation Source

**Kaggle:** `swoyam2609/fresh-and-stale-classification`
- **30,357 single-item images** (no boxes)
- Converted to **24,102 YOLO samples** via `scripts/augment_from_kaggle.py`
- Fills the `rotten tomato` gap (13 → 149 train boxes)
- **URL:** https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification

### Manual Annotations

**Roboflow Annotate** (Label Assist + manual review)
- **658 images** with 763 boxes
- Added `fresh tomato` class (new id 17)
- Merged via `scripts/merge_export.py`
- **Final dataset:** 9,941 images, 18 classes, ~2.3 boxes/image

### Class List (18)

| ID | Class | Train Boxes |
|----|-------|-------------|
| 0 | fresh apple | 1,279 |
| 1 | fresh banana | 1,007 |
| 2 | fresh bellpepper | 1,735 |
| 3 | fresh carrot | 1,638 |
| 4 | fresh cucumber | 1,185 |
| 5 | fresh mango | 797 |
| 6 | fresh orange | 984 |
| 7 | fresh potato | 1,412 |
| 8 | rotten apple | 645 |
| 9 | rotten banana | 574 |
| 10 | rotten carrot | 1,038 |
| 11 | rotten cucumber | 537 |
| 12 | rotten mango | 528 |
| 13 | rotten orange | 458 |
| 14 | rotten potato | 1,274 |
| 15 | rotten tomato | 149 |
| 16 | rottenbellpepper | 816 |
| 17 | fresh tomato | 159 |

---

## Installation

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Free [Roboflow](https://app.roboflow.com) account for dataset download

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/fresh-fruit-mlops.git
cd fresh-fruit-mlops

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Set Roboflow API key
export ROBOFLOW_API_KEY=your_key_here
```

---

## Quickstart

```bash
# 1. Download dataset (Roboflow → YOLOv8 format)
python scripts/download_dataset.py

# 2. Validate and prepare
python scripts/prepare_data.py

# 3. Run EDA (plots → data/processed/)
python scripts/eda.py

# 4. (Optional) Augment with Kaggle images
python scripts/augment_from_kaggle.py

# 5. Train
python src/train.py --model yolov8n.pt --epochs 50

# 6. Evaluate
python src/evaluate.py

# 7. Predict on a basket photo
python src/predict.py --image basket.jpg

# 8. Serve API
uvicorn api.app:app --reload

# 9. Launch dashboard
.venv/bin/python -m streamlit run dashboard/app.py
```

---

## Configuration

All settings live in [`configs/base.yaml`](configs/base.yaml). CLI flags override the file.

```yaml
data:
  yaml: data/raw/data.yaml
  nc: 18

train:
  model: yolov8n.pt        # yolov8n | yolov8s | yolo11n | yolo11s
  epochs: 50
  imgsz: 640               # 960 if nano misses small items
  batch: 16
  patience: 15
  close_mosaic: 10

evaluate:
  weights: runs/detect/train/weights/best.pt
  imgsz: 640
  target_map50: 0.579
  target_rotten_recall: 0.85

predict:
  conf: 0.35

api:
  conf: 0.35
  host: 0.0.0.0
  port: 8000
```

---

## Training

```bash
# Nano (phone/CPU demo)
python src/train.py --model yolov8n.pt --epochs 50

# Small (accuracy)
python src/train.py --model yolov8s.pt --epochs 80

# Custom config file
python src/train.py --config configs/base.yaml --epochs 100
```

Training logs to MLflow (`mlruns/`). Best weights saved to `runs/detect/train/weights/best.pt`.

---

## Evaluation

```bash
python src/evaluate.py
```

**Targets:**
- mAP@50 ≥ **0.579** (beat published YOLOv8s baseline)
- Rotten-class recall **> 0.85**

Per-class AP printed for all 18 classes. PASS/FAIL reported against targets.

---

## Inference

```bash
python src/predict.py --image basket.jpg --conf 0.35 --save pred_basket.jpg
```

Output: labeled image + console list of `label confidence xyxy` per box.

---

## API

```bash
uvicorn api.app:app --reload
```

**Endpoint:** `POST /predict`

**Request:** multipart form with `f` (image file), optional `conf` (float)

**Response:**
```json
{
  "detections": [
    {"label": "fresh apple", "conf": 0.94, "xyxy": [120, 80, 340, 300]},
    {"label": "rotten banana", "conf": 0.91, "xyxy": [400, 150, 580, 380]}
  ],
  "n": 2,
  "n_rotten": 1,
  "action": "REMOVE_ROTTEN"
}
```

Access from any phone browser on the same network.

---

## Dashboard

```bash
.venv/bin/python -m streamlit run dashboard/app.py
```

**Detection section:**
- **Overview** — dataset metrics, class table
- **EDA explorer** — grouped fresh/rotten bars, box-center scatter, sample browser, leakage/occlusion/color findings
- **Basket detector** — upload photo → boxes → REMOVE_ROTTEN/OK
- **Config & targets** — live config, model comparison table

---

## EDA Summary

Key findings from `scripts/eda.py` (see `data/processed/eda_summary.json`):

| Finding | Value | Implication |
|---------|-------|-------------|
| Images | 9,941 | Sufficient for YOLO training |
| Boxes/image | 2.3 | Real basket scenes |
| Fresh:rotten ratio | 1.7:1 | Acceptable; monitor per-class recall |
| Cross-split near-dupes | 1,660 train×valid | Leakage — valid mAP optimistic |
| Mean max box-IoU | 0.178 | Neatly arranged; default NMS fine |
| Box-crop brightness | fresh 0.46 / rotten 0.51 | Weak color cue; model must learn texture |
| Tiny boxes (<0.2%) | 41 | Consider `imgsz 960` if nano misses them |
| Thin eval classes | rotten tomato (4 test) | Don't trust until more test boxes |

---

## Model Selection

| Model | Params | Speed | Evidence |
|-------|--------|-------|----------|
| **YOLOv8n** | 3.2M | ~3ms | Start here — phone/CPU |
| **YOLOv8s** | 11M | ~6ms | Best generalization in fruit studies (mAP@50 83.8%) |
| **YOLO11s** | ~9M | ~15ms | Lower latency; 78.6% mAP at 70 FPS edge |
| **YOLOv9t** | 2.9M | ~8ms | Top fruit benchmark (0.981 mAP50) |
| **RF-DETR-M** | medium | slower | Transformer; pretrained on 22k fork |

**Not recommended:** Faster R-CNN — slowest + lowest mAP in every cited comparison.

---

## Results

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| mAP@50 | TBD | ≥ 0.579 | Pending training |
| Rotten recall | TBD | > 0.85 | Pending training |
| Inference (nano) | ~3ms | — | Verified |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

Dataset licenses:
- Roboflow base data: CC BY 4.0
- Kaggle augmentation data: Apache 2.0

---

## Acknowledgments

- **Roboflow Universe** — `college-74jj5/freshness-fruits-and-vegetables` dataset
- **Kaggle** — `swoyam2609/fresh-and-stale-classification` dataset
- **Ultralytics** — YOLOv8 framework
- **FAO** — food loss statistics (20–30% post-harvest)
- **MLOps course** — project structure and methodology
