# Fresh Fruit Basket Detection (YOLO) — MLOps starter

Detect fruit type + freshness in a whole basket photo with bounding boxes.
Example output: `[Apple-Fresh 0.94 @ (x1,y1,x2,y2), Banana-Rotten 0.91 @ (...)]`.

## Recommended dataset (exact source)
Roboflow Universe: `college-74jj5/freshness-fruits-and-vegetables`
- 9,283 images, 17 classes, CC BY 4.0, YOLOv8s baseline mAP@50 57.9%
- Classes are combined `fruit-freshness`: fresh apple/banana/bellpepper/carrot/
  cucumber/mango/orange/potato + rotten apple/banana/carrot/cucumber/mango/
  orange/potato/tomato/rottenbellpepper
- Alternatives: `saumil-evnba/freshness-fruits-and-vegetables-p8g5m` (22k imgs,
  RF-DETR), `kella/fresh-and-rottten-vegetable-dt85j` (9,164 imgs, veg-only),
  `master-pfe-2025/fruits-fresh-and-rotten` (51k imgs, 6 classes).

## Quickstart
1. `pip install -r requirements.txt`
2. Free Roboflow key at `app.roboflow.com` → `export ROBOFLOW_API_KEY=...`
3. `python scripts/download_dataset.py`  (YOLOv8 format into `data/raw/`)
4. `python scripts/prepare_data.py`
5. `python scripts/eda.py`              (plots + stats in `data/processed/`)
6. `python scripts/augment_from_kaggle.py`  (optional: 24k Kaggle imgs -> YOLO boxes, fixes rotten-tomato shortage)
7. `python src/train.py --model yolov8n.pt --epochs 50`
   All settings live in `configs/base.yaml`; any CLI flag overrides the file.
8. `python src/evaluate.py` / `python src/predict.py --image basket.jpg`
9. `uvicorn api.app:app --reload`
10. `.venv/bin/python -m streamlit run dashboard/app.py` — EDA explorer + detector UI
