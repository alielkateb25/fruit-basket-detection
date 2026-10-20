"""Basket inference. Config: configs/base.yaml (CLI overrides win)."""
import shutil
from pathlib import Path
from config import load

def main():
    p = load("predict")
    if not p.get("image"):
        raise SystemExit("Provide --image basket.jpg (or set predict.image in config).")
    from ultralytics import YOLO
    w = p["weights"] if Path(p["weights"]).exists() else "yolov8n.pt"
    m = YOLO(w)
    r = m.predict(p["image"], conf=p["conf"], save=True, project="runs", name="predict")[0]
    for b in r.boxes:
        print(f"{m.names[int(b.cls)]} {float(b.conf):.2f} xyxy={b.xyxy.tolist()}")
    saved = next(Path("runs/predict").glob(f"{Path(p['image']).stem}*"), None)
    if saved and saved.is_file():
        shutil.copy(saved, p["save"])
        print("saved:", p["save"])

if __name__ == "__main__":
    main()
