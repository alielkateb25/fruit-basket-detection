"""Validate weights: prints mAP50, mAP50-95, per-class AP. Config: configs/base.yaml."""
from pathlib import Path
from config import load, ROOT

def main():
    p = load("evaluate")
    w = p["weights"] if Path(p["weights"]).exists() else "yolov8n.pt"
    from ultralytics import YOLO
    m = YOLO(w)
    metrics = m.val(data=str(ROOT / "data" / "raw" / "data.yaml"), imgsz=p["imgsz"])
    print(f"mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} "
          f"P={metrics.box.mp:.3f} R={metrics.box.mr:.3f}")
    names = metrics.names if isinstance(metrics.names, dict) else dict(enumerate(metrics.names))
    for i, n in names.items():
        try:
            print(f"  {n}: AP50={metrics.box.ap50[i]:.3f}")
        except Exception:
            pass
    ok = metrics.box.map50 >= p.get("target_map50", 0)
    print(f"target mAP50>={p.get('target_map50')}: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
