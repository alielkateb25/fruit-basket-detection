"""Train a YOLO detector. Config: configs/base.yaml (CLI overrides win)."""
from pathlib import Path
from config import load, ROOT

def main():
    p = load("train")
    data_yaml = ROOT / p.get("data_yaml", "data/raw/data.yaml")

    if not data_yaml.exists():
        raise SystemExit(f"Missing {data_yaml}. Run download + prepare scripts first.")

    use_mlflow = False
    try:
        import mlflow
        mlflow.set_experiment(p.get("mlflow_experiment", "fresh-fruit-yolo"))
        mlflow.start_run()
        mlflow.log_params({k: v for k, v in p.items() if not k.startswith("_")})
        use_mlflow = True
    except ImportError:
        pass

    from ultralytics import YOLO
    m = YOLO(p["model"])
    train_kwargs = dict(data=str(data_yaml), epochs=p["epochs"], imgsz=p["imgsz"],
                        batch=p["batch"], patience=p.get("patience", 15),
                        close_mosaic=p.get("close_mosaic", 10),
                        project=p.get("project", "runs/detect"),
                        name=p.get("name", "train"))
    if p.get("device"):
        train_kwargs["device"] = p["device"]
    m.train(**train_kwargs)
    best = Path(p.get("project", "runs/detect")) / p.get("name", "train") / "weights" / "best.pt"
    print(f"best weights: {best}")
    if use_mlflow:
        import mlflow
        if best.exists():
            mlflow.log_artifact(str(best))
        mlflow.end_run()

if __name__ == "__main__":
    main()
