"""Validate YOLO structure and (re)write data.yaml with absolute paths."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def main():
    found = [s for s in ("train", "valid", "test", "val") if (RAW / s / "images").exists()]
    if not found:
        raise SystemExit(f"No YOLO splits in {RAW}. Run download_dataset.py first.")
    for split in found:
        imgdir = RAW / split / "images"
        lbldir = RAW / split / "labels"
        n_img = len([p for p in imgdir.iterdir() if p.is_file()])
        n_lbl = len(list(lbldir.glob("*.txt"))) if lbldir.exists() else 0
        print(f"{split}: {n_img} images, {n_lbl} label files")
    val = "valid" if (RAW / "valid").exists() else "val"
    cfg = {
        "path": str(RAW),
        "train": "train/images",
        "val": f"{val}/images",
    }
    if (RAW / "test" / "images").exists():
        cfg["test"] = "test/images"
    old = RAW / "data.yaml"
    if old.exists():
        cfg["names"] = yaml.safe_load(old.read_text()).get("names", {})
    (RAW / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Wrote", RAW / "data.yaml")

if __name__ == "__main__":
    main()
