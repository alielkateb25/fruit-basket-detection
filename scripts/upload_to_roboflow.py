"""Upload data/raw/ (images + YOLO labels) to a Roboflow project, boxes intact.

1. Create empty project at app.roboflow.com (Object Detection), e.g. fruit-basket-full
2. export ROBOFLOW_API_KEY=...  (or in .env)
3. python scripts/upload_to_roboflow.py --workspace ali-el-kateb --project fruit-basket-full [--limit 50]

Labels upload alongside images; class names are created from data.yaml.
Skips files already uploaded in a previous run (tracks .upload_log.txt).
"""
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sleep", type=float, default=0.2, help="seconds between uploads")
    a = ap.parse_args()

    import os
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("Set ROBOFLOW_API_KEY (export or .env).")
    from roboflow import Roboflow
    rf = Roboflow(api_key=key)
    project = rf.workspace(a.workspace).project(a.project)

    log = RAW / ".upload_log.txt"
    done = set(log.read_text().splitlines()) if log.exists() else set()
    n, skipped = 0, 0
    with log.open("a") as lf:
        for split in ("train", "valid", "test"):
            imgdir, lbldir = RAW / split / "images", RAW / split / "labels"
            for img in sorted(imgdir.iterdir()):
                if not img.is_file() or img.name in done:
                    skipped += 1
                    continue
                lab = lbldir / (img.stem + ".txt")
                kw = {"annotation_path": str(lab)} if lab.exists() and lab.stat().st_size else {}
                project.upload(str(img), split=split, **kw)
                lf.write(img.name + "\n"); lf.flush()
                n += 1
                if a.limit and n >= a.limit:
                    print(f"reached --limit {a.limit}")
                    return
                time.sleep(a.sleep)
    print(f"uploaded={n} skipped_already={skipped}")

if __name__ == "__main__":
    main()
