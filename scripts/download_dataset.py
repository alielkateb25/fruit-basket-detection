"""Download the Roboflow basket dataset in YOLOv8 format. Idempotent."""
import os
import sys
from pathlib import Path

WORKSPACE = "college-74jj5"
PROJECT = "freshness-fruits-and-vegetables"
VERSION = 7  # 9,283 images, 17 classes, CC BY 4.0
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Bigger alternatives (change PROJECT/VERSION):
#   ("saumil-evnba", "freshness-fruits-and-vegetables-p8g5m", 2)  # ~22k imgs
#   ("kella", "fresh-and-rottten-vegetable-dt85j", 1)             # veg-only 9k
#   ("master-pfe-2025", "fruits-fresh-and-rotten", 4)             # ~51k imgs

def main():
    if (RAW / "train" / "images").exists() and (RAW / "data.yaml").exists():
        print(f"Dataset already present in {RAW}, skipping download.")
        return
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print("Missing ROBOFLOW_API_KEY.")
        print("1) Sign up free at https://app.roboflow.com -> Account -> API key")
        print("2) export ROBOFLOW_API_KEY=...")
        print(f"3) Dataset: universe.roboflow.com/{WORKSPACE}/{PROJECT}")
        sys.exit(1)
    from roboflow import Roboflow
    RAW.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=key)
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8", location=str(RAW))
    print("Downloaded:", ds.location)

if __name__ == "__main__":
    main()
