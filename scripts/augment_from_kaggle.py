"""Convert Kaggle single-item classification images into YOLO samples.

Each image gets one full-frame box (0.5 0.5 0.98 0.98) with the matching
Roboflow class id. Images are SYMLINKED (no 3GB copy).
Output: data/raw/kaggle_yolo/images/*.jpg + labels/*.txt
Merge into training with: cat kaggle_yolo labels into train/ (see printout).

Skipped (not in the 17 Roboflow classes): bittergroud, capsicum, okra,
fresh tomato (no 'fresh tomato' class in Roboflow v7).
"""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
KAG = RAW / "kaggle_cls"
OUT = RAW / "kaggle_yolo"

# Kaggle folder -> Roboflow class name
MAP = {
    "freshapples": "fresh apple", "rottenapples": "rotten apple",
    "freshbanana": "fresh banana", "rottenbanana": "rotten banana",
    "freshcucumber": "fresh cucumber", "rottencucumber": "rotten cucumber",
    "freshoranges": "fresh orange", "rottenoranges": "rotten orange",
    "freshpotato": "fresh potato", "rottenpotato": "rotten potato",
    "freshpatato": "fresh potato", "rottenpatato": "rotten potato",
    "rottentomato": "rotten tomato", "rottentamto": "rotten tomato",
}

def main():
    names = yaml.safe_load((RAW / "data.yaml").read_text())["names"]
    name2id = {n: i for i, n in enumerate(names)}
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    kept, skipped = 0, 0
    per_class = {}
    for split in ("Train", "Test"):
        for clsdir in sorted((KAG / split).iterdir()):
            rf_name = MAP.get(clsdir.name.lower())
            if rf_name is None or rf_name not in name2id:
                skipped += sum(1 for _ in clsdir.iterdir())
                continue
            cid = name2id[rf_name]
            for img in sorted(clsdir.iterdir()):
                if not img.is_file():
                    continue
                link = OUT / "images" / f"{split}_{clsdir.name}_{img.name}"
                if not link.exists():
                    link.symlink_to(img.resolve())
                (OUT / "labels" / (link.stem + ".txt")).write_text(f"{cid} 0.5 0.5 0.98 0.98\n")
                kept += 1
                per_class[rf_name] = per_class.get(rf_name, 0) + 1
    print(f"kept={kept} skipped={skipped} (no matching Roboflow class)")
    for k in sorted(per_class):
        print(f"  {k}: +{per_class[k]}")
    print(f"\nTo merge into train:\n  cp {OUT}/images/* {RAW}/train/images/\n  cp {OUT}/labels/* {RAW}/train/labels/")

if __name__ == "__main__":
    main()
