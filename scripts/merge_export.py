"""Merge a Roboflow manual-annotation export into data/raw/ safely.

Workflow:
  1. In Roboflow Annotate, upload Kaggle images (bittergroud, capsicum, okra,
     fresh tomato, extra rotten tomato), draw boxes, export YOLOv8 -> zip.
  2. python scripts/merge_export.py --export /path/to/export.zip
Script appends NEW classes at the END of data.yaml (existing ids untouched),
remaps the export's label ids, and copies images+labels into
train/valid/test. Re-run prepare_data.py + eda.py afterwards.
"""
import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

def load_names(p):
    names = yaml.safe_load((p).read_text()).get("names", [])
    return list(names) if isinstance(names, list) else [names[str(i)] for i in range(len(names))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="Roboflow YOLOv8 export zip")
    a = ap.parse_args()

    cur_names = load_names(RAW / "data.yaml")
    tmp = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(a.export) as z:
        z.extractall(tmp)
    # find export root (contains data.yaml)
    exp_yaml = next(tmp.rglob("data.yaml"))
    exp_root = exp_yaml.parent
    exp_names = load_names(exp_yaml)

    mapping = {}  # old id -> new id
    for old_id, name in enumerate(exp_names):
        if name not in cur_names:
            cur_names.append(name)
            print(f"new class: {name} -> id {len(cur_names)-1}")
        mapping[old_id] = cur_names.index(name)

    n = 0
    for split in ("train", "valid", "test", "val"):
        src_img, src_lbl = exp_root / split / "images", exp_root / split / "labels"
        if not src_img.exists():
            continue
        dst = "valid" if split == "val" else split
        for img in src_img.iterdir():
            if not img.is_file():
                continue
            shutil.copy(img, RAW / dst / "images" / f"manual_{img.name}")
            lab = src_lbl / (img.stem + ".txt")
            lines = []
            if lab.exists():
                for line in lab.read_text().splitlines():
                    parts = line.split()
                    parts[0] = str(mapping[int(parts[0])])
                    lines.append(" ".join(parts))
            (RAW / dst / "labels" / f"manual_{img.stem}.txt").write_text("\n".join(lines) + "\n" if lines else "")
            n += 1
    cfg = yaml.safe_load((RAW / "data.yaml").read_text())
    cfg["names"] = cur_names
    (RAW / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.rmtree(tmp)
    print(f"merged {n} images, total classes now {len(cur_names)}")

if __name__ == "__main__":
    main()
