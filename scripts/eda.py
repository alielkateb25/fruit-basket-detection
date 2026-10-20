"""EDA for a YOLO-format detection dataset.

Reads data.yaml + train/valid labels and reports:
- images per split, boxes per class, boxes per image
- box size / aspect-ratio / center-position distributions
- images with zero labels, images per resolution
- NEW: cross-split near-duplicate check (dHash), fresh-vs-rotten color
  separation, crowdedness histogram, resolution spread
Saves plots + eda_summary.json to data/processed/.
Works on the real download AND on any YOLO fixture.
"""
import json
import random
from pathlib import Path
import os
from collections import Counter
from PIL import Image
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("YOLO_RAW", ROOT / "data" / "raw"))
OUT = Path(os.environ.get("YOLO_OUT", ROOT / "data" / "processed"))


def load_names():
    cfg = yaml.safe_load((RAW / "data.yaml").read_text())
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}, cfg
    return {int(k): v for k, v in names.items()}, cfg


def scan_split(split):
    imgdir = RAW / split / "images"
    lbldir = RAW / split / "labels"
    rows = []  # (img, w, h, cls, bw, bh, cx, cy)
    per_image = Counter()
    empty = 0
    for img in sorted(imgdir.glob("*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        with Image.open(img) as im:
            W, H = im.size
        lab = lbldir / (img.stem + ".txt")
        boxes = []
        if lab.exists() and lab.stat().st_size:
            for line in lab.read_text().splitlines():
                c, cx, cy, bw, bh = line.split()[:5]
                boxes.append((int(c), float(cx), float(cy), float(bw), float(bh)))
        if not boxes:
            empty += 1
        per_image[len(boxes)] += 1
        for c, cx, cy, bw, bh in boxes:
            rows.append({"img": img.name, "W": W, "H": H, "cls": c,
                         "bw": bw, "bh": bh, "cx": cx, "cy": cy,
                         "area": bw * bh, "ar": bw / max(bh, 1e-9)})
    return rows, empty, len(list(imgdir.glob("*"))), per_image


def dhash(path, size=9):
    """64-bit difference hash for near-duplicate detection."""
    with Image.open(path).convert("L").resize((size, 8), Image.BILINEAR) as im:
        px = list(im.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if px[r * size + c] > px[r * size + c + 1] else 0)
    return bits


def hash_split(split):
    imgdir = RAW / split / "images"
    out = {}
    for img in imgdir.iterdir():
        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                out[img.name] = dhash(img)
            except Exception:
                pass
    return out


def cross_split_dupes(hashes, thresh=6):
    """Count near-duplicate pairs across different splits (leakage check)."""
    res = {}
    splits = sorted(hashes)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            n = 0
            for name_a, ha in hashes[a].items():
                for name_b, hb in hashes[b].items():
                    if bin(ha ^ hb).count("1") <= thresh:
                        n += 1
                        break
            res[f"{a}x{b}"] = n
    return res


def fruit_of(name):
    s = name.lower().replace(" ", "")
    return s[5:] if s.startswith("fresh") else s[6:]


def cooccurrence(names):
    """Which fruits share basket images (guides realistic augmentation)."""
    fruits = sorted({fruit_of(n) for n in names.values()})
    idx = {f: i for i, f in enumerate(fruits)}
    mat = [[0] * len(fruits) for _ in fruits]
    for split in ("train", "valid", "test", "val"):
        imgdir = RAW / split / "images"
        if not imgdir.exists():
            continue
        for img in imgdir.iterdir():
            lab = (RAW / split / "labels") / (img.stem + ".txt")
            if not lab.exists():
                continue
            present = {fruit_of(names[int(l.split()[0])]) for l in lab.read_text().splitlines()}
            for a in present:
                for b in present:
                    mat[idx[a]][idx[b]] += 1
    return fruits, mat


def box_iou(a, b):
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax1, ay1 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx1, by1 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix, iy = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-9)


def occlusion_stats(names):
    """Mean pairwise box IoU + share of crowded images (NMS guidance)."""
    ious, crowded = [], 0
    n_img = 0
    for split in ("train", "valid", "test", "val"):
        imgdir = RAW / split / "images"
        if not imgdir.exists():
            continue
        for img in imgdir.iterdir():
            lab = (RAW / split / "labels") / (img.stem + ".txt")
            if not lab.exists():
                continue
            boxes = [tuple(map(float, l.split()[1:5])) for l in lab.read_text().splitlines()]
            n_img += 1
            if len(boxes) < 2:
                continue
            best = 0
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    best = max(best, box_iou(boxes[i], boxes[j]))
            ious.append(best)
            if best > 0.3:
                crowded += 1
    return {"images": n_img, "mean_max_iou": round(sum(ious) / max(len(ious), 1), 3),
            "crowded_share": round(crowded / max(n_img, 1), 3)}


def color_stats(names, n_sample=300, seed=0):
    """Mean brightness of fresh vs rotten box crops (learnability check)."""
    rng = random.Random(seed)
    groups = {"fresh": [], "rotten": []}
    for split in ("train", "valid", "test", "val"):
        imgdir = RAW / split / "images"
        if not imgdir.exists():
            continue
        for img in imgdir.iterdir():
            lab = (RAW / split / "labels") / (img.stem + ".txt")
            if not lab.exists():
                continue
            for line in lab.read_text().splitlines():
                p = line.split()
                kind = "rotten" if "rotten" in str(names.get(int(p[0]), "")).lower() else "fresh"
                groups[kind].append((img, tuple(map(float, p[1:5]))))
    out = {}
    for kind, items in groups.items():
        vals = []
        for img, (cx, cy, bw, bh) in rng.sample(items, min(n_sample, len(items))):
            try:
                with Image.open(img).convert("RGB") as im:
                    W, H = im.size
                    crop = im.crop((int((cx - bw / 2) * W), int((cy - bh / 2) * H),
                                    int((cx + bw / 2) * W), int((cy + bh / 2) * H)))
                    g = crop.convert("L").resize((32, 32))
                    vals.append(sum(g.getdata()) / (32 * 32 * 255))
            except Exception:
                pass
        out[kind] = {"n": len(vals),
                     "mean": round(sum(vals) / max(len(vals), 1), 3)}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names, cfg = load_names()
    print(f"classes ({len(names)}): {[names[i] for i in sorted(names)]}")
    all_rows, summary = [], {"classes": len(names), "splits": {}}
    crowded, resolutions = Counter(), {}
    for split in ("train", "valid", "val", "test"):
        if not (RAW / split / "images").exists():
            continue
        rows, empty, n_img, per_image = scan_split(split)
        all_rows += [dict(r, split=split) for r in rows]
        cnt = Counter(r["cls"] for r in rows)
        print(f"\n[{split}] images={n_img} boxes={len(rows)} empty={empty} "
              f"avg_boxes/img={len(rows)/max(n_img,1):.2f}")
        for c in sorted(cnt):
            print(f"  {names.get(c,c)}: {cnt[c]}")
        summary["splits"][split] = {"images": n_img, "boxes": len(rows), "empty": empty,
                                    "per_class": {names.get(c, c): cnt[c] for c in sorted(cnt)}}
        for k, v in per_image.items():
            crowded[k] += v
        ws = sorted({r["W"] for r in rows} | {r["H"] for r in rows})
        resolutions[split] = {"min_side": min(ws) if ws else 0, "max_side": max(ws) if ws else 0}
    if not all_rows:
        print("No boxes found.")
        return
    # plot 1: boxes per class (train only)
    tr = [r for r in all_rows if r["split"] == "train"] or all_rows
    cnt = Counter(r["cls"] for r in tr)
    labels = [names.get(c, c) for c in sorted(cnt)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, [cnt[c] for c in sorted(cnt)])
    ax.set_title("Boxes per class (train)")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(OUT / "eda_boxes_per_class.png", dpi=120, bbox_inches="tight")
    # plot 2: box area histogram
    fig, ax = plt.subplots()
    ax.hist([r["area"] for r in tr], bins=40)
    ax.set_title("Box area (fraction of image)")
    ax.set_xlabel("w*h"); ax.set_ylabel("count")
    fig.savefig(OUT / "eda_box_area.png", dpi=120, bbox_inches="tight")
    # plot 3: box centers (positional bias)
    fig, ax = plt.subplots()
    ax.scatter([r["cx"] for r in tr], [r["cy"] for r in tr], s=4)
    ax.set_title("Box centers (cx, cy)"); ax.set_xlim(0, 1); ax.set_ylim(1, 0)
    fig.savefig(OUT / "eda_box_centers.png", dpi=120, bbox_inches="tight")
    print(f"\nSaved plots to {OUT}")

    # integrity warnings
    small = sum(1 for r in tr if r["area"] < 0.002)
    print(f"tiny boxes (<0.2% of image): {small} — hard for nano models, consider imgsz 960")
    fresh = sum(1 for r in tr if "fresh" in str(names.get(r['cls'], '')).lower()
                and "rotten" not in str(names.get(r['cls'], '')).lower())
    rotten = sum(1 for r in tr if "rotten" in str(names.get(r['cls'], '')).lower())
    print(f"fresh boxes={fresh} rotten boxes={rotten} ratio={fresh/max(rotten,1):.2f}")

    # plot 4: crowdedness (boxes per image, all splits)
    fig, ax = plt.subplots()
    xs = sorted(crowded)
    ax.bar(xs, [crowded[x] for x in xs])
    ax.set_title("Crowdedness: images by box count")
    ax.set_xlabel("boxes/image"); ax.set_ylabel("images")
    fig.savefig(OUT / "eda_crowdedness.png", dpi=120, bbox_inches="tight")

    # resolutions
    print(f"resolutions: {resolutions}")

    # cross-split near-duplicates (leakage check)
    print("hashing images for duplicate check...")
    hashes = {s: hash_split(s) for s in summary["splits"]}
    dupes = cross_split_dupes(hashes)
    for k, v in dupes.items():
        print(f"  near-dupes {k}: {v}")
    summary["cross_split_dupes"] = dupes

    # fresh vs rotten color separation (learnability check)
    colors = color_stats(names)
    print(f"brightness fresh={colors['fresh']} rotten={colors['rotten']}")
    summary["brightness"] = colors
    summary["resolutions"] = resolutions
    summary["tiny_boxes"] = small

    # co-occurrence heatmap (which fruits share baskets)
    fruits, mat = cooccurrence(names)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mat)
    ax.set_xticks(range(len(fruits)), fruits, rotation=45, ha="right")
    ax.set_yticks(range(len(fruits)), fruits)
    ax.set_title("Basket co-occurrence (images sharing each pair)")
    fig.savefig(OUT / "eda_cooccurrence.png", dpi=120, bbox_inches="tight")
    summary["cooccurrence"] = {"fruits": fruits, "matrix": mat}

    # occlusion (NMS guidance)
    occ = occlusion_stats(names)
    print(f"occlusion: mean_max_iou={occ['mean_max_iou']} crowded_share={occ['crowded_share']}")
    summary["occlusion"] = occ

    # per-class median box area (size skew check)
    by_cls = {}
    for r in all_rows:
        by_cls.setdefault(names.get(r["cls"], r["cls"]), []).append(r["area"])
    med = {k: round(sorted(v)[len(v) // 2], 4) for k, v in sorted(by_cls.items())}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(list(med), list(med.values()))
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Median box area per class (fraction of image)")
    fig.savefig(OUT / "eda_size_per_class.png", dpi=120, bbox_inches="tight")
    summary["median_area"] = med

    # eval coverage (which test scores to trust)
    thin = []
    for split, need in (("valid", 20), ("test", 10)):
        for cls_name, c in summary["splits"].get(split, {}).get("per_class", {}).items():
            if c < need:
                thin.append(f"{cls_name} has {c} boxes in {split}")
    summary["thin_eval"] = thin
    print("thin eval:", thin if thin else "none")
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote eda_summary.json")


if __name__ == "__main__":
    main()
