"""Fresh Fruit MLOps — unified dashboard.
Detection (YOLO basket) + Tabular (Crop Recommendation) in one app.
Run: .venv/bin/python -m streamlit run dashboard/app.py
"""
from pathlib import Path
import sys
from collections import Counter
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
RAW = ROOT / "data" / "raw"
CSV_PATH = ROOT / "tabular_analysis" / "data" / "Crop_recommendation.csv"

st.set_page_config(page_title="Fresh Fruit MLOps", layout="wide")

# ══ DETECTION HELPERS ════════════════════════════════════════════════════════
@st.cache_data
def det_names():
    cfg = yaml.safe_load((RAW / "data.yaml").read_text())
    names = cfg.get("names", [])
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)

@st.cache_data
def det_stats():
    out = {}
    for split in ("train", "valid", "test"):
        imgdir, lbldir = RAW / split / "images", RAW / split / "labels"
        if not imgdir.exists():
            continue
        cnt, n_img, empty = Counter(), 0, 0
        for img in imgdir.iterdir():
            if not img.is_file():
                continue
            n_img += 1
            lab = lbldir / (img.stem + ".txt")
            if not lab.exists() or not lab.stat().st_size:
                empty += 1
                continue
            for line in lab.read_text().splitlines():
                cnt[int(line.split()[0])] += 1
        out[split] = {"images": n_img, "boxes": sum(cnt.values()),
                      "empty": empty, "per_class": dict(cnt)}
    return out

@st.cache_resource
def det_model():
    from ultralytics import YOLO
    w = ROOT / "runs/detect/train/weights/best.pt"
    return YOLO(str(w) if w.exists() else "yolov8n.pt"), (str(w) if w.exists() else "yolov8n.pt (COCO fallback)")

def draw_boxes(img, boxes, names):
    d = ImageDraw.Draw(img)
    for cls_id, conf, xyxy in boxes:
        x0, y0, x1, y1 = xyxy
        d.rectangle([x0, y0, x1, y1], outline="red", width=3)
        d.text((x0 + 2, y0 + 2), f"{names[cls_id]} {conf:.2f}", fill="red")
    return img

@st.cache_data
def box_centers(split, _names):
    rows = []
    imgdir, lbldir = RAW / split / "images", RAW / split / "labels"
    for img in imgdir.iterdir():
        if not img.is_file():
            continue
        lab = lbldir / (img.stem + ".txt")
        if not lab.exists():
            continue
        for line in lab.read_text().splitlines():
            p = line.split()
            c, cx, cy = int(p[0]), float(p[1]), float(p[2])
            kind = "rotten" if "rotten" in _names[c].lower() else "fresh"
            rows.append({"cx": cx, "cy": cy, "class": kind})
    return pd.DataFrame(rows)

# ══ TABULAR HELPERS ══════════════════════════════════════════════════════════
@st.cache_data
def tab_raw():
    return pd.read_csv(CSV_PATH)

def encode_target(s):
    return s.astype("category").cat.codes

def apply_pipeline(df, target, scaler, outliers, test_size, drop_cols):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    work = df.drop(columns=[c for c in drop_cols if c in df.columns and c != target], errors="ignore")
    num = work.select_dtypes(include=[np.number]).columns.tolist()
    if target in num:
        num.remove(target)
    X = work.drop(columns=[target])
    y = encode_target(work[target]) if work[target].dtype == object else work[target]
    if scaler != "none":
        sc = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[scaler]
        X[num] = sc().fit_transform(X[num])
    n_before = len(X)
    if outliers != "none":
        if outliers == "iqr":
            Q1, Q3 = X[num].quantile(0.25), X[num].quantile(0.75)
            mask = ~((X[num] < (Q1 - 1.5 * (Q3 - Q1))) | (X[num] > (Q3 + 1.5 * (Q3 - Q1)))).any(axis=1)
        else:
            from scipy.stats import zscore
            mask = (np.abs(zscore(X[num])) < 3).all(axis=1)
        X, y = X[mask], y[mask]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    m = {"rows_before": n_before, "rows_after": len(X),
         "removed": n_before - len(X), "pct_removed": round(100 * (n_before - len(X)) / max(n_before, 1), 1),
         "n_features": X.shape[1], "n_train": len(X_tr), "n_test": len(X_te),
         "scaler": scaler, "outliers": outliers, "test_size": test_size,
         "drop_cols": drop_cols, "target": target}
    return X_tr, X_te, y_tr, y_te, m, X, y

# ══ SIDEBAR NAVIGATION ═══════════════════════════════════════════════════════
st.sidebar.title("Fresh Fruit MLOps")
section = st.sidebar.radio("Section", ["🔍 Detection (YOLO)", "📊 Tabular (Crop)"])

# ══════════════════════════════════════════════════════════════════════════════
# DETECTION SECTION
# ══════════════════════════════════════════════════════════════════════════════
if section.startswith("🔍"):
    st.title("Fruit Basket Detection — YOLO")
    st.caption("Detect fruit type + freshness in a basket photo → REMOVE_ROTTEN / OK")
    page = st.sidebar.radio("Page", ["Overview", "EDA explorer", "Basket detector", "Config & targets"])
    names, stats = det_names(), det_stats()

    if page == "Overview":
        st.markdown("Detect **fruit type + freshness** in a basket photo. "
                    "`photo → boxes → REMOVE_ROTTEN / OK`")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images", sum(s["images"] for s in stats.values()))
        c2.metric("Boxes", sum(s["boxes"] for s in stats.values()))
        c3.metric("Classes", len(names))
        c4.metric("Fresh:rotten (train)",
                  f"{sum(v for k, v in stats['train']['per_class'].items() if 'fresh' in names[k].lower() and 'rotten' not in names[k].lower())}"
                  f":{sum(v for k, v in stats['train']['per_class'].items() if 'rotten' in names[k].lower())}")
        st.subheader("Classes")
        st.table([{"id": i, "class": n, "train boxes": stats.get("train", {}).get("per_class", {}).get(i, 0)}
                  for i, n in enumerate(names)])

    elif page == "EDA explorer":
        split = st.selectbox("Split", [s for s in ("train", "valid", "test") if s in stats])
        per = stats[split]["per_class"]
        norm = {k: names[k].lower().replace(" ", "") for k in per}
        fruits = sorted({s[5:] if s.startswith("fresh") else s[6:] for s in norm.values()})
        fv = [next((v for k, v in per.items() if norm[k] == f"fresh{f}"), 0) for f in fruits]
        rv = [next((v for k, v in per.items() if norm[k] == f"rotten{f}"), 0) for f in fruits]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(fruits))
        ax.bar([i - 0.2 for i in x], fv, width=0.4, color="#2ECC71", label="fresh")
        ax.bar([i + 0.2 for i in x], rv, width=0.4, color="#A93226", label="rotten")
        ax.set_xticks(list(x), fruits, rotation=45, ha="right")
        ax.set_title(f"Boxes per fruit: fresh vs rotten ({split})")
        ax.legend(); st.pyplot(fig); plt.close(fig)
        st.caption(f"{stats[split]['images']} images, {stats[split]['boxes']} boxes, "
                   f"{stats[split]['empty']} empty, avg {stats[split]['boxes']/max(stats[split]['images'],1):.2f} boxes/img")
        st.subheader("Browse samples")
        cls = st.selectbox("Class", names)
        cid = names.index(cls)
        shown = 0
        cols = st.columns(4)
        for img in sorted((RAW / split / "images").iterdir()):
            if shown >= 8:
                break
            lab = RAW / split / "labels" / (img.stem + ".txt")
            if not lab.exists():
                continue
            boxes = []
            for line in lab.read_text().splitlines():
                p = line.split()
                if int(p[0]) == cid:
                    boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            if not boxes:
                continue
            im = Image.open(img).convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            for cx, cy, bw, bh in boxes:
                d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H,
                             (cx + bw / 2) * W, (cy + bh / 2) * H], outline="red", width=3)
            cols[shown % 4].image(im, caption=img.name, width="stretch")
            shown += 1
        st.subheader("Where are the boxes? (center positions)")
        df = box_centers(split, names)
        st.scatter_chart(df, x="cx", y="cy", color="class")
        st.caption("Uniform cloud = no positional bias.")
        proc = ROOT / "data" / "processed"
        if (proc / "eda_crowdedness.png").exists():
            st.subheader("Crowdedness, leakage & color")
            st.image(str(proc / "eda_crowdedness.png"), caption="Images by box count")
        summ = {}
        if (proc / "eda_summary.json").exists():
            import json as _json
            summ = _json.loads((proc / "eda_summary.json").read_text())
        if summ.get("cross_split_dupes"):
            st.markdown("**Cross-split near-duplicates** (leakage inflates mAP):")
            st.table([{"pair": k, "pairs": v} for k, v in summ["cross_split_dupes"].items()])
        if summ.get("brightness"):
            b = summ["brightness"]
            st.markdown(f"**Box-crop brightness:** fresh {b['fresh']['mean']} vs rotten {b['rotten']['mean']}. "
                        "Clear gap = learnable cue; overlap = needs stronger augmentation.")
        if summ.get("occlusion"):
            o = summ["occlusion"]
            st.markdown(f"**Occlusion:** mean max box-IoU **{o['mean_max_iou']}**, crowded share **{o['crowded_share']}**.")
        if summ.get("thin_eval"):
            st.warning("Thin eval coverage — don't trust: " + "; ".join(summ["thin_eval"]))

    elif page == "Basket detector":
        model, wmsg = det_model()
        st.info(f"Weights: `{wmsg}`")
        conf = st.slider("Confidence", 0.05, 0.95, 0.35, 0.05)
        up = st.file_uploader("Upload a basket photo", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(BytesIO(up.read())).convert("RGB")
            r = model.predict(img, conf=conf, verbose=False)[0]
            dets = [(int(b.cls), float(b.conf), [float(v) for v in b.xyxy[0].tolist()]) for b in r.boxes]
            rotten = sum(1 for c, _, _ in dets if "rotten" in names[c].lower())
            c1, c2, c3 = st.columns(3)
            c1.metric("Detections", len(dets)); c2.metric("Rotten", rotten)
            c3.metric("Action", "REMOVE_ROTTEN" if rotten else "OK")
            st.image(draw_boxes(img.copy(), dets, names), width="stretch")
            st.table([{"label": names[c], "conf": round(cf, 3)} for c, cf, _ in dets])

    else:  # Config & targets
        st.subheader("configs/base.yaml")
        st.code((ROOT / "configs" / "base.yaml").read_text(), language="yaml")
        st.subheader("Evaluation targets")
        st.markdown("- mAP@50 ≥ **0.579** (published YOLOv8s baseline)\n"
                    "- Rotten-class recall **> 0.85**\n"
                    "- `python src/evaluate.py` prints PASS/FAIL")
        st.subheader("Candidate models")
        st.table([
            {"model": "YOLOv8n", "params": "3.2M", "speed": "~3ms", "use": "Start — phone/CPU", "status": "next"},
            {"model": "YOLOv8s", "params": "11M", "speed": "~6ms", "use": "Accuracy; best generalization in fruit studies", "status": "planned"},
            {"model": "YOLO11s", "params": "~9M", "speed": "~15ms", "use": "Lower latency; 78.6% mAP at 70 FPS edge", "status": "optional"},
            {"model": "YOLOv9t", "params": "2.9M", "speed": "~8ms", "use": "Top fruit benchmark (0.981 mAP50)", "status": "optional"},
            {"model": "RF-DETR-M", "params": "medium", "speed": "slower", "use": "Transformer; pretrained on 22k fork", "status": "optional"},
        ])
        st.caption("Skipped: Faster R-CNN — slowest + lowest mAP. Optimize rotten recall, not just mAP.")

# ══════════════════════════════════════════════════════════════════════════════
# TABULAR SECTION
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("Crop Recommendation — Tabular Console")
    st.caption("7 soil/climate features → best crop (22 classes). Target: `label` (crop name).")
    tpage = st.sidebar.radio("Page", ["Overview", "Pipeline", "Analysis"])
    raw, target = tab_raw(), "label"

    if tpage == "Overview":
        st.subheader("About this dataset")
        st.markdown("""
        **Crop Recommendation Dataset** (Kaggle: atharvaingle, Apache 2.0)

        **Purpose:** Help farmers and agricultural advisors decide which crop to
        plant on a given field by learning the soil-nutrient and climate conditions
        under which each crop thrives. The dataset was built by augmenting publicly
        available Indian rainfall, climate and fertilizer records into a clean
        tabular form.

        **Why it matters:** Choosing the wrong crop for a soil type or season leads
        to low yields, wasted fertilizer and financial loss — especially for
        smallholder farmers. A data-driven recommendation turns 7 simple field
        measurements into an actionable planting decision, and serves as a classic
        multi-class classification benchmark (22 crops).

        **Source:** Kaggle — `atharvaingle/crop-recommendation-dataset`
        (built from Indian rainfall / climate / fertilizer data).
        """)
        st.subheader("Feature meanings")
        st.markdown("""
        | Feature | Meaning | Unit | Role |
        |---|---|---|---|
        | **N** | Nitrogen content in soil | mg/kg | Nutrient — leaf/stem growth |
        | **P** | Phosphorus content in soil | mg/kg | Nutrient — root/flower development |
        | **K** | Potassium content in soil | mg/kg | Nutrient — disease resistance, water regulation |
        | **temperature** | Average air temperature | °C | Climate — growing-season warmth |
        | **humidity** | Average relative humidity | % | Climate — moisture availability |
        | **ph** | Soil pH (acidity/alkalinity) | pH scale | Soil — nutrient uptake range |
        | **rainfall** | Total rainfall | mm | Climate — water supply |
        | **label** | Recommended crop | 22 classes | **Target** — what to predict |

        **How to read a row:** `N=90, P=42, K=43, temp=20.9°C, humidity=82%, pH=6.5,
        rainfall=203mm → rice`. Each row is one field observation; the model learns
        the nutrient+climate signature of each crop.
        """)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", len(raw)); c2.metric("Columns", len(raw.columns))
        c3.metric("Missing cells", int(raw.isnull().sum().sum()))
        c4.metric("Target classes", raw[target].nunique())
        st.subheader("Data preview")
        st.dataframe(raw.head(20), width="stretch")

    elif tpage == "Pipeline":
        st.sidebar.markdown("**PARAMETERS**")
        drop_cols = st.sidebar.multiselect("Drop columns", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        scaler = st.sidebar.selectbox("Scaler", ["standard", "minmax", "robust", "none"])
        outliers = st.sidebar.radio("Outliers", ["none", "iqr", "zscore"])
        test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
        try:
            X_tr, X_te, y_tr, y_te, m, X_full, y_full = apply_pipeline(raw, target, scaler, outliers, test_size, drop_cols)
        except Exception as e:
            st.error(f"Pipeline error: {e}"); st.stop()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows before", m["rows_before"]); c2.metric("Rows after", m["rows_after"])
        c3.metric("Removed", f"{m['removed']} ({m['pct_removed']}%)"); c4.metric("Features", m["n_features"])
        cfg = {"data": {"path": str(CSV_PATH)}, "preprocessing": {"scaler": scaler, "drop_cols": drop_cols},
               "outliers": {"method": outliers}, "split": {"test_size": test_size, "random_state": 42}}
        st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")
        st.markdown(f"**Split:** {m['n_train']} train / {m['n_test']} test")
        st.subheader("Before vs After")
        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.markdown("**Raw**"); st.dataframe(raw.drop(columns=[target]).head(10), width="stretch")
        with col_proc:
            st.markdown("**Processed**")
            proc_df = pd.concat([X_full.head(10).reset_index(drop=True),
                                 pd.DataFrame({target: y_full.head(10).values})], axis=1)
            st.dataframe(proc_df, width="stretch")
        out = pd.concat([X_full, pd.Series(y_full, name=target)], axis=1)
        st.download_button("Download processed CSV", out.to_csv(index=False), "data_processed.csv", "text/csv")

    else:  # Analysis
        st.subheader("Distribution")
        feat = st.selectbox("Feature", ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        fig, ax = plt.subplots(figsize=(8, 3))
        raw[feat].hist(bins=30, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"{feat} distribution"); st.pyplot(fig); plt.close(fig)
        st.subheader("Class balance")
        vc = raw[target].value_counts()
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(vc.index, vc.values, color="coral"); ax.set_title("Samples per crop")
        plt.xticks(rotation=45, ha="right"); st.pyplot(fig); plt.close(fig)
        st.subheader("Correlation heatmap")
        num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        corr = raw[num_cols].corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
        ax.set_yticks(range(len(num_cols)), num_cols)
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax); st.pyplot(fig); plt.close(fig)
        st.subheader("Feature vs Target (ANOVA)")
        from scipy import stats
        rows = []
        for col in num_cols:
            groups = [g[col].values for _, g in raw.groupby(target)]
            f, p = stats.f_oneway(*groups)
            rows.append({"feature": col, "F": round(f, 1), "p": f"{p:.2e}"})
        st.table(pd.DataFrame(rows).sort_values("F", ascending=False))
        st.caption("Higher F = more discriminative for crop classification")
