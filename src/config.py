"""Shared config loader: yaml base + CLI overrides. CLI always wins."""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "base.yaml"


def load(section: str, argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    # generic overrides (only ones passed on CLI take effect)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="The amount of epochs for the model to train")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--save", default=None)
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text())
    params = dict(cfg.get(section, {}))
    for key in ("model", "epochs", "imgsz", "batch", "weights", "conf", "image", "save"):
        val = getattr(a, key)
        if val is not None:
            params[key] = val
    params["_config_file"] = a.config
    return params
