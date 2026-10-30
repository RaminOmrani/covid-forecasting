import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covidfc import data  # noqa: E402


def setup(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    df = data.load(None if cfg["data"] == "synthetic" else cfg["data"], seed=cfg.get("synthetic_seed", 0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return cfg, df, out


def anfis_cfg(cfg):
    return {**cfg["anfis"], **cfg["split"]}


def fmt(summary: dict) -> str:
    return "  ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in summary.items())
