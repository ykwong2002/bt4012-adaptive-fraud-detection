"""Shared CLI plumbing for scripts/: path setup, arg parsing, graph loading."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camo import ROOT, load_config


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--smoke", action="store_true", help="tiny subsample / fixture, one seed")
    p.add_argument("--source", choices=["raw", "mat", "auto"], default="auto",
                   help="raw = rebuilt from text release, mat = public graph file")
    return p


def resolve_source(source: str) -> str:
    cfg = load_config("base")
    if source != "auto":
        return source
    raw = ROOT / cfg["paths"]["raw_dir"]
    return "raw" if raw.exists() and any(p.name.startswith("metadata") for p in raw.glob("*")) else "mat"


def processed_path(source: str, smoke: bool) -> Path:
    cfg = load_config("base")
    d = ROOT / cfg["paths"]["processed_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d / f"graph_{source}{'_smoke' if smoke else ''}.npz"
