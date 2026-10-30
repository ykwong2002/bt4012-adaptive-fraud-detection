"""camo: camouflage-aware graph reconstruction for review-fraud detection (BT4012 Group 05)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "base") -> dict:
    import yaml

    with open(ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
