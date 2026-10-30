"""Download the public YelpChi.mat (CARE-GNN release) and check for the raw text files."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camo import ROOT, load_config  # noqa: E402


def main():
    cfg = load_config("base")
    mat_dir = ROOT / cfg["paths"]["mat_dir"]
    mat_dir.mkdir(parents=True, exist_ok=True)
    zpath = mat_dir / "YelpChi.zip"
    if not (mat_dir / "YelpChi.mat").exists():
        print("downloading", cfg["paths"]["mat_zip_url"])
        urllib.request.urlretrieve(cfg["paths"]["mat_zip_url"], zpath)
        sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if sha != cfg["paths"]["mat_zip_sha256"]:
            sys.exit(f"sha256 mismatch: {sha}")
        zipfile.ZipFile(zpath).extractall(mat_dir)
    print("ok:", mat_dir / "YelpChi.mat")

    raw = ROOT / cfg["paths"]["raw_dir"]
    have = [p.name for p in raw.glob("*")] if raw.exists() else []
    if any(n.startswith("metadata") for n in have):
        print("raw text present:", have)
    else:
        print("raw text NOT present; see data/README.md (email the dataset author). "
              "Text-dependent steps will use tests/fixtures/raw_mini in --smoke mode.")


if __name__ == "__main__":
    main()
