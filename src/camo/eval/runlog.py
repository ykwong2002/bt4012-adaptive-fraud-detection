"""Every script writes outputs/runs/<name>/run.json so the report's reproducibility claim is backed."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from camo import ROOT


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


class RunLog:
    def __init__(self, name: str, config: dict | None = None, seed: int | None = None):
        self.name, self.seed = name, seed
        self.t0 = time.time()
        self.dir = ROOT / "outputs" / "runs" / name
        self.dir.mkdir(parents=True, exist_ok=True)
        cfg = json.dumps(config or {}, sort_keys=True, default=str)
        self.info = {"name": name, "git_sha": _git_sha(), "config_hash": hashlib.sha256(cfg.encode()).hexdigest()[:12],
                     "seed": seed, "argv": sys.argv, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "notes": {}}

    def note(self, **kw):
        self.info["notes"].update({k: (v if isinstance(v, (int, float, str, list, dict, type(None))) else str(v)) for k, v in kw.items()})

    def finish(self) -> Path:
        self.info["wall_time_s"] = round(time.time() - self.t0, 1)
        p = self.dir / "run.json"
        p.write_text(json.dumps(self.info, indent=2, default=str))
        return p
