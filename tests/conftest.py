import subprocess
import sys
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "raw_mini"


@pytest.fixture(scope="session")
def raw_mini():
    if not (FIX / "metadata").exists():
        subprocess.check_call([sys.executable, str(FIX.parent / "make_raw_mini.py")])
    from camo.data.load_raw import load_raw

    return load_raw(FIX)
