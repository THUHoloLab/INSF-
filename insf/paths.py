import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _p(var: str, default: Path) -> Path:
    return Path(os.environ.get(var, str(default))).expanduser()


DATA = _p("INSF_DATA", REPO / "data")
OUT = _p("INSF_OUT", REPO / "outputs")


MASK = _p("INSF_MASK", DATA / "mask" / "mask_3D.mat")
QE_XLSX = _p("INSF_QE", DATA / "mask" / "filter_qe.xlsx")
CHANNEL_LAYOUT = _p("INSF_LAYOUT", DATA / "mask" / "channel_layout.json")
XYZBAR = _p("INSF_XYZBAR", DATA / "xyzbar.mat")


MEAS = _p("INSF_MEAS", DATA / "measurements")
MOVE1 = MEAS / "move1"
MOVE2 = MEAS / "move2"
STATIC = MEAS / "astastic"
COLORCHECKER_RAW = _p("INSF_COLORCHECKER", DATA / "colorchecker_raw")


HOLDOUT_MOVE2 = _p("INSF_HOLDOUT", DATA / "holdout_10pct_seed20261019.npy")


HARVARD_HOLDOUT = _p("INSF_HARVARD_HOLDOUT", DATA / "harvard_holdout.json")
COLORCHECKER_FRAMES = _p("INSF_CC_FRAMES", DATA / "colorchecker_frames.json")


HARVARD = _p("INSF_HARVARD", DATA / "public" / "harvard" / "CZ_hsdb")
CAVE = _p("INSF_CAVE", DATA / "public" / "cave" / "complete_ms_data")
KAIST = _p("INSF_KAIST", DATA / "public" / "kaist" / "exr")


THIRD_PARTY = _p("INSF_THIRD_PARTY", REPO.parent)
