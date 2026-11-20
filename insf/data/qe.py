import json
import numpy as np
from insf import paths

WAVELENGTHS_NM = np.arange(350, 951)


def load_pattern(path=None):
    p = np.array(json.load(open(path or paths.CHANNEL_LAYOUT))["pattern"], dtype=int)
    assert p.shape == (3, 3) and sorted(p.ravel()) == list(range(1, 10)), p
    return p


def read_qe_xlsx(path):
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True).active
    rows = [r for r in ws.iter_rows(values_only=True)][1:]
    d = np.array([[float(x) for x in r[:10]] for r in rows if isinstance(r[0], (int, float))])
    assert d.shape == (601, 10) and d[0, 0] == 350 and d[-1, 0] == 950, d.shape
    return d[:, 0], d[:, 1:]


def channel_map(h=1200, w=1600, r0=0, c0=0, pattern=None):
    pattern = load_pattern() if pattern is None else pattern
    rr, cc = np.meshgrid((np.arange(h) + r0) % 3, (np.arange(w) + c0) % 3, indexing="ij")
    return pattern[rr, cc] - 1
