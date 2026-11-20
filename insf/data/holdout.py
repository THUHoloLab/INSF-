#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
from insf import paths
h = np.random.default_rng(20261019).random((98, 512, 512)) < 0.1
paths.HOLDOUT_MOVE2.parent.mkdir(parents=True, exist_ok=True); np.save(paths.HOLDOUT_MOVE2, h)
print("wrote", paths.HOLDOUT_MOVE2, "held-out fraction %.4f" % h.mean())
