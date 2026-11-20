#!/usr/bin/env python3
import csv
import json
from pathlib import Path

from insf import paths
B = paths.OUT / "baselines"
rows, allsum = [], {}
for m in ("twist", "admm", "mst", "mstpp"):
    f = B / "runs" / m / "summary.json"
    if not f.exists():
        rows.append({"method": m, "status": "missing"})
        continue
    s = json.loads(f.read_text())
    allsum[m] = s
    for split in ("shared33", "shared_fast40_85"):
        d = s[split]
        rows.append({"method": m, "split": split, "n_frames": d.get("n_frames"),
                     "rmse_holdout": d.get("rmse_holdout"), "rmse_holdout_unsat": d.get("rmse_holdout_unsat"),
                     "rmse_train": d.get("rmse_train"), "fit_seconds_per_frame": d.get("fit_seconds"),
                     "complete": s.get("complete")})
ref = next(iter(allsum.values()))["reference_08"] if allsum else {}
for split in ("shared33", "shared_fast40_85"):
    r = ref.get(split, {})
    for m in ("e1h", "mc"):
        rows.append({"method": f"08_{m}", "split": split, "n_frames": r.get("n_frames"),
                     "rmse_holdout": r.get(f"{m}_rmse_holdout"), "rmse_holdout_unsat": r.get(f"{m}_rmse_holdout_unsat"),
                     "rmse_train": r.get(f"{m}_rmse_train")})
keys = []
for r in rows:
    keys += [k for k in r if k not in keys]
with open(B / "results_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
(B / "results_summary.json").write_text(json.dumps({"table": rows, "per_method": allsum}, indent=2, ensure_ascii=False))
for r in rows:
    print(r)
