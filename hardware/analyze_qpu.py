#!/usr/bin/env python3
"""Analysis of the IonQ Forte-1 run (Supplementary Fig. 1 and the numbers
quoted for it in the Discussion and Methods).

    python analyze_qpu.py            # run from hardware/
    HW=<dir> python analyze_qpu.py   # or point at another results directory

Reads plan_a_results_qpu_4096.json (the QPU job) and
plan_a_results_simulator.json (the IonQ cloud simulator job) and prints every
number the manuscript quotes for the run; nothing is typed in by hand.

The one-parameter correction is fitted leave-one-out: for molecule i the
intercept is the mean difference between hardware and noiseless predictions
over the other nine, applied to i. The qubit-level correction is the same, per
qubit, with a slope as well as an intercept.
"""
import json
import os

import numpy as np
from scipy import stats

HW = os.environ.get("HW", ".")

Q = json.load(open(os.path.join(HW, "plan_a_results_qpu_4096.json")))
S = json.load(open(os.path.join(HW, "plan_a_results_simulator.json")))
arr = lambda d, k: np.array([m[k] for m in d["per_molecule"]])

pn, pq, ps = arr(Q, "res_noiseless_eV"), arr(Q, "res_qpu_eV"), arr(S, "res_qpu_eV")
zn, zq = arr(Q, "z_noiseless"), arr(Q, "z_qpu")
sig = Q["qubit_level_metrics"]["shot_noise_stderr_theoretical"]
N, nq = zq.shape

# --- leave-one-out intercept at the prediction level ------------------------
loo = np.array([pq[i] - (pq[np.arange(N) != i] - pn[np.arange(N) != i]).mean()
                for i in range(N)])
mae_raw, mae_loo, mae_sim = (np.abs(pq - pn).mean(), np.abs(loo - pn).mean(),
                             np.abs(ps - pn).mean())
r2 = lambda a, b: 1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()

# --- global <Z> regression ----------------------------------------------------
m = np.abs(zn) > 0.05
slope, icpt, r, _, _ = stats.linregress(zn[m], zq[m])

# --- per-qubit shrinkage and leave-one-out qubit-level correction -----------
shift = (zq - zn).mean(0)
dc = np.zeros_like(zq)
for i in range(N):
    keep = np.arange(N) != i
    for q in range(nq):
        mm = keep & (np.abs(zn[:, q]) > 0.05)
        if mm.sum() >= 4:
            s_, b_, *_ = stats.linregress(zn[mm, q], zq[mm, q])
            dc[i, q] = (zq[i, q] - b_) / s_ - zn[i, q]
        else:
            dc[i, q] = (zq[i, q] - zn[i, q]) - (zq[keep, q] - zn[keep, q]).mean()
within_raw = 100 * (np.abs(zq - zn) <= 2 * sig).mean()
within_loo = 100 * (np.abs(dc) <= 2 * sig).mean()
within_sim = S["qubit_level_metrics"]["pct_within_2sigma_bound"]

print(f"  backend {Q['backend']}  shots {Q['shots']}  molecules {N}  qubits {nq}")
print(f"  shift {(pq - pn).mean():+.3f} eV (sd {(pq - pn).std(ddof=1):.3f})")
print(f"  <Z>: slope {slope:.3f} intercept {icpt:+.4f} R2 {r * r:.3f} n={m.sum()}")
print(f"  per-qubit slopes: {[round(float(stats.linregress(zn[np.abs(zn[:, q]) > 0.05, q], zq[np.abs(zn[:, q]) > 0.05, q])[0]), 3) if (np.abs(zn[:, q]) > 0.05).sum() >= 4 else None for q in range(nq)]}")
print(f"  Pearson {np.corrcoef(pq, pn)[0, 1]:.3f}  Spearman {stats.spearmanr(pq, pn).statistic:.3f}")
print(f"  MAE vs noiseless: raw {mae_raw:.3f}  LOO {mae_loo:.3f}  simulator {mae_sim:.3f}")
print(f"  R2  vs noiseless: raw {r2(pq, pn):.3f}  LOO {r2(loo, pn):.3f}  simulator {r2(ps, pn):.3f}")
print(f"  within 2 s.e. (qubit level): raw {within_raw:.1f}%  LOO {within_loo:.1f}%  simulator {within_sim:.1f}%")
print(f"  residual qubit scatter after LOO / shot s.e.: {dc.std() / sig:.2f}")
