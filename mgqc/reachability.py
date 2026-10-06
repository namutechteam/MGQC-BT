#!/usr/bin/env python3
"""Reachability of the graph layer -- the single source for the Fig. 1b numbers and for
every reachability number quoted in the manuscript and the Supplementary Note.

    python -m mgqc.reachability > results/reachability.json

The test is the one described in the text: freeze everything except the
coupling coefficient, scale that coefficient over nine orders of magnitude,
and record the largest change in any measured expectation.  A value at the
double-precision floor means the coefficient cannot be seen at all.

The circuit is the five-qubit arrangement drawn in Fig. 1a (C, C, O, H, H with
bonds C-C, C-O, C-H, C-H), so the figure and the numbers describe the same
object.  Nothing here is fitted; the encoding angles and the variational
angles are fixed by a seeded draw so the test is deterministic.
"""
import json, sys
import numpy as np
import pennylane as qml

N = 5
BONDS = [(0, 1), (1, 2), (0, 3), (0, 4)]
SCALES = np.logspace(-4.5, 4.5, 25)          # nine orders of magnitude
rng = np.random.default_rng(7)
ENC = rng.uniform(0.2, np.pi - 0.2, N)       # encoding angles, fixed
VAR = rng.uniform(-np.pi, np.pi, (N, 2))     # variational angles, fixed
J = np.array([0.31, 0.27, 0.18, 0.18])       # fixed prefactors J_ij

dev = qml.device("default.qubit", wires=N)


def _encode():
    for q in range(N):
        qml.RY(ENC[q], wires=q)


def _couple(alpha, kind):
    for (i, j), j_bond in zip(BONDS, J):
        if kind == "zz":
            qml.IsingZZ(j_bond * alpha, wires=[i, j])
        elif kind == "xy":
            qml.IsingXY(j_bond * alpha, wires=[i, j])


def _tail(kind):
    if kind == "cnot":
        for q in range(N - 1):
            qml.CNOT(wires=[q, q + 1])
    elif kind == "s":
        for q in range(N):
            qml.S(wires=q)
        for q in range(N - 1):
            qml.CNOT(wires=[q, q + 1])
    elif kind == "h":
        for q in range(N):
            qml.Hadamard(wires=q)
        for q in range(N - 1):
            qml.CNOT(wires=[q, q + 1])


def make(coupling, rotate, tail):
    @qml.qnode(dev)
    def c(alpha):
        _encode()
        _couple(alpha, coupling)
        if rotate:
            for q in range(N):
                qml.RY(VAR[q, 0], wires=q)
                qml.RZ(VAR[q, 1], wires=q)
        _tail(tail)
        return [qml.expval(qml.PauliZ(q)) for q in range(N)]
    return c


def sweep(c):
    """Largest change in any measured expectation over the coupling scan."""
    expvals = np.array([np.asarray(c(s), dtype=float) for s in SCALES])
    return float(np.abs(expvals - expvals[0]).max())


CASES = {
    # Fig. 1b, top to bottom
    "zz_cnot":     dict(coupling="zz", rotate=False, tail="cnot"),
    "zz_ry_cnot":  dict(coupling="zz", rotate=True,  tail="cnot"),
    "xy_cnot":     dict(coupling="xy", rotate=False, tail="cnot"),
    # Supplementary Note 1: which Clifford tails lift the constraint
    "zz_s_tail":   dict(coupling="zz", rotate=False, tail="s"),
    "zz_h_tail":   dict(coupling="zz", rotate=False, tail="h"),
}

out = {k: sweep(make(**v)) for k, v in CASES.items()}
out["_scan_decades"] = float(np.log10(SCALES[-1] / SCALES[0]))
out["_n_qubits"] = N
json.dump(out, sys.stdout, indent=1)
print(file=sys.stderr)
for k, v in out.items():
    if not k.startswith("_"):
        print(f"  {k:14s} {v:.3e}", file=sys.stderr)
