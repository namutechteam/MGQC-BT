# Bond-type-conditioned coupling in graph-structured quantum circuits

Code and results for the manuscript *"Bond-type-conditioned coupling in
graph-structured quantum circuits for molecular property prediction"*
(H. Jo, J. Lee, M. Yeom).

---

## Contents

```
mgqc/
  core.py              the circuit, the dataset, the trainer, the statistics
  cli.py               one entry point for every experiment
  selftest.py          gradient and measurement checks
  reachability.py      the five-qubit observability demonstration behind Fig. 1b
  experiments/         the experiments, four modules grouped by question
results/               per-seed output of every experiment
data/                  preprocessed QM7-X molecule caches
supp_data/             Supplementary Data 1 and 2 (CSV)
hardware/              the IonQ Forte-1 run: circuits, weights, raw results, analysis
```

`core.py` holds the physics and the statistics. The experiment modules define
the arms each comparison runs and hand them to `core.run_arm`; none of them
reimplements the circuit or the tests. `_gnn.py` is the SchNet implementation
used by the `schnet` and `tuning` experiments.

```bash
python -m mgqc list              # every experiment and what it does
python -m mgqc show grouping     # what one writes, and which items it feeds
python -m mgqc run grouping --help
python -m mgqc selftest
```

## Requirements

```
python >= 3.11
pennylane == 0.45.1          # lightning.qubit
numpy, scipy, scikit-learn
torch                        # classical baselines and SchNet
```

```bash
pip install -r requirements.txt
```

## Quick check

```bash
python -m mgqc selftest
```

This verifies the gradient implementation against central differences and
against the parameter-shift rule, and checks the entanglement measures against
a product state and a Bell state. It takes about a minute and needs no GPU.

---

## Checking the reported numbers against the included results

`results/` holds the per-seed output of every experiment, one CSV per run, with
the configuration, seed, arm and metric in each row. Every table entry in the
manuscript is a paired statistic over those rows (see *Notes on reading the
results* below); the two large tables are also provided as `supp_data/`.

The five-qubit observability demonstration behind Fig. 1b runs in seconds and
prints the reachability numbers quoted in the text:

```bash
python -m mgqc.reachability > results/reachability.json
```

---

## The experiments

The experiments live in four modules under `mgqc/experiments/`, one per
question below; `python -m mgqc run <name>` runs one experiment and writes its
CSV into `results/`. They are independent and can be run in any order. Times are for 26 parallel workers on
CPU; a full sweep is on the order of a day. `python -m mgqc show <name>` prints
which manuscript items an experiment feeds.

**Does bond-type-conditioned coupling help, and against what?** (`baselines.py`)

- `accuracy` — the main comparison. On each of the seven configurations the
  circuit is trained against the same-split, same-seed alternatives: the
  composition model, ridge and kernel ridge on spectral features, the circuit
  with coupling switched off, with one shared coefficient, and with the bond
  types randomly reassigned. Also records the learned per-type coefficients.
- `classical` — two classical message-passing models built to match the
  circuit's parameter count and node information, so the comparison isolates
  the circuit rather than the parameter budget.
- `schnet` — a deep graph network (SchNet) trained on the circuit's own
  splits, as the reference for the parameter-efficiency claim.
- `tuning` — hyper-parameter grids for the circuit and both classical
  baselines on the same footing, so no comparison rests on an untuned model.

**Where does the gain come from?** (`grouping.py`)

- `grouping` — what the parameter-sharing groups are indexed by. Bonds are
  regrouped by chemistry (element pair + bond order), by element pair only, by
  bond length only, or arbitrarily, so the randomized control's loss can be
  attributed to chemical alignment or to mere consistency of sharing.
- `grouping-verdict` — the decision rules for `grouping`, fixed before the run.
- `chemistry` — whether the learned coefficients line up with
  electronegativity difference and bond enthalpy, which the model is never
  given; partial correlation removes the nuclear-charge prefactor.
- `bondcut`, `bondcut-cv` — whether the result depends on the bond-order
  thresholds that define the types: a sensitivity scan with a randomized
  control at each cut, and the threshold chosen by cross-validation inside the
  training split.

**What is the quantum structure doing?** (`quantum_structure.py`)

- `entangler` — alternative entangling generators (XY, XXZ, Trotterized
  transverse field) in place of ZZ.
- `entanglement` — concurrence and mutual information of the trained circuit,
  and how much of the signal survives when it is removed.
- `angles` — the distribution of the coupling angle actually applied to each
  bond, i.e. how far the circuit sits inside the perturbative regime.
- `angle-scan` — the coupling angle scaled out of that regime, to see whether
  the circuit separates from its classical counterpart there.
- `optimizer` — adjoint-gradient Adam against the simultaneous-perturbation
  loop, with controls at zero variational weights.

**Beyond atomization energy on the full set** (`generalization.py`)

- `isomer` — error within groups of isomers and top-1 selection of the
  lowest-energy isomer, stratified by group size.
- `targets` — HOMO–LUMO gap and dipole moment, targets that the composition
  model cannot explain.

Example:

```bash
OMP_NUM_THREADS=1 python -m mgqc run accuracy \
    --configs A1,A2,B1,B2,B3,B4,B5 --seeds 20 \
    --canonical --types-from-train --jobs 26
```

Most experiments take `--configs`, `--seeds` and `--jobs`; `python -m mgqc run
<name> --help` prints the rest. Paths can be redirected with `MGQC_DATA` and
`MGQC_RESULTS`, which is how the smoke tests keep output out of `results/`.

---

## Hardware run

`hardware/` holds everything behind Supplementary Fig. 1: the Qiskit circuit
builder and IonQ submission code (`mgqc_ionq_batch.py`, `plan_a_fidelity.py`),
the weights of the variant trained without readout standardization
(`mgqc_bt_bundle_A1_shotrobust.npz`, produced by `train_mgqc_bt.py
--shot-robust`), the ten test molecules, and the two result files returned by
the QPU job and the IonQ cloud-simulator job (`plan_a_results_qpu_4096.json`,
`plan_a_results_simulator.json`). Every number quoted for the run
comes from `hardware/analyze_qpu.py` reading those two files:

```bash
cd hardware
python analyze_qpu.py
```

`verify_30min.py` reproduces the shot-noise amplification of the standardized
readout on a local Aer simulator without an IonQ account. Re-running the QPU
job needs `requirements-ionq.txt`, an IonQ API key and a budget; the job ids
are in `ionq_jobs.jsonl`.

## Data

`data/bench_<config>/cache_atoms<N>_seed42.pkl` holds the preprocessed
molecules for one configuration: nuclear charges, coordinates, the adjacency
and distance matrices, the Coulomb-weighted Laplacian spectrum, and the three
target properties (atomization energy, HOMO–LUMO gap, dipole moment).

These derive from **QM7-X**, which is public at
<https://zenodo.org/record/4288677> and is not redistributed here beyond the
subset needed to reproduce the reported runs. Duplicates listed in the
accompanying file are removed and only equilibrium geometries are used.
Configuration definitions (atom-count ceiling, heavy-atom range, molecule
counts) are in Supplementary Table 1.

> **Note.** The caches are Python pickles produced by an earlier stage of this
> project, and the script that built them from the raw QM7-X HDF5 is not part
> of this release. Loading a pickle executes code, so treat them as you would
> any downloaded artifact. Rebuilding them from the public QM7-X archive is
> straightforward from the field list above and is on the list of things to
> add.

---

## Notes on reading the results

- **Arm names.** `zz_bytype_r3` is the circuit of the paper; `zz_shuffled_r3`
  is the randomized-grouping control; `none_r3` is the α = 0 control;
  `zz_fixed_r3` is the single-coefficient variant; `krr_res` is the kernel
  baseline. `C3_eigfeat` and `C3_msgpass` are the two parameter-matched
  classical models.
- **B5 is run at 40 seeds**, A2 at 10, everything else at 20. Use
  `E1_sevenconfig_b5s40*.csv` for B5, not the 20-seed file.
- **`E1_alpha_learned_*.csv`.** Only the two `*tft` files feed the learned-
  coefficient analyses. Other files of that name in a working results directory
  belong to different protocols and must not be pooled with them; the scripts
  name the files they read rather than globbing.
- **Paired statistics** are the mean of per-seed relative differences, with a
  Wilcoxon signed-rank test as support. With five seeds the smallest two-sided
  p the test can return is 0.0625, so the five-seed scans are reported as
  scans, not as confirmatory tests.

## Labels in file names and code comments

The short codes that prefix the files in `results/` (`E1_`, `R10_`, ...) and
appear in code comments are the working labels the analyses carried during the
project. They are kept so that the files match the scripts that wrote them.

| Label | Experiment (`python -m mgqc run ...`) | Manuscript item |
|---|---|---|
| `E1` | `accuracy` | Table 1, Table 2, Figs 2–4, Supp. Tables 6, 7, 12, Supp. Data 1 |
| `E2` | `classical` | Methods, Supp. Table 5, Fig. 8 |
| `H3` | `schnet` | Supp. Table 5, Fig. 8 |
| `H4` | `isomer` | Figs 5, 6, Supp. Table 3 |
| `N1` | `targets` | Fig. 7, Supp. Tables 4, 10 |
| `N2` | `bondcut` | Supp. Table 8 |
| `N2b` | `bondcut-cv` | Methods, bond typing |
| `N3` | `tuning` | Supp. Table 11 |
| `I1` | `entanglement` | Supp. Table 9 |
| `G0` | `optimizer` | Discussion, training |
| `G1` | `entangler` | Results, Supp. Table 17 |
| `R5` | `angles` | Supp. Table 13 |
| `R7` | `angle-scan` | Discussion, Supp. Table 14, Supp. Data 2 |
| `R10` | `grouping`, `grouping-verdict` | Results, Supp. Table 15 |
| `R11` | `chemistry` | Results, Supp. Table 16 |

In `core.py`, `G0`–`G4` also name the parts of the circuit that the revision
changed, and the `Arch` fields are grouped under them: `G0` training (adjoint
gradients and Adam), `G1` the entangler, `G2` the bond-type-conditioned
coupling, `G3` the encoding, `G4` the readout and evaluation. Labels that
appear in comments but not in this table refer to earlier-stage analyses that
are not part of this release.

## License

MIT. See `LICENSE`. The QM7-X data derived in `data/` remain under the terms of
the original QM7-X release.
