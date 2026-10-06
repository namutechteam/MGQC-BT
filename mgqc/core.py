#!/usr/bin/env python3
"""
=============================================================================
MGQC circuit strengthening -- shared core
=============================================================================
Implements the audit that followed the first submission:

  G0  training recovery      analytic adjoint gradients + Adam
  G1  open the information path   non-diagonal (XY) entangler
  G2  chemical inductive bias     bond-type conditional learnable coupling
  G3  encoding                    drop lambda_1, angle range, atom identity,
                                  data re-uploading
  G4  readout / evaluation axis   standardisation, isomer-group metrics

Why this file exists
--------------------
The superseded implementation evaluated the circuit through one QNode per
molecule on `default.qubit` with parameter-shift, and optimised with SPSA.
The circuit audit showed the SPSA loop is a no-op: a circuit
with `var_weights = 0` matches a 100-epoch trained one to the third decimal.

Two things are fixed here.

  1.  Gradients.  We build the tapes by hand and call
      `device.compute_derivatives` with `gradient_method="adjoint"`.  That is
      one reverse pass per molecule instead of 2*n_params forward passes, and
      it skips the QNode's per-call Python overhead (measured 12.6 ms -> 1-3 ms
      per molecule).  Angles that are products (theta = J_ij * alpha_tau) are
      differentiated by an explicit chain rule; `selftest.py` checks it against
      finite differences and against PennyLane's own parameter-shift.

  2.  The entangler.  IsingZZ is diagonal in the computational basis and Layer
      C's CNOT chain is Clifford, so the coupling strength provably cannot move
      any Z-string expectation (measured delta = 4e-16).  IsingXY is
      excitation-preserving and does not commute with Z_i, so local <Z_q>
      reacts to it and the existing readout works unchanged.

Backend note
------------
`lightning.gpu` is deliberately NOT used.  At 13 qubits the statevector is 8192
amplitudes; the run time is dominated by Python-side circuit construction, not
by the linear algebra, and the host-device transfer would make a GPU slower.
the speed target is met by adjoint + batched device execution on
`lightning.qubit`.
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # <repo>/mgqc -> <repo>
# Layout of this release:  <root>/src  <root>/data  <root>/results
# Both can be overridden, which is what the manuscript build scripts do.
PAPER = os.environ.get("MGQC_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("MGQC_RESULTS", os.path.join(ROOT, "results"))
os.makedirs(RESULTS, exist_ok=True)

# Paper/ contains a directory literally named h5py/, which shadows the real
# package whenever Paper/ sits at the front of sys.path.  Append, never prepend.
if PAPER not in sys.path:
    sys.path.append(PAPER)

import pennylane as qml
from pennylane.devices import ExecutionConfig

ADJOINT_CFG = ExecutionConfig(gradient_method="adjoint")

COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 16: 1.05, 17: 1.02}
ATOM_SYMBOLS = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}

CONFIG_ATOMS = {"A1": 11, "A2": 13, "B1": 13, "B2": 13,
                "B3": 13, "B4": 13, "B5": 13}


# =============================================================================
# 1.  Architecture description
# =============================================================================

@dataclass
class Arch:
    """Everything that defines the circuit.  Defaults reproduce the published
    MGQC+QSM-CL circuit exactly (entangler zz, fixed alpha 0.1, one round,
    one re-upload, CNOT chain on, no standardisation in the readout)."""

    n_qubits: int = 13

    # --- G1: entangler -------------------------------------------------
    entangler: str = "zz"          # zz | xy | xxz | trotter | none
    alpha: float = 0.1             # coupling scale (init value when trained)
    alpha_mode: str = "fixed"      # fixed | scalar | bytype | bytype_shuffled
    gamma_zz: float = 0.5          # ZZ fraction for the xxz arm
    trotter_layers: int = 2
    trotter_beta: float = 0.3      # RX angle between Trotter slices (fixed)
    pad_entangle: bool = True      # baseline Layer B-2 on unused qubits

    # --- G2: message-passing depth -------------------------------------
    n_rounds: int = 1
    bond_type_scheme: str = "elem_order"   # elem | elem_order

    # --- R7: pushing the coupling out of the perturbative regime ---------
    # R5-B measured that >99% of the applied angles sit where sin(x) = x to
    # within 1%, i.e. the coupling acts as a weak perturbation.  `theta_scale`
    # multiplies every entangling angle globally so the scan can leave that
    # regime; `alpha_norm="rms"` fixes |alpha| so the optimiser cannot simply
    # shrink alpha and cancel the scale (the usual way this experiment dies).
    # The DIRECTION of alpha -- the relative pattern across bond types, which
    # is the chemistry we claim -- stays trained.
    theta_scale: float = 1.0
    alpha_norm: str = "free"       # free | rms  (rms: only for bytype modes)

    # --- R10: what the coefficients are shared ACROSS -------------------
    # The published shuffled control redraws a fake type for EVERY BOND
    # independently, so it destroys chemical alignment and consistent
    # parameter sharing at the same time.  These modes separate the two:
    #   chem   the published vocabulary (element pair + bond order)
    #   elem   merged to the element pair only -- chemistry kept, coarser
    #   arb    merged into groups with the SAME size distribution as `elem`,
    #          but the members drawn at random -- sharing is just as
    #          consistent, the grouping simply is not chemical
    #   lenq   indexed by reduced bond length quantile only, no element
    #          identity at all -- separates chemistry from geometry
    #   bijection  a 1:1 relabelling of the full vocabulary; the hypothesis
    #          space is unchanged, so it must reproduce `chem` exactly.  The
    #          null control for this whole mechanism.
    group_mode: str = "chem"       # chem | elem | arb | lenq | bijection
    group_seed_offset: int = 70_000
    lenq_bins: int = 0             # 0 = match the chemical vocabulary size
    max_types: int = 0             # cap the vocabulary at the K most frequent
                                   # bond types (0 = no cap).  Holding K fixed
                                   # across a learning-curve sweep separates
                                   # "more molecules" from "more chemical
                                   # variety", which otherwise grow together.
    types_from_train: bool = False # build the bond-type vocabulary from the
                                   # TRAINING molecules only (an unseen type at
                                   # test time falls into a reserved bucket
                                   # whose coupling stays at its init value)

    # --- data-scale sweep --------------------------------------------------
    train_frac: float = 1.0        # fraction of the training split actually used

    # --- target ------------------------------------------------------------
    target: str = "eAT"            # which property to predict
    residual: bool = True          # subtract a composition model first.  For a
                                   # target the composition model cannot explain
                                   # (dipole moment: R2 ~ 0) that subtraction
                                   # removes nothing and the two-stage design
                                   # stops making sense -- see I3_target_scan.

    # --- G3: encoding ---------------------------------------------------
    encode_source: str = "eig"     # eig (published, spectral) | diag (atom-aligned)
    drop_q0: int = 0               # how many leading CL eigenvalues to skip
    angle_lo: float = 0.0
    angle_hi: float = np.pi
    rz_source: str = "none"        # none | Znum (atom identity) | diag (CL diagonal) | eig (spectrum)
    canonical_order: bool = False  # deterministic atom -> qubit assignment
    n_reupload: int = 1

    # --- readout / structure --------------------------------------------
    use_cnot: bool = True
    observable: str = "Z"          # Z | ZXY  (G4 R-b)
    readout_scaler: bool = False   # False = published behaviour (unstandardised readout)
    readout_alpha: float = 1.0     # ridge penalty; RidgeCV is harmful at these sample sizes
    readout_kind: str = "ridge"    # ridge | krr

    # --- optimisation ----------------------------------------------------
    epochs: int = 100
    lr: float = 0.05
    optimizer: str = "adam"        # adam | spsa (published) | none
    train_var: bool = True         # False -> var_weights stay at 0

    def n_var_params(self):
        return self.n_reupload * self.n_rounds * self.n_qubits * 2

    def n_features(self):
        return self.n_qubits * (3 if self.observable == "ZXY" else 1)


# =============================================================================
# 2.  Data
# =============================================================================

def load_config(config, bench_root=PAPER):
    import pickle
    path = os.path.join(bench_root, f"bench_{config}",
                        f"cache_atoms{CONFIG_ATOMS[config]}_seed42.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as fp:
        cache = pickle.load(fp)
    return cache["molecules"]


def group_labels(molecules, level):
    if level == "structure":
        return np.arange(len(molecules))
    if level == "isomer":
        keys = [(m["mol_id"], m["iso_idx"]) for m in molecules]
    elif level == "molecule":
        keys = [m["mol_id"] for m in molecules]
    else:
        raise ValueError(level)
    lookup, out = {}, np.empty(len(keys), dtype=np.int64)
    for i, k in enumerate(keys):
        out[i] = lookup.setdefault(k, len(lookup))
    return out


def make_split(molecules, level, seed, test_size=0.25):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    idx = np.arange(len(molecules))
    if level == "structure":
        return train_test_split(idx, test_size=test_size, random_state=seed)
    g = group_labels(molecules, level)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(idx, groups=g))
    return idx[tr], idx[te]


# --- bond typing ------------------------------------------------------------
#
# QM7-X gives no bond orders, so a length proxy stands in for one.  With
# r = d_ij / (r_cov_i + r_cov_j), a C-C single bond (1.54 A) sits at r = 1.01,
# a double bond (1.34 A) at 0.88 and a triple bond (1.20 A) at 0.79.  The cuts
# below are the midpoints of those, held fixed so no target information can
# enter the type assignment.
_ORDER_CUTS = (0.84, 0.95)


def bond_order_proxy(d, zi, zj):
    r = d / (COVALENT_RADII.get(zi, 1.0) + COVALENT_RADII.get(zj, 1.0))
    if r < _ORDER_CUTS[0]:
        return 3
    if r < _ORDER_CUTS[1]:
        return 2
    return 1


def bond_type_key(zi, zj, d, scheme):
    a, b = (int(zi), int(zj)) if zi <= zj else (int(zj), int(zi))
    if scheme == "elem":
        return (a, b)
    return (a, b, bond_order_proxy(d, a, b))


def type_label(key):
    if len(key) == 2:
        return f"{ATOM_SYMBOLS.get(key[0], '?')}-{ATOM_SYMBOLS.get(key[1], '?')}"
    order = {1: "-", 2: "=", 3: "#"}[key[2]]
    return f"{ATOM_SYMBOLS.get(key[0], '?')}{order}{ATOM_SYMBOLS.get(key[1], '?')}"


def extract_bonds(mol, n_qubits, scheme, max_bonds=30, with_r=False):
    """Bonded pairs inside the qubit window, with the published Z-weighted,
    bond-count-normalised coupling and a chemical type key.

    `with_r=True` appends the reduced length r = d / (r_cov_i + r_cov_j), the
    quantity the bond-order proxy thresholds.  R10's `lenq` mode indexes the
    coefficients by its quantile instead of by chemistry."""
    adj, Z, d = mol["adj"], mol["Z"], mol["dist"]
    n = min(int(mol["n_atoms"]), n_qubits)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] > 0]
    n_bonds = max(len(pairs), 1)
    out = []
    for (i, j) in pairs[:max_bonds]:
        J = Z[i] * Z[j] / (Z[i] + Z[j]) / np.sqrt(n_bonds)
        rec = (i, j, float(J), bond_type_key(Z[i], Z[j], d[i, j], scheme))
        if with_r:
            a, b = int(Z[i]), int(Z[j])
            rec = rec + (float(d[i, j] / (COVALENT_RADII.get(a, 1.0)
                                          + COVALENT_RADII.get(b, 1.0))),)
        out.append(rec)
    return out


# The cache stores three properties under lower-case names; the QM7-X file
# spells them differently.  Accept either.
TARGET_ALIAS = {"HLgap": "hlgap", "DIP": "dip", "eAT": "eAT",
                "hlgap": "hlgap", "dip": "dip"}


def target_value(mol, name):
    key = TARGET_ALIAS.get(name, name)
    if key not in mol:
        raise KeyError(
            f"property {name!r} is not in the cached molecules "
            f"(available: eAT, hlgap, dip). Other QM7-X properties need the "
            f"raw HDF5, and only 8000.hdf5 (config B5) is present here.")
    return float(np.asarray(mol[key]).ravel()[0])


def canonical_order(mol):
    """A deterministic atom -> qubit assignment.

    The published pipeline uses whatever atom order the HDF5 file happened to
    store, so qubit q means a different thing in every molecule.  Layer A is
    blind to this (a sorted spectrum is permutation invariant) but Layer B is
    not: it entangles qubit i with qubit j whenever ATOMS i and j are bonded.
    An arbitrary order therefore scatters the same chemistry across different
    wires from one molecule to the next.

    The key sorts by nuclear charge, then coordination number, then Coulomb
    local strength, with the original index last so ties resolve stably and the
    map is a genuine function of the molecule rather than of the file.
    """
    Z = np.asarray(mol["Z"], int)
    deg = np.asarray(mol["adj"]).sum(1)
    coulomb_sum = np.asarray(mol["W_coulomb"], float).sum(1)
    # np.lexsort takes the LAST key as primary.
    return np.lexsort((np.arange(len(Z)), -coulomb_sum, -deg, -Z))


def reorder_molecule(mol, perm):
    """Relabel one molecule's atoms.  Everything atom-indexed moves together;
    permutation-invariant quantities (spectra, energies) are untouched."""
    out = dict(mol)
    idx = np.ix_(perm, perm)
    out["Z"] = np.asarray(mol["Z"])[perm]
    out["adj"] = np.asarray(mol["adj"])[idx]
    out["W_coulomb"] = np.asarray(mol["W_coulomb"], float)[idx]
    out["dist"] = np.asarray(mol["dist"], float)[idx]
    out["xyz"] = np.asarray(mol["xyz"], float)[perm]
    return out


def apply_canonical_order(molecules):
    return [reorder_molecule(m, canonical_order(m)) for m in molecules]


def build_bond_types(molecules, n_qubits, scheme):
    """Global type vocabulary, ordered by descending frequency."""
    cnt = Counter()
    for m in molecules:
        for (_, _, _, k) in extract_bonds(m, n_qubits, scheme):
            cnt[k] += 1
    keys = [k for k, _ in cnt.most_common()]
    return {k: i for i, k in enumerate(keys)}, cnt


# =============================================================================
# 3.  Dataset assembly
# =============================================================================

class Dataset:
    """Per-molecule circuit inputs plus the composition-residual target."""

    def __init__(self, config, arch: Arch, split_level="molecule", seed=42,
                 bench_root=PAPER, molecules=None):
        self.config, self.arch, self.seed = config, arch, seed
        mols = molecules if molecules is not None else load_config(config, bench_root)
        if arch.canonical_order:
            mols = apply_canonical_order(mols)
        self.mols = mols
        nq = arch.n_qubits
        tr_full, te = make_split(mols, split_level, seed)
        # Learning-curve subsampling.  The TEST set is fixed by make_split and
        # never touched, so MAE is comparable across fractions; the training
        # subsets are nested (one permutation per seed, taken as a prefix), so
        # a smaller fraction is a subset of every larger one and the paired
        # comparison across fractions is clean.
        if arch.train_frac < 1.0:
            perm = np.random.default_rng(90_000 + seed).permutation(len(tr_full))
            k = max(2, int(round(len(tr_full) * arch.train_frac)))
            tr_full = tr_full[np.sort(perm[:k])]
        self.train_idx, self.test_idx = tr_full, te
        tr = self.train_idx

        self.y_raw = np.array([target_value(m, arch.target) for m in mols])

        # --- composition model (train-only fit), the published Plan A ---
        from sklearn.linear_model import LinearRegression
        elements = sorted({int(z) for m in mols for z in m["Z"]})
        elem_index = {e: i for i, e in enumerate(elements)}
        Xc = np.zeros((len(mols), len(elements)))
        for i, m in enumerate(mols):
            for z, c in Counter(int(z) for z in m["Z"]).items():
                Xc[i, elem_index[z]] = c
        if arch.residual:
            comp = LinearRegression().fit(Xc[tr], self.y_raw[tr])
            self.y_comp = comp.predict(Xc)
            self.comp_r2 = float(comp.score(Xc[tr], self.y_raw[tr]))
        else:
            # No composition stage: the circuit predicts the target directly.
            self.y_comp = np.zeros(len(mols))
            self.comp_r2 = 0.0
        resid = self.y_raw - self.y_comp
        self.resid = resid
        self.res_mean, self.res_std = resid[tr].mean(), resid[tr].std() + 1e-10
        self.y_target = (resid - self.res_mean) / self.res_std

        # --- Layer A angles (G3: encode_source, drop_q0, angle range) ---
        #
        # "eig" is the published QSM-CL encoding: qubit q carries the q-th
        # Coulomb-Laplacian eigenvalue.  But Layer B entangles qubit i with
        # qubit j whenever ATOMS i and j are bonded, so the same wire is asked
        # to mean an eigenmode index in one layer and an atom index in the
        # next.  XY hopping then mixes sorted eigenvalues along an unrelated
        # adjacency pattern, which is why the G1 diagnosis sees the coupling
        # reach the readout (delta > 0) while destroying the correlation with
        # the target (ridge R2 falls, spread falls).
        #
        # "diag" removes the mismatch: qubit q carries the Coulomb-Laplacian
        # DIAGONAL of atom q, sum_j Z_q Z_j / (d_qj + 0.1) -- an atom-local
        # quantity on an atom-indexed wire.  Same matrix, same physics, read
        # along the axis the entangler actually uses.
        raw = np.zeros((len(mols), nq))
        for i, m in enumerate(mols):
            if arch.encode_source == "diag":
                e = np.asarray(m["W_coulomb"], float).sum(1)
            else:
                e = np.asarray(m["lap_eig_coulomb"], float)[arch.drop_q0:]
            raw[i, :min(len(e), nq)] = e[:nq]
        self.gmax = float(np.max(np.abs(raw[tr]))) + 1e-10
        frac = raw / self.gmax
        self.ang = arch.angle_lo + (arch.angle_hi - arch.angle_lo) * frac

        # --- second encoding angle, applied as RZ (G3 N-d) ---------------
        # RZ is diagonal, so it cannot change any <Z_q> on its own: adding it
        # costs the published readout exactly nothing.  It only becomes visible
        # through a NON-diagonal entangler, which is the point -- it tells XY
        # hopping which atom each wire is, without disturbing the spectral
        # encoding that Layer A already put there.
        self.zang = np.zeros((len(mols), nq))
        if arch.rz_source == "Znum":
            zmax = max(int(z) for m in mols for z in m["Z"])
            for i, m in enumerate(mols):
                for q in range(min(int(m["n_atoms"]), nq)):
                    self.zang[i, q] = np.pi * int(m["Z"][q]) / zmax
        elif arch.rz_source == "diag":
            diag_raw = np.zeros((len(mols), nq))
            for i, m in enumerate(mols):
                e = np.asarray(m["W_coulomb"], float).sum(1)
                diag_raw[i, :min(len(e), nq)] = e[:nq]
            self.zang = np.pi * diag_raw / (np.max(np.abs(diag_raw[tr])) + 1e-10)
        elif arch.rz_source == "eig":
            # The complement of `encode_source="diag"`: RY carries the
            # atom-aligned quantity so the entangler reads the axis it indexes,
            # and RZ carries the spectrum.  Both meanings ride the same wire,
            # and RZ costs Layer A nothing because it is diagonal.
            eig_raw = np.zeros((len(mols), nq))
            for i, m in enumerate(mols):
                e = np.asarray(m["lap_eig_coulomb"], float)[arch.drop_q0:]
                eig_raw[i, :min(len(e), nq)] = e[:nq]
            self.zang = np.pi * eig_raw / (np.max(np.abs(eig_raw[tr])) + 1e-10)

        # --- bonds and type vocabulary ---
        # Built over every molecule by default, which is how all earlier results
        # were produced.  That is a mild support leak -- the SIZE of the
        # vocabulary depends on molecules in the test split, though no target
        # does -- and it also makes n_types constant under a learning-curve
        # sweep, which would defeat the purpose.  `types_from_train` builds it
        # from the training molecules alone and sends any unseen bond type to a
        # reserved bucket; no training bond touches that bucket, so its coupling
        # keeps its initial value and acts as the fallback.
        if arch.types_from_train:
            self.type_index, self.type_count = build_bond_types(
                [mols[i] for i in tr], nq, arch.bond_type_scheme)
            self.other_type = len(self.type_index)      # reserved fallback slot
        else:
            self.type_index, self.type_count = build_bond_types(
                mols, nq, arch.bond_type_scheme)
            self.other_type = None
        if arch.max_types and len(self.type_index) > arch.max_types:
            # build_bond_types orders by descending frequency, so truncating
            # keeps the commonest types and sends the tail to the fallback.
            keep = {k: i for k, i in self.type_index.items()
                    if i < arch.max_types}
            self.type_index = keep
            self.type_count = {k: self.type_count[k] for k in keep}
            self.other_type = arch.max_types
        # --- R10: relabel the vocabulary into coefficient GROUPS ---------
        # Everything downstream (alpha vector length, the shuffled control, the
        # readout) only ever sees `type_index` values, so remapping them here
        # makes the circuit behave as if it had been given a coarser
        # vocabulary.  `chem` leaves it untouched and is bit-identical to the
        # published behaviour.
        self.group_size_dist = ""
        if arch.group_mode in ("elem", "arb", "bijection"):
            full_keys = [k for k, _ in sorted(self.type_index.items(),
                                              key=lambda kv: kv[1])]
            sizes = Counter((k[0], k[1]) for k in full_keys)
            size_list = sorted(sizes.values(), reverse=True)
            if arch.group_mode == "elem":
                group_id = {group_key: i for i, group_key in enumerate(sorted(sizes))}
                remap = {self.type_index[k]: group_id[(k[0], k[1])] for k in full_keys}
            else:
                if arch.group_mode == "bijection":
                    size_list = [1] * len(full_keys)     # 1:1, same hypothesis space
                g = np.random.default_rng(arch.group_seed_offset + seed)
                order = g.permutation(len(full_keys))
                remap, pos = {}, 0
                for group_idx, group_size in enumerate(size_list):
                    for t in order[pos:pos + group_size]:
                        remap[int(t)] = group_idx
                    pos += group_size
            self.type_index = {k: remap[self.type_index[k]] for k in full_keys}
            if self.other_type is not None:
                self.other_type = len(size_list)
            self.group_size_dist = str(size_list)

        # `type_index` may now be many-to-one, so the bookkeeping below has to
        # be per GROUP: one representative key per coefficient, and counts
        # summed over the members.  Under `chem` this is the identity.
        reps, group_count = {}, {}
        for k, t in sorted(self.type_index.items(), key=lambda kv: kv[1]):
            reps.setdefault(t, k)
            group_count[t] = group_count.get(t, 0) + self.type_count[k]
        self.type_members = {t: [k for k in self.type_index if self.type_index[k] == t]
                             for t in reps}
        self.type_keys = [reps[t] for t in sorted(reps)]
        self.type_count = {reps[t]: group_count[t] for t in sorted(reps)}
        self.n_types_train = len(reps)
        self.n_types = self.n_types_train + (1 if arch.types_from_train else 0)

        # --- R10 `lenq`: index by reduced-length quantile, no chemistry -----
        self.len_cuts = None
        if arch.group_mode == "lenq":
            nbin = arch.lenq_bins or self.n_types_train
            r_all = [b[4] for i in tr
                     for b in extract_bonds(mols[i], nq, arch.bond_type_scheme,
                                            with_r=True)]
            # quantile edges from TRAINING molecules only
            self.len_cuts = np.quantile(r_all, np.linspace(0, 1, nbin + 1)[1:-1])
            binned = np.searchsorted(self.len_cuts, r_all)
            self.type_keys = [("LENQ", b) for b in range(nbin)]
            self.type_count = {("LENQ", b): int((binned == b).sum()) or 1
                               for b in range(nbin)}
            self.type_members = {b: [("LENQ", b)] for b in range(nbin)}
            self.other_type = None
            self.n_types_train = nbin
            self.n_types = nbin
            self.group_size_dist = str([1] * nbin)

        rng = np.random.default_rng(10_000 + seed)
        probs = np.array([self.type_count[k] for k in self.type_keys], float)
        if self.other_type is not None:
            probs = np.append(probs, 0.0)             # never shuffle INTO the bucket
        probs /= probs.sum()

        self.bonds = []
        lenq = arch.group_mode == "lenq"
        for m in mols:
            raw_bonds = extract_bonds(m, nq, arch.bond_type_scheme, with_r=lenq)
            typed_bonds = []
            for b in raw_bonds:
                i, j, J, key = b[0], b[1], b[2], b[3]
                if lenq:
                    t = int(np.searchsorted(self.len_cuts, b[4]))
                else:
                    t = self.type_index.get(key, self.other_type)
                if t is None:
                    raise KeyError(key)
                # G2 control arm: same parameter budget, chemistry destroyed.
                t_sh = int(rng.choice(self.n_types, p=probs))
                typed_bonds.append((i, j, J, t, t_sh))
            self.bonds.append(typed_bonds)

        self.n_active = np.array([min(int(m["n_atoms"]), nq) for m in mols])
        self.formula_key = [m["formula_key"] for m in mols]

    def angle_max(self, alpha_vec):
        """Largest entangling angle actually applied -- watch for 2pi wrap."""
        max_angle = 0.0
        for mol_bonds in self.bonds:
            for (_, _, J, t, _) in mol_bonds:
                max_angle = max(max_angle, abs(J * float(alpha_vec[t % len(alpha_vec)])))
        return max_angle * self.arch.theta_scale / max(self.arch.n_rounds, 1)


# =============================================================================
# 4.  Circuit blueprints
# =============================================================================
#
# A blueprint is the static op list of one molecule: (gate class, wires,
# provenance).  Provenance says where the gate's angle comes from and is used
# both to rebuild the angle each epoch and to route the adjoint gradient back:
#
#   ("const", value)                fixed angle, not trained
#   ("var",   flat_index)           d theta / d var[flat] = 1
#   ("ent",   type_id, factor)      theta = factor * alpha[type_id]
#
# Only "var" and "ent" slots are declared trainable on the tape, so the adjoint
# pass costs one reverse sweep per trained parameter and nothing for the data.

def _entangler_ops(ops, prov, i, j, tid, J, arch, scale):
    """Append the chosen two-qubit entangler for one bond."""
    f = J * scale
    if arch.entangler == "zz":
        ops.append((qml.IsingZZ, [i, j])); prov.append(("ent", tid, f))
    elif arch.entangler == "xy":
        ops.append((qml.IsingXY, [i, j])); prov.append(("ent", tid, f))
    elif arch.entangler == "xxz":
        ops.append((qml.IsingXY, [i, j])); prov.append(("ent", tid, f))
        ops.append((qml.IsingZZ, [i, j])); prov.append(("ent", tid, f * arch.gamma_zz))
    elif arch.entangler == "trotter":
        pass  # handled by the caller, which interleaves RX slices
    elif arch.entangler == "none":
        pass
    else:
        raise ValueError(arch.entangler)


def blueprint(ds: Dataset, mol_i, arch: Arch, shuffle_types=False):
    """Static op list for one molecule."""
    nq = arch.n_qubits
    ops, prov = [], []
    bonds = ds.bonds[mol_i]
    n_act = ds.n_active[mol_i]
    type_field = 4 if shuffle_types else 3    # index of the type field

    scale = arch.theta_scale / max(arch.n_rounds, 1)

    for u in range(arch.n_reupload):
        # ---- Layer A: QSM encoding (globally normalised CL eigenvalues) ----
        for q in range(nq):
            ops.append((qml.RY, [q])); prov.append(("enc", mol_i, q))
        if arch.rz_source != "none":
            for q in range(nq):
                ops.append((qml.RZ, [q])); prov.append(("encz", mol_i, q))

        for r in range(arch.n_rounds):
            # ---- Layer B: MGQC entangler on bonded pairs ----
            if arch.entangler == "trotter":
                L = max(arch.trotter_layers, 1)
                for _l in range(L):
                    for (i, j, J, t, ts) in bonds:
                        ops.append((qml.IsingZZ, [i, j]))
                        prov.append(("ent", (ts if shuffle_types else t), J * scale / L))
                    for q in range(nq):
                        ops.append((qml.RX, [q])); prov.append(("const", arch.trotter_beta))
            else:
                for b in bonds:
                    i, j, J = b[0], b[1], b[2]
                    _entangler_ops(ops, prov, i, j, b[type_field], J, arch, scale)

            # ---- Layer B-2: padding entanglement on unused qubits ----
            if arch.pad_entangle and arch.entangler != "none":
                gate = qml.IsingXY if arch.entangler in ("xy", "xxz") else qml.IsingZZ
                for q in range(n_act, nq):
                    anchor_q = min(q, n_act - 1) if n_act > 0 else 0
                    if anchor_q != q:
                        ops.append((gate, [anchor_q, q]))
                        prov.append(("const", 0.05 * scale))

            # ---- Layer C-local: variational refinement ----
            for q in range(nq):
                base = (((u * arch.n_rounds) + r) * nq + q) * 2
                ops.append((qml.RY, [q])); prov.append(("var", base))
                ops.append((qml.RZ, [q])); prov.append(("var", base + 1))

        # ---- Layer C-global: CNOT chain, once per re-upload block ----
        if arch.use_cnot:
            for q in range(nq - 1):
                ops.append((qml.CNOT, [q, q + 1])); prov.append(None)

    trainable = [k for k, p in enumerate(prov) if p is not None and p[0] in ("var", "ent")]
    # index among *parametric* ops only (CNOT contributes no tape parameter)
    par_pos, c = {}, 0
    for k, p in enumerate(prov):
        if p is not None:
            par_pos[k] = c
            c += 1
    return ops, prov, [par_pos[k] for k in trainable], trainable


class Compiled:
    """Blueprints for every molecule, reused across epochs."""

    def __init__(self, ds: Dataset, arch: Arch):
        self.ds, self.arch = ds, arch
        self.shuffle = (arch.alpha_mode == "bytype_shuffled")
        self.bp = [blueprint(ds, i, arch, self.shuffle) for i in range(len(ds.mols))]

    def n_alpha(self):
        if self.arch.alpha_mode in ("bytype", "bytype_shuffled"):
            return self.ds.n_types
        return 1

    def alpha_vec(self, alpha_params, normalize=True):
        """Broadcast the trained alpha(s) to a per-type vector.

        With `arch.alpha_norm == "rms"` the vector is rescaled to unit RMS just
        before it enters the circuit, so `theta_scale` alone sets the coupling
        magnitude and gradient descent can only move the direction.  The stored
        `alpha` is left untouched, so everything that reports alpha (the E1
        dumps, R5-B's theta distribution) keeps reading the raw values.
        `normalize=False` returns the un-normalised vector, which the R7
        non-linearity probe needs in order to sweep a multiplier.
        """
        if self.arch.alpha_mode in ("bytype", "bytype_shuffled"):
            v = alpha_params
        else:
            v = np.full(max(self.ds.n_types, 1), float(alpha_params[0]))
        if normalize and self.arch.alpha_norm == "rms":
            v = v / (np.sqrt(np.mean(np.asarray(v, float) ** 2)) + 1e-12)
        return v

    def alpha_norm_vjp(self, alpha_params, g_eff):
        """Pull d loss / d(normalised alpha) back to d loss / d(raw alpha).

        For a_eff = a / r with r = sqrt(mean(a^2)),
            d a_eff_i / d a_j = (delta_ij - a_eff_i a_eff_j / n) / r
        so the raw gradient is the effective one with its component along
        a_eff removed and divided by r.  Without this the optimiser keeps
        pushing on a direction the circuit cannot see, |a| drifts, and Adam's
        per-coordinate step silently stops moving the direction we do care
        about."""
        if self.arch.alpha_norm != "rms":
            return g_eff
        a = np.asarray(alpha_params, float)
        n = a.size
        r = np.sqrt(np.mean(a ** 2)) + 1e-12
        a_eff = a / r
        return (g_eff - a_eff * float(a_eff @ g_eff) / n) / r

    def tape(self, i, var, alpha_vec, obs):
        ops, prov, tp_idx, _ = self.bp[i]
        ang, zang = self.ds.ang[i], self.ds.zang[i]
        qops = []
        for (G, w), p in zip(ops, prov):
            if p is None:
                qops.append(G(wires=w))
            elif p[0] == "enc":
                qops.append(G(ang[p[2]], wires=w))
            elif p[0] == "encz":
                qops.append(G(zang[p[2]], wires=w))
            elif p[0] == "const":
                qops.append(G(p[1], wires=w))
            elif p[0] == "var":
                qops.append(G(var[p[1]], wires=w))
            else:                                    # ("ent", tid, factor)
                qops.append(G(p[2] * alpha_vec[p[1]], wires=w))
        t = qml.tape.QuantumScript(qops, obs)
        t.trainable_params = tp_idx
        return t

    def chain(self, i, grads):
        """Map d loss / d(gate angle) back onto (var, alpha) parameters."""
        _, prov, _, trainable = self.bp[i]
        grad_var = np.zeros(self.arch.n_var_params())
        grad_alpha = np.zeros(self.n_alpha())
        bytype = self.arch.alpha_mode in ("bytype", "bytype_shuffled")
        for g, k in zip(np.atleast_1d(np.asarray(grads, float).ravel()), trainable):
            p = prov[k]
            if p[0] == "var":
                grad_var[p[1]] += g
            else:
                grad_alpha[p[1] if bytype else 0] += g * p[2]
        return grad_var, grad_alpha


# =============================================================================
# 5.  Readout
# =============================================================================

class Readout:
    """Ridge (or RBF-KRR) on the measured features.

    `scaler=True` is the one-line readout fix: qubit-wise <Z> standard deviations span
    a factor of 40-220 within one configuration, and a single ridge penalty
    over unscaled features effectively deletes the low-variance qubits.
    Zero-variance qubits (lambda_1 = 0 always gives <Z_0> = 1) are guarded.
    """

    def __init__(self, arch: Arch):
        self.arch = arch
        self.mu = self.sd = self.w = None
        self.b = 0.0
        self.krr = None

    def _prep(self, Z, fit=False):
        if not self.arch.readout_scaler:
            if fit:
                self.mu = np.zeros(Z.shape[1]); self.sd = np.ones(Z.shape[1])
            return Z
        if fit:
            self.mu = Z.mean(0)
            sd = Z.std(0)
            sd[sd < 1e-12] = 1.0
            self.sd = sd
        return (Z - self.mu) / self.sd

    def fit(self, Z, y):
        Zs = self._prep(Z, fit=True)
        if self.arch.readout_kind == "krr":
            from sklearn.kernel_ridge import KernelRidge
            from sklearn.model_selection import GridSearchCV
            grid = {"alpha": [1e-3, 1e-2, 1e-1, 1.0], "gamma": [1e-3, 1e-2, 1e-1, 1.0]}
            gs = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=5,
                              scoring="neg_mean_absolute_error")
            gs.fit(Zs, y)
            self.krr = gs.best_estimator_
            return self
        Za = np.column_stack([Zs, np.ones(len(Zs))])
        reg = self.arch.readout_alpha * np.eye(Za.shape[1]); reg[-1, -1] = 0.0
        sol = np.linalg.solve(Za.T @ Za + reg, Za.T @ y)
        self.w, self.b = sol[:-1], float(sol[-1])
        return self

    def predict(self, Z):
        Zs = self._prep(Z)
        if self.krr is not None:
            return self.krr.predict(Zs)
        return Zs @ self.w + self.b

    def obs_coeffs(self):
        """d pred / d <Z_q>, the coefficients of the effective Hamiltonian."""
        if self.krr is not None:
            return None
        return self.w / self.sd


# =============================================================================
# 6.  Trainer
# =============================================================================

class Trainer:
    """Alternating optimisation: closed-form ridge readout, then one Adam step
    on the circuit parameters against that readout.

    The gradient of the readout output w.r.t. the circuit is
        d pred_i / d theta = sum_q (w_q / sigma_q) d <Z_q>_i / d theta,
    so instead of 13 separate adjoint sweeps we measure a single observable
    H = sum_q (w_q / sigma_q) Z_q.  That is one reverse pass per molecule.
    """

    def __init__(self, ds: Dataset, arch: Arch, seed=42, device="lightning.qubit"):
        self.ds, self.arch, self.seed = ds, arch, seed
        self.dev = qml.device(device, wires=arch.n_qubits)
        self.comp = Compiled(ds, arch)
        rng = np.random.default_rng(seed)
        self.var0 = (rng.normal(size=arch.n_var_params()) * 0.3
                     if arch.train_var else np.zeros(arch.n_var_params()))
        self.var = self.var0.copy()
        self.alpha = np.full(self.comp.n_alpha(), float(arch.alpha))
        self.alpha0 = self.alpha.copy()
        self.trainable_alpha = arch.alpha_mode != "fixed"
        self.ops_readout = [qml.PauliZ(q) for q in range(arch.n_qubits)]
        if arch.observable == "ZXY":
            # Only meaningful with a non-diagonal entangler AND without Layer C's
            # CNOT chain: the chain turns a single-qubit X into a 13-body X-string
            # whose expectation is a product of 13 sines, numerically zero for
            # every molecule (13-body X string).
            self.ops_readout += [qml.PauliX(q) for q in range(arch.n_qubits)]
            self.ops_readout += [qml.PauliY(q) for q in range(arch.n_qubits)]
        self.Zobs = [qml.expval(o) for o in self.ops_readout]
        self.traj = []

    # -- forward ---------------------------------------------------------
    def features(self, idx, var=None, alpha=None):
        var = self.var if var is None else var
        alpha_eff = self.comp.alpha_vec(self.alpha if alpha is None else alpha)
        tapes = [self.comp.tape(i, var, alpha_eff, self.Zobs) for i in idx]
        res = self.dev.execute(tapes)
        return np.array([np.asarray(r, float).ravel() for r in res])

    def features_vec(self, idx, alpha_vector, var=None):
        """Readout features for an explicit per-type coupling vector.

        Bypasses `alpha_vec`, so the R7 probe can sweep a global multiplier
        without the RMS normalisation cancelling it.  theta = factor * alpha
        is linear in alpha, so multiplying this vector by k is exactly
        equivalent to multiplying every entangling angle by k."""
        var = self.var if var is None else var
        alpha_eff = np.asarray(alpha_vector, float)
        tapes = [self.comp.tape(i, var, alpha_eff, self.Zobs) for i in idx]
        res = self.dev.execute(tapes)
        return np.array([np.asarray(r, float).ravel() for r in res])

    # -- gradient --------------------------------------------------------
    def _grads(self, idx, coef, weights):
        alpha_eff = self.comp.alpha_vec(self.alpha)
        obs = [qml.expval(qml.dot(list(coef), self.ops_readout))]
        tapes = [self.comp.tape(i, self.var, alpha_eff, obs) for i in idx]
        d = self.dev.compute_derivatives(tuple(tapes), ADJOINT_CFG)
        grad_var = np.zeros_like(self.var)
        grad_alpha = np.zeros_like(self.alpha)
        for k, i in enumerate(idx):
            a, b = self.comp.chain(i, d[k])
            grad_var += weights[k] * a
            grad_alpha += weights[k] * b
        # `chain` returns d loss / d(alpha as the circuit saw it).  Under
        # alpha_norm="rms" that is the NORMALISED vector, so project back.
        grad_alpha = self.comp.alpha_norm_vjp(self.alpha, grad_alpha)
        return grad_var, grad_alpha

    # -- training loop ---------------------------------------------------
    def fit(self, epochs=None, lr=None, verbose=False):
        arch = self.arch
        epochs = arch.epochs if epochs is None else epochs
        lr = arch.lr if lr is None else lr
        tr = self.ds.train_idx
        y = self.ds.y_target[tr]
        n = len(tr)

        train_anything = (arch.train_var or self.trainable_alpha) and epochs > 0
        if train_anything and arch.optimizer == "spsa":
            return self._fit_spsa(epochs, verbose)
        mv = np.zeros_like(self.var); vv = np.zeros_like(self.var)
        ma = np.zeros_like(self.alpha); va = np.zeros_like(self.alpha)
        b1, b2, eps = 0.9, 0.999, 1e-8

        # The circuit is always trained against a LINEAR readout: the gradient
        # needs d pred / d <Z_q>, which only a linear map provides in closed
        # form.  A non-linear readout (KRR) is a choice about how the finished
        # features are read, so it is fitted once at the end instead.
        import copy as _copy
        grad_arch = _copy.copy(arch)
        grad_arch.readout_kind = "ridge"
        self.readout = Readout(grad_arch)
        loss_first = loss_last = np.nan

        for ep in range(max(epochs, 1)):
            Z = self.features(tr)
            self.readout.fit(Z, y)
            pred = self.readout.predict(Z)
            loss = float(np.mean((pred - y) ** 2))
            if ep == 0:
                loss_first = loss
            loss_last = loss

            gnorm = np.nan
            if train_anything and ep < epochs:
                coef = self.readout.obs_coeffs()
                w = 2.0 * (pred - y) / n
                grad_var, grad_alpha = self._grads(tr, coef, w)
                gnorm = float(np.sqrt((grad_var ** 2).sum() + (grad_alpha ** 2).sum()))
                t = ep + 1
                if arch.train_var:
                    mv = b1 * mv + (1 - b1) * grad_var
                    vv = b2 * vv + (1 - b2) * grad_var ** 2
                    self.var -= lr * (mv / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + eps)
                if self.trainable_alpha:
                    ma = b1 * ma + (1 - b1) * grad_alpha
                    va = b2 * va + (1 - b2) * grad_alpha ** 2
                    self.alpha -= lr * (ma / (1 - b1 ** t)) / (np.sqrt(va / (1 - b2 ** t)) + eps)

            self.traj.append(dict(
                epoch=ep, loss=loss, grad_norm=gnorm,
                param_shift=float(np.linalg.norm(self.var - self.var0)
                                  + np.linalg.norm(self.alpha - self.alpha0)),
                var_std=float(self.var.std()),
                alpha_mean=float(self.alpha.mean())))
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"    ep {ep:3d}  loss {loss:.5f}  |g| {gnorm:.3e}  "
                      f"shift {self.traj[-1]['param_shift']:.4f}")

        # final readout on the trained circuit, in the requested class
        Z = self.features(tr)
        self.readout = Readout(arch).fit(Z, y)
        self.loss_first, self.loss_last = loss_first, loss_last
        self.init_norm = float(np.linalg.norm(self.var0) + np.linalg.norm(self.alpha0))
        return self

    # -- published optimiser, kept for the G0 "before" measurement --------
    def _fit_spsa(self, epochs, verbose):
        """Reproduces the superseded implementation: PennyLane's
        SPSAOptimizer(maxiter=1) stepped once per epoch against a ridge readout
        that is refitted at the top of every epoch.  Only the backend differs,
        so this is the published training dynamics at ~200x the speed."""
        from pennylane import numpy as pnp
        arch = self.arch
        tr = self.ds.train_idx
        y = self.ds.y_target[tr]
        np.random.seed(self.seed)
        opt = qml.SPSAOptimizer(maxiter=1)
        import copy as _copy
        grad_arch = _copy.copy(arch)
        grad_arch.readout_kind = "ridge"
        self.readout = Readout(grad_arch)
        loss_first = loss_last = np.nan
        w = pnp.array(self.var, requires_grad=True)

        for ep in range(epochs):
            self.var = np.asarray(w, float)
            Z = self.features(tr)
            self.readout.fit(Z, y)
            pred = self.readout.predict(Z)
            loss = float(np.mean((pred - y) ** 2))
            if ep == 0:
                loss_first = loss
            loss_last = loss

            def cost(ww):
                Zc = self.features(tr, var=np.asarray(ww, float))
                return np.mean((self.readout.predict(Zc) - y) ** 2)

            w = opt.step(cost, w)
            self.var = np.asarray(w, float)
            self.traj.append(dict(
                epoch=ep, loss=loss, grad_norm=np.nan,
                param_shift=float(np.linalg.norm(self.var - self.var0)),
                var_std=float(self.var.std()), alpha_mean=float(self.alpha.mean())))
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(f"    ep {ep:3d}  loss {loss:.5f}  shift "
                      f"{self.traj[-1]['param_shift']:.4f}  (SPSA)")

        Z = self.features(tr)
        self.readout = Readout(arch).fit(Z, y)
        self.loss_first, self.loss_last = loss_first, loss_last
        self.init_norm = float(np.linalg.norm(self.var0) + np.linalg.norm(self.alpha0))
        return self

    # -- evaluation -------------------------------------------------------
    def predict_eV(self, idx):
        Z = self.features(idx)
        p = self.readout.predict(Z) * self.ds.res_std + self.ds.res_mean
        return self.ds.y_comp[idx] + p

    def evaluate(self):
        from sklearn.metrics import mean_absolute_error, r2_score
        te = self.ds.test_idx
        pred = self.predict_eV(te)
        true = self.ds.y_raw[te]
        out = dict(mae=float(mean_absolute_error(true, pred)),
                   r2=float(r2_score(true, pred)))
        out.update(isomer_metrics(self.ds, te, pred))
        return out, pred

    def evaluate_readouts(self, kinds=("ridge", "krr")):
        """Several readout classes off ONE trained circuit.

        The readout is fitted on the finished features, so comparing ridge
        against RBF kernel ridge costs one extra fit, not one extra training
        run.  Only the training-time readout (always linear -- the gradient
        needs d pred / d <Z_q> in closed form) is fixed.
        """
        from sklearn.metrics import mean_absolute_error, r2_score
        import copy as _copy
        tr, te = self.ds.train_idx, self.ds.test_idx
        y = self.ds.y_target[tr]
        Ztr, Zte = self.features(tr), self.features(te)
        true = self.ds.y_raw[te]
        out = {}
        for kind in kinds:
            a = _copy.copy(self.arch)
            a.readout_kind = kind
            readout = Readout(a).fit(Ztr, y)
            pred = self.ds.y_comp[te] + readout.predict(Zte) * self.ds.res_std + self.ds.res_mean
            m = dict(mae=float(mean_absolute_error(true, pred)),
                     r2=float(r2_score(true, pred)))
            m.update(isomer_metrics(self.ds, te, pred))
            out[kind] = (m, pred)
        return out


# =============================================================================
# 7.  Isomer-group evaluation axis (G4)
# =============================================================================

def isomer_metrics(ds: Dataset, test_idx, pred):
    """Inside a formula group composition is constant, so only the structural
    signal survives.  Two variants are reported:

      isomer_mae   deviations from each test group's own mean, on both the
                   truth and the prediction.  Group means cancel, so this is a
                   pure within-group discrimination measure and never feeds
                   back into training.
      isomer_mae_tr  the leak-free variant: the group mean is
                   estimated from TRAINING members only.
      isomer_rho   size-weighted mean Spearman rho inside each group.
    """
    from scipy import stats
    y = ds.y_raw
    tr_by_g = defaultdict(list)
    for i in ds.train_idx:
        tr_by_g[ds.formula_key[i]].append(i)
    te_by_g = defaultdict(list)
    pos = {i: k for k, i in enumerate(test_idx)}
    for i in test_idx:
        te_by_g[ds.formula_key[i]].append(i)

    dev_true, dev_pred, rhos, wts = [], [], [], []
    dev_true_tr, dev_pred_tr = [], []
    n_groups = 0
    for g, members in te_by_g.items():
        if len(members) < 2:
            continue
        n_groups += 1
        y_group = np.array([y[i] for i in members])
        pred_group = np.array([pred[pos[i]] for i in members])
        dev_true.extend(y_group - y_group.mean()); dev_pred.extend(pred_group - pred_group.mean())
        if len(set(np.round(y_group, 9))) > 1:
            r = stats.spearmanr(y_group, pred_group).statistic
            if np.isfinite(r):
                rhos.append(r); wts.append(len(members))
        if g in tr_by_g:
            mu = np.mean([y[i] for i in tr_by_g[g]])
            dev_true_tr.extend(y_group - mu); dev_pred_tr.extend(pred_group - mu)

    out = dict(n_isomer_groups=n_groups)
    out["isomer_mae"] = float(np.mean(np.abs(np.array(dev_true) - np.array(dev_pred)))) if dev_true else np.nan
    out["isomer_mae_tr"] = (float(np.mean(np.abs(np.array(dev_true_tr) - np.array(dev_pred_tr))))
                            if dev_true_tr else np.nan)
    out["isomer_rho"] = (float(np.average(rhos, weights=wts)) if rhos else np.nan)
    return out


# =============================================================================
# 8.  One arm = one (arch, config, seed) run
# =============================================================================

COMMON_COLUMNS = [
    "config", "target", "residual", "split", "entangler", "alpha", "alpha_mode",
    "theta_scale", "alpha_norm", "n_rounds",
    "bond_param_mode", "encode_source", "drop_q0", "angle_range",
    "rz_source", "canonical", "n_reupload",
    "observable", "use_cnot", "readout", "scaler", "optimizer", "diff_method", "epochs",
    "seed", "n_qubits", "n_params", "n_types", "n_train", "n_test",
    "mae", "r2", "isomer_mae", "isomer_mae_tr", "isomer_rho", "n_isomer_groups",
    "loss_first", "loss_last", "loss_drop", "param_shift", "param_shift_rel",
    "grad_norm_last", "alpha_mean", "angle_max", "comp_r2", "time_s",
]


def run_arm(config, arch: Arch, seed=42, split_level="molecule",
            ds_cache=None, verbose=False, extra=None, readouts=None):
    t0 = time.time()
    key = (config, split_level, seed, arch.n_qubits, arch.encode_source,
           arch.rz_source, arch.canonical_order, arch.drop_q0,
           arch.angle_lo, arch.angle_hi, arch.bond_type_scheme,
           arch.types_from_train, arch.max_types, arch.train_frac,
           arch.target, arch.residual)
    ds = None
    if ds_cache is not None:
        ds = ds_cache.get(key)
    if ds is None:
        ds = Dataset(config, arch, split_level, seed)
        if ds_cache is not None:
            ds_cache[key] = ds
    ds.arch = arch

    tr = Trainer(ds, arch, seed=seed).fit(verbose=verbose)
    if readouts:
        multi = tr.evaluate_readouts(readouts)
        metrics, pred = multi[readouts[0]]
    else:
        multi = None
        metrics, pred = tr.evaluate()

    n_params = (arch.n_var_params() * int(arch.train_var)
                + (tr.comp.n_alpha() if arch.alpha_mode != "fixed" else 0)
                + arch.n_features() + 1)
    loss_drop_rel = (tr.loss_first - tr.loss_last) / max(abs(tr.loss_first), 1e-12)
    shift = float(np.linalg.norm(tr.var - tr.var0) + np.linalg.norm(tr.alpha - tr.alpha0))

    row = dict(
        config=config, target=arch.target, residual=int(arch.residual),
        split=split_level, entangler=arch.entangler,
        alpha=arch.alpha, alpha_mode=arch.alpha_mode,
        theta_scale=arch.theta_scale, alpha_norm=arch.alpha_norm,
        n_rounds=arch.n_rounds,
        bond_param_mode=arch.alpha_mode, encode_source=arch.encode_source,
        drop_q0=arch.drop_q0,
        angle_range=f"[{arch.angle_lo:.3f},{arch.angle_hi:.3f}]",
        rz_source=arch.rz_source, canonical=int(arch.canonical_order),
        n_reupload=arch.n_reupload,
        observable=arch.observable, readout=arch.readout_kind,
        use_cnot=int(arch.use_cnot), scaler=int(arch.readout_scaler),
        optimizer=(arch.optimizer
                   if (arch.train_var or arch.alpha_mode != "fixed") and arch.epochs > 0
                   else "none"),
        diff_method=("adjoint" if arch.optimizer == "adam" else "none"),
        epochs=arch.epochs, seed=seed,
        n_qubits=arch.n_qubits, n_params=n_params, n_types=ds.n_types,
        n_types_train=ds.n_types_train, max_types=arch.max_types,
        train_frac=arch.train_frac,
        types_from_train=int(arch.types_from_train),
        n_train=len(ds.train_idx), n_test=len(ds.test_idx),
        loss_first=tr.loss_first, loss_last=tr.loss_last, loss_drop=loss_drop_rel,
        param_shift=shift,
        param_shift_rel=shift / max(tr.init_norm, 1e-12),
        grad_norm_last=tr.traj[-1]["grad_norm"],
        alpha_mean=float(tr.alpha.mean()),
        angle_max=ds.angle_max(tr.comp.alpha_vec(tr.alpha)),
        comp_r2=ds.comp_r2, time_s=round(time.time() - t0, 2))
    row.update(metrics)
    if multi:
        for kind, (m, _) in multi.items():
            for k, v in m.items():
                row[f"{k}_{kind}"] = v
    if extra:
        row.update(extra)
    return row, tr, pred


# =============================================================================
# 9.  Small helpers
# =============================================================================

def write_csv(path, rows, columns=None):
    import csv
    if not rows:
        return
    cols = columns or list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"  -> {path}  ({len(rows)} rows)")


def paired_test(a, b, test="wilcoxon"):
    """Paired comparison of two arms across seeds.

    `test` selects the p value that is returned: "wilcoxon" for the Wilcoxon
    signed-rank test (the manuscript's stated test, and the safer choice at
    10-40 seeds where normality is not established) or "t" for the paired
    t-test.  Both are computed and returned as p_wilcoxon / p_t so a run can be
    reported either way without recomputation."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return dict(mean_a=float(a.mean()) if len(a) else np.nan, std_a=np.nan,
                    mean_b=float(b.mean()) if len(b) else np.nan, std_b=np.nan,
                    rel=float((a.mean() - b.mean()) / b.mean()) if len(a) else np.nan,
                    wins=int((a < b).sum()), n=len(a), p=np.nan,
                    p_wilcoxon=np.nan, p_t=np.nan)
    p_t = float(stats.ttest_rel(a, b).pvalue)
    d = a - b
    if np.allclose(d, 0):
        p_wilcoxon = 1.0
    else:
        try:
            p_wilcoxon = float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
        except ValueError:
            p_wilcoxon = float("nan")
    return dict(mean_a=float(a.mean()), std_a=float(a.std()),
                mean_b=float(b.mean()), std_b=float(b.std()),
                rel=float((a.mean() - b.mean()) / b.mean()),
                wins=int((a < b).sum()), n=len(a),
                p=(p_wilcoxon if test == "wilcoxon" else p_t),
                p_wilcoxon=p_wilcoxon, p_t=p_t)


def qubits_for(config):
    """No hard cap: one qubit per atom at the configuration's atom limit."""
    return CONFIG_ATOMS[config]


def baseline_arch(n_qubits=13, **kw):
    """The published circuit, byte for byte."""
    a = Arch(n_qubits=n_qubits)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def parallel_jobs(fn, arglist, n_jobs=1, desc=""):
    """Run `fn(*args)` over arglist.  Each call returns a list of result rows.

    Jobs are grouped by (config, seed) so one worker builds the Dataset once
    and reuses it across every architecture arm -- dataset assembly costs more
    than a short training run does.
    """
    import multiprocessing as mp
    rows = []
    t0 = time.time()
    if n_jobs <= 1:
        for k, a in enumerate(arglist):
            rows.extend(fn(*a))
            print(f"  [{k+1}/{len(arglist)}] {desc} {a}  ({time.time()-t0:.0f}s)",
                  flush=True)
        return rows
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(n_jobs, len(arglist))) as pool:
        for k, r in enumerate(pool.starmap(fn, arglist, chunksize=1)):
            rows.extend(r)
            print(f"  [{k+1}/{len(arglist)}] {desc} done  ({time.time()-t0:.0f}s)",
                  flush=True)
    return rows


def summarise(rows, group_keys, value="mae"):
    """Mean/std of `value` over seeds, for every distinct combination of
    `group_keys`."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[tuple(r[k] for k in group_keys)].append(r)
    out = []
    for k, bucket_rows in buckets.items():
        v = np.array([r[value] for r in bucket_rows], float)
        d = dict(zip(group_keys, k))
        d.update({f"{value}_mean": float(np.nanmean(v)),
                  f"{value}_std": float(np.nanstd(v)),
                  "n_seeds": len(v)})
        out.append(d)
    return out
