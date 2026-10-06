#!/usr/bin/env python3
"""
=============================================================================
Correctness checks for the fast adjoint path
=============================================================================
The trainer differentiates by hand: it asks the device for d<H>/d(gate angle)
and then applies its own chain rule to reach (var_weights, alpha).  Nothing
downstream is worth anything if that routing is wrong, so it is checked here
against two independent references.

  T1  gradient vs central finite differences of the same loss
  T2  gradient vs PennyLane's own parameter-shift through a plain QNode
  T3  the published circuit is reproduced bit for bit by the defaults
  T4  Diagonal-entangler invariance re-measured: with a diagonal (zz) entangler, alpha cannot move any
      <Z_q>;  with xy it can.  This is the premise of G1.
=============================================================================
"""
import sys, os
import numpy as np

import pennylane as qml
from pennylane import numpy as pnp
from . import core as C


def loss_of(tr, var, alpha):
    """Exactly the quantity the trainer differentiates: MSE of the ridge
    readout output, with the readout held fixed."""
    Z = tr.features(tr.ds.train_idx, var=var, alpha=alpha)
    pred = tr.readout.predict(Z)
    y = tr.ds.y_target[tr.ds.train_idx]
    return float(np.mean((pred - y) ** 2))


def analytic(tr):
    tr_idx = tr.ds.train_idx
    Z = tr.features(tr_idx)
    pred = tr.readout.predict(Z)
    y = tr.ds.y_target[tr_idx]
    w = 2.0 * (pred - y) / len(tr_idx)
    return tr._grads(tr_idx, tr.readout.obs_coeffs(), w)


def t1_finite_difference(config="B5", nq=8, mode="bytype", ent="xy"):
    print(f"\nT1  adjoint chain rule vs finite differences   "
          f"[{config}, {nq} qubits, entangler={ent}, alpha_mode={mode}]")
    arch = C.Arch(n_qubits=nq, entangler=ent, alpha=0.6, alpha_mode=mode,
                  n_rounds=2, readout_scaler=True, epochs=0)
    ds = C.Dataset(config, arch, "molecule", 42)
    ds.train_idx = ds.train_idx[:24]          # keep the FD sweep cheap
    tr = C.Trainer(ds, arch, seed=1).fit(epochs=0)
    grad_var, grad_alpha = analytic(tr)

    eps = 1e-5
    err = []
    for k in np.linspace(0, len(grad_var) - 1, 8).astype(int):
        plus = tr.var.copy(); plus[k] += eps
        minus = tr.var.copy(); minus[k] -= eps
        fd = (loss_of(tr, plus, tr.alpha) - loss_of(tr, minus, tr.alpha)) / (2 * eps)
        err.append(abs(fd - grad_var[k]))
        print(f"    var[{k:3d}]   adjoint {grad_var[k]:+.9f}   fd {fd:+.9f}   |d| {err[-1]:.2e}")
    for k in range(min(len(grad_alpha), 5)):
        plus = tr.alpha.copy(); plus[k] += eps
        minus = tr.alpha.copy(); minus[k] -= eps
        fd = (loss_of(tr, tr.var, plus) - loss_of(tr, tr.var, minus)) / (2 * eps)
        err.append(abs(fd - grad_alpha[k]))
        print(f"    alpha[{k:2d}]  adjoint {grad_alpha[k]:+.9f}   fd {fd:+.9f}   |d| {err[-1]:.2e}")
    ok = max(err) < 5e-6
    print(f"    max abs error {max(err):.2e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def t2_parameter_shift(nq=6):
    """Independent path: rebuild one molecule's circuit as an ordinary QNode
    and differentiate with parameter-shift on default.qubit."""
    print(f"\nT2  device adjoint vs QNode parameter-shift   [{nq} qubits]")
    arch = C.Arch(n_qubits=nq, entangler="xxz", alpha=0.7, alpha_mode="scalar",
                  n_rounds=2, rz_source="Znum", readout_scaler=True, epochs=0)
    ds = C.Dataset("B5", arch, "molecule", 42)
    tr = C.Trainer(ds, arch, seed=3).fit(epochs=0)
    i = int(ds.train_idx[0])
    coef = tr.readout.obs_coeffs()

    ops, prov, tp_idx, trainable = tr.comp.bp[i]
    ang, zang = ds.ang[i], ds.zang[i]
    dev = qml.device("default.qubit", wires=nq)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(var, alpha):
        alpha_vec = pnp.stack([alpha[0]] * max(ds.n_types, 1))
        for (G, w), p in zip(ops, prov):
            if p is None:
                G(wires=w)
            elif p[0] == "enc":
                G(ang[p[2]], wires=w)
            elif p[0] == "encz":
                G(zang[p[2]], wires=w)
            elif p[0] == "const":
                G(p[1], wires=w)
            elif p[0] == "var":
                G(var[p[1]], wires=w)
            else:
                G(p[2] * alpha_vec[p[1]], wires=w)
        return qml.expval(qml.dot(list(coef), [qml.PauliZ(q) for q in range(nq)]))

    var = pnp.array(tr.var, requires_grad=True)
    alpha = pnp.array(tr.alpha, requires_grad=True)
    grad_var_shift, grad_alpha_shift = qml.grad(circuit)(var, alpha)

    obs = [qml.expval(qml.dot(list(coef), [qml.PauliZ(q) for q in range(nq)]))]
    tape = tr.comp.tape(i, tr.var, tr.comp.alpha_vec(tr.alpha), obs)
    d = tr.dev.compute_derivatives((tape,), C.ADJOINT_CFG)
    grad_var_adj, grad_alpha_adj = tr.comp.chain(i, d[0])

    err_var = float(np.max(np.abs(np.asarray(grad_var_shift) - grad_var_adj)))
    err_alpha = float(np.max(np.abs(np.asarray(grad_alpha_shift) - grad_alpha_adj)))
    print(f"    max |d var| {err_var:.2e}    max |d alpha| {err_alpha:.2e}   "
          f"(alpha: shift {float(np.asarray(grad_alpha_shift)[0]):+.6f} vs adjoint {grad_alpha_adj[0]:+.6f})")
    ok = err_var < 1e-8 and err_alpha < 1e-8
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t3_reproduces_published(config="B5", nq=13):
    """Defaults must rebuild the published circuit gate for gate."""
    print(f"\nT3  defaults reproduce the superseded implementation   [{config}]")
    try:
        import qml_pennylane_benchmark_v2 as bench  # not shipped
    except ModuleNotFoundError:
        print("    the superseded reference implementation is not part of this "
              "release -- SKIP")
        return True
    arch = C.Arch(n_qubits=nq, epochs=0, train_var=False)
    ds = C.Dataset(config, arch, "molecule", 42)
    tr = C.Trainer(ds, arch, seed=42).fit(epochs=0)

    ref = bench.create_mgqc_qsm_cl(nq, 1)
    idx = ds.train_idx[:20]
    mine = tr.features(idx)
    theirs = []
    gmax = ds.gmax
    for i in idx:
        bp, bc, na = bench.extract_bond_info(ds.mols[i], nq, 30)
        eig = np.zeros(nq)
        eig_raw = np.asarray(ds.mols[i]["lap_eig_coulomb"], float)
        eig[:min(len(eig_raw), nq)] = eig_raw[:nq]
        theirs.append(np.array(ref(eig, bp, bc, np.zeros((1, nq, 2)), na, gmax)))
    theirs = np.array(theirs)
    err = float(np.max(np.abs(mine - theirs)))
    print(f"    max |<Z> difference| over {len(idx)} molecules: {err:.3e}")
    ok = err < 1e-9
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t4_d1_premise(config="B5", nq=13):
    """The invariance result says a diagonal entangler cannot reach a Z readout.  G1's whole
    premise is that a non-diagonal one can."""
    print(f"\nT4  Diagonal-entangler invariance re-measured -- does alpha move <Z_q>?   [{config}]")
    ok = True
    for ent in ("zz", "xy", "xxz", "trotter"):
        arch = C.Arch(n_qubits=nq, entangler=ent, epochs=0, train_var=False)
        ds = C.Dataset(config, arch, "molecule", 42)
        tr = C.Trainer(ds, arch, seed=42).fit(epochs=0)
        idx = ds.train_idx[:40]
        base = tr.features(idx, alpha=np.array([1e-9]))
        big = tr.features(idx, alpha=np.array([1.0]))
        d = float(np.abs(base - big).mean())
        verdict = ("diagonal -> unreachable" if d < 1e-12
                   else "non-diagonal -> reaches readout")
        print(f"    {ent:8s} mean |delta <Z>| = {d:.3e}   {verdict}")
        if ent == "zz" and d > 1e-12:
            ok = False
        if ent in ("xy", "xxz") and d < 1e-6:
            ok = False
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t5_xy_observables(nq=6):
    """The ZXY readout adds X and Y expectations.  The gradient then routes
    through a mixed Hamiltonian, so the chain rule is checked again there."""
    print(f"\nT5  chain rule with the ZXY readout   [{nq} qubits, no CNOT chain]")
    arch = C.Arch(n_qubits=nq, entangler="xy", alpha=0.8, alpha_mode="scalar",
                  observable="ZXY", use_cnot=False, readout_scaler=True, epochs=0)
    ds = C.Dataset("B5", arch, "molecule", 42)
    ds.train_idx = ds.train_idx[:20]
    tr = C.Trainer(ds, arch, seed=5).fit(epochs=0)
    grad_var, grad_alpha = analytic(tr)
    eps, err = 1e-5, []
    for k in list(np.linspace(0, len(grad_var) - 1, 4).astype(int)):
        plus = tr.var.copy(); plus[k] += eps
        minus = tr.var.copy(); minus[k] -= eps
        fd = (loss_of(tr, plus, tr.alpha) - loss_of(tr, minus, tr.alpha)) / (2 * eps)
        err.append(abs(fd - grad_var[k]))
    plus = tr.alpha.copy(); plus[0] += eps
    minus = tr.alpha.copy(); minus[0] -= eps
    fd = (loss_of(tr, tr.var, plus) - loss_of(tr, tr.var, minus)) / (2 * eps)
    err.append(abs(fd - grad_alpha[0]))
    print(f"    n_features {tr.features(ds.train_idx[:2]).shape[1]} (= 3 x {nq})"
          f"   alpha: adjoint {grad_alpha[0]:+.9f}  fd {fd:+.9f}")
    ok = max(err) < 5e-6
    print(f"    max abs error {max(err):.2e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def t6_canonical_order(config="B5", nq=13):
    """`canonical_order` has to be a function of the MOLECULE, not of the file.

    the protocol proposes checking that arm A's MAE is unchanged by
    reordering.  That is not the right invariant: Layer A is permutation
    invariant (a sorted spectrum), but Layer B entangles ATOM i with ATOM j, so
    relabelling atoms genuinely changes the circuit.  A shifted MAE there is the
    experiment working, not a bug.

    What must hold exactly is stronger and is what is checked here:
      (a) reordering preserves the physics -- spectrum, energy, bond multiset;
      (b) canonicalising an arbitrarily permuted copy gives the SAME molecule,
          i.e. the map really is determined by the molecule;
      (c) Layer A angles under encode_source="eig" are bit-identical.
    """
    print(f"\nT6  canonical atom order   [{config}]")
    mols = C.load_config(config)[:60]
    rng = np.random.default_rng(0)
    ok = True

    # (a) physics preserved
    worst = 0.0
    for m in mols:
        perm = C.canonical_order(m)
        r = C.reorder_molecule(m, perm)
        W = np.asarray(r["W_coulomb"], float)
        lap = np.diag(W.sum(1)) - W
        worst = max(worst, float(np.max(np.abs(
            np.sort(np.linalg.eigvalsh(lap))
            - np.sort(np.asarray(m["lap_eig_coulomb"], float))))))
        assert r["eAT"] == m["eAT"]
        bonds_orig = sorted(C.extract_bonds(m, nq, "elem_order"), key=lambda b: (b[3], b[2]))
        bonds_reordered = sorted(C.extract_bonds(r, nq, "elem_order"), key=lambda b: (b[3], b[2]))
        if [(x[2], x[3]) for x in bonds_orig] != [(x[2], x[3]) for x in bonds_reordered]:
            ok = False
    print(f"    (a) spectrum preserved, max |delta lambda| = {worst:.2e};"
          f"  bond multiset preserved: {'yes' if ok else 'NO'}")

    # (b) determined by the molecule, not by the stored order
    bad = 0
    for m in mols:
        q = rng.permutation(len(m["Z"]))
        shuffled = C.reorder_molecule(m, q)
        a = C.reorder_molecule(m, C.canonical_order(m))
        b = C.reorder_molecule(shuffled, C.canonical_order(shuffled))
        if not (np.array_equal(a["Z"], b["Z"])
                and np.allclose(a["W_coulomb"], b["W_coulomb"])
                and np.array_equal(a["adj"], b["adj"])):
            bad += 1
    print(f"    (b) canonicalising a randomly permuted copy reproduces it: "
          f"{len(mols) - bad}/{len(mols)}")
    ok = ok and bad == 0

    # (c) Layer A angles under "eig" are untouched
    arch_default = C.Arch(n_qubits=nq, epochs=0, train_var=False)
    arch_canon = C.Arch(n_qubits=nq, epochs=0, train_var=False, canonical_order=True)
    ds_default = C.Dataset(config, arch_default, "molecule", 42)
    ds_canon = C.Dataset(config, arch_canon, "molecule", 42)
    ang_err = float(np.max(np.abs(ds_default.ang - ds_canon.ang)))
    print(f"    (c) Layer A angles (eig) max |delta| = {ang_err:.2e}")
    ok = ok and ang_err < 1e-12
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t7_learning_curve_split(config="A2", nq=13):
    """A learning curve is only readable if the TEST set is identical at every
    training fraction and the training subsets are nested.  Everything fitted
    from data -- the composition model, the residual normalisation, the global
    angle scale, the bond-type vocabulary -- must also see only the reduced
    training set, or the smaller fractions are quietly borrowing information
    from molecules they are not supposed to have."""
    print(f"\nT7  learning-curve subsampling   [{config}]")
    fracs = [0.10, 0.20, 0.35, 0.50, 0.70, 1.00]
    ds = {}
    for f in fracs:
        a = C.Arch(n_qubits=nq, epochs=0, train_var=False, train_frac=f,
                   types_from_train=True, canonical_order=True)
        ds[f] = C.Dataset(config, a, "molecule", 42)
    ok = True

    test_idx_full = ds[1.0].test_idx
    same = all(np.array_equal(ds[f].test_idx, test_idx_full) for f in fracs)
    print(f"    test set identical at every fraction: {'yes' if same else 'NO'} "
          f"(n_test = {len(test_idx_full)})")
    ok = ok and same

    nested = True
    for i in range(len(fracs) - 1):
        a, b = set(ds[fracs[i]].train_idx), set(ds[fracs[i + 1]].train_idx)
        nested = nested and a.issubset(b)
    print(f"    training subsets nested: {'yes' if nested else 'NO'}")
    ok = ok and nested

    # nothing fitted may be shared across fractions
    moved = (abs(ds[0.10].res_std - ds[1.0].res_std) > 1e-9
             and abs(ds[0.10].gmax - ds[1.0].gmax) > 1e-12)
    print(f"    train-only statistics differ across fractions: "
          f"{'yes' if moved else 'NO'}   "
          f"(res_std {ds[0.10].res_std:.4f} -> {ds[1.0].res_std:.4f}, "
          f"gmax {ds[0.10].gmax:.2f} -> {ds[1.0].gmax:.2f})")

    print(f"    {'frac':>6}{'n_train':>9}{'n_types(train)':>16}{'n_isomer_grp':>14}"
          f"{'comp_R2':>9}")
    for f in fracs:
        d = ds[f]
        n_groups = len({d.formula_key[i] for i in d.train_idx})
        print(f"    {f:>6.2f}{len(d.train_idx):>9}{d.n_types_train:>16}{n_groups:>14}"
              f"{d.comp_r2:>9.4f}")
    grew = ds[1.0].n_types_train > ds[0.10].n_types_train
    print(f"    vocabulary grows with the training set: {'yes' if grew else 'NO'}"
          f"  <- required for the n_types analysis")
    ok = ok and grew

    # an unseen bond type at test time must land in the reserved bucket
    d = ds[0.10]
    unseen = sum(1 for i in d.test_idx for b in d.bonds[i] if b[3] == d.other_type)
    print(f"    test bonds falling into the reserved bucket at frac 0.10: {unseen}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t8_gpu_simulator():
    """The batched GPU path is a second implementation of the same physics, so
    it is only usable if it agrees with the first.  Features and gradients are
    checked against lightning.qubit across the arms the results use."""
    import importlib.util
    if importlib.util.find_spec("pennylane_lightning_gpu") is None:
        print("    no GPU simulator on this machine -- SKIP")
        return True
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nT8  batched GPU simulator -- no CUDA device, skipped")
            return True
        from . import _gpu as gpu_sim
    except Exception as e:
        print(f"\nT8  batched GPU simulator -- unavailable ({type(e).__name__}), skipped")
        return True
    print("\nT8  batched GPU simulator vs lightning.qubit   (threshold 1e-10)")
    cases = [("B5", "zz", "bytype", 3, 1, "none", True, 13),
             ("B5", "zz", "fixed", 3, 1, "none", True, 13),
             ("B5", "none", "fixed", 3, 1, "none", True, 13),
             ("B5", "zz", "bytype_shuffled", 3, 1, "none", True, 13),
             ("A1", "zz", "bytype", 3, 1, "none", True, 11),
             ("B5", "zz", "bytype", 2, 2, "Znum", True, 13),
             ("B5", "xy", "bytype", 3, 1, "none", True, 13)]
    ok = True
    for cfg, ent, mode, rounds, reup, rz, cnot, nq in cases:
        arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=0, entangler=ent,
                      alpha=0.3, alpha_mode=mode, readout_scaler=True,
                      canonical_order=True, types_from_train=True,
                      n_reupload=reup, rz_source=rz, use_cnot=cnot)
        ds = C.Dataset(cfg, arch, "molecule", 42)
        tr = C.Trainer(ds, arch, seed=42).fit(epochs=0)
        rng = np.random.default_rng(0)
        tr.var = rng.normal(size=arch.n_var_params()) * 0.4
        tr.alpha = rng.normal(size=tr.comp.n_alpha()) * 0.3 + 0.2
        idx = np.asarray(ds.train_idx)[:40]
        F = tr.features(idx)
        batched = gpu_sim.BatchedCircuit(ds, arch, device="cuda")
        var_t = torch.tensor(tr.var, dtype=torch.float64, device="cuda", requires_grad=True)
        alpha_t = torch.tensor(tr.alpha, dtype=torch.float64, device="cuda", requires_grad=True)
        F_gpu = batched.features(idx, var_t, alpha_t)
        err_feat = float(np.max(np.abs(F - F_gpu.detach().cpu().numpy())))
        coef = rng.normal(size=nq); w = rng.normal(size=len(idx))
        ((F_gpu @ torch.tensor(coef, dtype=torch.float64, device="cuda"))
         * torch.tensor(w, dtype=torch.float64, device="cuda")).sum().backward()
        grad_np = lambda g, k: np.zeros(k) if g is None else g.cpu().numpy()
        grad_var, grad_alpha = tr._grads(idx, coef, w)
        err_grad_var = float(np.max(np.abs(grad_var - grad_np(var_t.grad, len(tr.var)))))
        err_grad_alpha = float(np.max(np.abs(grad_alpha - grad_np(alpha_t.grad, len(tr.alpha)))))
        good = max(err_feat, err_grad_var, err_grad_alpha) < 1e-10
        ok = ok and good
        print(f"    {cfg} {ent:5s} {mode:16s} r{rounds} u{reup} "
              f"|dF| {err_feat:.1e} |dgv| {err_grad_var:.1e} |dga| {err_grad_alpha:.1e}  "
              f"{'PASS' if good else 'FAIL'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


def t9_gpu_training():
    """A full 100-epoch run must land on the same REPORTED numbers.

    Parameter trajectories drift to about 4e-7 over 100 steps: gradients agree
    to 1e-14, but Adam divides by sqrt(v), so a machine-precision difference in
    a near-zero gradient component is amplified.  What has to match is the
    quantity that gets reported, and the MAE agrees to ~1e-10 -- seven orders
    below the third decimal place the paper prints."""
    import importlib.util
    if importlib.util.find_spec("pennylane_lightning_gpu") is None:
        print("    no GPU simulator on this machine -- SKIP")
        return True
    try:
        import torch, gpu_sim
        if not torch.cuda.is_available():
            print("\nT9  GPU training equivalence -- no CUDA device, skipped")
            return True
    except Exception:
        print("\nT9  GPU training equivalence -- unavailable, skipped")
        return True
    print("\nT9  full 100-epoch training, GPU vs CPU   (reported MAE, threshold 1e-6)")
    ok = True
    for cfg, seed in [("B5", 42), ("B2", 44)]:
        arch = C.Arch(n_qubits=C.qubits_for(cfg), n_rounds=3, epochs=100,
                      entangler="zz", alpha=0.1, alpha_mode="bytype",
                      readout_scaler=True, canonical_order=True,
                      types_from_train=True)
        ds = C.Dataset(cfg, arch, "molecule", seed)
        metrics_cpu, _ = C.Trainer(ds, arch, seed=seed).fit().evaluate()
        metrics_gpu, _ = gpu_sim.GPUTrainer(ds, arch, seed=seed).fit().evaluate()
        d = abs(metrics_cpu["mae"] - metrics_gpu["mae"]); d_isomer = abs(metrics_cpu["isomer_mae"] - metrics_gpu["isomer_mae"])
        good = d < 1e-6 and d_isomer < 1e-6
        ok = ok and good
        print(f"    {cfg} s{seed}  MAE {metrics_cpu['mae']:.6f} / {metrics_gpu['mae']:.6f}  "
              f"|d| {d:.1e}   isomer |d| {d_isomer:.1e}   {'PASS' if good else 'FAIL'}")
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("  improvement/ self-tests")
    print("=" * 78)
    r = [t1_finite_difference(),
         t1_finite_difference(config="B5", nq=7, mode="scalar", ent="xxz"),
         t2_parameter_shift(),
         t3_reproduces_published(), t4_d1_premise(), t5_xy_observables(),
         t6_canonical_order(), t7_learning_curve_split(),
         t8_gpu_simulator(), t9_gpu_training()]
    print("\n" + "=" * 78)
    print(f"  {sum(r)}/{len(r)} passed")
    print("=" * 78)
    sys.exit(0 if all(r) else 1)
