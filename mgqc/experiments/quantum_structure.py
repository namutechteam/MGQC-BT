"""What is the quantum structure doing?

entangler     Alternative entangling generators (XY, XXZ, Trotterized
              transverse field) in place of ZZ.
entanglement  Concurrence and mutual information of the trained circuit, and
              how much of the signal survives when it is removed.
angles        The distribution of the coupling angle actually applied to each
              bond, i.e. how far the circuit sits inside the perturbative
              regime.
angle_scan    The coupling angle scaled out of that regime; the response
              exponent (`probe`) measures how far from linear the circuit is.
optimizer     Adjoint-gradient Adam against the simultaneous-perturbation loop,
              with controls at zero variational weights.
"""
import argparse, os, time
import numpy as np
import pennylane as qml

from .. import core as C


# ============================================================================
# entangler
# ============================================================================
ENTANGLERS = ("zz", "xy", "xxz", "trotter")


def diagnose(config, nq, alphas, split, seed, scaler=True, sources=("eig",)):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    rows = []
    cache = {}
    for src in sources:
      base = None
      for ent in ENTANGLERS:
        for al in alphas:
            arch = C.Arch(n_qubits=nq, entangler=ent, alpha=al, epochs=0,
                          train_var=False, readout_scaler=scaler,
                          encode_source=src)
            key = (config, split, seed, nq, src, arch.rz_source, 0, 0.0,
                   float(np.pi), "elem_order")
            ds = cache.get(key) or C.Dataset(config, arch, split, seed)
            cache[key] = ds
            ds.arch = arch
            tr = C.Trainer(ds, arch, seed=seed).fit(epochs=0)

            F = tr.features(np.arange(len(ds.mols)))
            if base is None:
                base = F.copy()          # zz @ smallest alpha == the published circuit
            delta = float(np.abs(F - base).mean())
            spread = float(F.std())

            feature_scaler = StandardScaler().fit(F[ds.train_idx])
            ridge = Ridge(alpha=1.0).fit(feature_scaler.transform(F[ds.train_idx]),
                                         ds.y_target[ds.train_idx])
            pred_z = ridge.predict(feature_scaler.transform(F[ds.test_idx]))
            r2 = r2_score(ds.y_target[ds.test_idx], pred_z)
            pred_eV = ds.y_comp[ds.test_idx] + (pred_z * ds.res_std + ds.res_mean)
            mae = mean_absolute_error(ds.y_raw[ds.test_idx], pred_eV)
            max_corr = float(np.nanmax([abs(np.corrcoef(F[:, q], ds.resid)[0, 1])
                                        for q in range(nq)]))
            max_angle = ds.angle_max(np.full(max(ds.n_types, 1), al))
            rows.append(dict(config=config, n_qubits=nq, encode_source=src,
                             entangler=ent, alpha=al,
                             delta_vs_base=delta, spread=spread,
                             max_abs_corr_resid=max_corr, ridge_r2_resid=float(r2),
                             mae_eV=float(mae), angle_max=max_angle,
                             wrap=int(max_angle > np.pi), seed=seed, split=split,
                             scaler=int(scaler)))
            print(f"  {src:5s} {ent:9s} a={al:<5} delta {delta:9.2e}  spread {spread:7.4f}"
                  f"  r_res {max_corr:6.3f}  ridge_R2 {r2:+7.3f}  MAE {mae:6.3f}"
                  f"  ang_max {max_angle:5.2f}{'  WRAP' if max_angle > np.pi else ''}",
                  flush=True)
    return rows


def main_arms(nq, alphas, epochs, scaler, sources):
    arms = []
    for src in sources:
        arms.append((f"{src}_off", C.Arch(n_qubits=nq, entangler="none", alpha=0.0,
                                          epochs=epochs, readout_scaler=scaler,
                                          encode_source=src)))
        for ent in ENTANGLERS:
            for al in alphas:
                arms.append((f"{src}_{ent}_a{al}",
                             C.Arch(n_qubits=nq, entangler=ent, alpha=al,
                                    epochs=epochs, readout_scaler=scaler,
                                    encode_source=src)))
    return arms


def entangler_job(config, seed, nq, alphas, epochs, split, scaler, sources):
    cache, rows = {}, []
    for name, arch in main_arms(nq, alphas, epochs, scaler, sources):
        row, _, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                              ds_cache=cache, extra=dict(arm=name))
        rows.append(row)
        print(f"    {config} s{seed} {name:14s} MAE {row['mae']:.3f}  "
              f"iso {row['isomer_mae']:.3f}  rho {row['isomer_rho']:+.3f}  "
              f"({row['time_s']:.0f}s)", flush=True)
    return rows


def report(rows, configs):
    maes_by_arm = {}
    for r in rows:
        maes_by_arm.setdefault((r["config"], r["arm"]), []).append(r["mae"])
    print("\n" + "=" * 78)
    print("  G1 -- entangler arms vs the alpha = 0 control (condition 2)")
    print("=" * 78)
    print(f"  {'config':<8}{'arm':<16}{'MAE mean+-std':>20}{'vs off':>10}"
          f"{'wins':>7}{'p':>9}")
    for c in configs:
      for src in ("eig", "diag"):
        maes_off = maes_by_arm.get((c, f"{src}_off"), [])
        if not maes_off:
            continue
        for k in sorted(maes_by_arm):
            if k[0] != c or not k[1].startswith(src + "_") or k[1].endswith("_off"):
                continue
            arm_maes = maes_by_arm[k]
            test = C.paired_test(arm_maes, maes_off)
            flag = " *" if (test["p"] == test["p"] and test["p"] < 0.05 and test["rel"] < 0) else ""
            print(f"  {c:<8}{k[1]:<16}{np.mean(arm_maes):>12.3f} +-{np.std(arm_maes):<6.3f}"
                  f"{test['rel']:>+9.1%}{test['wins']:>5}/{test['n']}{test['p']:>9.4f}{flag}")
        print(f"  {c:<8}{src + '_off (a=0)':<16}{np.mean(maes_off):>12.3f} "
              f"+-{np.std(maes_off):<6.3f}")


def entangler_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="diag", choices=["diag", "main", "both"])
    p.add_argument("--configs", default="B5")
    p.add_argument("--alphas", default="0.1,0.5,1.0")
    p.add_argument("--sources", default="eig,diag",
                   help="Layer A encoding source(s): eig (published) and/or diag")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-qubits", type=int, default=13)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--no-scaler", action="store_true",
                   help="use the published (unstandardised) readout")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    alphas = [float(x) for x in a.alphas.split(",")]
    scaler = not a.no_scaler

    if a.stage in ("diag", "both"):
        print("=" * 78)
        print(f"  G1 stage 1 -- untrained entangler diagnosis   {configs}   "
              f"scaler={int(scaler)}")
        print("=" * 78)
        rows = []
        for c in configs:
            print(f"\n  [{c}]")
            rows += diagnose(c, a.n_qubits, alphas, a.split, a.seed0, scaler,
                             tuple(a.sources.split(",")))
        C.write_csv(os.path.join(a.out, "G1_entangler_diag.csv"), rows)
        print("\n  Decision rule (the pre-registered rule):")
        for c in configs:
          for src in a.sources.split(","):
            for ent in ENTANGLERS:
                cell_rows = [r for r in rows if r["config"] == c and r["entangler"] == ent
                      and r["encode_source"] == src]
                cell_rows.sort(key=lambda r: r["alpha"])
                d = np.mean([r["delta_vs_base"] for r in cell_rows])
                r2 = [r["ridge_r2_resid"] for r in cell_rows]
                if d < 1e-12:
                    verdict = "delta = 0  -> still diagonal, unreachable"
                elif r2[-1] > r2[0]:
                    verdict = "ridge_R2 rises with alpha  -> PROCEED to the main run"
                else:
                    verdict = "ridge_R2 falls with alpha  -> information is uncorrelated; fix G3 first"
                print(f"    {c} {src:5s} {ent:9s} {verdict}")

    if a.stage in ("main", "both"):
        seeds = list(range(a.seed0, a.seed0 + a.seeds))
        args = [(c, s, a.n_qubits, alphas, a.epochs, a.split, scaler,
                 tuple(a.sources.split(","))) for c in configs for s in seeds]
        print("\n" + "=" * 78)
        print(f"  G1 stage 2 -- trained   {configs}   seeds {seeds}   "
              f"epochs {a.epochs}")
        print("=" * 78)
        rows = C.parallel_jobs(entangler_job, args, a.jobs, desc="G1")
        C.write_csv(os.path.join(a.out, "G1_entangler.csv"), rows,
                    C.COMMON_COLUMNS + ["arm"])
        report(rows, configs)


# ============================================================================
# entanglement
# ============================================================================
def two_qubit_rdm(psi, n, i, j):
    """Reduced density matrix of qubits (i, j).  PennyLane orders the
    statevector with wire 0 as the most significant bit, which is the same
    convention as reshaping into [2]*n and indexing axis k for wire k."""
    t = np.asarray(psi).reshape([2] * n)
    rest = [k for k in range(n) if k not in (i, j)]
    t = np.transpose(t, [i, j] + rest).reshape(4, -1)
    return t @ t.conj().T


def vn_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-(eigvals * np.log2(eigvals)).sum()) if len(eigvals) else 0.0


def concurrence(rho):
    """Wootters concurrence of a two-qubit state, 0 (separable) to 1."""
    Y = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(Y, Y)
    R = rho @ YY @ rho.conj() @ YY
    # eigenvalues of R are real and non-negative up to numerical noise
    eigvals = np.sort(np.real(np.linalg.eigvals(R)))[::-1]
    eigvals = np.sqrt(np.clip(eigvals, 0.0, None))
    return float(max(0.0, eigvals[0] - eigvals[1] - eigvals[2] - eigvals[3]))


def single_qubit_entropy(psi, n, q):
    t = np.asarray(psi).reshape([2] * n)
    rest = [k for k in range(n) if k != q]
    t = np.transpose(t, [q] + rest).reshape(2, -1)
    return vn_entropy(t @ t.conj().T)


def entanglement_features(psi, n, bonds, max_bonds=30):
    """Fixed-length feature vector.  Absent bonds are padded with zeros, which
    is the right filler here: no bond means no entanglement on that slot."""
    ent = np.array([single_qubit_entropy(psi, n, q) for q in range(n)])
    concurrence_per_bond = np.zeros(max_bonds)
    mutual_info_per_bond = np.zeros(max_bonds)
    for k, b in enumerate(bonds[:max_bonds]):
        i, j = int(b[0]), int(b[1])
        rij = two_qubit_rdm(psi, n, i, j)
        concurrence_per_bond[k] = concurrence(rij)
        rij_tensor = rij.reshape(2, 2, 2, 2)
        ri = np.trace(rij_tensor, axis1=1, axis2=3)
        rj = np.trace(rij_tensor, axis1=0, axis2=2)
        mutual_info_per_bond[k] = vn_entropy(ri) + vn_entropy(rj) - vn_entropy(rij)
    n_bonds = max(len(bonds), 1)
    scalars = np.array([ent.mean(), ent.max(), ent.sum(),
                        concurrence_per_bond.sum() / n_bonds, concurrence_per_bond.max(), concurrence_per_bond.sum(),
                        mutual_info_per_bond.sum() / n_bonds, mutual_info_per_bond.max(), mutual_info_per_bond.sum()])
    return np.concatenate([ent, concurrence_per_bond, mutual_info_per_bond, scalars]), scalars, dict(
        mean_entropy=float(ent.mean()), total_entropy=float(ent.sum()),
        max_concurrence=float(concurrence_per_bond.max()), mean_concurrence=float(concurrence_per_bond.sum() / n_bonds),
        max_mutual_info=float(mutual_info_per_bond.max()), mean_mutual_info=float(mutual_info_per_bond.sum() / n_bonds))


def state_features(tr, idx, max_bonds=30):
    """One statevector per molecule from the trained circuit."""
    dev = qml.device("default.qubit", wires=tr.arch.n_qubits)
    alpha_eff = tr.comp.alpha_vec(tr.alpha)
    feats, aggs, scalars = [], [], []
    for i in idx:
        tape = tr.comp.tape(i, tr.var, alpha_eff, [qml.state()])
        psi = np.asarray(dev.execute((tape,))[0]).ravel()
        f, agg, s = entanglement_features(psi, tr.arch.n_qubits,
                                          tr.ds.bonds[i], max_bonds)
        feats.append(f); aggs.append(agg); scalars.append(s)
    return np.array(feats), np.array(aggs), scalars


def entanglement_selftest(nq=6):
    """S1 product state -> 0, S2 Bell state -> 1, S3 unit-trace RDMs."""
    out, ok = [], True
    rng = np.random.default_rng(0)
    ang = rng.random(nq) * np.pi
    dev = qml.device("default.qubit", wires=nq)

    ops = [qml.RY(float(a), wires=q) for q, a in enumerate(ang)]   # product state
    psi = np.asarray(dev.execute((qml.tape.QuantumScript(ops, [qml.state()]),))[0]).ravel()
    concurrences = [concurrence(two_qubit_rdm(psi, nq, i, j))
                    for i in range(nq) for j in range(i + 1, nq)]
    s1 = max(concurrences)
    # Concurrence is defined through a SQUARE ROOT of eigenvalues, so a state
    # that is a product to machine precision (amplitude error ~1e-16) shows a
    # concurrence of order sqrt(1e-16) = 1e-8.  the 1e-12 threshold
    # is not numerically reachable for this quantity; 1e-7 is.
    out.append(f"  S1 product state (RY only)      max concurrence {s1:.3e}"
               f"   {'PASS' if s1 < 1e-7 else 'FAIL'}   (sqrt of eigenvalues:"
               f" 1e-8 is the floor)")
    ok = ok and s1 < 1e-7

    ops = ([qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])]
           + [qml.Identity(wires=q) for q in range(2, nq)])
    psi = np.asarray(dev.execute((qml.tape.QuantumScript(ops, [qml.state()]),))[0]).ravel()
    c = concurrence(two_qubit_rdm(psi, nq, 0, 1))
    pair_entropy = vn_entropy(two_qubit_rdm(psi, nq, 0, 1))
    out.append(f"  S2 Bell state on (0,1)          concurrence {c:.6f}"
               f"   entropy of the pair {pair_entropy:.3e}   {'PASS' if abs(c - 1) < 1e-9 else 'FAIL'}")
    ok = ok and abs(c - 1) < 1e-9

    trace_err = abs(np.trace(two_qubit_rdm(psi, nq, 0, 2)) - 1.0)
    out.append(f"  S3 reduced density matrix trace  |tr - 1| = {trace_err:.3e}"
               f"   {'PASS' if trace_err < 1e-10 else 'FAIL'}")
    ok = ok and trace_err < 1e-10
    return ok, out


ENTANGLEMENT_ARMS = [
    ("E1_zz_trained",  dict(entangler="zz", alpha=0.1, alpha_mode="bytype")),
    ("E0_alpha0",      dict(entangler="none", alpha=0.0, alpha_mode="fixed")),
    ("E2_xy_trained",  dict(entangler="xy", alpha=0.1, alpha_mode="bytype")),
    ("E3_untrained_zz", dict(entangler="zz", alpha=0.1, alpha_mode="fixed",
                             train_var=False)),
]


def entanglement_job(config, seed, split, epochs, rounds):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    nq = C.qubits_for(config)
    cache, rows, feats = {}, [], {}
    for name, kw in ENTANGLEMENT_ARMS:
        arm_epochs = 0 if kw.get("train_var") is False else epochs
        arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=arm_epochs,
                      readout_scaler=True, canonical_order=True,
                      types_from_train=True, **kw)
        row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                               ds_cache=cache, extra=dict(arm=name))
        ds = tr.ds
        F, A, scalars = state_features(tr, np.arange(len(ds.mols)))
        feats[name] = F

        trn, te = ds.train_idx, ds.test_idx

        def ridge_r2(X):
            """Held-out R^2 of a ridge on the given entanglement features.

            Columns that are constant on the TRAINING split are dropped first.
            The per-bond block is padded to a fixed 30 slots, so a slot only
            some molecules reach is constant in train and non-constant in test;
            standardising such a column divides by a scale of 1 and hands the
            ridge an arbitrarily large input, which is what produced R^2 of
            -7 to -122 before this guard."""
            keep = X[trn].std(0) > 1e-6 * max(X[trn].std(0).max(), 1e-12)
            if keep.sum() == 0:
                return np.nan, np.nan
            X_kept = X[:, keep]
            scaler = StandardScaler().fit(X_kept[trn])
            Xtr, Xte = scaler.transform(X_kept[trn]), scaler.transform(X_kept[te])
            from sklearn.linear_model import Ridge
            r2_ridge1 = float(r2_score(ds.y_target[te],
                                       Ridge(alpha=1.0).fit(Xtr, ds.y_target[trn]).predict(Xte)))
            r2_ridge_cv = float(r2_score(ds.y_target[te],
                                         RidgeCV(alphas=np.logspace(-3, 3, 13))
                                         .fit(Xtr, ds.y_target[trn]).predict(Xte)))
            return r2_ridge1, r2_ridge_cv

        r2fix, r2cv = ridge_r2(F)
        # The number the entanglement R^2 has to be read against: what the
        # circuit's OWN readout features (<Z_q>) explain on the same split.
        # r2_ent alone is not evidence that entanglement carries signal -- the
        # entanglement is a function of the same encoding angles that already
        # predict the target.  What matters is whether it carries MORE than the
        # readout already reaches.
        Z_features = tr.features(np.arange(len(ds.mols)))
        r2_readout, _ = ridge_r2(Z_features)
        # the scalar block is permutation invariant and needs no padding, so it
        # is the cleaner test of "do the entanglement magnitudes predict?"
        r2agg, r2agg_cv = ridge_r2(A)

        from scipy import stats
        rho = 0.0
        for k in range(F.shape[1]):
            if F[trn, k].std() > 1e-10:
                rho_k = stats.spearmanr(F[trn, k], ds.resid[trn]).statistic
                if np.isfinite(rho_k):
                    rho = max(rho, abs(rho_k))
        agg = {k: float(np.mean([s[k] for s in scalars])) for k in scalars[0]}
        r2 = max(r2fix, r2agg)
        rows.append(dict(config=config, seed=seed, arm=name, mae=row["mae"],
                         n_train=len(trn), n_test=len(te),
                         r2_ent=r2, r2_ent_ridge1=r2fix, r2_ent_cv=r2cv,
                         r2_ent_agg=r2agg, r2_ent_agg_cv=r2agg_cv,
                         r2_readout=r2_readout,
                         max_abs_spearman=float(rho), **agg))
        print(f"    {config} s{seed} {name:16s} MAE {row['mae']:.3f}  "
              f"r2(all) {r2fix:+.3f} r2(agg) {r2agg:+.3f} "
              f"r2(readout) {r2_readout:+.3f} |rho| {rho:.3f}  "
              f"maxC {agg['max_concurrence']:.3e}",
              flush=True)

    for r in rows:
        base = feats["E0_alpha0"]
        r["delta_ent"] = float(np.abs(feats[r["arm"]] - base).mean())
        r["r2_ent_off"] = [x["r2_ent"] for x in rows if x["arm"] == "E0_alpha0"][0]
    return rows


def entanglement_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3,A1")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.split, a.epochs, a.rounds) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  I1 -- entanglement quantified, and regressed on the target")
    print(f"  configs {configs}   seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(entanglement_job, args, a.jobs, desc="I1")
    C.write_csv(os.path.join(a.out, "I1_entanglement_signal.csv"), rows)

    # ---- self-test ----
    ok, lines = entanglement_selftest()
    txt = ["I1 self-test -- the entanglement measurement code", "",
           "An earlier protocol proposed checking that the untrained circuit with a",
           "diagonal entangler has zero concurrence.  That premise is wrong:",
           "IsingZZ is diagonal but still entangling (so is CZ), and Layer C's",
           "CNOT chain is present even at var_weights = 0.  the diagonal-entangler invariance constrains the",
           "OBSERVABLE, not the state.  The tests below are the ones that must",
           "hold.", ""] + lines
    e3 = [r for r in rows if r["arm"] == "E3_untrained_zz"]
    if e3:
        txt += ["",
                f"  reference: untrained zz circuit, max concurrence "
                f"{max(r['max_concurrence'] for r in e3):.3e}",
                "  (real entanglement from IsingZZ and the CNOT chain, not a bug)"]
    txt += ["", f"  -> {'PASS' if ok else 'FAIL -- the measurement code is wrong'}"]
    print("\n" + "\n".join(txt))
    with open(os.path.join(a.out, "I1_selftest.txt"), "w") as fh:
        fh.write("\n".join(txt) + "\n")
    if not ok:
        print("\n  Everything below is meaningless until this passes.")

    rows_by_arm = {}
    for r in rows:
        rows_by_arm.setdefault((r["config"], r["arm"]), []).append(r)
    print("\n" + "=" * 78)
    print("  I1 -- does Layer B change the entanglement, and does it predict?")
    print("=" * 78)
    print(f"  {'config':<8}{'arm':<17}{'MAE':>8}{'delta_ent':>11}{'r2_ent':>9}"
          f"{'r2 agg':>9}{'r2 readout':>12}{'max C':>11}{'|rho|':>8}")
    for c in configs:
        for name, _ in ENTANGLEMENT_ARMS:
            cell_rows = rows_by_arm.get((c, name), [])
            if not cell_rows:
                continue
            arm_mean = lambda k: np.mean([r[k] for r in cell_rows])
            print(f"  {c:<8}{name:<17}{arm_mean('mae'):>8.3f}{arm_mean('delta_ent'):>11.2e}"
                  f"{arm_mean('r2_ent_ridge1'):>9.3f}{arm_mean('r2_ent_agg'):>9.3f}"
                  f"{arm_mean('r2_readout'):>12.3f}{arm_mean('max_concurrence'):>11.2e}"
                  f"{arm_mean('max_abs_spearman'):>8.3f}")
        print()

    print("=" * 78)
    print("  Verdict (the pre-registered rule)")
    print("=" * 78)
    for c in configs:
        e1 = rows_by_arm.get((c, "E1_zz_trained"), []); e0 = rows_by_arm.get((c, "E0_alpha0"), [])
        e2 = rows_by_arm.get((c, "E2_xy_trained"), [])
        if not (e1 and e0):
            continue
        d = np.mean([r["delta_ent"] for r in e1])
        r1 = np.mean([r["r2_ent_agg"] for r in e1])
        r0 = np.mean([r["r2_ent_agg"] for r in e0])
        r2x = np.mean([r["r2_ent_agg"] for r in e2]) if e2 else np.nan
        r2_readout = np.mean([r["r2_readout"] for r in e1])
        if d < 1e-12:
            verdict = "Layer B does not change the entanglement at all"
        elif r1 <= 0.02:
            verdict = "entanglement is UNCORRELATED with the target -> section 4-a"
        elif abs(r1 - r0) < 0.02:
            verdict = "entanglement present, but its molecule-dependent part adds nothing"
        elif r1 > r2_readout:
            verdict = ("entanglement carries MORE than the readout reaches "
                       "-> experiment beta")
        else:
            verdict = ("entanglement carries real but LESS information than the "
                       "readout already extracts -> no headroom in the observable")
        print(f"  {c:<6} delta_ent {d:.2e}   r2_ent(zz) {r1:+.3f}   "
              f"r2_ent(off) {r0:+.3f}   r2_ent(xy) {r2x:+.3f}   "
              f"r2_readout {r2_readout:+.3f}\n         -> {verdict}")


# ============================================================================
# angles
# ============================================================================
ALPHA_FILES = ["E1_alpha_learned_tft.csv", "E1_alpha_learned_A2tft.csv"]


ARM = "zz_bytype_r3"


N_ROUNDS = 3


ALPHA_INIT = 0.1          # the fallback slot keeps its initial value


LINEAR_1PCT = 2 * np.sqrt(0.06)    # 0.4899 rad -- |sin(x)-x|/x < 1% for x=theta/2


LINEAR_5PCT = 2 * np.sqrt(0.30)    # 1.0954 rad -- same bound at 5%


def load_alpha(results):
    """(config, seed) -> {type_id: (alpha, label)}"""
    import csv
    out = {}
    for fn in ALPHA_FILES:
        p = os.path.join(results, fn)
        if not os.path.exists(p):
            print("  (missing, skipped)", fn); continue
        for r in csv.DictReader(open(p)):
            if r.get("arm") != ARM:
                continue
            k = (r["config"], int(r["seed"]))
            out.setdefault(k, {})[int(r["type_id"])] = (
                float(r["alpha_learned"]), r["type"])
    return out


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def angles_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="molecule")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-bond-out", default=None)
    a = ap.parse_args(argv)

    alphas = load_alpha(C.RESULTS)
    keys = sorted(alphas, key=lambda k: (k[0], k[1]))
    print(f"{len(keys)} (config, seed) pairs with learned alpha for arm {ARM}")

    rows, bond_rows, ds_cache = [], [], {}
    for (config, seed) in keys:
        alpha_map = alphas[(config, seed)]
        nq = C.qubits_for(config)
        if (config, seed) not in ds_cache:
            arch = C.Arch(n_qubits=nq, epochs=0, train_var=False,
                          canonical_order=True, types_from_train=True)
            ds_cache[(config, seed)] = C.Dataset(config, arch, a.split, seed)
        ds = ds_cache[(config, seed)]

        by_type, n_fallback = {}, 0
        for mol_bonds in ds.bonds:
            for (i, j, J, t, _ts) in mol_bonds:
                if t in alpha_map:
                    alpha_t, label = alpha_map[t]
                else:
                    alpha_t, label, n_fallback = ALPHA_INIT, "OTHER", n_fallback + 1
                by_type.setdefault((t, label), []).append(J * alpha_t)

        all_theta = []
        for (t, label), th in sorted(by_type.items()):
            th = np.asarray(th, float)
            theta_gate = th / N_ROUNDS
            all_theta.append(th)
            rows.append(dict(
                config=config, seed=seed, type_id=t, type=label, n_bonds=len(th),
                alpha=alpha_map.get(t, (ALPHA_INIT, label))[0],
                J_median=q(np.abs(th) / abs(alpha_map.get(t, (ALPHA_INIT, label))[0] or 1), 50),
                theta_total_median=q(np.abs(th), 50),
                theta_total_q25=q(np.abs(th), 25), theta_total_q75=q(np.abs(th), 75),
                theta_gate_median=q(np.abs(theta_gate), 50),
                theta_gate_q25=q(np.abs(theta_gate), 25), theta_gate_q75=q(np.abs(theta_gate), 75),
                theta_gate_max=float(np.abs(theta_gate).max()),
                frac_gate_gt_pi=float((np.abs(theta_gate) > np.pi).mean()),
                frac_gate_gt_2pi=float((np.abs(theta_gate) > 2 * np.pi).mean()),
                frac_total_gt_pi=float((np.abs(th) > np.pi).mean()),
                frac_total_gt_2pi=float((np.abs(th) > 2 * np.pi).mean()),
                frac_gate_linear_1pct=float((np.abs(theta_gate) < LINEAR_1PCT).mean()),
                frac_gate_linear_5pct=float((np.abs(theta_gate) < LINEAR_5PCT).mean()),
                n_fallback_bonds=n_fallback))
        theta_total_abs = np.abs(np.concatenate(all_theta)); theta_gate_abs = theta_total_abs / N_ROUNDS
        bond_rows.append(dict(
            config=config, seed=seed, n_bonds=len(theta_total_abs), n_types=ds.n_types,
            n_fallback_bonds=n_fallback,
            theta_total_median=q(theta_total_abs, 50), theta_total_q25=q(theta_total_abs, 25),
            theta_total_q75=q(theta_total_abs, 75), theta_total_q95=q(theta_total_abs, 95),
            theta_total_max=float(theta_total_abs.max()),
            theta_gate_median=q(theta_gate_abs, 50), theta_gate_q95=q(theta_gate_abs, 95),
            theta_gate_max=float(theta_gate_abs.max()),
            frac_gate_gt_pi=float((theta_gate_abs > np.pi).mean()),
            frac_gate_gt_2pi=float((theta_gate_abs > 2 * np.pi).mean()),
            frac_total_gt_pi=float((theta_total_abs > np.pi).mean()),
            frac_total_gt_2pi=float((theta_total_abs > 2 * np.pi).mean()),
            frac_gate_linear_1pct=float((theta_gate_abs < LINEAR_1PCT).mean()),
            frac_gate_linear_5pct=float((theta_gate_abs < LINEAR_5PCT).mean()),
            frac_total_linear_1pct=float((theta_total_abs < LINEAR_1PCT).mean()),
            frac_total_linear_5pct=float((theta_total_abs < LINEAR_5PCT).mean())))
        print(f"  {config} s{seed}: {len(theta_total_abs):6d} bonds  "
              f"theta_gate med {q(theta_gate_abs,50):.4f}  max {theta_gate_abs.max():.4f}  "
              f"wrap(>pi) {(theta_gate_abs>np.pi).mean():.4f}  "
              f"linear<1% {(theta_gate_abs<LINEAR_1PCT).mean():.4f}  fallback {n_fallback}")

    out = a.out or os.path.join(C.RESULTS, "R5_theta_dist.csv")
    out2 = a.per_bond_out or os.path.join(C.RESULTS, "R5_theta_summary.csv")
    C.write_csv(out, rows); C.write_csv(out2, bond_rows)

    # ---- what goes in the Supplementary column: pooled over seeds, per type
    from collections import defaultdict
    pool = defaultdict(list)
    for r in rows:
        pool[(r["config"], r["type"])].append(r)
    print("\npooled per (config, type), median over seeds of the per-type median:")
    print(f"  {'config':7s} {'type':8s} {'n_bond':>7s} {'alpha':>8s} "
          f"{'th_gate':>8s} {'th_tot':>8s} {'>pi':>7s} {'lin1%':>7s}")
    for (cfg, label), cell_rows in sorted(pool.items()):
        print(f"  {cfg:7s} {label:8s} {int(np.median([x['n_bonds'] for x in cell_rows])):7d} "
              f"{np.median([x['alpha'] for x in cell_rows]):8.3f} "
              f"{np.median([x['theta_gate_median'] for x in cell_rows]):8.4f} "
              f"{np.median([x['theta_total_median'] for x in cell_rows]):8.4f} "
              f"{np.median([x['frac_gate_gt_pi'] for x in cell_rows]):7.4f} "
              f"{np.median([x['frac_gate_linear_1pct'] for x in cell_rows]):7.4f}")
    print("\nwrote", out, "and", out2)


# ============================================================================
# angle_scan
# ============================================================================
def theta_stats(ds, arch, alpha_eff):
    """The angles actually applied, per gate, at this theta_scale."""
    gate_angles = []
    for mol_bonds in ds.bonds:
        for (_i, _j, J, t, _ts) in mol_bonds:
            gate_angles.append(J * float(alpha_eff[t % len(alpha_eff)]))
    if not gate_angles:
        return {}
    gate_angles = np.abs(np.asarray(gate_angles)) * arch.theta_scale / max(arch.n_rounds, 1)
    return dict(theta_gate_median=float(np.median(gate_angles)),
                theta_gate_q95=float(np.percentile(gate_angles, 95)),
                theta_gate_max=float(gate_angles.max()),
                frac_gate_gt_pi=float((gate_angles > np.pi).mean()),
                frac_gate_gt_2pi=float((gate_angles > 2 * np.pi).mean()),
                frac_gate_linear_1pct=float((gate_angles < LINEAR_1PCT).mean()))


def angle_scan_job(config, seed, scale, norm, epochs, rounds, split, canonical, types_from_train):
    nq = C.qubits_for(config)
    arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs,
                  entangler="zz", alpha=0.1, alpha_mode="bytype",
                  readout_scaler=True, canonical_order=canonical,
                  types_from_train=types_from_train,
                  theta_scale=float(scale), alpha_norm=norm)
    t0 = time.time()
    row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                           readouts=("ridge", "krr"),
                           extra=dict(arm=f"zz_bytype_r{rounds}", kind="quantum"))
    ds = tr.ds
    a_raw = np.asarray(tr.alpha, float)
    a_eff = tr.comp.alpha_vec(tr.alpha)
    a0 = np.asarray(tr.alpha0, float)
    alpha_cos = float(a_raw @ a0 / (np.linalg.norm(a_raw) * np.linalg.norm(a0) + 1e-12))
    row.update(theta_stats(ds, arch, a_eff))
    row.update(probe(tr, ds.test_idx))
    row.update(alpha_rms_init=float(np.sqrt(np.mean(a0 ** 2))),
               alpha_rms_final=float(np.sqrt(np.mean(a_raw ** 2))),
               alpha_cos_init=alpha_cos, scan_time_s=round(time.time() - t0, 1))
    print(f"    {config} s{seed} scale={scale:<6g} {norm:<4s} "
          f"MAE {row['mae']:.3f}  th_med {row.get('theta_gate_median', float('nan')):.3f} "
          f"exp {row.get('response_exponent', float('nan')):.2f} "
          f"(asy {row.get('exponent_asymptotic', float('nan')):.2f})  "
          f"D {row.get('perturbative_strength', float('nan')):.3f}  "
          f"drop {row.get('loss_drop', float('nan')):.3f}  "
          f"({row['time_s']:.0f}s)", flush=True)
    return [row]


def angle_scan_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--scales", default="1,2,4,8,16,32")
    p.add_argument("--alpha-norm", default="free,rms")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--no-canonical", action="store_true")
    p.add_argument("--no-tft", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    scales = [float(x) for x in a.scales.split(",")]
    norms = a.alpha_norm.split(",")
    args = [(c, s, scale, nm, a.epochs, a.rounds, a.split,
             not a.no_canonical, not a.no_tft)
            for c in a.configs.split(",") for s in seeds
            for scale in scales for nm in norms]
    print("=" * 78)
    print(f"  R7 -- theta scale scan   configs {a.configs}  seeds {len(seeds)}")
    print(f"  scales {scales}   alpha_norm {norms}   {len(args)} runs")
    print("=" * 78)
    rows = C.parallel_jobs(angle_scan_job, args, a.jobs, desc="R7")
    out = os.path.join(a.out, f"R7_theta_scan{a.tag}.csv")
    C.write_csv(out, rows)
    print("\nwrote", out, len(rows), "rows")


# ============================================================================
# _nonlinearity
# ============================================================================
def response_exponent(feats_fn, s=1.0, ratio=2.0):
    """Local power-law exponent of |f(s) - f(0)| in s.

    feats_fn(mult) -> array : readout features with the coupling scaled by
    `mult` (mult = 1 is the circuit as trained).
    """
    f0 = np.asarray(feats_fn(0.0), float)
    d_lo = np.linalg.norm(np.asarray(feats_fn(s / ratio), float) - f0)
    d_hi = np.linalg.norm(np.asarray(feats_fn(s), float) - f0)
    if d_lo <= 0 or d_hi <= 0:
        return float("nan")
    return float(np.log(d_hi / (d_lo + 1e-300)) / np.log(ratio))


def perturbative_strength(feats_fn, s=1.0):
    """How large the coupling's effect is, relative to the uncoupled output."""
    f0 = np.asarray(feats_fn(0.0), float)
    return float(np.linalg.norm(np.asarray(feats_fn(s), float) - f0)
                 / (np.linalg.norm(f0) + 1e-12))


def probe(trainer, idx, ratio=2.0):
    """Both diagnostics for one trained circuit, at its own operating point.

    `idx` should be the test split -- the probe must not read training labels,
    and it reads no labels at all, only <Z>.
    """
    base = trainer.comp.alpha_vec(trainer.alpha)      # what the circuit uses

    def feats_fn(mult):
        return trainer.features_vec(idx, base * float(mult))

    return dict(response_exponent=response_exponent(feats_fn, 1.0, ratio),
                exponent_asymptotic=response_exponent(feats_fn, 0.01, ratio),
                perturbative_strength=perturbative_strength(feats_fn, 1.0),
                perturbative_strength_lo=perturbative_strength(feats_fn, 0.5),
                exponent_ratio=ratio)


def nonlinearity_selftest():
    """The four asymptotic orders the docstring tabulates, on a 5-qubit toy.

    Measured at s = 0.01, inside the perturbative regime, where the structural
    order is exposed.  At the operating point s = 1 the same circuits read
    lower (2.000 and 0.789) because the response has already begun to bend --
    which is the effect R7 exists to detect, not an error.
    """
    import pennylane as qml
    NQ, EDGES = 5, [(0, 1), (1, 2), (0, 3), (0, 4)]
    ENC = np.array([0.7, 1.3, 2.1, 0.4, 1.8])
    dev = qml.device("default.qubit", wires=NQ)
    rng = np.random.default_rng(7)
    var = rng.normal(size=(NQ, 2)) * 0.4

    def make(rounds, with_rz):
        @qml.qnode(dev)
        def f(mult):
            for q in range(NQ):
                qml.RY(ENC[q], wires=q)
            for _ in range(rounds):
                for (i, j) in EDGES:
                    qml.IsingZZ(0.05 * mult, wires=[i, j])
                for q in range(NQ):
                    qml.RY(var[q, 0], wires=q)
                    if with_rz:
                        qml.RZ(var[q, 1], wires=q)
            return [qml.expval(qml.PauliZ(q)) for q in range(NQ)]
        return lambda m: np.array(f(m))

    cases = [("1 round,  no RZ", 1, False, 2.0),
             ("1 round,  RZ   ", 1, True,  2.0),
             ("3 rounds, no RZ", 3, False, 2.0),
             ("3 rounds, RZ   ", 3, True,  1.0)]
    ok = True
    print("  asymptotic (s = 0.01):")
    for lbl, r, rz, want in cases:
        exponent = response_exponent(make(r, rz), 0.01)
        good = abs(exponent - want) < 0.05
        ok &= good
        print(f"    {lbl}  exponent = {exponent:.3f}   expect {want:.1f}   "
              f"{'ok' if good else 'FAIL'}")
    print("  at the operating point (s = 1), for contrast:")
    for lbl, r, rz, _ in cases[::3]:
        print(f"    {lbl}  exponent = {response_exponent(make(r, rz), 1.0):.3f}")
    print("  selftest:", "PASS" if ok else "FAIL")
    return ok


# ============================================================================
# optimizer
# ============================================================================
OPTIMIZER_ARMS = [
    ("zero_published", dict(optimizer="none", train_var=False, readout_scaler=False)),
    ("spsa_published", dict(optimizer="spsa", train_var=True,  readout_scaler=False)),
    ("adam_published", dict(optimizer="adam", train_var=True,  readout_scaler=False)),
    ("zero_scaled",    dict(optimizer="none", train_var=False, readout_scaler=True)),
    ("adam_scaled",    dict(optimizer="adam", train_var=True,  readout_scaler=True)),
]


def optimizer_job(config, seed, epochs, nq, split):
    rows, traj = [], []
    cache = {}
    for name, kw in OPTIMIZER_ARMS:
        arch = C.Arch(n_qubits=nq, epochs=(0 if kw["optimizer"] == "none" else epochs),
                      **kw)
        row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                               ds_cache=cache, extra=dict(arm=name))
        rows.append(row)
        for t in tr.traj:
            traj.append(dict(config=config, seed=seed, arm=name, **t))
        print(f"    {config} s{seed} {name:16s} MAE {row['mae']:.3f}  "
              f"loss {row['loss_first']:.4f}->{row['loss_last']:.4f}  "
              f"shift_rel {row['param_shift_rel']:.3f}  ({row['time_s']:.0f}s)",
              flush=True)
    return [("row", r) for r in rows] + [("traj", t) for t in traj]


def optimizer_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B3,B5")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-qubits", type=int, default=13)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.epochs, a.n_qubits, a.split) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  G0 -- training recovery   configs {configs}   seeds {seeds}   "
          f"epochs {a.epochs}")
    print("=" * 78)

    tagged = C.parallel_jobs(optimizer_job, args, a.jobs, desc="G0")
    rows = [r for k, r in tagged if k == "row"]
    traj = [r for k, r in tagged if k == "traj"]

    C.write_csv(os.path.join(a.out, "G0_training_recovery.csv"), rows,
                C.COMMON_COLUMNS + ["arm"])
    C.write_csv(os.path.join(a.out, "G0_trajectory.csv"), traj,
                ["config", "seed", "arm", "epoch", "loss", "grad_norm",
                 "param_shift", "var_std", "alpha_mean"])

    # ---------------- gate ----------------
    print("\n" + "=" * 78)
    print("  G0 gate")
    print("=" * 78)
    rows_by_arm = {}
    for r in rows:
        rows_by_arm.setdefault((r["config"], r["arm"]), []).append(r)

    print(f"\n  {'config':<8}{'arm':<18}{'MAE mean+-std':>20}{'loss drop':>12}"
          f"{'shift/|init|':>14}")
    for c in configs:
        for name, _ in OPTIMIZER_ARMS:
            cell_rows = rows_by_arm.get((c, name), [])
            if not cell_rows:
                continue
            mae = np.array([r["mae"] for r in cell_rows])
            loss_drop = np.nanmean([r["loss_drop"] for r in cell_rows])
            shift_rel = np.nanmean([r["param_shift_rel"] for r in cell_rows])
            print(f"  {c:<8}{name:<18}{mae.mean():>12.3f} +-{mae.std():<6.3f}"
                  f"{loss_drop:>12.1%}{shift_rel:>14.1%}")

    print(f"\n  {'config':<8}{'comparison':<34}{'rel':>9}{'wins':>7}{'p':>10}  gate")
    gate_pass = True
    for c in configs:
        for tr_arm, base_arm in [("adam_published", "zero_published"),
                                 ("adam_scaled", "zero_scaled"),
                                 ("spsa_published", "zero_published")]:
            A = [r["mae"] for r in rows_by_arm.get((c, tr_arm), [])]
            B = [r["mae"] for r in rows_by_arm.get((c, base_arm), [])]
            if not A or not B:
                continue
            test = C.paired_test(A, B)
            sig = test["p"] < 0.05
            mark = "OK" if sig else "--"
            if tr_arm.startswith("adam") and not sig:
                gate_pass = False
            print(f"  {c:<8}{tr_arm + ' vs ' + base_arm:<34}{test['rel']:>+8.1%}"
                  f"{test['wins']:>5}/{test['n']}{test['p']:>10.4f}  {mark}")

    for c in configs:
        cell_rows = rows_by_arm.get((c, "adam_published"), []) + rows_by_arm.get((c, "adam_scaled"), [])
        if not cell_rows:
            continue
        loss_drop = np.nanmean([r["loss_drop"] for r in cell_rows])
        shift_rel = np.nanmean([r["param_shift_rel"] for r in cell_rows])
        c1, c2 = loss_drop >= 0.05, shift_rel >= 0.10
        print(f"\n  {c}: (1) loss drop {loss_drop:.1%} {'>= 5% OK' if c1 else '< 5% FAIL'}"
              f"   (2) param shift {shift_rel:.1%} {'>= 10% OK' if c2 else '< 10% FAIL'}")
        gate_pass = gate_pass and c1 and c2

    print("\n  G0 " + ("PASSED -- learning is live; G1 may proceed."
                       if gate_pass else
                       "NOT fully passed -- see condition (3) note in the header."))
    print("=" * 78)
