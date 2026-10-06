"""Beyond atomization energy on the full set.

isomer   Error within groups of isomers and top-1 selection of the
         lowest-energy isomer, stratified by group size.
targets  HOMO-LUMO gap and dipole moment, targets that the composition model
         cannot explain.
"""
import argparse, os
import numpy as np

from .. import core as C
from .baselines import c3_message_passing


# ============================================================================
# isomer
# ============================================================================
def mol_features(ds, i, nq, scheme):
    m = ds.mols[i]
    bonds = C.extract_bonds(m, nq, scheme)
    n_atoms = int(min(m["n_atoms"], nq))
    adj = np.asarray(m["adj"])[:n_atoms, :n_atoms]
    return dict(
        n_atoms=n_atoms,
        n_bonds=len(bonds),
        n_bond_types=len({b[3] for b in bonds}),
        n_rings=max(len(bonds) - n_atoms + 1, 0),
        max_degree=int(adj.sum(1).max()) if n_atoms else 0,
        n_heavy=int(sum(1 for z in m["Z"][:n_atoms] if int(z) > 1)),
        formula=m["formula"],
        comp_resid=float(abs(ds.resid[i])),
    )


def isomer_job(config, seed, split, epochs, rounds):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    nq = C.qubits_for(config)
    scheme = "elem_order"
    cache, rows = {}, []

    arch = C.Arch(n_qubits=nq, entangler="zz", alpha=0.1, alpha_mode="bytype",
                  n_rounds=rounds, epochs=epochs, readout_scaler=True,
                  canonical_order=True)
    row, tr, pred_q = C.run_arm(config, arch, seed=seed, split_level=split,
                                ds_cache=cache, extra=dict(arm="circuit"))
    ds = tr.ds
    te = ds.test_idx
    print(f"    {config} s{seed} circuit MAE {row['mae']:.3f} "
          f"({row['time_s']:.0f}s)", flush=True)

    # classical equivalent, same node information and the circuit's budget
    r_c3 = c3_message_passing(ds, config, seed, nq, budget=row["n_params"],
                                 rounds=rounds, epochs=epochs, node_feat="eig")
    # rebuild its per-molecule predictions
    preds = {"circuit": pred_q}
    import torch
    torch.manual_seed(seed)

    # KRR-res on the same split
    cm = np.zeros((len(ds.mols), nq))
    for k, m in enumerate(ds.mols):
        e = np.asarray(m["cm_eig"], float)
        cm[k, :min(len(e), nq)] = e[:nq]
    scaler = StandardScaler().fit(cm[ds.train_idx])
    kr = KernelRidge(alpha=0.001, kernel="rbf", gamma=0.01)
    kr.fit(scaler.transform(cm[ds.train_idx]), ds.y_target[ds.train_idx])
    pred_scaled = kr.predict(scaler.transform(cm[te]))
    preds["krr_res"] = ds.y_comp[te] + pred_scaled * ds.res_std + ds.res_mean
    preds["composition"] = ds.y_comp[te]

    groups = {}
    for k in te:
        groups.setdefault(ds.formula_key[k], []).append(k)

    y = ds.y_raw
    for pos, k in enumerate(te):
        f = mol_features(ds, k, nq, scheme)
        base = dict(config=config, seed=seed, mol_index=int(k),
                    y_true=float(y[k]),
                    in_isomer_grp=int(len(groups[ds.formula_key[k]]) > 1),
                    isomer_grp_size=len(groups[ds.formula_key[k]]), **f)
        for arm, p in preds.items():
            rows.append(dict(base, arm=arm, pred=float(p[pos]),
                             err=float(abs(p[pos] - y[k]))))
    rows.append(dict(config=config, seed=seed, arm="C3_eigfeat_summary",
                     mol_index=-1, err=float(r_c3["mae"]), y_true=np.nan,
                     n_atoms=0, n_bonds=0, n_bond_types=0, n_rings=0,
                     max_degree=0, n_heavy=0, formula="", comp_resid=np.nan,
                     in_isomer_grp=0, isomer_grp_size=0, pred=np.nan))
    return rows


def tertile_table(rows, var, arms, label):
    """Split the pooled test molecules into tertiles of `var` and compare."""
    mols = [r for r in rows if r["arm"] == arms[0] and r["mol_index"] >= 0]
    if not mols:
        return
    vals = np.array([r[var] for r in mols], float)
    qs = np.quantile(vals, [1 / 3, 2 / 3])
    print(f"\n  {label}  (tertile cuts at {qs[0]:.1f}, {qs[1]:.1f})")
    print(f"    {'tertile':<16}{'n':>7}" + "".join(f"{a:>14}" for a in arms)
          + f"{'circuit/ref':>14}")
    key = {}
    for r in rows:
        if r["mol_index"] < 0:
            continue
        key[(r["arm"], r["config"], r["seed"], r["mol_index"])] = r["err"]
    for lo, hi, name in ((-np.inf, qs[0], "low"), (qs[0], qs[1], "mid"),
                         (qs[1], np.inf, "high")):
        selected = [r for r in mols if lo < r[var] <= hi] if lo != -np.inf else\
              [r for r in mols if r[var] <= hi]
        if not selected:
            continue
        cells, first = [], None
        for a in arms:
            e = [key.get((a, r["config"], r["seed"], r["mol_index"]))
                 for r in selected]
            e = [x for x in e if x is not None]
            cells.append(np.mean(e) if e else np.nan)
            if first is None:
                first = cells[0]
        ratio = cells[0] / cells[1] if len(cells) > 1 and cells[1] else np.nan
        print(f"    {name:<16}{len(selected):>7}" +
              "".join(f"{c:>14.3f}" for c in cells) + f"{ratio:>13.2f}x")


def isomer_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="A1,A2,B1,B2,B3,B4,B5")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=14)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.split, a.epochs, a.rounds) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  H4 -- stratified error analysis   configs {configs}   "
          f"seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(isomer_job, args, a.jobs, desc="H4")
    C.write_csv(os.path.join(a.out, "H4_stratified.csv"), rows,
                ["config", "seed", "arm", "mol_index", "formula", "y_true",
                 "pred", "err", "n_atoms", "n_heavy", "n_bonds",
                 "n_bond_types", "n_rings", "max_degree", "in_isomer_grp",
                 "isomer_grp_size", "comp_resid"])

    arms = ["circuit", "krr_res", "composition"]
    print("\n" + "=" * 78)
    print("  H4 -- mean absolute error by stratum (all configurations pooled)")
    print("=" * 78)
    for var, lab in (("n_bond_types", "bond-type variety  <- the condition (4) mechanism"),
                     ("n_rings", "ring count (cyclomatic number)"),
                     ("n_bonds", "molecule size"),
                     ("max_degree", "branching")):
        tertile_table(rows, var, arms, lab)

    key = {}
    for r in rows:
        if r["mol_index"] >= 0:
            key[(r["arm"], r["config"], r["seed"], r["mol_index"])] = r["err"]
    mols = [r for r in rows if r["arm"] == "circuit" and r["mol_index"] >= 0]
    print("\n  isomer-group membership")
    print(f"    {'stratum':<16}{'n':>7}" + "".join(f"{a:>14}" for a in arms))
    for v, name in ((1, "in a group"), (0, "singleton")):
        selected = [r for r in mols if r["in_isomer_grp"] == v]
        if not selected:
            continue
        cells = [np.mean([key[(a, r["config"], r["seed"], r["mol_index"])]
                          for r in selected]) for a in arms]
        print(f"    {name:<16}{len(selected):>7}" + "".join(f"{c:>14.3f}" for c in cells))

    from scipy import stats
    print("\n  Spearman(stratum variable, circuit error) and the same for KRR-res")
    for var in ("n_bond_types", "n_rings", "n_bonds", "max_degree", "comp_resid"):
        x = np.array([r[var] for r in mols], float)
        err_circuit = np.array([key[("circuit", r["config"], r["seed"], r["mol_index"])]
                                for r in mols])
        err_krr = np.array([key[("krr_res", r["config"], r["seed"], r["mol_index"])]
                            for r in mols])
        rho_circuit = stats.spearmanr(x, err_circuit); rho_krr = stats.spearmanr(x, err_krr)
        rho_logratio = stats.spearmanr(x, np.log((err_circuit + 1e-9) / (err_krr + 1e-9)))
        print(f"    {var:<14} circuit {rho_circuit.statistic:+.3f}  krr {rho_krr.statistic:+.3f}"
              f"   log-ratio {rho_logratio.statistic:+.3f} (p={rho_logratio.pvalue:.2e})")
    print("\n  A NEGATIVE log-ratio correlation means the circuit's relative")
    print("  advantage grows with that variable.")


# ============================================================================
# targets
# ============================================================================
QUANTUM = [
    ("zz_bytype_r3",   dict(entangler="zz", alpha=0.1, alpha_mode="bytype")),
    ("xy_bytype_r3",   dict(entangler="xy", alpha=0.1, alpha_mode="bytype")),
    ("none_r3",        dict(entangler="none", alpha=0.0, alpha_mode="fixed")),
    ("xy_shuffled_r3", dict(entangler="xy", alpha=0.1,
                            alpha_mode="bytype_shuffled")),
]


def classical(ds, config, seed, nq, budget, rounds, epochs, target, residual):
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    tr, te = ds.train_idx, ds.test_idx
    rows = []
    cm = np.zeros((len(ds.mols), nq))
    for i, m in enumerate(ds.mols):
        e = np.asarray(m["cm_eig"], float)
        cm[i, :min(len(e), nq)] = e[:nq]
    scaler = StandardScaler().fit(cm[tr])
    krr = KernelRidge(alpha=0.001, kernel="rbf", gamma=0.01)
    krr.fit(scaler.transform(cm[tr]), ds.y_target[tr])
    pred_scaled = krr.predict(scaler.transform(cm[te]))
    pred = ds.y_comp[te] + pred_scaled * ds.res_std + ds.res_mean
    base = dict(config=config, seed=seed, target=target,
                residual=int(residual), n_train=len(tr), n_test=len(te))
    r = dict(base, arm="krr_res", n_params=len(tr),
             mae=float(mean_absolute_error(ds.y_raw[te], pred)),
             r2=float(r2_score(ds.y_raw[te], pred)))
    r.update(C.isomer_metrics(ds, te, pred)); rows.append(r)

    pred = ds.y_comp[te] if residual else np.full(len(te), ds.y_raw[tr].mean())
    r = dict(base, arm="composition", n_params=0,
             mae=float(mean_absolute_error(ds.y_raw[te], pred)),
             r2=float(r2_score(ds.y_raw[te], pred)))
    r.update(C.isomer_metrics(ds, te, pred)); rows.append(r)

    for node_feat, name in (("eig", "C3_eigfeat"), ("atom", "C3_msgpass")):
        c3_row = c3_message_passing(ds, config, seed, nq, budget=budget,
                                   rounds=rounds, epochs=epochs, node_feat=node_feat)
        c3_row.update(base); c3_row["arm"] = name
        rows.append(c3_row)
    return rows


def targets_job(config, seed, target, residual, split, epochs, rounds, types_from_train=False):
    nq = C.qubits_for(config)
    cache, rows = {}, []
    budget, ds = None, None
    for name, kw in QUANTUM:
        arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs,
                      readout_scaler=True, canonical_order=True,
                      types_from_train=types_from_train, target=target, residual=residual,
                      **kw)
        row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                               ds_cache=cache, readouts=("ridge", "krr"),
                               extra=dict(arm=name))
        rows.append(row)
        if name == "zz_bytype_r3":
            budget, ds = row["n_params"], tr.ds
        print(f"    {config} s{seed} {target:6s} res{int(residual)} "
              f"{name:15s} MAE {row['mae']:.4f}  iso {row['isomer_mae']:.4f} "
              f"({row['time_s']:.0f}s)", flush=True)
    rows += classical(ds, config, seed, nq, budget, rounds, epochs, target,
                      residual)
    return rows


def targets_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3,A1")
    p.add_argument("--targets", default="HLgap,DIP")
    p.add_argument("--residual", default="1,0",
                   help="1 = subtract a composition model, 0 = predict directly")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=15)
    p.add_argument("--types-from-train", action="store_true")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    targets = a.targets.split(",")
    resids = [bool(int(x)) for x in a.residual.split(",")]
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, t, r, a.split, a.epochs, a.rounds, a.types_from_train)
            for t in targets for r in resids for c in configs for s in seeds]

    print("=" * 78)
    print(f"  I3 gamma-1 -- targets {targets}   residual {resids}   "
          f"configs {configs}   seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(targets_job, args, a.jobs, desc="I3")
    C.write_csv(os.path.join(a.out, f"N1_alt_targets{a.tag}.csv"), rows,
                C.COMMON_COLUMNS + ["arm", "mae_krr", "isomer_mae_krr"])

    rows_by_cell = {}
    for r in rows:
        rows_by_cell.setdefault((r["target"], int(r["residual"]), r["config"], r["arm"]),
                      []).append(r["mae"])

    def v(t, residual, c, arm):
        return np.array(rows_by_cell.get((t, residual, c, arm), []), float)

    for t in targets:
        for residual in [int(x) for x in a.residual.split(",")]:
            print("\n" + "=" * 78)
            print(f"  target {t}   residual={residual}")
            print("=" * 78)
            arms = [x[0] for x in QUANTUM] + ["krr_res", "C3_eigfeat",
                                              "C3_msgpass", "composition"]
            print(f"  {'arm':<17}" + "".join(f"{c:>11}" for c in configs))
            for arm in arms:
                cells = [f"{v(t,residual,c,arm).mean():.4f}" if len(v(t,residual,c,arm))
                         else "-" for c in configs]
                print(f"  {arm:<17}" + "".join(f"{x:>11}" for x in cells))
            print(f"\n  {'config':<8}{'comparison':<34}{'rel':>9}{'wins':>8}{'p':>10}")
            for c in configs:
                for x, y, lab in (("zz_bytype_r3", "none_r3", "(2) vs alpha = 0"),
                                  ("xy_bytype_r3", "zz_bytype_r3", "(5) xy vs zz"),
                                  ("zz_bytype_r3", "xy_shuffled_r3", "(4) vs shuffled"),
                                  ("zz_bytype_r3", "krr_res", "  * vs KRR-res"),
                                  ("zz_bytype_r3", "C3_eigfeat", "(6b) vs C3_eigfeat"),
                                  ("zz_bytype_r3", "C3_msgpass", "(6a) vs C3_msgpass")):
                    A, B = v(t, residual, c, x), v(t, residual, c, y)
                    n = min(len(A), len(B))
                    if n < 3:
                        continue
                    st = C.paired_test(A[:n], B[:n])
                    flag = " *" if (st["p"] == st["p"] and st["p"] < 0.05
                                    and st["rel"] < 0) else ""
                    print(f"  {c:<8}{lab:<34}{st['rel']:>+9.1%}"
                          f"{st['wins']:>6}/{n}{st['p']:>10.4f}{flag}")
    paired = []
    for t in targets:
        for residual in [int(x) for x in a.residual.split(",")]:
            for c in configs:
                for x, y, lab in (("zz_bytype_r3", "none_r3", "cond2_alpha0"),
                                  ("xy_bytype_r3", "zz_bytype_r3", "cond5_xy_vs_zz"),
                                  ("zz_bytype_r3", "xy_shuffled_r3", "cond4_shuffled"),
                                  ("zz_bytype_r3", "krr_res", "vs_krr_res"),
                                  ("zz_bytype_r3", "C3_eigfeat", "cond6b_C3_eigfeat"),
                                  ("zz_bytype_r3", "C3_msgpass", "cond6a_C3_msgpass")):
                    A, B = v(t, residual, c, x), v(t, residual, c, y)
                    n = min(len(A), len(B))
                    if n < 3:
                        continue
                    st = C.paired_test(A[:n], B[:n])
                    paired.append(dict(target=t, residual=residual, config=c,
                                       comparison=lab, arm_a=x, arm_b=y,
                                       mae_a=float(A[:n].mean()),
                                       mae_b=float(B[:n].mean()),
                                       rel=st["rel"], wins=st["wins"],
                                       n_seeds=n, p=st["p"]))
    C.write_csv(os.path.join(a.out, f"N1_paired{a.tag}.csv"), paired)

    print("\n  Compare against the same conditions on eAT (verdict_final_v4.txt):")
    print("    (2) 7/7 significant   (5) 0/7   (6a) 0/7   (6b) 2/7")
    print("  If (2) and (5) both pass here, entanglement is task dependent and")
    print("  the paper's conclusion changes.  If the pattern repeats, the cause")
    print("  is the architecture, not the target.")
