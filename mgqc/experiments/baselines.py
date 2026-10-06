"""Does bond-type-conditioned coupling help, and against what?

accuracy   The main comparison. On each configuration the circuit is trained
           against same-split, same-seed alternatives: the composition model,
           ridge and kernel ridge on spectral features, the circuit with the
           coupling off, with one shared coefficient, and with the bond types
           randomly reassigned. Also records the learned per-type coefficients.
classical  Two classical message-passing models matched to the circuit's
           parameter count and node information.
schnet     A deep graph network (SchNet) trained on the circuit's own splits.
tuning     Hyper-parameter grids for the circuit and both classical baselines.

Every arm of every comparison shares the split and the seed, so each paired
statistic in the manuscript is a within-seed difference.
"""
import argparse, csv, os, sys, time
from collections import defaultdict
import numpy as np

from .. import core as C


# ============================================================================
# accuracy
# ============================================================================
QUANTUM = [
    ("zz_bytype_r3",   dict(entangler="zz", alpha=0.1, alpha_mode="bytype")),
    ("xy_bytype_r3",   dict(entangler="xy", alpha=0.1, alpha_mode="bytype")),
    ("none_r3",        dict(entangler="none", alpha=0.0, alpha_mode="fixed")),
    ("zz_fixed_r3",    dict(entangler="zz", alpha=0.1, alpha_mode="fixed")),
    ("xy_shuffled_r3", dict(entangler="xy", alpha=0.1, alpha_mode="bytype_shuffled")),
    # R6-A.  The original control shuffled the type ids AND swapped the
    # entangler, so "zz_bytype vs xy_shuffled" moved two things at once.  ZZ
    # genuinely beats XY on A1 (-12.2%) and B1 (-9.1%), and that difference was
    # being charged to the bond-type effect.  This arm closes the comparison
    # inside ZZ.  t_sh is untouched: `Dataset` draws it from
    # default_rng(10_000 + seed) with the same arch, so this arm sees exactly
    # the same random regrouping xy_shuffled_r3 saw.
    ("zz_shuffled_r3", dict(entangler="zz", alpha=0.1, alpha_mode="bytype_shuffled")),
]


# Every comparison the verdict needs, as (quantum arm, reference).
COMPARISONS = [
    ("krr_res",        "the contest -- condition on which the paper's form turns"),
    ("none_r3",        "condition (2): entanglement vs its own alpha = 0"),
    ("zz_fixed_r3",    "condition (5) decomposed: per-type vs one fixed alpha"),
    ("zz_shuffled_r3", "condition (4), matched: same entangler, random groups"),
    ("xy_shuffled_r3", "condition (4), confounded: also swaps ZZ -> XY"),
    ("krr_cos_cl",     "condition (3): best classical feature transform"),
    ("ridge_raw_eig",  "condition (1): the stated classical equivalent"),
]


def classical_rows(ds, config, seed, nq):
    """Classical references on this exact split.  Hyper-parameters are either
    fixed or chosen by CV inside the training split only."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score

    tr, te = ds.train_idx, ds.test_idx
    eig = np.zeros((len(ds.mols), nq)); cm = np.zeros((len(ds.mols), nq))
    for i, m in enumerate(ds.mols):
        e = np.asarray(m["lap_eig_coulomb"], float)
        eig[i, :min(len(e), nq)] = e[:nq]
        cm_spectrum = np.asarray(m["cm_eig"], float)
        cm[i, :min(len(cm_spectrum), nq)] = cm_spectrum[:nq]
    gmax = np.max(np.abs(eig[tr])) + 1e-10
    cos_eig = np.cos(eig * np.pi / gmax)

    out = []

    def record(name, X, model, n_params):
        scaler = StandardScaler().fit(X[tr])
        model.fit(scaler.transform(X[tr]), ds.y_target[tr])
        pred_z = model.predict(scaler.transform(X[te]))
        pred = ds.y_comp[te] + pred_z * ds.res_std + ds.res_mean
        r = dict(config=config, seed=seed, arm=name, kind="classical",
                 n_qubits=nq, n_params=n_params,
                 n_train=len(tr), n_test=len(te),
                 mae=float(mean_absolute_error(ds.y_raw[te], pred)),
                 r2=float(r2_score(ds.y_raw[te], pred)))
        r.update(C.isomer_metrics(ds, te, pred))
        # A classical arm has one number; it is carried under both readout
        # keys so the quantum-with-KRR-readout table pairs against the same
        # reference rather than a differently-sized subset.
        for k in ("ridge", "krr"):
            r[f"mae_{k}"] = r["mae"]
            r[f"isomer_mae_{k}"] = r["isomer_mae"]
            r[f"isomer_rho_{k}"] = r["isomer_rho"]
        out.append(r)

    # KRR-res: the published baseline's hyper-parameters, unchanged.
    record("krr_res", cm, KernelRidge(alpha=0.001, kernel="rbf", gamma=0.01),
           len(tr))
    # Best classical transform of the same spectrum the circuit encodes.
    grid = {"alpha": [1e-3, 1e-2, 1e-1, 1.0], "gamma": [1e-3, 1e-2, 5e-2, 1e-1, 1.0]}
    record("krr_cos_cl", cos_eig,
           GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=5,
                        scoring="neg_mean_absolute_error"), len(tr))
    record("ridge_raw_eig", eig, Ridge(alpha=1.0), nq + 1)

    from sklearn.metrics import mean_absolute_error as mae_f
    pred = ds.y_comp[te]
    r = dict(config=config, seed=seed, arm="composition", kind="classical",
             n_qubits=nq, n_params=0, n_train=len(tr), n_test=len(te),
             mae=float(mae_f(ds.y_raw[te], pred)),
             r2=float(r2_score(ds.y_raw[te], pred)))
    r.update(C.isomer_metrics(ds, te, pred))
    for k in ("ridge", "krr"):
        r[f"mae_{k}"] = r["mae"]
        r[f"isomer_mae_{k}"] = r["isomer_mae"]
        r[f"isomer_rho_{k}"] = r["isomer_rho"]
    out.append(r)
    return out


def accuracy_job(config, seed, epochs, split, rounds, only_arm=None, canonical=False,
        types_from_train=False):
    """One (config, seed).  `only_arm` narrows it to a single arm so that a
    large configuration can be spread over arm-level jobs instead of being
    limited to one worker per seed -- A2 is 2,639 training molecules, where
    five arms in series is ~3 h on one core and ~1 h spread out."""
    nq = C.qubits_for(config)
    cache, rows, alpha_rows = {}, [], []
    arms = QUANTUM if only_arm is None else [x for x in QUANTUM
                                             if x[0] == only_arm]
    for name, kw in arms:
        arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs,
                      readout_scaler=True, canonical_order=canonical,
                      types_from_train=types_from_train, **kw)
        row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                               ds_cache=cache, readouts=("ridge", "krr"),
                               extra=dict(arm=name, kind="quantum"))
        rows.append(row)
        if kw["alpha_mode"] == "bytype":
            for t, key in enumerate(tr.ds.type_keys):
                alpha_rows.append(dict(config=config, seed=seed, arm=name,
                                       n_qubits=nq, type_id=t,
                                       type=C.type_label(key),
                                       n_bonds=tr.ds.type_count[key],
                                       alpha_learned=float(tr.alpha[t])))
        print(f"    {config} s{seed} {name:16s} MAE {row['mae']:.3f} "
              f"(krr {row.get('mae_krr', float('nan')):.3f})  "
              f"iso {row['isomer_mae']:.3f}  np {row['n_params']}  "
              f"({row['time_s']:.0f}s)", flush=True)
        ds = tr.ds
    # The classical references are seconds; attach them to one arm only so
    # they are not recomputed once per arm-level job.
    if only_arm in (None, QUANTUM[0][0]):
        rows += classical_rows(ds, config, seed, nq)
        print(f"    {config} s{seed} classical: " +
              "  ".join(f"{r['arm']} {r['mae']:.3f}" for r in rows[-4:]),
              flush=True)
    return [("row", r) for r in rows] + [("alpha", r) for r in alpha_rows]


def report(rows, configs, out_dir, readout_key="mae"):
    rows_by_cell = {}
    for r in rows:
        rows_by_cell.setdefault((r["config"], r["arm"]), []).append(r)

    def vals(cfg, arm):
        cell_rows = rows_by_cell.get((cfg, arm), [])
        return np.array([r[readout_key] for r in cell_rows
                         if r.get(readout_key) is not None
                         and r.get(readout_key) == r.get(readout_key)], float)

    print("\n" + "=" * 78)
    print(f"  E1 -- MAE by configuration and arm   [{readout_key}]")
    print("=" * 78)
    all_arms = [a for a, _ in QUANTUM] + ["krr_res", "krr_cos_cl",
                                          "ridge_raw_eig", "composition"]
    print(f"  {'arm':<17}" + "".join(f"{c:>11}" for c in configs))
    for a in all_arms:
        cells = []
        for c in configs:
            v = vals(c, a)
            cells.append(f"{v.mean():.3f}" if len(v) else "-")
        print(f"  {a:<17}" + "".join(f"{x:>11}" for x in cells))

    paired = []
    print("\n" + "=" * 78)
    print("  E1 -- paired comparisons, per configuration")
    print("=" * 78)
    for ref, why in COMPARISONS:
        print(f"\n  vs {ref}   ({why})")
        print(f"    {'config':<8}{'arm':<17}{'quantum':>9}{'ref':>9}"
              f"{'rel':>9}{'wins':>8}{'p':>9}")
        for c in configs:
            for q, _ in QUANTUM[:2]:
                A, B = vals(c, q), vals(c, ref)
                n = min(len(A), len(B))
                if n < 2:
                    continue
                paired_stats = C.paired_test(A[:n], B[:n])
                flag = " *" if (paired_stats["p"] == paired_stats["p"] and paired_stats["p"] < 0.05
                                and paired_stats["rel"] < 0) else ""
                print(f"    {c:<8}{q:<17}{A[:n].mean():>9.3f}{B[:n].mean():>9.3f}"
                      f"{paired_stats['rel']:>+9.1%}{paired_stats['wins']:>6}/{n}{paired_stats['p']:>9.4f}{flag}")
                paired.append(dict(config=c, quantum=q, reference=ref,
                                   readout=readout_key,
                                   mae_quantum=float(A[:n].mean()),
                                   mae_reference=float(B[:n].mean()),
                                   rel=paired_stats["rel"], wins=paired_stats["wins"], n_seeds=n,
                                   p=paired_stats["p"], reason=why))

    print("\n" + "=" * 78)
    print("  E1 -- seven-configuration tally (sign test over configurations)")
    print("=" * 78)
    from scipy import stats
    for ref, _ in COMPARISONS:
        for q, _ in QUANTUM[:2]:
            wins = sig = n_configs = 0
            for c in configs:
                A, B = vals(c, q), vals(c, ref)
                n = min(len(A), len(B))
                if n < 2:
                    continue
                n_configs += 1
                paired_stats = C.paired_test(A[:n], B[:n])
                if paired_stats["rel"] < 0:
                    wins += 1
                    if paired_stats["p"] < 0.05:
                        sig += 1
            if n_configs:
                sign_p = stats.binomtest(wins, n_configs, 0.5).pvalue
                print(f"  {q:<17} vs {ref:<17}  {wins}/{n_configs} configs won "
                      f"({sig} significant)   sign-test p={sign_p:.4f}")
    return paired


def accuracy_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B2,B3,A1,B1,B4")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=28)
    p.add_argument("--canonical", action="store_true",
                   help="deterministic atom -> qubit map (F1 arm F)")
    p.add_argument("--split-arms", action="store_true",
                   help="one job per (config, seed, arm) -- use for A2")
    p.add_argument("--tag", default="")
    p.add_argument("--arms", default=None,
                   help="comma-separated subset of the quantum arms to run "
                        "(default: all).  Each named arm becomes its own job.")
    p.add_argument("--types-from-train", action="store_true",
                   help="build the bond-type vocabulary from the training "
                        "molecules only (removes the support leak)")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    if a.arms:
        known = dict(QUANTUM)
        selected_arms = a.arms.split(",")
        bad = [x for x in selected_arms if x not in known]
        if bad:
            raise SystemExit(f"unknown arm(s) {bad}; choose from {list(known)}")
        args = [(c, s, a.epochs, a.split, a.rounds, q, a.canonical,
                 a.types_from_train)
                for c in configs for s in seeds for q in selected_arms]
    elif a.split_arms:
        args = [(c, s, a.epochs, a.split, a.rounds, q, a.canonical,
                 a.types_from_train)
                for c in configs for s in seeds for q, _ in QUANTUM]
    else:
        args = [(c, s, a.epochs, a.split, a.rounds, None, a.canonical,
                 a.types_from_train)
                for c in configs for s in seeds]

    print("=" * 78)
    print(f"  E1 -- seven-configuration head to head")
    print(f"  configs {configs}   seeds {len(seeds)}   rounds {a.rounds}   "
          f"epochs {a.epochs}")
    print("=" * 78)
    tagged = C.parallel_jobs(accuracy_job, args, a.jobs, desc="E1")
    rows = [r for k, r in tagged if k == "row"]
    alpha_rows = [r for k, r in tagged if k == "alpha"]

    C.write_csv(os.path.join(a.out, f"E1_sevenconfig{a.tag}.csv"), rows,
                ["config", "arm", "kind", "seed", "n_qubits", "n_params",
                 "n_train", "n_test", "mae", "mae_ridge", "mae_krr", "r2",
                 "isomer_mae", "isomer_mae_krr", "isomer_rho", "isomer_rho_krr",
                 "n_isomer_groups", "loss_drop", "param_shift_rel", "time_s"])
    C.write_csv(os.path.join(a.out, f"E1_alpha_learned{a.tag}.csv"), alpha_rows)

    paired = report(rows, configs, a.out, "mae")
    paired += report(rows, configs, a.out, "mae_krr")
    C.write_csv(os.path.join(a.out, f"E1_paired{a.tag}.csv"), paired)


# ============================================================================
# classical
# ============================================================================
def bond_arrays(ds, nq, shuffled=False):
    """Per-molecule (i, j, J, type_id) as flat arrays, for fast rebuilds."""
    out = []
    type_col = 4 if shuffled else 3
    for mol_bonds in ds.bonds:
        if mol_bonds:
            i = np.array([b[0] for b in mol_bonds]); j = np.array([b[1] for b in mol_bonds])
            J = np.array([b[2] for b in mol_bonds], float)
            t = np.array([b[type_col] for b in mol_bonds])
        else:
            i = j = t = np.zeros(0, dtype=int); J = np.zeros(0)
        out.append((i, j, J, t))
    return out


def ridge_head(F, y, alpha=1.0):
    """The same closed-form standardised ridge the circuit readout uses."""
    mu, sd = F.mean(0), F.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Fs = (F - mu) / sd
    A = np.column_stack([Fs, np.ones(len(Fs))])
    reg = alpha * np.eye(A.shape[1]); reg[-1, -1] = 0.0
    sol = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return sol[:-1], float(sol[-1]), mu, sd


def apply_head(F, w, b, mu, sd):
    return ((F - mu) / sd) @ w + b


def score(ds, pred_norm_te, name, config, seed, n_params, extra=None):
    from sklearn.metrics import mean_absolute_error, r2_score
    te = ds.test_idx
    pred = ds.y_comp[te] + pred_norm_te * ds.res_std + ds.res_mean
    r = dict(config=config, seed=seed, arm=name, n_params=int(n_params),
             n_train=len(ds.train_idx), n_test=len(te),
             mae=float(mean_absolute_error(ds.y_raw[te], pred)),
             r2=float(r2_score(ds.y_raw[te], pred)))
    r.update(C.isomer_metrics(ds, te, pred))
    if extra:
        r.update(extra)
    return r


def c1_transplant(ds, config, seed, nq, alpha_by_type, weight="J"):
    """Freeze alpha_tau, build a classical weighted graph, use its spectrum.

    weight="J"       edge = alpha_tau * J_ij, the circuit's own coupling
                     (Z_i Z_j / (Z_i + Z_j) / sqrt(n_bonds)) -- the faithful
                     transplant of what the circuit actually learned.
    weight="coulomb" edge = alpha_tau * Z_i Z_j / (d_ij + 0.1), the form named
                     in the protocol.
    """
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler

    bonds = bond_arrays(ds, nq)
    F = np.zeros((len(ds.mols), nq))
    for m, (i, j, J, t) in enumerate(bonds):
        n = int(ds.n_active[m])
        if n == 0:
            continue
        W = np.zeros((n, n))
        mol = ds.mols[m]
        for k in range(len(i)):
            if i[k] >= n or j[k] >= n:
                continue
            alpha_t = alpha_by_type.get(int(t[k]), 0.0)
            if weight == "coulomb":
                Z = mol["Z"]
                e = alpha_t * Z[i[k]] * Z[j[k]] / (mol["dist"][i[k], j[k]] + 0.1)
            else:
                e = alpha_t * J[k]
            W[i[k], j[k]] = W[j[k], i[k]] = e
        L = np.diag(W.sum(1)) - W
        lam = np.sort(np.linalg.eigvalsh(L))
        F[m, :min(n, nq)] = lam[:nq]

    tr, te = ds.train_idx, ds.test_idx
    scaler = StandardScaler().fit(F[tr])
    grid = {"alpha": [1e-3, 1e-2, 1e-1, 1.0], "gamma": [1e-3, 1e-2, 5e-2, 1e-1, 1.0]}
    gs = GridSearchCV(KernelRidge(kernel="rbf"), grid, cv=5,
                      scoring="neg_mean_absolute_error")
    gs.fit(scaler.transform(F[tr]), ds.y_target[tr])
    pred_z = gs.predict(scaler.transform(F[te]))
    return score(ds, pred_z, f"C1_transplant_{weight}", config, seed,
                 len(alpha_by_type) + len(tr))


def c2_end_to_end(ds, config, seed, nq, epochs=100, lr=0.05, alpha_init=0.1,
                  shuffled=False, jitter=1e-6, clip=5.0):
    import torch
    torch.manual_seed(seed)
    bonds = bond_arrays(ds, nq, shuffled)
    n_types = ds.n_types
    alpha = torch.full((n_types,), float(alpha_init), requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=lr)
    tr, te = ds.train_idx, ds.test_idx
    y_tr = ds.y_target[tr]

    # A multiple of the identity would shift every eigenvalue equally and lift
    # no degeneracy; a non-uniform diagonal does, which is what keeps
    # d lambda / d W finite when two eigenvalues collide.
    def feats(idx):
        rows = []
        for m in idx:
            i, j, J, t = bonds[m]
            n = int(ds.n_active[m])
            if n == 0:
                rows.append(torch.zeros(nq)); continue
            W = torch.zeros((n, n), dtype=torch.float64)
            for k in range(len(i)):
                if i[k] >= n or j[k] >= n:
                    continue
                e = alpha[int(t[k])].double() * float(J[k])
                W[i[k], j[k]] = e
                W[j[k], i[k]] = e
            L = torch.diag(W.sum(1)) - W
            L = L + torch.diag(torch.arange(1, n + 1, dtype=torch.float64) * jitter)
            lam = torch.linalg.eigvalsh(L)
            pad = torch.zeros(nq, dtype=torch.float64)
            pad[:min(n, nq)] = lam[:nq]
            rows.append(pad)
        return torch.stack(rows)

    loss_first = loss_last = np.nan
    for ep in range(epochs):
        Ftr = feats(tr)
        Fn = Ftr.detach().numpy()
        w, b, mu, sd = ridge_head(Fn, y_tr)
        pred = apply_head(Ftr, torch.tensor(w), float(b),
                          torch.tensor(mu), torch.tensor(sd))
        loss = ((pred - torch.tensor(y_tr)) ** 2).mean()
        if ep == 0:
            loss_first = float(loss)
        loss_last = float(loss)
        opt.zero_grad()
        loss.backward()
        if alpha.grad is not None and torch.isfinite(alpha.grad).all():
            torch.nn.utils.clip_grad_norm_([alpha], clip)
            opt.step()

    with torch.no_grad():
        Ftr = feats(tr).numpy(); Fte = feats(te).numpy()
    w, b, mu, sd = ridge_head(Ftr, y_tr)
    pred_z = apply_head(Fte, w, b, mu, sd)
    name = "C2_shuffled" if shuffled else "C2_end2end"
    return score(ds, pred_z, name, config, seed, n_types + nq + 1,
                 dict(loss_first=loss_first, loss_last=loss_last,
                      alpha_mean=float(alpha.mean()),
                      alpha_std=float(alpha.std()))),\
        {t: float(alpha[t]) for t in range(n_types)}


def width_for_budget(budget, count_fn):
    """Largest hidden width whose parameter count still fits the budget.

    `count_fn(d)` must return the model's exact parameter count at width d, so
    that "parameter matched" means matched and not approximately matched --
    otherwise the comparison invites the reply that the classical model simply
    had more capacity."""
    d = 1
    while count_fn(d + 1) <= budget:
        d += 1
    return max(d, 1)


def c3_message_passing(ds, config, seed, nq, budget=110, rounds=3, epochs=100,
                       lr=0.05, alpha_init=0.1, shuffled=False,
                       node_feat="atom", return_pred=False):
    """node_feat decides what each node starts with, which is the one place the
    circuit and a classical message-passing model are NOT automatically
    comparable:

      atom   a learned embedding of the atomic number.  This is what the
             protocol specifies, and it is MORE than the circuit gets:
             the circuit never sees an atom-resolved identity.
      eig    node q starts from the q-th CL eigenvalue -- exactly the circuit's
             Layer A.  Node q therefore means "eigenmode q" while the edges
             mean "atom q", the same mismatch the G1 diagnosis found.
      diag   node q starts from the CL diagonal of atom q, an atom-local
             quantity on an atom-indexed node.

    Running all three separates "quantum vs classical" from "the circuit's
    encoding choice", which are otherwise confounded.
    """
    import torch
    torch.manual_seed(seed)
    bonds = bond_arrays(ds, nq, shuffled)
    elements = sorted({int(z) for m in ds.mols for z in m["Z"]})
    elem_index = {e: k for k, e in enumerate(elements)}
    n_types, n_elem = ds.n_types, len(elements)
    # embedding path: n_types + n_elem*d (embedding) + d + 1 (ridge head)
    # lift path:      n_types + 2d (Linear(1,d)) + d + 1 (ridge head)
    if node_feat == "atom":
        d = width_for_budget(budget, lambda k: n_types + n_elem * k + k + 1)
    else:
        d = width_for_budget(budget, lambda k: n_types + 3 * k + 1)

    alpha = torch.full((n_types,), float(alpha_init), requires_grad=True)

    if node_feat == "atom":
        emb = torch.nn.Embedding(n_elem, d)
        torch.nn.init.normal_(emb.weight, std=0.3)
        params = list(emb.parameters()) + [alpha]
        n_params = n_types + n_elem * d + d + 1
        elem_ids = [torch.tensor([elem_index[int(z)] for z in m["Z"][:int(ds.n_active[k])]],
                                 dtype=torch.long)
              for k, m in enumerate(ds.mols)]

        def node0(m):
            return emb(elem_ids[m])
    else:
        # A scalar per node, lifted to width d by a learned affine map.  The
        # scalar is whatever the circuit itself puts on that wire.
        lift = torch.nn.Linear(1, d)
        params = list(lift.parameters()) + [alpha]
        n_params = n_types + 2 * d + d + 1
        src = ds.ang if node_feat == "eig" else None
        if node_feat == "diag":
            diag_raw = np.zeros((len(ds.mols), nq))
            for k, m in enumerate(ds.mols):
                e = np.asarray(m["W_coulomb"], float).sum(1)
                diag_raw[k, :min(len(e), nq)] = e[:nq]
            src = np.pi * diag_raw / (np.max(np.abs(diag_raw[ds.train_idx])) + 1e-10)
        src_t = torch.tensor(src, dtype=torch.float32)

        def node0(m):
            n = int(ds.n_active[m])
            return lift(src_t[m, :n].unsqueeze(-1))

    opt = torch.optim.Adam(params, lr=lr)

    def feats(idx):
        rows = []
        for m in idx:
            i, j, J, t = bonds[m]
            n = int(ds.n_active[m])
            if n == 0:
                rows.append(torch.zeros(d)); continue
            A = torch.zeros((n, n))
            for k in range(len(i)):
                if i[k] >= n or j[k] >= n:
                    continue
                e = alpha[int(t[k])] * float(J[k])
                A[i[k], j[k]] = e
                A[j[k], i[k]] = e
            h = node0(m)
            for _r in range(rounds):
                h = torch.tanh(h + A @ h)
            rows.append(h.mean(0))
        return torch.stack(rows)

    tr, te = ds.train_idx, ds.test_idx
    y_tr = ds.y_target[tr]
    loss_first = loss_last = np.nan
    for ep in range(epochs):
        Ftr = feats(tr)
        w, b, mu, sd = ridge_head(Ftr.detach().numpy(), y_tr)
        pred = apply_head(Ftr, torch.tensor(w, dtype=torch.float32), float(b),
                          torch.tensor(mu, dtype=torch.float32),
                          torch.tensor(sd, dtype=torch.float32))
        loss = ((pred - torch.tensor(y_tr, dtype=torch.float32)) ** 2).mean()
        if ep == 0:
            loss_first = float(loss)
        loss_last = float(loss)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    with torch.no_grad():
        Ftr = feats(tr).numpy(); Fte = feats(te).numpy()
    w, b, mu, sd = ridge_head(Ftr, y_tr)
    pred_z = apply_head(Fte, w, b, mu, sd)
    tag = {"atom": "msgpass", "eig": "eigfeat", "diag": "diagfeat"}[node_feat]
    name = f"C3_{tag}_shuffled" if shuffled else f"C3_{tag}"
    row = score(ds, pred_z, name, config, seed, n_params,
                dict(hidden_width=d, node_feat=node_feat,
                     loss_first=loss_first, loss_last=loss_last))
    if return_pred:
        # R8 needs the per-molecule prediction in eV, on the same test rows and
        # in the same units as the circuit's -- exactly what `score` inverts.
        return row, ds.y_comp[ds.test_idx] + pred_z * ds.res_std + ds.res_mean
    return row


def load_learned_alpha(path, config, seed, arm="zz_bytype_r3"):
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r["config"] == config and int(r["seed"]) == seed
                    and r["arm"] == arm):
                out[int(r["type_id"])] = float(r["alpha_learned"])
    return out or None


def classical_job(config, seed, epochs, split, rounds, budget, alpha_csv, arm,
        types_from_train=False):
    nq = C.qubits_for(config)
    # The circuit and its classical equivalent must share the vocabulary rule;
    # tightening only one side would make condition (6) unfair.
    arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=0, train_var=False,
                  canonical_order=True, types_from_train=types_from_train)
    ds = C.Dataset(config, arch, split, seed)
    rows, alphas = [], []
    t0 = time.time()

    learned = load_learned_alpha(alpha_csv, config, seed, arm)
    if learned:
        for wt in ("J", "coulomb"):
            rows.append(c1_transplant(ds, config, seed, nq, learned, wt))

    r, a_learned = c2_end_to_end(ds, config, seed, nq, epochs=epochs)
    rows.append(r)
    for t, v in a_learned.items():
        # With a training-set vocabulary there is one extra slot at the end --
        # the fallback bucket for bond types unseen in training.  It has no key
        # and no training bonds, so it is labelled rather than indexed.
        if t < len(ds.type_keys):
            key = ds.type_keys[t]
            label, n_bonds = C.type_label(key), ds.type_count[key]
        else:
            label, n_bonds = "OTHER", 0
        alphas.append(dict(config=config, seed=seed, model="C2",
                           type_id=t, type=label, n_bonds=n_bonds,
                           alpha_learned=v))
    rows.append(c2_end_to_end(ds, config, seed, nq, epochs=epochs,
                              shuffled=True)[0])
    for nf in ("atom", "eig", "diag"):
        rows.append(c3_message_passing(ds, config, seed, nq, budget, rounds,
                                       epochs, node_feat=nf))
    rows.append(c3_message_passing(ds, config, seed, nq, budget, rounds, epochs,
                                   shuffled=True, node_feat="atom"))

    for r in rows:
        r["time_s"] = round(time.time() - t0, 1)
        print(f"    {config} s{seed} {r['arm']:22s} MAE {r['mae']:.3f}  "
              f"iso {r['isomer_mae']:.3f}  np {r['n_params']}", flush=True)
    return [("row", r) for r in rows] + [("alpha", a) for a in alphas]


def classical_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--budget", type=int, default=110,
                   help="parameter budget C3 must not exceed")
    p.add_argument("--alpha-csv", default=os.path.join(C.RESULTS,
                                                       "E1_alpha_learned.csv"))
    p.add_argument("--alpha-arm", default="zz_bytype_r3")
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--tag", default="")
    p.add_argument("--types-from-train", action="store_true",
                   help="build the bond-type vocabulary from the training "
                        "molecules only (removes the support leak)")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.epochs, a.split, a.rounds, a.budget, a.alpha_csv,
             a.alpha_arm, a.types_from_train)
            for c in configs for s in seeds]

    print("=" * 78)
    print(f"  E2 -- learnable-type-weight classical equivalents")
    print(f"  configs {configs}   seeds {len(seeds)}   budget {a.budget} params")
    print("=" * 78)
    tagged = C.parallel_jobs(classical_job, args, a.jobs, desc="E2")
    rows = [r for k, r in tagged if k == "row"]
    alpha_rows = [r for k, r in tagged if k == "alpha"]
    C.write_csv(os.path.join(a.out, f"E2_classical_equivalent{a.tag}.csv"), rows,
                ["config", "seed", "arm", "n_params", "hidden_width",
                 "n_train", "n_test", "mae", "r2", "isomer_mae",
                 "isomer_mae_tr", "isomer_rho", "n_isomer_groups",
                 "loss_first", "loss_last", "alpha_mean", "alpha_std", "time_s"])
    C.write_csv(os.path.join(a.out, f"E2_alpha_classical{a.tag}.csv"), alpha_rows)

    rows_by_cell = defaultdict(list)
    for r in rows:
        rows_by_cell[(r["config"], r["arm"])].append(r["mae"])
    print("\n" + "=" * 78)
    print("  E2 -- classical equivalents")
    print("=" * 78)
    for c in configs:
        print(f"\n  [{c}]  {'arm':<24}{'MAE':>18}{'n_params':>10}")
        for k in sorted(rows_by_cell, key=lambda k: np.mean(rows_by_cell[k])):
            if k[0] != c:
                continue
            v = np.array(rows_by_cell[k])
            npar = [r["n_params"] for r in rows
                    if r["config"] == c and r["arm"] == k[1]]
            print(f"  {'':<6}{k[1]:<24}{v.mean():>10.3f} +-{v.std():<6.3f}"
                  f"{int(np.mean(npar)):>10}")
        print(f"\n  {'':<6}{'condition (4) for the classical models':<44}")
        for base, shuffled_arm in (("C2_end2end", "C2_shuffled"),
                                   ("C3_msgpass", "C3_msgpass_shuffled")):
            A, B = rows_by_cell.get((c, base), []), rows_by_cell.get((c, shuffled_arm), [])
            if A and B:
                paired_stats = C.paired_test(A, B)
                ok = paired_stats["p"] == paired_stats["p"] and paired_stats["p"] < 0.05 and paired_stats["rel"] < 0
                print(f"  {'':<6}{base + ' vs ' + shuffled_arm:<34}{paired_stats['rel']:>+8.1%}"
                      f"{paired_stats['wins']:>5}/{paired_stats['n']}{paired_stats['p']:>9.4f}"
                      f"  -> {'uses chemistry' if ok else 'does not'}")
    print("\n  Join against the circuit with E1_sevenconfig.csv "
          "(same configs, splits and seeds).")


# ============================================================================
# schnet
# ============================================================================
sys.path.append(C.PAPER)


def count_params(model_name, hidden, n_basis, n_inter, cutoff):
    import torch
    from . import _gnn as G
    cls = G.SchNet if model_name == "schnet" else G.PaiNN
    m = cls(hidden=hidden, n_basis=n_basis, n_inter=n_inter, cutoff=cutoff)
    return int(sum(p.numel() for p in m.parameters()))


def schnet_job(config, seed, model_name, epochs, hidden, n_inter, split, device):
    import torch
    from . import _gnn as G
    from sklearn.metrics import mean_absolute_error, r2_score

    mols = C.load_config(config)
    tr, te = C.make_split(mols, split, seed)
    np.random.seed(seed); torch.manual_seed(seed)

    t0 = time.time()
    preds, _ = G.train_gnn(mols, tr, te, model_name=model_name, epochs=epochs,
                           batch_size=32, lr=1e-3, hidden=hidden, n_basis=32,
                           n_inter=n_inter, cutoff=5.0, residual=True,
                           device=device, verbose=False)
    elapsed = time.time() - t0

    y = np.array([m["eAT"] for m in mols])
    arch = C.Arch(n_qubits=C.qubits_for(config), epochs=0, train_var=False)
    ds = C.Dataset(config, arch, split, seed)   # only for the isomer metrics
    row = dict(config=config, seed=seed, arm=f"{model_name}_h{hidden}_l{n_inter}",
               model=model_name, hidden=hidden, n_inter=n_inter, epochs=epochs,
               split=split, n_train=len(tr), n_test=len(te),
               n_params=count_params(model_name, hidden, 32, n_inter, 5.0),
               mae=float(mean_absolute_error(y[te], preds)),
               r2=float(r2_score(y[te], preds)), time_s=round(elapsed, 1))
    row.update(C.isomer_metrics(ds, te, preds))
    print(f"    {config} s{seed} {row['arm']:16s} MAE {row['mae']:.3f}  "
          f"iso {row['isomer_mae']:.3f}  np {row['n_params']}  ({elapsed:.0f}s)",
          flush=True)
    return [row]


def schnet_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="A1,A2,B1,B2,B3,B4,B5")
    p.add_argument("--models", default="schnet")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-inter", type=int, default=3)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--split", default="molecule")
    p.add_argument("--device", default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, m, a.epochs, a.hidden, a.n_inter, a.split, a.device)
            for m in a.models.split(",") for c in configs for s in seeds]

    print("=" * 78)
    print(f"  H3 -- GNN baseline on the circuit's molecule-level splits")
    print(f"  models {a.models}  hidden {a.hidden}  layers {a.n_inter}  "
          f"epochs {a.epochs}  seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(schnet_job, args, a.jobs, desc="H3")
    C.write_csv(os.path.join(a.out, "H3_gnn_baseline.csv"), rows)

    import csv, glob
    from collections import defaultdict
    circuit = defaultdict(dict)
    for f in glob.glob(os.path.join(a.out, "E1_sevenconfig*canon*.csv")):
        for r in csv.DictReader(open(f)):
            if r.get("mae") and r["arm"] == "zz_bytype_r3":
                circuit[r["config"]][int(r["seed"])] = float(r["mae"])
                if r.get("isomer_mae"):
                    circuit[r["config"] + "|iso"][int(r["seed"])] = float(r["isomer_mae"])
                if r.get("n_params"):
                    circuit[r["config"] + "|np"][int(r["seed"])] = float(r["n_params"])

    rows_by_cell = defaultdict(list)
    for r in rows:
        rows_by_cell[(r["config"], r["arm"])].append(r)
    print("\n" + "=" * 78)
    print("  H3 -- GNN vs the circuit, same splits and seeds")
    print("=" * 78)
    print(f"  {'config':<8}{'GNN MAE':>10}{'circuit':>10}{'ratio':>9}"
          f"{'GNN iso':>10}{'circ iso':>10}{'GNN np':>10}{'circ np':>9}")
    for c in configs:
        for k in sorted({k[1] for k in rows_by_cell if k[0] == c}):
            cell_rows = rows_by_cell[(c, k)]
            gnn_mae = np.mean([r["mae"] for r in cell_rows])
            gnn_iso = np.nanmean([r["isomer_mae"] for r in cell_rows])
            gnn_np = int(np.mean([r["n_params"] for r in cell_rows]))
            circ_mae = np.mean(list(circuit.get(c, {}).values())) if circuit.get(c) else np.nan
            circ_iso = np.nanmean(list(circuit.get(c + "|iso", {}).values())) if circuit.get(c + "|iso") else np.nan
            circ_np = int(np.mean(list(circuit.get(c + "|np", {}).values()))) if circuit.get(c + "|np") else 0
            print(f"  {c:<8}{gnn_mae:>10.3f}{circ_mae:>10.3f}{circ_mae / gnn_mae:>9.2f}x"
                  f"{gnn_iso:>10.3f}{circ_iso:>10.3f}{gnn_np:>10d}{circ_np:>9d}")
    print("\n  'ratio' is circuit MAE / GNN MAE -- how many times the GNN's error")
    print("  the circuit makes, at 1/1000 the parameters.  Report both numbers.")


# ============================================================================
# tuning
# ============================================================================
def inner_split(ds, frac=0.75, seed=0):
    """A validation fold carved out of the TRAINING molecules only."""
    tr = np.asarray(ds.train_idx)
    groups = C.group_labels([ds.mols[i] for i in tr], "molecule")
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - frac, random_state=seed)
    fit_pos, val_pos = next(gss.split(np.arange(len(tr)), groups=groups))
    return tr[fit_pos], tr[val_pos]


def circuit_grid():
    return [dict(n_rounds=r, lr=lr, epochs=ep)
            for r in (3, 4, 5) for lr in (0.02, 0.05, 0.1) for ep in (100, 300)]


def c3_grid():
    return [dict(rounds=r, lr=lr, epochs=ep)
            for r in (2, 3, 4) for lr in (0.02, 0.05, 0.1) for ep in (100, 300)]


def _trainer(ds, arch, seed, device):
    """CPU adjoint path or the batched GPU path.  selftest T8/T9 require the
    two to agree to 1e-14 on gradients and ~1e-10 on the reported MAE, so the
    choice is a speed decision, not a numerical one."""
    if device == "cuda":
        from .. import _gpu as gpu_sim
        return gpu_sim.GPUTrainer(ds, arch, seed=seed)
    return C.Trainer(ds, arch, seed=seed)


def eval_circuit(config, seed, split, hparams, tr_idx, va_idx, nq, device="cpu"):
    """Fit on tr_idx, score on va_idx, without touching the real test split."""
    from sklearn.metrics import mean_absolute_error
    arch = C.Arch(n_qubits=nq, entangler="zz", alpha=0.1, alpha_mode="bytype",
                  readout_scaler=True, canonical_order=True,
                  types_from_train=True, **hparams)
    ds = C.Dataset(config, arch, split, seed)
    ds.train_idx, ds.test_idx = tr_idx, va_idx
    t = _trainer(ds, arch, seed, device).fit()
    pred = t.predict_eV(va_idx)
    return float(mean_absolute_error(ds.y_raw[va_idx], pred)), arch


def run_circuit(config, seed, split, arch, device="cpu"):
    """Train on the real training split and score on the real test split."""
    from sklearn.metrics import mean_absolute_error
    ds = C.Dataset(config, arch, split, seed)
    t = _trainer(ds, arch, seed, device).fit()
    pred = t.predict_eV(ds.test_idx)
    n_par = (arch.n_var_params() * int(arch.train_var)
             + (ds.n_types if arch.alpha_mode != "fixed" else 0)
             + arch.n_features() + 1)
    return float(mean_absolute_error(ds.y_raw[ds.test_idx], pred)), n_par


def tuning_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3,A1")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--split", default="molecule")
    p.add_argument("--models", default="circuit,c3,schnet")
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    models = a.models.split(",")
    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.split, models, a.device) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  N3 -- matched hyper-parameter search, selection by inner CV only")
    print(f"  models {models}   configs {configs}   seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(tuning_job, args, a.jobs, desc="N3")
    C.write_csv(os.path.join(a.out, f"N3_baseline_tuning{a.tag}.csv"), rows)

    rows_by_cell = {}
    for r in rows:
        rows_by_cell.setdefault((r["config"], r["model"]), []).append(r)
    print("\n" + "=" * 78)
    print("  N3 -- selected configuration and its test MAE")
    print("=" * 78)
    print(f"  {'config':<8}{'model':<10}{'test MAE (tuned)':>19}"
          f"{'test MAE (default)':>21}{'change':>10}")
    for c in configs:
        for m in models:
            cell_rows = rows_by_cell.get((c, m), [])
            if not cell_rows:
                continue
            tuned = np.mean([r["mae_tuned"] for r in cell_rows])
            default = np.mean([r["mae_default"] for r in cell_rows])
            print(f"  {c:<8}{m:<10}{tuned:>13.3f} +-{np.std([r['mae_tuned'] for r in cell_rows]):<4.3f}"
                  f"{default:>15.3f} +-{np.std([r['mae_default'] for r in cell_rows]):<4.3f}"
                  f"{(tuned - default) / default:>+10.1%}")
        print()
    print("=" * 78)
    print("  Conditions (6) and (7) with BOTH sides tuned")
    print("=" * 78)
    for c in configs:
        q = {r["seed"]: r["mae_tuned"] for r in rows_by_cell.get((c, "circuit"), [])}
        for m, lab in (("c3", "(6b) vs C3_eigfeat"), ("schnet", "(7) vs SchNet")):
            other = {r["seed"]: r["mae_tuned"] for r in rows_by_cell.get((c, m), [])}
            s = sorted(set(q) & set(other))
            if len(s) < 3:
                continue
            paired_stats = C.paired_test(np.array([q[i] for i in s]),
                                         np.array([other[i] for i in s]))
            flag = " *" if paired_stats["p"] < 0.05 and paired_stats["rel"] < 0 else ""
            print(f"  {c:<6}{lab:<22}{paired_stats['ma']:>8.3f}{paired_stats['mb']:>9.3f}"
                  f"{paired_stats['rel']:>+9.1%}{paired_stats['wins']:>5}/{len(s)}{paired_stats['p']:>9.4f}{flag}")


def tuning_job(config, seed, split, models, device="cpu"):
    from sklearn.metrics import mean_absolute_error
    nq = C.qubits_for(config)
    rows = []
    base = C.Arch(n_qubits=nq, n_rounds=3, epochs=100, entangler="zz",
                  alpha=0.1, alpha_mode="bytype", readout_scaler=True,
                  canonical_order=True, types_from_train=True)
    ds0 = C.Dataset(config, base, split, seed)
    tr_in, va = inner_split(ds0, seed=seed)
    t0 = time.time()

    if "circuit" in models:
        best, best_hp = np.inf, None
        for hparams in circuit_grid():
            try:
                v, _ = eval_circuit(config, seed, split, hparams, tr_in, va, nq, device)
            except Exception:
                continue
            if v < best:
                best, best_hp = v, hparams
        arch = C.Arch(n_qubits=nq, entangler="zz", alpha=0.1,
                      alpha_mode="bytype", readout_scaler=True,
                      canonical_order=True, types_from_train=True, **best_hp)
        mae_tuned, n_par = run_circuit(config, seed, split, arch, device)
        mae_default, _ = run_circuit(config, seed, split, base, device)
        rows.append(dict(config=config, seed=seed, model="circuit",
                         selected=str(best_hp), inner_mae=best,
                         mae_tuned=mae_tuned, mae_default=mae_default,
                         n_params=n_par, time_s=round(time.time() - t0, 1)))
        print(f"    {config} s{seed} circuit  {best_hp}  "
              f"tuned {mae_tuned:.3f} vs default {mae_default:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    if "c3" in models:
        budget = 116
        best, best_hp = np.inf, None
        ds_val = C.Dataset(config, base, split, seed)
        ds_val.train_idx, ds_val.test_idx = tr_in, va
        for hparams in c3_grid():
            try:
                r = c3_message_passing(ds_val, config, seed, nq, budget=budget,
                                          node_feat="eig", **hparams)
            except Exception:
                continue
            if r["mae"] < best:
                best, best_hp = r["mae"], hparams
        r = c3_message_passing(ds0, config, seed, nq, budget=budget,
                                  node_feat="eig", **best_hp)
        d = c3_message_passing(ds0, config, seed, nq, budget=budget,
                                  node_feat="eig", rounds=3, epochs=100, lr=0.05)
        rows.append(dict(config=config, seed=seed, model="c3",
                         selected=str(best_hp), inner_mae=best,
                         mae_tuned=r["mae"], mae_default=d["mae"],
                         n_params=r["n_params"], time_s=round(time.time() - t0, 1)))
        print(f"    {config} s{seed} c3       {best_hp}  "
              f"tuned {r['mae']:.3f} vs default {d['mae']:.3f}", flush=True)

    if "schnet" in models:
        sys.path.append(C.PAPER)
        from . import _gnn as G
        y = np.array([m["eAT"] for m in ds0.mols])
        best, best_hp = np.inf, None
        for hparams in [dict(hidden=h, n_inter=l, lr=lr, epochs=ep)
                   for h in (32, 64, 128) for l in (2, 3, 4)
                   for lr in (1e-3, 5e-4) for ep in (300,)]:
            try:
                pr, _ = G.train_gnn(ds0.mols, tr_in, va, model_name="schnet",
                                    batch_size=32, n_basis=32, cutoff=5.0,
                                    residual=True, verbose=False, **hparams)
            except Exception:
                continue
            v = float(mean_absolute_error(y[va], pr))
            if v < best:
                best, best_hp = v, hparams
        pr, _ = G.train_gnn(ds0.mols, ds0.train_idx, ds0.test_idx,
                            model_name="schnet", batch_size=32, n_basis=32,
                            cutoff=5.0, residual=True, verbose=False, **best_hp)
        pred_default, _ = G.train_gnn(ds0.mols, ds0.train_idx, ds0.test_idx,
                                      model_name="schnet", batch_size=32, n_basis=32,
                                      cutoff=5.0, residual=True, verbose=False,
                                      hidden=64, n_inter=3, lr=1e-3, epochs=300)
        te = ds0.test_idx
        rows.append(dict(config=config, seed=seed, model="schnet",
                         selected=str(best_hp), inner_mae=best,
                         mae_tuned=float(mean_absolute_error(y[te], pr)),
                         mae_default=float(mean_absolute_error(y[te], pred_default)),
                         n_params=0, time_s=round(time.time() - t0, 1)))
        print(f"    {config} s{seed} schnet   {best_hp}  "
              f"tuned {rows[-1]['mae_tuned']:.3f} vs "
              f"default {rows[-1]['mae_default']:.3f}", flush=True)
    return rows
