"""Where does the gain come from?

grouping          What the parameter-sharing groups are indexed by: bonds are
                  regrouped by chemistry (element pair + bond order), element
                  pair only, bond length only, or arbitrarily.
grouping_verdict  The decision rules for `grouping`, fixed before the run.
chemistry         Whether the learned coefficients line up with
                  electronegativity difference and bond enthalpy, which the
                  model is never given; a partial correlation removes the
                  nuclear-charge prefactor.
bondcut           Sensitivity to the bond-order thresholds that define the
                  types, with a randomized control at each cut.
bondcut_cv        The thresholds chosen by cross-validation inside the
                  training split.
"""
import argparse, collections, csv, os, time
import numpy as np
from scipy import stats

from .. import core as C
from ..core import RESULTS


# ============================================================================
# grouping
# ============================================================================
MODE = {"chem": "chem", "elem": "elem", "arb": "arb", "lenq": "lenq",
        "bijection": "bijection"}


def grouping_job(config, seed, arms, epochs, rounds, split, lenq_bins):
    nq = C.qubits_for(config)
    rows = []
    for arm in arms:
        t0 = time.time()
        arch = C.Arch(n_qubits=nq, entangler="zz", alpha=0.1,
                      alpha_mode="bytype", n_rounds=rounds, epochs=epochs,
                      readout_scaler=True, canonical_order=True,
                      types_from_train=True, group_mode=MODE[arm],
                      lenq_bins=lenq_bins)
        row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                               readouts=("ridge", "krr"),
                               extra=dict(arm=arm, kind="quantum"))
        ds = tr.ds
        row.update(group_mode=MODE[arm], n_groups=ds.n_types_train,
                   group_size_dist=ds.group_size_dist,
                   lenq_bins=int(arch.lenq_bins),
                   group_seed_offset=int(arch.group_seed_offset),
                   n_len_cuts=(0 if ds.len_cuts is None else len(ds.len_cuts)))
        rows.append(row)
        print(f"    {config} s{seed} {arm:<10} MAE {row['mae']:.4f}  "
              f"groups {ds.n_types_train:3d}  sizes {ds.group_size_dist[:34]:<34} "
              f"drop {row['loss_drop']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


def grouping_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="A1,A2,B1,B2,B3,B4,B5")
    p.add_argument("--arms", default="elem,arb")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--lenq-bins", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)
    arms = a.arms.split(",")
    bad = [x for x in arms if x not in MODE]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(MODE)}")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, arms, a.epochs, a.rounds, a.split, a.lenq_bins)
            for c in a.configs.split(",") for s in seeds]
    print("=" * 78)
    print(f"  R10 -- grouping control   configs {a.configs}   arms {arms}   "
          f"seeds {len(seeds)}   {len(args)} jobs")
    print("=" * 78)
    rows = C.parallel_jobs(grouping_job, args, a.jobs, desc="R10")
    out = os.path.join(a.out, f"R10_grouping{a.tag}.csv")
    C.write_csv(out, rows)
    print("\nwrote", out, len(rows), "rows")


# ============================================================================
# grouping_verdict
# ============================================================================
CONFIGS = ["A1", "A2", "B1", "B2", "B3", "B4", "B5"]


# existing arms: (file, arm name) per config
OLD = {c: ("E1_sevenconfig_tft.csv", "E1_sevenconfig_zzshuf.csv")
       for c in ["A1", "B1", "B2", "B3", "B4"]}


OLD["A2"] = ("E1_sevenconfig_A2tft.csv", "E1_sevenconfig_A2zzshuf.csv")


OLD["B5"] = ("E1_sevenconfig_b5s40.csv", "E1_sevenconfig_b5s40zzshuf.csv")


def load(files):
    out = {}
    for fn in files:
        p = os.path.join(C.RESULTS, fn)
        if not os.path.exists(p):
            print(f"  !! missing {fn}"); continue
        for r in csv.DictReader(open(p)):
            if r.get("mae") in (None, "", "nan"):
                continue
            out.setdefault((r["config"], r["arm"]), {})[int(r["seed"])] = float(r["mae"])
    return out


def paired(mae_table, cfg, a, b):
    A, B = mae_table.get((cfg, a)), mae_table.get((cfg, b))
    if not A or not B:
        return None
    s = sorted(set(A) & set(B))
    if len(s) < 2:
        return None
    x = np.array([A[i] for i in s]); y = np.array([B[i] for i in s])
    st = C.paired_test(x, y); st["n"] = len(s)
    st["mean_a"], st["mean_b"] = x.mean(), y.mean()
    return st


def verdict_main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="R10_grouping.csv")
    ap.add_argument("--extra", default="")
    a = ap.parse_args(argv)
    files = [a.inp] + [x for x in a.extra.split(",") if x]
    files += sorted({f for pair in OLD.values() for f in pair})
    mae_table = load(files)
    arm_rename = {"zz_bytype_r3": "chem_true", "zz_shuffled_r3": "shuffled"}
    for (cfg, arm), v in list(mae_table.items()):
        if arm in arm_rename:
            mae_table[(cfg, arm_rename[arm])] = v

    # ---- null control
    print("=" * 96)
    print("  NULL CONTROL -- bijection must reproduce chem (hypothesis space unchanged)")
    print("=" * 96)
    for c in CONFIGS:
        st = paired(mae_table, c, "bijection", "chem")
        if st:
            ok = abs(st["mean_a"] - st["mean_b"]) < 5e-4
            print(f"  {c:<4} bijection {st['mean_a']:.4f}  chem {st['mean_b']:.4f}  "
                  f"|d| {abs(st['mean_a']-st['mean_b']):.2e}  n={st['n']}  "
                  f"{'PASS' if ok else 'FAIL'}")

    # ---- experiment A
    print("\n" + "=" * 96)
    print("  EXPERIMENT A -- chemical merge (elem) vs arbitrary merge (arb)")
    print("=" * 96)
    print(f"  {'config':<7}{'n':>3}{'elem':>9}{'arb':>9}{'rel(arb)':>11}"
          f"{'p':>9}{'groups e/a':>13}")
    n_rel_ge_10pct = n_rel_lt_5pct = n_configs = 0
    for c in CONFIGS:
        st = paired(mae_table, c, "arb", "elem")
        if not st:
            print(f"  {c:<7}  (missing)"); continue
        n_configs += 1
        rel = (st["mean_a"] - st["mean_b"]) / st["mean_b"]
        n_rel_ge_10pct += rel >= 0.10
        n_rel_lt_5pct += abs(rel) < 0.05
        groups_elem = groups_arb = "-"
        for fn in [a.inp] + [x for x in a.extra.split(",") if x]:
            p = os.path.join(C.RESULTS, fn)
            if not os.path.exists(p):
                continue
            for r in csv.DictReader(open(p)):
                if r["config"] == c and r.get("group_mode") == "elem":
                    groups_elem = r.get("n_groups", "-")
                if r["config"] == c and r.get("group_mode") == "arb":
                    groups_arb = r.get("n_groups", "-")
        print(f"  {c:<7}{st['n']:>3}{st['mean_b']:>9.4f}{st['mean_a']:>9.4f}"
              f"{rel:>+11.1%}{st['p_wilcoxon']:>9.4f}{f'{groups_elem}/{groups_arb}':>13}")
    if n_configs:
        v = ("SUCCESS -- the grouping must be chemical" if n_rel_ge_10pct >= 5 else
             "FAILURE -- only the consistency of sharing matters" if n_rel_lt_5pct >= 5 else
             "PARTIAL -- report both, withhold interpretation")
        print(f"\n  rel >= +10%: {n_rel_ge_10pct}/{n_configs}   |rel| < 5%: {n_rel_lt_5pct}/{n_configs}   VERDICT: {v}")

    # ---- experiment B
    print("\n" + "=" * 96)
    print("  EXPERIMENT B -- chemistry vs geometry   gain = (shuffled - arm) / arm")
    print("=" * 96)
    print(f"  {'config':<7}{'shuffled':>10}{'chem':>9}{'lenq':>9}"
          f"{'gain(chem)':>12}{'gain(lenq)':>12}{'ratio':>8}")
    ratios = []
    for c in CONFIGS:
        gains = {}
        for arm in ("chem_true", "lenq"):
            st = paired(mae_table, c, "shuffled", arm)
            gains[arm] = None if st is None else (
                (st["mean_a"] - st["mean_b"]) / st["mean_b"], st["mean_b"], st["mean_a"])
        if not gains["chem_true"] or not gains["lenq"]:
            print(f"  {c:<7}  (missing)"); continue
        gain_chem, mae_chem, mae_shuffled = gains["chem_true"]; gain_lenq, mae_lenq, _ = gains["lenq"]
        r = gain_lenq / gain_chem if gain_chem else float("nan")
        ratios.append(r)
        print(f"  {c:<7}{mae_shuffled:>10.4f}{mae_chem:>9.4f}{mae_lenq:>9.4f}"
              f"{gain_chem:>+12.1%}{gain_lenq:>+12.1%}{r:>8.2f}")
    if ratios:
        m = float(np.median(ratios))
        v = ("CHEMISTRY" if m <= 0.30 else "GEOMETRY" if m >= 0.70 else "MIXED")
        print(f"\n  median gain(lenq)/gain(chem) = {m:.2f}   VERDICT: {v}")


# ============================================================================
# chemistry
# ============================================================================
CFG = ["A1", "A2", "B1", "B2", "B3", "B4", "B5"]


ARM = "zz_bytype_r3"


MIN_BONDS = 10


# Pauling electronegativity
EN = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "S": 2.58, "Cl": 3.16}


# nuclear charge
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Cl": 17}


# mean bond enthalpy, kJ/mol; None where no dependable tabulated value exists
BDE = {
    "H-C": 413, "H-N": 391, "H-O": 463,
    "C-C": 348, "C-N": 293, "C-O": 358, "C-S": 259, "C-Cl": 328,
    "C=C": 614, "C=N": 615, "C=O": 799, "C=S": 573,
    "C#C": 839, "C#N": 891, "C#O": 1072,
    "N-N": 163, "N-O": 201, "N-S": None,
    "N=N": 418, "N=O": 607,
    "N#N": 945,
    "O-S": None, "O=S": 522, "O#S": None,
}


def split_type(t):
    """'C=O' -> ('C', 'O'). The separator encodes the bond-order proxy."""
    for s in "-=#":
        if s in t:
            a, b = t.split(s)
            return a, b
    raise ValueError(t)


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with the rank of z removed from both.

    Ranks are taken first, then z's rank is regressed out of the other two by
    ordinary least squares and the correlation of the residuals is returned.
    The p value is Student's t on n - 3 degrees of freedom, the usual test for
    a first-order partial correlation.
    """
    n = len(x)
    if n < 5:
        return float("nan"), float("nan"), n
    rx, ry, rz = (stats.rankdata(v).astype(float) for v in (x, y, z))
    A = np.c_[np.ones(n), rz]

    def resid(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]

    r = float(np.corrcoef(resid(rx), resid(ry))[0, 1])
    r = min(max(r, -0.999999), 0.999999)
    t = r * np.sqrt((n - 3) / (1 - r * r))
    p = float(2 * stats.t.sf(abs(t), n - 3))
    return r, p, n


# The two files the rest of the manuscript build reads for the learned
# coefficients: the types-from-train runs, six configurations plus A2. Other
# E1_alpha_learned_*.csv files in a working results directory belong to other
# protocols (canonical ordering, the 40-seed B5 sweep) and must not be pooled
# with these, so the set is named rather than globbed.
ALPHA_FILES = ("E1_alpha_learned_tft.csv", "E1_alpha_learned_A2tft.csv")


def load_alpha(res):
    """mean learned coefficient and mean occurrence count, per config and type."""
    alpha_lists = collections.defaultdict(lambda: collections.defaultdict(list))
    nbond_lists = collections.defaultdict(lambda: collections.defaultdict(list))
    files = [os.path.join(res, n) for n in ALPHA_FILES
             if os.path.exists(os.path.join(res, n))]
    if not files:
        raise SystemExit(f"none of {ALPHA_FILES} under {res}")
    for f in files:
        for r in csv.DictReader(open(f)):
            if r.get("arm") != ARM:
                continue
            try:
                alpha_lists[r["config"]][r["type"]].append(float(r["alpha_learned"]))
                nbond_lists[r["config"]][r["type"]].append(float(r["n_bonds"]))
            except (ValueError, KeyError):
                pass
    return ({c: {t: float(np.mean(v)) for t, v in d.items()} for c, d in alpha_lists.items()},
            {c: {t: float(np.mean(v)) for t, v in d.items()} for c, d in nbond_lists.items()})


def vocabulary(alpha, n_bonds, cfg, prop, floor, common_only, common):
    """The types entering one configuration's correlation, after the filters."""
    out = []
    for t in sorted(alpha.get(cfg, {})):
        if common_only and t not in common:
            continue
        if n_bonds[cfg].get(t, 0) < floor:
            continue
        if prop == "bde" and BDE.get(t) is None:
            continue
        out.append(t)
    return out


def one_cell(alpha, n_bonds, prop, floor, common_only, common):
    """Per-configuration partial correlations for one setting, plus Fisher."""
    rows, ps = [], []
    for c in CFG:
        types = vocabulary(alpha, n_bonds, c, prop, floor, common_only, common)
        if len(types) < 5:
            rows.append((c, len(types), float("nan"), float("nan")))
            continue
        alpha_vals = [alpha[c][t] for t in types]
        pref, prop_vals = [], []
        for t in types:
            e1, e2 = split_type(t)
            pref.append(Z[e1] * Z[e2] / (Z[e1] + Z[e2]))
            prop_vals.append(abs(EN[e1] - EN[e2]) if prop == "en" else BDE[t])
        r, p, n = partial_spearman(alpha_vals, prop_vals, pref)
        rows.append((c, n, r, p))
        if p == p:
            ps.append(p)
    fisher = float(stats.combine_pvalues(ps, method="fisher")[1]) if len(ps) > 1 else float("nan")
    valid_rhos = [r for _, _, r, _ in rows if r == r]
    return rows, float(np.mean(valid_rhos)) if valid_rhos else float("nan"), fisher, sum(r < 0 for r in valid_rhos), len(valid_rhos)


def raw_prefactor_correlation(alpha, n_bonds, floor, common_only, common):
    """Do the coefficients simply invert the fixed prefactor? (uncorrected)"""
    rows, ps = [], []
    for c in CFG:
        types = vocabulary(alpha, n_bonds, c, "en", floor, common_only, common)
        if len(types) < 5:
            continue
        alpha_vals = [alpha[c][t] for t in types]
        pref = []
        for t in types:
            e1, e2 = split_type(t)
            pref.append(Z[e1] * Z[e2] / (Z[e1] + Z[e2]))
        r = stats.spearmanr(alpha_vals, pref)
        rows.append((c, len(types), float(r.statistic), float(r.pvalue)))
        ps.append(float(r.pvalue))
    fisher = float(stats.combine_pvalues(ps, method="fisher")[1]) if len(ps) > 1 else float("nan")
    return rows, float(np.mean([x[2] for x in rows])), fisher


def chemistry_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="mgqc run chemistry",
        description="Partial rank correlation of the learned coefficients "
                    "against electronegativity difference and mean bond "
                    "enthalpy, with the fixed prefactor partialled out.")
    ap.add_argument("--results", default=RESULTS,
                    help="directory holding E1_alpha_learned*.csv")
    ap.add_argument("--out", default=None, help="output CSV (default: RESULTS/R11_chemistry.csv)")
    ap.add_argument("--min-bonds", type=int, default=MIN_BONDS,
                    help=f"primary occurrence floor (default {MIN_BONDS})")
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.results, "R11_chemistry.csv")

    alpha, n_bonds = load_alpha(a.results)
    have = [c for c in CFG if c in alpha]
    common = set.intersection(*[set(alpha[c]) for c in have]) if have else set()

    settings = [("per-config", a.min_bonds, False), ("per-config", 0, False),
                ("per-config", 20, False), ("common-9", 0, True)]
    primary = settings[0]

    print(f"\n  learned coefficients: {len(have)} configurations, "
          f"{len(set().union(*[set(alpha[c]) for c in have]))} bond types, "
          f"{len(common)} common to all\n")
    print("  prefactor partialled out: Z_i Z_j / (Z_i + Z_j)")
    print(f"  primary setting: per-configuration vocabulary, "
          f"occurrence floor {a.min_bonds}\n")

    records = []
    for prop, label in (("en", "electronegativity difference"),
                        ("bde", "mean bond enthalpy")):
        print(f"  {'=' * 84}\n  {label}\n  {'=' * 84}")
        for vocab, floor, common_only in settings:
            rows, mean_rho, fisher_p, neg, n_configs = one_cell(alpha, n_bonds, prop, floor, common_only, common)
            tag = "primary" if (vocab, floor, common_only) == primary else ""
            name = f"{vocab}, floor {floor}"
            print(f"\n  {name:<26s} mean partial rho = {mean_rho:+.3f}   "
                  f"Fisher p = {fisher_p:.2e}   negative in {neg}/{n_configs}  {tag}")
            print("    " + "  ".join(f"{c}" for c in CFG))
            print("    n     " + " ".join(f"{n:>5d}" if n else "    -" for _, n, _, _ in rows))
            print("    rho   " + " ".join(f"{r:+.2f}" if r == r else "    -" for _, _, r, _ in rows))
            print("    p     " + " ".join(f"{p:.3f}" if p == p else "    -" for _, _, _, p in rows))
            for c, n, r, p in rows:
                records.append(dict(property=prop, vocabulary=vocab, min_bonds=floor,
                                    config=c, n_types=n, partial_rho=r, p=p,
                                    mean_partial_rho=mean_rho, fisher_p=fisher_p,
                                    n_negative=neg, n_configs=n_configs,
                                    primary=int((vocab, floor, common_only) == primary)))

    print(f"\n  {'=' * 84}\n  control: coefficient against the fixed prefactor, "
          f"uncorrected\n  {'=' * 84}")
    rows, mean_rho, fisher_p = raw_prefactor_correlation(alpha, n_bonds, a.min_bonds, False, common)
    print(f"\n  mean Spearman rho = {mean_rho:+.3f}   Fisher p = {fisher_p:.2e}")
    print("    " + " ".join(f"{c}:{r:+.2f}" for c, _, r, _ in rows))
    for c, n, r, p in rows:
        records.append(dict(property="prefactor", vocabulary="per-config",
                            min_bonds=a.min_bonds, config=c, n_types=n,
                            partial_rho=r, p=p, mean_partial_rho=mean_rho, fisher_p=fisher_p,
                            n_negative=sum(x[2] < 0 for x in rows), n_configs=len(rows),
                            primary=1))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(f"\n  -> {out}  ({len(records)} rows)\n")
    return 0


# ============================================================================
# bondcut
# ============================================================================
CUTS = [("m06", (0.78, 0.89)), ("m03", (0.81, 0.92)), ("base", (0.84, 0.95)),
        ("p03", (0.87, 0.98)), ("p06", (0.90, 1.01))]


def bondcut_job(config, seed, split, epochs, rounds):
    nq = C.qubits_for(config)
    rows = []
    settings = [(n, cut, "elem_order") for n, cut in CUTS] +\
               [("elem_only", (0.84, 0.95), "elem")]
    for name, cut, scheme in settings:
        C._ORDER_CUTS = cut
        cache = {}
        mae_by_arm = {}
        for arm, mode in (("bytype", "bytype"), ("shuffled", "bytype_shuffled"),
                          ("none", "fixed")):
            arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs,
                          entangler=("none" if arm == "none" else "zz"),
                          alpha=(0.0 if arm == "none" else 0.1),
                          alpha_mode=mode, readout_scaler=True,
                          canonical_order=True, types_from_train=True,
                          bond_type_scheme=scheme)
            row, tr, _ = C.run_arm(config, arch, seed=seed, split_level=split,
                                   ds_cache=cache,
                                   extra=dict(arm=arm, cut_name=name,
                                              cut_lo=cut[0], cut_hi=cut[1],
                                              scheme=scheme))
            rows.append(row); mae_by_arm[arm] = row["mae"]
            n_types = row["n_types_train"]
        print(f"    {config} s{seed} {name:10s} {scheme:10s} types {n_types:>3} "
              f"bytype {mae_by_arm['bytype']:.3f}  shuffled {mae_by_arm['shuffled']:.3f}  "
              f"none {mae_by_arm['none']:.3f}", flush=True)
    C._ORDER_CUTS = (0.84, 0.95)
    return rows


def bondcut_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3,A1")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=15)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.split, a.epochs, a.rounds) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  N2 -- bond-order threshold sensitivity   configs {configs}   "
          f"seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(bondcut_job, args, a.jobs, desc="N2")
    C.write_csv(os.path.join(a.out, "N2_bondcut_sensitivity.csv"), rows,
                C.COMMON_COLUMNS + ["arm", "cut_name", "cut_lo", "cut_hi",
                                    "scheme"])

    rows_by_cell = {}
    for r in rows:
        rows_by_cell.setdefault((r["config"], r["cut_name"], r["arm"]), []).append(r)
    names = [n for n, _ in CUTS] + ["elem_only"]
    print("\n" + "=" * 78)
    print("  N2 -- MAE and condition (4) under each threshold")
    print("=" * 78)
    for c in configs:
        base = np.array([r["mae"] for r in rows_by_cell.get((c, "base", "bytype"), [])])
        print(f"\n  [{c}]  {'cut':<11}{'n_types':>9}{'bytype MAE':>14}"
              f"{'vs base':>10}{'(4) vs shuffled':>18}{'p':>9}")
        for n in names:
            mae_bytype = np.array([r["mae"] for r in rows_by_cell.get((c, n, "bytype"), [])])
            mae_shuffled = np.array([r["mae"] for r in rows_by_cell.get((c, n, "shuffled"), [])])
            if not len(mae_bytype):
                continue
            n_types_mean = int(np.mean([r["n_types_train"] for r in rows_by_cell[(c, n, "bytype")]]))
            rel = (mae_bytype.mean() - base.mean()) / base.mean() if len(base) else np.nan
            st = C.paired_test(mae_bytype, mae_shuffled) if len(mae_shuffled) else None
            flag = " *" if (st and st["p"] < 0.05 and st["rel"] < 0) else ""
            print(f"  {'':<6}{n:<11}{n_types_mean:>9}{mae_bytype.mean():>9.3f} +-{mae_bytype.std():<4.3f}"
                  f"{rel:>+10.1%}{st['rel']:>+18.1%}{st['p']:>9.4f}{flag}")

    print("\n" + "=" * 78)
    print("  Verdict")
    print("=" * 78)
    for c in configs:
        vals = {n: np.mean([r["mae"] for r in rows_by_cell.get((c, n, "bytype"), [np.nan])])
                for n in names}
        cutvals = [vals[n] for n, _ in CUTS if np.isfinite(vals[n])]
        spread = (max(cutvals) - min(cutvals)) / np.mean(cutvals)
        ok4 = all((C.paired_test(
            np.array([r["mae"] for r in rows_by_cell.get((c, n, "bytype"), [])]),
            np.array([r["mae"] for r in rows_by_cell.get((c, n, "shuffled"), [])]))["rel"] < 0)
            for n in names if rows_by_cell.get((c, n, "bytype")) and rows_by_cell.get((c, n, "shuffled")))
        elem = vals.get("elem_only", np.nan); mae_base = vals.get("base", np.nan)
        print(f"  {c:<5} MAE spread over the five cuts: {spread:.1%}"
              f"   condition (4) holds at every cut: {'yes' if ok4 else 'NO'}")
        print(f"        elem-only {elem:.3f} vs base {mae_base:.3f}  "
              f"({(elem - mae_base) / mae_base:+.1%}) -> "
              f"{'the bond-order proxy is not needed' if abs(elem - mae_base) / mae_base < 0.05 else 'the bond-order proxy matters'}")


# ============================================================================
# bondcut_cv
# ============================================================================
CV_CUTS = {"m06": (0.78, 0.89), "m03": (0.81, 0.92), "base": (0.84, 0.95),
        "p03": (0.87, 0.98), "p06": (0.90, 1.01)}


def inner_split(mols, train_idx, seed, frac=0.75):
    """A validation fold carved out of the TRAINING molecules only."""
    from sklearn.model_selection import GroupShuffleSplit
    tr = np.asarray(train_idx)
    groups = C.group_labels([mols[i] for i in tr], "molecule")
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - frac, random_state=seed)
    fit_pos, val_pos = next(gss.split(np.arange(len(tr)), groups=groups))
    return tr[fit_pos], tr[val_pos]


def fit_score(config, seed, split, cut, nq, epochs, rounds,
              train_idx=None, eval_idx=None):
    from sklearn.metrics import mean_absolute_error
    C._ORDER_CUTS = cut
    arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs, entangler="zz",
                  alpha=0.1, alpha_mode="bytype", readout_scaler=True,
                  canonical_order=True, types_from_train=True)
    ds = C.Dataset(config, arch, split, seed)
    if train_idx is not None:
        ds.train_idx, ds.test_idx = train_idx, eval_idx
    trainer = C.Trainer(ds, arch, seed=seed).fit()
    idx = ds.test_idx
    pred = trainer.predict_eV(idx)
    return float(mean_absolute_error(ds.y_raw[idx], pred)), ds


def bondcut_cv_job(config, seed, split, epochs, rounds):
    nq = C.qubits_for(config)
    base = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs, entangler="zz",
                  alpha=0.1, alpha_mode="bytype", readout_scaler=True,
                  canonical_order=True, types_from_train=True)
    ds0 = C.Dataset(config, base, split, seed)
    tr_in, val_idx = inner_split(ds0.mols, ds0.train_idx, seed)

    t0 = time.time()
    inner = {}
    for name, cut in CV_CUTS.items():
        try:
            v, _ = fit_score(config, seed, split, cut, nq, epochs, rounds,
                             tr_in, val_idx)
        except Exception:
            continue
        inner[name] = v
    chosen = min(inner, key=inner.get)

    test_sel, _ = fit_score(config, seed, split, CV_CUTS[chosen], nq, epochs, rounds)
    test_base, _ = fit_score(config, seed, split, CV_CUTS["base"], nq, epochs, rounds)
    C._ORDER_CUTS = CV_CUTS["base"]

    print(f"    {config} s{seed}  inner picks {chosen:<5} "
          f"(inner MAE {inner[chosen]:.3f})   test {test_sel:.3f}   "
          f"published cut {test_base:.3f}   ({time.time() - t0:.0f}s)", flush=True)
    return [dict(config=config, seed=seed, chosen=chosen,
                 cut_lo=CV_CUTS[chosen][0], cut_hi=CV_CUTS[chosen][1],
                 inner_mae=inner[chosen], mae_cv_selected=test_sel,
                 mae_published_cut=test_base,
                 **{f"inner_{k}": v for k, v in inner.items()})]


def bondcut_cv_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="B5,B3,A1")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--split", default="molecule")
    p.add_argument("--jobs", type=int, default=15)
    p.add_argument("--out", default=C.RESULTS)
    a = p.parse_args(argv)

    configs = a.configs.split(",")
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    args = [(c, s, a.split, a.epochs, a.rounds) for c in configs for s in seeds]

    print("=" * 78)
    print(f"  N2b -- bond-order cut selected by inner CV   configs {configs}   "
          f"seeds {len(seeds)}")
    print("=" * 78)
    rows = C.parallel_jobs(bondcut_cv_job, args, a.jobs, desc="N2b")
    C.write_csv(os.path.join(a.out, "N2b_cut_cv.csv"), rows)

    from collections import Counter
    print("\n" + "=" * 78)
    print("  N2b -- CV-selected cut vs the published cut")
    print("=" * 78)
    print(f"  {'config':<8}{'CV-selected':>14}{'published':>12}{'rel':>9}"
          f"{'wins':>8}{'p':>9}   choices")
    for c in configs:
        cfg_rows = [r for r in rows if r["config"] == c]
        if not cfg_rows:
            continue
        mae_cv_selected = np.array([r["mae_cv_selected"] for r in cfg_rows])
        mae_published_cut = np.array([r["mae_published_cut"] for r in cfg_rows])
        st = C.paired_test(mae_cv_selected, mae_published_cut)
        cnt = Counter(r["chosen"] for r in cfg_rows)
        flag = " *" if st["p"] < 0.05 and st["rel"] < 0 else ""
        print(f"  {c:<8}{mae_cv_selected.mean():>9.3f} +-{mae_cv_selected.std():<4.3f}"
              f"{mae_published_cut.mean():>8.3f} +-{mae_published_cut.std():<4.3f}{st['rel']:>+9.1%}"
              f"{st['wins']:>6}/{st['n']}{st['p']:>9.4f}{flag}   "
              + "  ".join(f"{k}:{v}" for k, v in cnt.most_common()))

    print("\n  The point is not that CV selection wins -- it is that the")
    print("  threshold is no longer a free choice made by the authors.")
    print("  Report whichever way it comes out.")
