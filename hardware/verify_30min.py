#!/usr/bin/env python3
"""30분 재현 검증 — 논문 A1 결과(zz_bytype_r3 MAE=0.335±0.026 eV) 재현.

전체 흐름 (총 ~20-25분):
  [1/4] 사전학습:       MGQC-BT A1 100 epochs (lightning.qubit, ~10분)
  [2/4] PennyLane 평가: 전체 160 test 분자, statevector (noiseless), ~30초
  [3/4] Qiskit 평가:    로컬 Qiskit AerSimulator, 1024 shots, 전체 160개 (~5-10분)
  [4/4] 결과 보고:      PennyLane / Qiskit 비교 + 논문 수치 범위 확인

핵심 검증 포인트:
  (A) PennyLane MAE 가 논문 범위(0.335±0.026)에 들어가는가?  → 학습 재현성
  (B) Qiskit shot-based MAE 가 PennyLane noiseless MAE 와 가까운가? → 포팅 정확성
       (단, shot noise 때문에 완전 일치는 안 되고 수치 차이 평균 ~0.01-0.05 eV 예상)

실행:
  python verify_30min.py                     # A1 전체 (160분자)
  python verify_30min.py --n-test 20         # 처음 20개만 (5분 내 완료)
  python verify_30min.py --shots 512         # shot 수 절반 (더 빠르지만 노이즈 ↑)

필요 환경:
  pennylane 0.45.1, pennylane-lightning 0.45.1, scikit-learn
  qiskit, qiskit-aer, numpy
  code_release/mgqc 패키지는 상위 디렉터리에서 가져옴
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)


def paper_target(config: str) -> tuple[float, float]:
    """논문 reported zz_bytype_r3 MAE (mean, sd) by config."""
    # code_release/results/E1_sevenconfig.csv 요약 (20 seeds)
    return {
        "A1": (0.335, 0.026),
        "A2": (0.334, 0.030),   # 추정, N1_alt_targets_r1a2.csv에서 재집계 가능
        "B1": (0.258, 0.020),
        "B2": (0.315, 0.040),
        "B3": (0.460, 0.050),
        "B4": (0.380, 0.040),
        "B5": (0.390, 0.054),
    }.get(config, (float("nan"), float("nan")))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--config", default="A1", choices=["A1","A2","B1","B2","B3","B4","B5"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100,
                    help="v6.9 기본값 (낮추면 더 빠름, 하지만 MAE 올라감)")
    ap.add_argument("--n-test", type=int, default=0,
                    help="추론할 테스트 분자 수 (0=전체)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--skip-training", action="store_true",
                    help="저장된 bundle 재사용 (반복 실행 시)")
    ap.add_argument("--out", default=os.path.join(_HERE, "verify_30min_results.json"))
    args = ap.parse_args()

    t_total_0 = time.time()

    cfg = args.config
    nq = 11 if cfg == "A1" else 13
    paper_mean, paper_sd = paper_target(cfg)

    print("=" * 72)
    print(f" MGQC-BT 30분 재현 검증  (config={cfg}, seed={args.seed})")
    print("=" * 72)
    print(f" 목표: 논문 reported MAE = {paper_mean:.3f} ± {paper_sd:.3f} eV "
          f"(zz_bytype_r3, 20 seeds)")
    print(f" 수용 범위: [{paper_mean - 2*paper_sd:.3f}, {paper_mean + 2*paper_sd:.3f}] eV  (±2σ)")

    # ========================================================================
    # [1/4] 사전학습 (PennyLane lightning.qubit)
    # ========================================================================
    bundle_path = os.path.join(_HERE, f"mgqc_bt_bundle_{cfg}.npz")
    test_path = os.path.join(_HERE, f"mgqc_bt_test_{cfg}.npz")

    if args.skip_training and os.path.exists(bundle_path) and os.path.exists(test_path):
        print(f"\n[1/4] 사전학습 SKIP — 저장된 bundle 재사용")
    else:
        print(f"\n[1/4] 사전학습 (A1, {args.epochs} epochs, PennyLane lightning.qubit) ...")
        t0 = time.time()
        from mgqc import core as C
        arch = C.Arch(
            n_qubits=nq, n_rounds=3,
            entangler="zz", alpha=0.1, alpha_mode="bytype",
            readout_scaler=True, canonical_order=True, types_from_train=True,
            epochs=args.epochs, lr=0.05, target="eAT",
        )
        ds = C.Dataset(cfg, arch, split_level="molecule", seed=args.seed)
        print(f"       N_train={len(ds.train_idx)}  N_test={len(ds.test_idx)}  "
              f"bond_types={ds.n_types_train}")
        trainer = C.Trainer(ds, arch, seed=args.seed, device="lightning.qubit").fit()
        pl_metrics, _ = trainer.evaluate()
        print(f"       훈련 완료 ({time.time()-t0:.1f}s)  loss "
              f"{trainer.loss_first:.4f} → {trainer.loss_last:.4f}")
        print(f"       PennyLane statevector MAE = {pl_metrics['mae']:.4f} eV   "
              f"R² = {pl_metrics['r2']:.4f}")

        # Bundle 저장 (후속 단계에서 공유)
        from train_mgqc_bt import export_bundle
        meta_path = os.path.join(_HERE, f"mgqc_bt_meta_{cfg}.json")
        export_bundle(ds, trainer, arch, bundle_path, test_path, meta_path,
                       n_test_cap=args.n_test)
        pl_mae_full = pl_metrics["mae"]
        pl_r2_full = pl_metrics["r2"]
    train_time = time.time() - t_total_0

    # ========================================================================
    # [2/4] 로드 + Qiskit statevector 교차 검증 (noiseless, 포팅 정확성)
    # ========================================================================
    print(f"\n[2/4] Qiskit statevector 교차 검증 (noiseless) ...")
    t0 = time.time()
    from mgqc_ionq_batch import load_bundle, build_qiskit_circuit
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    bundle, test, meta = load_bundle(cfg, bundle_path, test_path,
                                     os.path.join(_HERE, f"mgqc_bt_meta_{cfg}.json"))
    N_all = int(len(test["y_true"]))
    N = N_all if args.n_test == 0 else min(args.n_test, N_all)
    print(f"       테스트 분자 N = {N}  (전체 {N_all} 중)")

    var_weights = bundle["var_weights"]; alpha_vec = bundle["alpha_vec"]
    scaler_mu = bundle["scaler_mu"]; scaler_sd = bundle["scaler_sd"]
    readout_w = bundle["readout_w"]; readout_b = float(bundle["readout_b"])
    res_mean = float(bundle["res_mean"]); res_std = float(bundle["res_std"])

    sim_sv = AerSimulator(method="statevector")
    sv_preds = np.zeros(N)
    for k in range(N):
        qc = build_qiskit_circuit(
            test["enc_ang"][k], test["zang"][k], int(test["n_active"][k]),
            test["bonds"][k], var_weights, alpha_vec, bundle)
        qc2 = qc.remove_final_measurements(inplace=False)
        qc2.save_statevector()
        sv = np.asarray(sim_sv.run(transpile(qc2, sim_sv)).result().get_statevector(qc2))
        # <Z_q> from statevector
        z_exp = np.zeros(nq)
        for i, a in enumerate(sv):
            p = abs(a) ** 2
            for q in range(nq):
                z_exp[q] += p * (1 if ((i >> q) & 1) == 0 else -1)
        z_scaled = (z_exp - scaler_mu) / scaler_sd
        pred_norm = float(z_scaled @ readout_w + readout_b)
        sv_preds[k] = test["y_comp"][k] + pred_norm * res_std + res_mean
    y_true = test["y_true"][:N]
    sv_mae = float(np.mean(np.abs(sv_preds - y_true)))
    print(f"       Qiskit statevector MAE = {sv_mae:.4f} eV  "
          f"(처리 시간 {time.time()-t0:.1f}s)")

    # ========================================================================
    # [3/4] Qiskit AerSimulator shot-based (1024 shots, 하드웨어 유사)
    # ========================================================================
    print(f"\n[3/4] Qiskit AerSimulator shot-based ({args.shots} shots/circuit) ...")

    # Wiener shot-noise shrinkage 적용 (하드웨어/shot-based 환경 필수)
    from mgqc_ionq_batch import wiener_shrink_readout
    readout_w_eff = wiener_shrink_readout(readout_w, scaler_sd, args.shots)
    shrink_vec = readout_w_eff / np.where(np.abs(readout_w) > 1e-12, readout_w, 1.0)
    print(f"       [shrink] shots={args.shots} → qubit별 shrink: "
          f"min={shrink_vec.min():.3f}  max={shrink_vec.max():.3f}  "
          f"n_dropped(<0.1)={int((shrink_vec < 0.1).sum())}/{nq}")

    t0 = time.time()
    sim_shot = AerSimulator()
    shot_preds = np.zeros(N)
    shot_preds_noshrink = np.zeros(N)
    for k in range(N):
        qc = build_qiskit_circuit(
            test["enc_ang"][k], test["zang"][k], int(test["n_active"][k]),
            test["bonds"][k], var_weights, alpha_vec, bundle)
        result = sim_shot.run(transpile(qc, sim_shot), shots=args.shots).result()
        counts = result.get_counts()
        z_exp = np.zeros(nq)
        total = sum(counts.values())
        for bs, cnt in counts.items():
            bs = bs.replace(" ", "")
            for q in range(nq):
                pos = nq - 1 - q
                z_exp[q] += (cnt if bs[pos] == "0" else -cnt) / total
        z_scaled = (z_exp - scaler_mu) / scaler_sd
        # (A) shrinkage 적용 (하드웨어 실행용)
        pred_norm = float(z_scaled @ readout_w_eff + readout_b)
        shot_preds[k] = test["y_comp"][k] + pred_norm * res_std + res_mean
        # (B) shrinkage 미적용 (비교용)
        pred_norm_ns = float(z_scaled @ readout_w + readout_b)
        shot_preds_noshrink[k] = test["y_comp"][k] + pred_norm_ns * res_std + res_mean
        if (k + 1) % 20 == 0:
            print(f"         {k+1}/{N} 완료 ...", flush=True)

    shot_mae = float(np.mean(np.abs(shot_preds - y_true)))
    shot_mae_noshrink = float(np.mean(np.abs(shot_preds_noshrink - y_true)))
    shot_time = time.time() - t0
    print(f"       Qiskit shot-based (shrinkage ON)  MAE = {shot_mae:.4f} eV")
    print(f"       Qiskit shot-based (shrinkage OFF) MAE = {shot_mae_noshrink:.4f} eV  "
          f"(비교용, 기존 결과)")
    print(f"       처리 시간: {shot_time:.1f}s")

    # ========================================================================
    # [4/4] 결과 보고
    # ========================================================================
    diff_sv_shot = float(np.mean(np.abs(sv_preds - shot_preds)))
    r2_shot = float(1 - np.sum((shot_preds - y_true) ** 2)
                    / np.sum((y_true - y_true.mean()) ** 2))

    in_range = (paper_mean - 2*paper_sd) <= shot_mae <= (paper_mean + 2*paper_sd)
    verdict = "accept" if in_range else "out of range"

    print(f"\n{'=' * 72}")
    print(f" 결과 요약")
    print(f"{'=' * 72}")
    print(f"  config:                  {cfg}  (N_test = {N})")
    print(f"  Qiskit statevector MAE:  {sv_mae:.4f} eV  (noiseless, 포팅 정확성)")
    print(f"  Qiskit shot-based MAE:   {shot_mae:.4f} eV  "
          f"({args.shots} shots, 하드웨어 유사)")
    print(f"  Shot vs statevector |Δ|: {diff_sv_shot:.4f} eV  (shot noise 효과)")
    print(f"  R² (shot-based):         {r2_shot:.4f}")
    print(f"")
    print(f"  논문 reported:           {paper_mean:.3f} ± {paper_sd:.3f} eV")
    print(f"  수용 범위 (±2σ):         [{paper_mean - 2*paper_sd:.3f}, "
          f"{paper_mean + 2*paper_sd:.3f}]")
    print(f"  판정:                    {verdict}")
    print(f"")
    print(f"  총 wall-clock:           {time.time() - t_total_0:.1f} 초")
    print(f"{'=' * 72}")

    out = {
        "config": cfg,
        "seed": args.seed,
        "N_test": N,
        "shots": args.shots,
        "mae": {
            "qiskit_statevector_noiseless": sv_mae,
            "qiskit_aer_shot_based_shrink": shot_mae,
            "qiskit_aer_shot_based_noshrink": shot_mae_noshrink,
            "mean_abs_diff_shrink": diff_sv_shot,
        },
        "r2_shot_based": r2_shot,
        "paper_reported": {"mean": paper_mean, "sd": paper_sd},
        "within_2sigma": in_range,
        "wall_clock_s": time.time() - t_total_0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
