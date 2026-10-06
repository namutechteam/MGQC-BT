#!/usr/bin/env python3
"""Plan A — IonQ 하드웨어 fidelity 검증 (10 분자 × 4096 shots, ~24분 QPU).

과학적 질문:
  "IonQ 양자컴퓨터가 우리의 MGQC-BT 회로를 정확히 실행하는가?"

판정 지표 (잔차 기반 — 조성 모델 지배 제거):

  총 E = y_comp (선형 조성, 97%+ variance) + E_res (양자, ~3% variance, ±1 eV)
  총 E 기준 R²는 y_comp 에 지배되어 'quantum=0' 모델도 0.997+ 통과 → fidelity 테스트 부적합.

  대신 양자 신호 격리 지표를 사용:
    [주] res-R²   = R²(E_res_qpu, E_res_noiseless)          목표 ≥ 0.95
    [주] res-MAE  = mean |E_res_qpu − E_res_noiseless|       목표 ≤ 0.15 eV
    [보조] ⟨Z⟩-MAE = 큐빗별 <Z_q>_qpu vs <Z_q>_noiseless     이론 bound ≈ 1/√shots

워크플로:
  1. Shot-robust bundle 로드 (mgqc_bt_bundle_A1_shotrobust.npz)
  2. 10 대표 분자 선정 (원자 수 전 범위에서 균등 샘플링)
  3. 각 분자의 **noiseless 예측** (Qiskit statevector) 계산 → reference
  4. [하드웨어] 10 분자 → Qiskit 회로 → IonQ Aria-1에 1 batched job (4096 shots)
  5. 결과 분석:
       - 분자별 QPU 예측값
       - QPU vs noiseless 산점도 데이터
       - R² (상관계수)
       - 샷 노이즈 이론 경계 안착률
       - MAE (QPU vs true, QPU vs noiseless)
  6. 분석 결과 JSON 저장 (논문/보고서 figure용 데이터 포함)

사용:
  # 로컬 dry-run (하드웨어 submit 전 사전 체크, 무료)
  python plan_a_fidelity.py --dry-run

  # 로컬 AerSim으로 하드웨어 결과 예측 (shot noise 포함, 참고용)
  python plan_a_fidelity.py --backend aer

  # 실제 IonQ Aria-1 하드웨어 실행
  export IONQ_API_KEY='your_token'
  python plan_a_fidelity.py --backend qpu.aria-1

QPU 예산:
  10 분자 × 4096 shots × ~36 ms/shot (Aria-1 11q) ≈ 24.6 분
  큐 대기 (가변): 30분~2시간
  비용 (Aria-1 via Braket, $0.03/shot): ~$1,230
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# ============================================================================
# 1. 대표 분자 선정 (원자 수 전 범위 균등 샘플링)
# ============================================================================
def select_representative(test_data, n_select: int = 10, seed: int = 42):
    """원자 수 전 범위에서 균등하게 n_select 분자를 선정.

    A1 test set은 원자 수 4~11 분포. 각 bin에서 비슷하게 뽑아 다양한 분자 크기
    대표성을 확보한다. 단순 랜덤 샘플링은 n_atoms=10~11에 치우침 (대부분이 11원자).
    """
    n_atoms = test_data["n_atoms"]
    unique_sizes = sorted(set(int(n) for n in n_atoms))
    rng = np.random.default_rng(seed)

    # 크기별 균등 분배
    per_bin = max(1, n_select // len(unique_sizes))
    picked = []
    for atom_count in unique_sizes:
        idxs = np.where(n_atoms == atom_count)[0]
        rng.shuffle(idxs)
        picked.extend(idxs[:per_bin].tolist())
    picked = picked[:n_select]
    # 부족분을 전체에서 랜덤 보충
    if len(picked) < n_select:
        pool = [i for i in range(len(n_atoms)) if i not in picked]
        rng.shuffle(pool)
        picked.extend(pool[:n_select - len(picked)])
    return sorted(picked)


# ============================================================================
# 2. Noiseless 예측 (Qiskit statevector)
# ============================================================================
def noiseless_predictions(test_subset, bundle, pick_idx, n_qubits):
    """Qiskit statevector로 분자별 noiseless 예측값 계산."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from mgqc_ionq_batch import build_qiskit_circuit

    sim = AerSimulator(method="statevector")
    scaler_mu = bundle["scaler_mu"]; scaler_sd = bundle["scaler_sd"]
    readout_w = bundle["readout_w"]; readout_b = float(bundle["readout_b"])
    res_mean = float(bundle["res_mean"]); res_std = float(bundle["res_std"])

    preds = np.zeros(len(pick_idx))
    z_tables = np.zeros((len(pick_idx), n_qubits))
    for k, i in enumerate(pick_idx):
        qc = build_qiskit_circuit(
            test_subset["enc_ang"][i], test_subset["zang"][i],
            int(test_subset["n_active"][i]), test_subset["bonds"][i],
            bundle["var_weights"], bundle["alpha_vec"], bundle)
        qc2 = qc.remove_final_measurements(inplace=False)
        qc2.save_statevector()
        sv = np.asarray(sim.run(transpile(qc2, sim)).result().get_statevector(qc2))
        z = np.zeros(n_qubits)
        for j, a in enumerate(sv):
            p = abs(a) ** 2
            for q in range(n_qubits):
                z[q] += p * (1 if ((j >> q) & 1) == 0 else -1)
        z_tables[k] = z
        z_scaled = (z - scaler_mu) / scaler_sd
        preds[k] = test_subset["y_comp"][i] + (
            float(z_scaled @ readout_w + readout_b) * res_std + res_mean
        )
    return preds, z_tables


# ============================================================================
# 3. QPU 예측 (실제 하드웨어 또는 AerSim shot-based)
# ============================================================================
def qpu_predictions(test_subset, bundle, pick_idx, n_qubits, args):
    """하드웨어(또는 AerSim shot-based)로 분자별 예측값 계산."""
    from mgqc_ionq_batch import build_qiskit_circuit

    scaler_mu = bundle["scaler_mu"]; scaler_sd = bundle["scaler_sd"]
    readout_w = bundle["readout_w"]; readout_b = float(bundle["readout_b"])
    res_mean = float(bundle["res_mean"]); res_std = float(bundle["res_std"])

    # 회로 생성
    circuits = []
    for i in pick_idx:
        qc = build_qiskit_circuit(
            test_subset["enc_ang"][i], test_subset["zang"][i],
            int(test_subset["n_active"][i]), test_subset["bonds"][i],
            bundle["var_weights"], bundle["alpha_vec"], bundle,
            name=f"mol_{int(test_subset['orig_idx'][i])}")
        circuits.append(qc)

    print(f"  회로 생성 완료: N={len(circuits)}, depth={circuits[0].depth()}, "
          f"size={circuits[0].size()}")

    if args.dry_run:
        print(f"  [dry-run] submit 안 함. QPU 전송 시 예상 wall-clock: "
              f"~{len(circuits) * args.shots * 0.036 / 60:.1f}분 (Aria-1 QPU time)")
        return None, 0.0, None, None

    # 실행
    counts_all = []
    if args.backend == "aer":
        # 로컬 AerSim shot-based (하드웨어 proxy)
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        print(f"  [local AerSim] {args.shots} shots × {len(circuits)} 회로 ...")
        t0 = time.time()
        for qc in circuits:
            counts_all.append(sim.run(transpile(qc, sim), shots=args.shots).result().get_counts())
        wall = time.time() - t0
        job_id = "aer-local"
    else:
        # IonQ 하드웨어
        from mgqc_ionq_batch import submit_ionq_batched, fetch_ionq_cost
        if not args.ionq_token:
            sys.exit("ERROR: --ionq-token 또는 환경변수 IONQ_API_KEY 필요")
        job, result, wall = submit_ionq_batched(
            circuits, args.backend, args.ionq_token, args.shots,
            poll_interval=args.poll_interval, timeout_s=args.timeout,
            retrieve_job_id=args.retrieve or None,
            ionq_dry_run=args.ionq_dry_run, job_log=args.job_log,
        )
        job_id = job.job_id()
        if args.ionq_dry_run:
            cost = fetch_ionq_cost(job_id, args.ionq_token)
            print(f"  [ionq-dry-run] job_id={job_id}")
            print(f"  [ionq-dry-run] IonQ 예상 비용: {json.dumps(cost, ensure_ascii=False)}")
            return None, wall, job_id, None
        for k, qc in enumerate(circuits):
            try:
                counts_all.append(result.get_counts(qc))
            except Exception:
                counts_all.append(result.get_counts(k))

    # counts → <Z_q> → prediction
    preds = np.zeros(len(pick_idx))
    z_tables = np.zeros((len(pick_idx), n_qubits))
    for k, counts in enumerate(counts_all):
        z = np.zeros(n_qubits)
        tot = sum(counts.values())
        for bs, cnt in counts.items():
            bs = bs.replace(" ", "")
            for q in range(n_qubits):
                z[q] += (cnt if bs[n_qubits - 1 - q] == "0" else -cnt) / tot
        z_tables[k] = z
        z_scaled = (z - scaler_mu) / scaler_sd
        i = pick_idx[k]
        preds[k] = test_subset["y_comp"][i] + (
            float(z_scaled @ readout_w + readout_b) * res_std + res_mean
        )
    return preds, wall, job_id, z_tables


# ============================================================================
# 4. 분석: QPU vs noiseless 등가성 메트릭 (잔차 기반 — 핵심)
# ============================================================================
def _r2(a, b):
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def analyze_fidelity(preds_qpu, preds_noiseless, y_true, y_comp,
                     z_qpu, z_noiseless, shots):
    """하드웨어 등가성 지표 계산.

    주 판정은 양자 잔차 기준 (조성 모델 지배 제거):
      res = pred - y_comp     (조성 모델이 설명하지 못하는 양자 부분, ±1 eV 수준)
      res_R²  = R²(res_qpu, res_noiseless)
      res_MAE = mean |res_qpu - res_noiseless|

    보조로 큐빗별 <Z> 비교와 총 E 지표(참고용)도 함께 보고.
    """
    N, n_qubits = z_qpu.shape

    # ---- 양자 잔차 (조성 모델 제거) ----
    res_true = y_true - y_comp
    res_noiseless = preds_noiseless - y_comp
    res_qpu = preds_qpu - y_comp

    res_r2_qpu_vs_nl = _r2(res_qpu, res_noiseless)
    res_r2_nl_vs_true = _r2(res_noiseless, res_true)
    res_r2_qpu_vs_true = _r2(res_qpu, res_true)
    res_mae_qpu_vs_nl = float(np.mean(np.abs(res_qpu - res_noiseless)))
    res_mae_nl_vs_true = float(np.mean(np.abs(res_noiseless - res_true)))
    res_mae_qpu_vs_true = float(np.mean(np.abs(res_qpu - res_true)))
    res_pearson = float(np.corrcoef(res_qpu, res_noiseless)[0, 1])

    # Spearman (잔차 순위 보존)
    from scipy.stats import spearmanr
    res_spearman = float(spearmanr(res_qpu, res_noiseless).statistic)

    # ---- 큐빗별 <Z> 비교 ----
    z_diff = z_qpu - z_noiseless                      # (N, n_qubits)
    z_abs_diff = np.abs(z_diff)
    z_mae_per_qubit = z_abs_diff.mean(axis=0)         # (n_qubits,)
    z_mae_overall = float(z_abs_diff.mean())
    z_mae_per_molecule = z_abs_diff.mean(axis=1)      # (N,)
    # 이론 bound: Var(<Z>) ≤ 1/shots, stderr ≤ 1/√shots
    shot_stderr = 1.0 / np.sqrt(max(shots, 1))
    # qubit당 2σ bound 안착률
    pct_within_2sigma = float((z_abs_diff <= 2 * shot_stderr).mean()) * 100

    # ---- 총 E 지표 (참고용, 조성 지배로 fidelity 테스트 부적합) ----
    total_r2_qpu_vs_nl = _r2(preds_qpu, preds_noiseless)
    total_r2_null = _r2(y_comp, y_true)   # quantum=0 널 모델 (엔지니어 지적)
    total_mae_qpu_vs_true = float(np.mean(np.abs(preds_qpu - y_true)))
    total_mae_nl_vs_true = float(np.mean(np.abs(preds_noiseless - y_true)))
    total_mae_qpu_vs_nl = float(np.mean(np.abs(preds_qpu - preds_noiseless)))

    return {
        "N": N,
        "shots": shots,
        "n_qubits": int(n_qubits),

        # ====== 주 판정 지표 (양자 잔차 기준) ======
        "primary_metrics": {
            "res_R2_qpu_vs_noiseless": res_r2_qpu_vs_nl,     # 목표 ≥ 0.95
            "res_MAE_qpu_vs_noiseless_eV": res_mae_qpu_vs_nl,  # 목표 ≤ 0.15 eV
            "res_pearson_qpu_vs_noiseless": res_pearson,
            "res_spearman_qpu_vs_noiseless": res_spearman,
        },

        # ====== 보조: 큐빗별 <Z> (저수준) ======
        "qubit_level_metrics": {
            "Z_MAE_overall": z_mae_overall,
            "Z_MAE_per_qubit": z_mae_per_qubit.tolist(),
            "shot_noise_stderr_theoretical": float(shot_stderr),
            "pct_within_2sigma_bound": pct_within_2sigma,
        },

        # ====== 보조: 양자 파트 절대 성능 ======
        "residual_absolute": {
            "res_R2_noiseless_vs_true": res_r2_nl_vs_true,
            "res_R2_qpu_vs_true": res_r2_qpu_vs_true,
            "res_MAE_noiseless_vs_true_eV": res_mae_nl_vs_true,
            "res_MAE_qpu_vs_true_eV": res_mae_qpu_vs_true,
        },

        # ====== 참고용: 총 E 지표 (조성 지배, fidelity 테스트 부적합) ======
        "total_E_reference_only": {
            "total_R2_qpu_vs_noiseless": total_r2_qpu_vs_nl,
            "total_R2_null_model": total_r2_null,       # y_comp 단독 (quantum=0)
            "total_MAE_qpu_vs_true_eV": total_mae_qpu_vs_true,
            "total_MAE_noiseless_vs_true_eV": total_mae_nl_vs_true,
            "total_MAE_qpu_vs_noiseless_eV": total_mae_qpu_vs_nl,
            "note": "총 E 기준 R²는 y_comp 에 지배됨. quantum=0 모델도 R²≈" +
                    f"{total_r2_null:.4f} 통과. 하드웨어 fidelity 판정은 primary_metrics 사용.",
        },

        # ====== 분자별 원자료 (재분석 가능) ======
        "per_molecule": [
            {
                "idx": k,
                "true_eV": float(y_true[k]),
                "y_comp_eV": float(y_comp[k]),
                "pred_noiseless_eV": float(preds_noiseless[k]),
                "pred_qpu_eV": float(preds_qpu[k]),
                "res_true_eV": float(res_true[k]),
                "res_noiseless_eV": float(res_noiseless[k]),
                "res_qpu_eV": float(res_qpu[k]),
                "res_diff_qpu_noiseless_eV": float(res_qpu[k] - res_noiseless[k]),
                "z_qpu": z_qpu[k].tolist(),
                "z_noiseless": z_noiseless[k].tolist(),
                "z_mae_molecule": float(z_mae_per_molecule[k]),
            } for k in range(N)
        ],
    }


# ============================================================================
# 5. Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--bundle", default="mgqc_bt_bundle_A1_shotrobust.npz")
    ap.add_argument("--test", default="mgqc_bt_test_A1_shotrobust.npz")
    ap.add_argument("--meta", default="mgqc_bt_meta_A1_shotrobust.json")
    ap.add_argument("--n-molecules", type=int, default=10)
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--backend", default="aer",
                    help="aer (로컬 shot-based proxy), simulator (IonQ cloud sim), "
                         "qpu.aria-1 / qpu.aria-2 / qpu.forte (실제 하드웨어)")
    ap.add_argument("--ionq-token", default=os.environ.get("IONQ_API_KEY", ""))
    ap.add_argument("--poll-interval", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=259200)   # 72시간. 실측 대기 20.5h (2026-09-30 Bell)
    ap.add_argument("--dry-run", action="store_true",
                    help="회로만 생성, 하드웨어 submit 전 종료")
    ap.add_argument("--ionq-dry-run", action="store_true",
                    help="IonQ에 dry_run 으로 제출 — 컴파일·예상비용만 받고 대기열에 안 들어감 (무료)")
    ap.add_argument("--retrieve", default="",
                    help="이미 제출한 job_id. 재제출 없이 결과만 받아 분석")
    ap.add_argument("--job-log", default=os.path.join(_HERE, "ionq_jobs.jsonl"),
                    help="제출 즉시 job_id 를 한 줄씩 기록하는 파일")
    ap.add_argument("--seed", type=int, default=42, help="분자 선정 시드")
    ap.add_argument("--out", default=os.path.join(_HERE, "plan_a_results.json"))
    args = ap.parse_args()
    if args.ionq_dry_run and not args.backend.startswith("qpu."):
        sys.exit("ERROR: --ionq-dry-run 은 --backend qpu.forte-1 처럼 실기기 백엔드와 같이 써야 함")
    if args.retrieve and args.backend == "aer":
        sys.exit("ERROR: --retrieve 는 IonQ 백엔드(simulator / qpu.*)와 같이 써야 함")

    bundle_path = os.path.join(_HERE, args.bundle)
    test_path = os.path.join(_HERE, args.test)
    meta_path = os.path.join(_HERE, args.meta)

    print("=" * 72)
    print(" Plan A — IonQ 하드웨어 fidelity 검증")
    print("=" * 72)
    print(f" bundle:    {args.bundle}")
    print(f" backend:   {args.backend}  shots: {args.shots}")
    print(f" molecules: {args.n_molecules}")

    # ---- 1. 로드 ----
    from mgqc_ionq_batch import load_bundle
    bundle, test, meta = load_bundle("A1", bundle_path, test_path, meta_path)
    n_qubits = int(bundle["n_qubits"])
    print(f"\n[bundle] n_qubits={n_qubits}  n_rounds={int(bundle['n_rounds'])}  "
          f"N_test_in_bundle={len(test['y_true'])}  readout_scaler_sd range: "
          f"{bundle['scaler_sd'].min():.3f}~{bundle['scaler_sd'].max():.3f}")

    # ---- 2. 10 대표 분자 선정 ----
    pick = select_representative(test, args.n_molecules, args.seed)
    print(f"\n[선정] {args.n_molecules} 분자 (seed {args.seed}, 원자 수 균등):")
    for k, i in enumerate(pick):
        print(f"       [{k+1:2d}] orig_idx={int(test['orig_idx'][i]):4d}  "
              f"n_atoms={int(test['n_atoms'][i]):2d}  "
              f"true_eV={float(test['y_true'][i]):7.3f}")

    # ---- 3. Noiseless 예측 ----
    print(f"\n[1/3] Qiskit statevector noiseless 예측 ...")
    t0 = time.time()
    preds_noiseless, z_noiseless = noiseless_predictions(test, bundle, pick, n_qubits)
    print(f"      완료 ({time.time()-t0:.1f}s)")

    # ---- 4. QPU/AerSim 예측 ----
    print(f"\n[2/3] {args.backend} 추론 (shots={args.shots}) ...")
    preds_qpu, qpu_wall, job_id, z_qpu = qpu_predictions(
        test, bundle, pick, n_qubits, args)
    if preds_qpu is None:
        print("\n[dry-run] 종료.")
        return

    # ---- 5. 분석 (잔차 기반) ----
    print(f"\n[3/3] 등가성 분석 (잔차 기준) ...")
    y_true = np.array([float(test["y_true"][i]) for i in pick])
    y_comp = np.array([float(test["y_comp"][i]) for i in pick])
    analysis = analyze_fidelity(preds_qpu, preds_noiseless, y_true, y_comp,
                                 z_qpu, z_noiseless, args.shots)
    analysis["molecules"] = [
        {
            "pick_idx": k,
            "orig_idx": int(test["orig_idx"][i]),
            "n_atoms": int(test["n_atoms"][i]),
        } for k, i in enumerate(pick)
    ]
    analysis["backend"] = args.backend
    analysis["job_id"] = job_id
    analysis["qpu_wall_clock_s"] = qpu_wall

    # ---- 6. 리포트 ----
    primary = analysis["primary_metrics"]
    qubit_metrics = analysis["qubit_level_metrics"]
    residual_abs = analysis["residual_absolute"]
    total_ref = analysis["total_E_reference_only"]

    print(f"\n{'=' * 76}")
    print(f" 결과 요약 — Plan A ({args.n_molecules} 분자 × {args.shots} shots)")
    print(f"{'=' * 76}")
    print(f"\n per-molecule — 양자 잔차(res = pred - y_comp) 비교:")
    print(f" {'orig':>5} {'n':>3} {'true':>8} {'y_comp':>8} {'res_true':>9} "
          f"{'res_nl':>9} {'res_qpu':>9} {'|Δres|':>8} {'Z_mae':>7}")
    for k in range(len(pick)):
        i = pick[k]
        d = analysis["per_molecule"][k]
        print(f" {int(test['orig_idx'][i]):>5} {int(test['n_atoms'][i]):>3} "
              f"{d['true_eV']:>8.3f} {d['y_comp_eV']:>8.3f} {d['res_true_eV']:>9.3f} "
              f"{d['res_noiseless_eV']:>9.3f} {d['res_qpu_eV']:>9.3f} "
              f"{abs(d['res_diff_qpu_noiseless_eV']):>8.4f} {d['z_mae_molecule']:>7.4f}")

    print(f"\n ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" ① 주 fidelity 지표 (양자 잔차, 조성 지배 제거)")
    print(f" ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    r2_pass = "PASS" if primary['res_R2_qpu_vs_noiseless'] >= 0.95 else "FAIL"
    mae_pass = "PASS" if primary['res_MAE_qpu_vs_noiseless_eV'] <= 0.15 else "FAIL"
    print(f"  res-R² (QPU vs noiseless):     {primary['res_R2_qpu_vs_noiseless']:.4f}   "
          f"목표 ≥ 0.95  {r2_pass}")
    print(f"  res-MAE (QPU vs noiseless):    {primary['res_MAE_qpu_vs_noiseless_eV']:.4f} eV "
          f"목표 ≤ 0.15 {mae_pass}")
    print(f"  res-Pearson corr:              {primary['res_pearson_qpu_vs_noiseless']:.4f}")
    print(f"  res-Spearman ρ:                {primary['res_spearman_qpu_vs_noiseless']:.4f}")

    print(f"\n ② 보조: 큐빗별 ⟨Z⟩ 비교 (저수준, readout 전)")
    print(f"  Z-MAE (overall):               {qubit_metrics['Z_MAE_overall']:.4f}")
    print(f"  shot noise 이론 stderr:        {qubit_metrics['shot_noise_stderr_theoretical']:.4f}  "
          f"(= 1/√{args.shots})")
    print(f"  이론 ±2σ bound 안착률:         {qubit_metrics['pct_within_2sigma_bound']:.1f}%")
    z_mae_per_q = qubit_metrics['Z_MAE_per_qubit']
    print(f"  큐빗별 Z-MAE: " + " ".join(f"{v:.3f}" for v in z_mae_per_q))

    print(f"\n ③ 양자 파트 절대 성능 (res vs true)")
    print(f"  res-R² (noiseless vs true):    {residual_abs['res_R2_noiseless_vs_true']:.4f}  "
          f"← 양자가 잔차 설명하는 능력")
    print(f"  res-R² (QPU vs true):          {residual_abs['res_R2_qpu_vs_true']:.4f}")
    print(f"  res-MAE (noiseless vs true):   {residual_abs['res_MAE_noiseless_vs_true_eV']:.4f} eV")
    print(f"  res-MAE (QPU vs true):         {residual_abs['res_MAE_qpu_vs_true_eV']:.4f} eV")

    print(f"\n ④ 참고용: 총 E 기준 (조성 모델 지배, fidelity 판정에 사용하지 않음)")
    print(f"  total-R² (quantum=0 null):     {total_ref['total_R2_null_model']:.4f}  "
          f"← 양자 안 써도 통과")
    print(f"  total-R² (QPU vs noiseless):   {total_ref['total_R2_qpu_vs_noiseless']:.4f}")
    print(f"  total-MAE (QPU vs true):       {total_ref['total_MAE_qpu_vs_true_eV']:.4f} eV")

    print(f"\n ⑤ 메타")
    print(f"  wall-clock ({args.backend}):      {qpu_wall:.1f}s")
    print(f"  job_id:                        {job_id}")

    with open(args.out, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n결과 저장: {args.out}")


if __name__ == "__main__":
    main()
