#!/usr/bin/env python3
"""MGQC-BT 추론 — IonQ 하드웨어 batched 실행 (v6.9 Headline Arm).

이 스크립트는 IonQ 실행 환경에서 사용합니다. 다음 3개 파일만 있으면 됩니다:
  mgqc_bt_bundle_<cfg>.npz  — 학습된 var_weights, alpha_vec, readout
  mgqc_bt_test_<cfg>.npz    — 테스트 분자별 encoding/bond 데이터
  mgqc_bt_meta_<cfg>.json   — 메타데이터 (옵션, 로그용)

의존성: qiskit, qiskit-ionq, numpy (requirements-ionq.txt 참조).
학습 서버(PennyLane 등)와 분리된 최소 환경.

워크플로:
  1. bundle + test_data 로드
  2. 각 test 분자에 대해 Qiskit QuantumCircuit 생성
     (학습된 α_τ × J_ij 를 미리 곱해 각도 상수로 하드코딩 → 하드웨어는 추론만)
  3. qiskit-ionq IonQProvider로 전체 N 분자 회로를 **단일 job**으로 batched submit
  4. counts → ⟨Z_q⟩ → readout(scaler + ridge) → 역정규화 + composition → E_pred
  5. MAE/R² 리포트 + JSON 저장

사용:
  export IONQ_API_KEY='your_token'   # https://cloud.ionq.com/settings/keys

  # IonQ 클라우드 시뮬레이터 (무료, 검증용)
  python mgqc_ionq_batch.py --config A1 --backend simulator --shots 1024

  # IonQ Aria-1 (실제 QPU, 25q)
  python mgqc_ionq_batch.py --config A1 --backend qpu.aria-1 --shots 1024

  # IonQ Forte (최신, 36q, 2Q fidelity 99.7%)
  python mgqc_ionq_batch.py --config A1 --backend qpu.forte --shots 1024

  # 비용 확인 (dry-run: 회로만 생성, submit 안 함)
  python mgqc_ionq_batch.py --config A1 --dry-run

v6.9 회로 구조 (code_release/mgqc/core.py:594-672의 blueprint와 1:1 대응):
  n_reupload=1 (한 번만 반복):
    Layer A: RY(encode_angle[q]) × n_qubits   (QSM encoding, 정규화된 CL 고유값)
    for round in range(n_rounds):             # n_rounds=3
       Layer B-1: IsingZZ(θ=J_ij·α_τ(i,j)·scale)   # scale=1/n_rounds, bond-type별 α
       Layer B-2: IsingZZ(0.05·scale)        # padding entanglement
       Layer C-local: RY(var[k]), RZ(var[k+1]) × n_qubits
    Layer C-global: CNOT chain q↔q+1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 1. Bundle 로드 + 검증
# ============================================================================
def load_bundle(config, bundle_path, test_path, meta_path=None):
    bundle = np.load(bundle_path, allow_pickle=False)
    test = np.load(test_path, allow_pickle=False)
    meta = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # 필수 키 검증
    required_bundle = ["n_qubits", "n_rounds", "n_reupload", "theta_scale",
                       "pad_entangle", "use_cnot",
                       "var_weights", "alpha_vec",
                       "scaler_mu", "scaler_sd", "readout_w", "readout_b",
                       "res_mean", "res_std"]
    for k in required_bundle:
        if k not in bundle.files:
            raise KeyError(f"bundle 누락: {k}")
    required_test = ["orig_idx", "enc_ang", "zang", "n_active", "bonds",
                     "y_true", "y_comp", "n_atoms"]
    for k in required_test:
        if k not in test.files:
            raise KeyError(f"test 누락: {k}")

    return bundle, test, meta


# ============================================================================
# 2. v6.9 회로 → Qiskit QuantumCircuit
# ============================================================================
def build_qiskit_circuit(enc_ang, zang, n_active, bonds,
                          var_weights, alpha_vec, bundle, name=None):
    """code_release/mgqc/core.py:blueprint 와 1:1 대응되는 Qiskit 포팅.

    - IsingZZ(θ) = Qiskit의 qc.rzz(θ, i, j) — exp(-iθ/2 Z⊗Z)로 동일
    - RY(θ), RZ(θ), CNOT — 자명한 대응
    - 측정: 전 큐빗 Z 기저 → counts → <Z_q> 후처리
    """
    from qiskit import QuantumCircuit

    n_qubits = int(bundle["n_qubits"])
    n_rounds = int(bundle["n_rounds"])
    n_reupload = int(bundle["n_reupload"])
    theta_scale = float(bundle["theta_scale"])
    pad_entangle = bool(int(bundle["pad_entangle"]))
    use_cnot = bool(int(bundle["use_cnot"]))

    scale = theta_scale / max(n_rounds, 1)
    n_act = int(n_active)
    rz_has_values = bool(np.any(np.abs(zang) > 1e-12))

    qc = QuantumCircuit(n_qubits, n_qubits, name=name or "mgqc_bt")

    for u in range(n_reupload):
        # ---- Layer A: QSM encoding ----
        for q in range(n_qubits):
            qc.ry(float(enc_ang[q]), q)
        if rz_has_values:
            for q in range(n_qubits):
                qc.rz(float(zang[q]), q)

        for r in range(n_rounds):
            # ---- Layer B-1: MGQC-BT ZZ coupling on bonded pairs ----
            # θ_ij = J_ij × α_τ(i,j) × scale (훈련 완료, 하드웨어는 상수)
            for b in bonds:
                i = int(b[0]); j = int(b[1])
                if i < 0 or j < 0:
                    continue
                J = float(b[2])
                bond_type_id = int(b[3])
                a = float(alpha_vec[bond_type_id]) if 0 <= bond_type_id < len(alpha_vec) else 0.0
                theta = J * scale * a
                qc.rzz(theta, i, j)

            # ---- Layer B-2: padding entanglement ----
            if pad_entangle:
                for q in range(n_act, n_qubits):
                    anchor_qubit = min(q, n_act - 1) if n_act > 0 else 0
                    if anchor_qubit != q:
                        qc.rzz(0.05 * scale, anchor_qubit, q)

            # ---- Layer C-local: variational RY, RZ ----
            for q in range(n_qubits):
                base = (((u * n_rounds) + r) * n_qubits + q) * 2
                qc.ry(float(var_weights[base]), q)
                qc.rz(float(var_weights[base + 1]), q)

        # ---- Layer C-global: CNOT chain ----
        if use_cnot:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

    # 전 큐빗 Z 측정
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# ============================================================================
# 2b. Shot-noise Wiener shrinkage (하드웨어 실행 필수)
# ============================================================================
def wiener_shrink_readout(readout_w, scaler_sd, shots):
    """Shot noise에 대한 최적 shrinkage를 각 qubit의 readout weight에 적용.

    논문의 readout_scaler (train sd가 작은 qubit도 standardize) 가
    noiseless statevector에서는 유리하지만 유한-shot 환경에서는
    저분산 qubit의 shot noise를 1/(sd·√shots) 배로 증폭 → MAE 폭발.

    각 qubit별 신호/노이즈 분산:
        signal_var ≈ 1   (standardized feature)
        noise_var  ≈ 1 / (shots · sd²)
    Wiener filter 최적 shrinkage (ML estimator):
        shrink_q = signal / (signal + noise) = 1 / (1 + 1 / (shots · sd_q²))

    저분산 qubit(sd 작음)은 shrink ≈ 0 → 사실상 제외
    고분산 qubit(sd 큼)   은 shrink ≈ 1 → 그대로 사용

    노이즈 없는 (statevector) 평가에서는 shots=∞ 로 두어 비활성화됨.
    """
    if shots is None or shots <= 0 or not np.isfinite(shots):
        return readout_w.copy()
    scaler_sd = np.asarray(scaler_sd, dtype=np.float64)
    denom = shots * (scaler_sd ** 2)
    shrink = 1.0 / (1.0 + 1.0 / np.maximum(denom, 1e-30))
    return np.asarray(readout_w, dtype=np.float64) * shrink


# ============================================================================
# 3. counts → ⟨Z_q⟩
# ============================================================================
def z_expectation_from_counts(counts, qubit_idx, n_qubits):
    """<Z_q> = (N_{bit=0} - N_{bit=1}) / N_total.
    Qiskit bitstring 규약: 가장 오른쪽 문자가 qubit 0 (little-endian)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    pos = n_qubits - 1 - qubit_idx
    acc = 0
    for bs, cnt in counts.items():
        bs = bs.replace(" ", "")
        acc += cnt if bs[pos] == "0" else -cnt
    return acc / total


# ============================================================================
# 4. IonQ batched submit (qiskit-ionq)
# ============================================================================
def _record_job(job_log, record):
    """제출 직후 job 정보를 파일에 남긴다. 스크립트가 죽어도 job_id는 남도록."""
    if not job_log:
        return
    with open(job_log, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[submit] job 기록: {job_log}")


def submit_ionq_batched(circuits, backend_name, api_key, shots,
                        poll_interval=15, timeout_s=14400,
                        retrieve_job_id=None, ionq_dry_run=False, job_log=None):
    """qiskit-ionq로 N개 회로를 1 job에 batched submit.

    retrieve_job_id 가 주어지면 새로 제출하지 않고 기존 job을 불러와 기다린다.
    ionq_dry_run=True 면 IonQ 쪽 컴파일·비용계산만 하고 대기열에 넣지 않는다 (무과금).
    """
    try:
        from qiskit_ionq import IonQProvider
    except ImportError:
        sys.exit("qiskit-ionq 미설치. 설치: pip install qiskit qiskit-ionq")

    print(f"[submit] IonQProvider 초기화")
    provider = IonQProvider(token=api_key)
    print(f"[submit] 백엔드 선택: {backend_name}")
    backend = provider.get_backend(backend_name)
    # qiskit-ionq >= 1.0 (BackendV2) 에서 name 은 메서드가 아니라 속성
    print(f"[submit] 백엔드 확인: {backend.name}")

    t0 = time.time()
    if retrieve_job_id:
        print(f"[submit] 기존 job 불러오기: {retrieve_job_id} (재제출 안 함)")
        job = backend.retrieve_job(retrieve_job_id)
    else:
        print(f"[submit] {len(circuits)}개 회로 batched 제출 (shots={shots}"
              f"{', IonQ dry_run' if ionq_dry_run else ''}) ...")
        run_kw = {"dry_run": True} if ionq_dry_run else {}
        job = backend.run(circuits, shots=shots, **run_kw)
        print(f"[submit] 제출 완료. job_id = {job.job_id()}")
        _record_job(job_log, {
            "job_id": job.job_id(),
            "backend": backend_name,
            "shots": shots,
            "n_circuits": len(circuits),
            "ionq_dry_run": ionq_dry_run,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        })

    print(f"[submit] 결과 대기 (poll every {poll_interval}s, timeout {timeout_s}s)...")
    last_status = None
    while not job.in_final_state():
        if time.time() - t0 > timeout_s:
            sys.exit(f"[submit] 타임아웃: {timeout_s}s 초과. job은 IonQ에서 계속 진행됨. "
                     f"나중에 --retrieve {job.job_id()} 로 결과만 받을 것 (재제출 금지)")
        time.sleep(poll_interval)
        try:
            status_name = job.status().name
        except Exception as e:
            status_name = f"ERR:{e}"
        if status_name != last_status:
            print(f"[submit]   t={time.time()-t0:6.1f}s  status={status_name}")
            last_status = status_name

    wall = time.time() - t0
    print(f"[submit] 완료. wall-clock={wall:.1f}s  final_status={job.status().name}")
    if ionq_dry_run:
        return job, None, wall      # dry run 은 측정 결과가 없음
    return job, job.result(), wall


def fetch_ionq_cost(job_id, api_key):
    """IonQ v0.4 /jobs/{id}/cost 조회. dry run 이면 예상 비용, 완료 job 이면 실제 비용."""
    import requests
    r = requests.get(f"https://api.ionq.co/v0.4/jobs/{job_id}/cost",
                     headers={"Authorization": f"apiKey {api_key}"}, timeout=30)
    r.raise_for_status()
    return r.json()


# ============================================================================
# 5. Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--config", default="A1",
                    help="학습 bundle의 config 이름 (A1 / A2 / B1~B5)")
    ap.add_argument("--bundle", default="",
                    help="mgqc_bt_bundle_<cfg>.npz 경로 (비우면 자동 유추)")
    ap.add_argument("--test", default="",
                    help="mgqc_bt_test_<cfg>.npz 경로 (비우면 자동 유추)")
    ap.add_argument("--meta", default="",
                    help="mgqc_bt_meta_<cfg>.json 경로 (옵션)")
    ap.add_argument("--backend", default="simulator",
                    help="IonQ 백엔드: simulator / qpu.aria-1 / qpu.aria-2 / qpu.forte")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--n-test", type=int, default=0,
                    help="추론할 분자 수 (0=bundle에 저장된 전체)")
    ap.add_argument("--ionq-token", default=os.environ.get("IONQ_API_KEY", ""))
    ap.add_argument("--poll-interval", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=259200)   # 72시간. 실측 대기 20.5h (2026-09-30 Bell)
    ap.add_argument("--dry-run", action="store_true",
                    help="회로만 생성하고 submit 전 종료 (연결/비용 사전 확인용)")
    ap.add_argument("--no-shrink", action="store_true",
                    help="Wiener shot-noise shrinkage 비활성화 (noiseless/고-shot 전용)")
    ap.add_argument("--save-qasm", default="",
                    help="회로를 OpenQASM 2.0 파일로 저장할 디렉토리 (디버깅)")
    ap.add_argument("--out", default="",
                    help="결과 JSON 경로 (비우면 mgqc_ionq_results_<cfg>.json)")
    args = ap.parse_args()

    cfg = args.config
    bundle_path = args.bundle or os.path.join(_HERE, f"mgqc_bt_bundle_{cfg}.npz")
    test_path = args.test or os.path.join(_HERE, f"mgqc_bt_test_{cfg}.npz")
    meta_path = args.meta or os.path.join(_HERE, f"mgqc_bt_meta_{cfg}.json")
    out_path = args.out or os.path.join(_HERE, f"mgqc_ionq_results_{cfg}.json")

    print("=" * 70)
    print(f" MGQC-BT IonQ Batched Inference  (v6.9 Headline Arm)")
    print("=" * 70)
    print(f" config:  {cfg}")
    print(f" bundle:  {bundle_path}")
    print(f" test:    {test_path}")
    print(f" backend: {args.backend}  shots: {args.shots}")

    # ---- 1. 로드 ----
    bundle, test, meta = load_bundle(cfg, bundle_path, test_path, meta_path)
    n_qubits = int(bundle["n_qubits"])
    n_rounds = int(bundle["n_rounds"])
    var_weights = bundle["var_weights"]
    alpha_vec = bundle["alpha_vec"]
    scaler_mu = bundle["scaler_mu"]
    scaler_sd = bundle["scaler_sd"]
    readout_w = bundle["readout_w"]
    readout_b = float(bundle["readout_b"])
    res_mean = float(bundle["res_mean"])
    res_std = float(bundle["res_std"])
    print(f"\n[bundle] n_qubits={n_qubits}  n_rounds={n_rounds}  "
          f"var_weights {var_weights.shape}  alpha {alpha_vec.shape}")
    print(f"[bundle] readout_scaler: mu range {scaler_mu.min():.3f}~{scaler_mu.max():.3f},  "
          f"sd range {scaler_sd.min():.3f}~{scaler_sd.max():.3f}")
    print(f"[bundle] res_mean={res_mean:.3f}  res_std={res_std:.3f}")

    # Shot-noise Wiener shrinkage
    if args.no_shrink:
        readout_w_eff = readout_w
        print(f"[shrink] 비활성화 (--no-shrink)")
    else:
        readout_w_eff = wiener_shrink_readout(readout_w, scaler_sd, args.shots)
        shrink_vec = readout_w_eff / np.where(np.abs(readout_w) > 1e-12, readout_w, 1.0)
        print(f"[shrink] shots={args.shots} 기준 Wiener shrinkage 적용")
        print(f"[shrink]   qubit별 shrink 계수: "
              f"min={shrink_vec.min():.3f}  max={shrink_vec.max():.3f}  "
              f"n_dropped(<0.1)={int((shrink_vec < 0.1).sum())}/{n_qubits}")

    N_all = int(len(test["y_true"]))
    N = N_all if args.n_test == 0 else min(args.n_test, N_all)
    print(f"\n[test] 전체 {N_all} 분자 중 {N}개 사용")

    # ---- 2. Qiskit 회로 생성 ----
    print(f"\n[build] Qiskit 회로 생성 중...")
    t0 = time.time()
    circuits = []
    for k in range(N):
        qc = build_qiskit_circuit(
            enc_ang=test["enc_ang"][k],
            zang=test["zang"][k],
            n_active=int(test["n_active"][k]),
            bonds=test["bonds"][k],
            var_weights=var_weights,
            alpha_vec=alpha_vec,
            bundle=bundle,
            name=f"mol_{int(test['orig_idx'][k])}",
        )
        circuits.append(qc)
    print(f"[build] 완료 ({time.time()-t0:.1f}s). "
          f"depth={circuits[0].depth()}  size={circuits[0].size()}  "
          f"qubits={circuits[0].num_qubits}")

    if args.save_qasm:
        os.makedirs(args.save_qasm, exist_ok=True)
        for k, qc in enumerate(circuits):
            fp = os.path.join(args.save_qasm, f"{qc.name}.qasm")
            try:
                from qiskit.qasm2 import dumps
                with open(fp, "w") as f: f.write(dumps(qc))
            except ImportError:
                with open(fp, "w") as f: f.write(qc.qasm())
        print(f"[build] QASM 저장: {args.save_qasm}/")

    if args.dry_run:
        print(f"\n[dry-run] 종료. {N}개 회로 준비 완료 (submit 안 함).")
        return

    # ---- 3. IonQ batched submit ----
    if not args.ionq_token:
        sys.exit("ERROR: --ionq-token 또는 환경변수 IONQ_API_KEY 필요.")

    job, result, wall_s = submit_ionq_batched(
        circuits, args.backend, args.ionq_token, args.shots,
        poll_interval=args.poll_interval, timeout_s=args.timeout,
    )

    # ---- 4. 결과 처리: counts → <Z_q> → readout → eV ----
    print(f"\n[post] 결과 처리 ...")
    rows = []
    for k, qc in enumerate(circuits):
        try:
            counts = result.get_counts(qc)
        except Exception:
            counts = result.get_counts(k)

        z = np.array([
            z_expectation_from_counts(counts, q, n_qubits) for q in range(n_qubits)
        ])
        # readout_scaler: (z - mu) / sd  → @ (w·shrink) + b
        z_scaled = (z - scaler_mu) / scaler_sd
        pred_norm = float(z_scaled @ readout_w_eff + readout_b)
        pred_res = pred_norm * res_std + res_mean
        pred_eV = float(test["y_comp"][k] + pred_res)
        true_eV = float(test["y_true"][k])
        rows.append({
            "orig_idx": int(test["orig_idx"][k]),
            "n_atoms": int(test["n_atoms"][k]),
            "pred_eV": pred_eV,
            "true_eV": true_eV,
            "abs_err_eV": abs(pred_eV - true_eV),
            "z_exp": z.tolist(),
        })

    preds = np.array([r["pred_eV"] for r in rows])
    trues = np.array([r["true_eV"] for r in rows])
    mae = float(np.mean(np.abs(preds - trues)))
    ss_res = np.sum((preds - trues) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    print(f"\n{'orig_idx':>9} {'n_atoms':>8} {'pred (eV)':>11} {'true (eV)':>11} {'|err|':>8}")
    for r in rows[:20]:
        print(f"{r['orig_idx']:>9} {r['n_atoms']:>8} "
              f"{r['pred_eV']:>11.3f} {r['true_eV']:>11.3f} {r['abs_err_eV']:>8.3f}")
    if len(rows) > 20:
        print(f" ... ({len(rows) - 20} more)")

    print(f"\n==== 요약 ====")
    print(f"backend:     {args.backend}")
    print(f"shots:       {args.shots}")
    print(f"N:           {len(rows)}")
    print(f"wall-clock:  {wall_s:.1f} s  ({wall_s/max(len(rows),1):.2f}s/molecule)")
    print(f"MAE (eV):    {mae:.4f}")
    print(f"R²:          {r2:.4f}")

    out = {
        "config": cfg,
        "backend": args.backend,
        "shots": args.shots,
        "n": len(rows),
        "wall_clock_s": wall_s,
        "mae_eV": mae,
        "r2": r2,
        "job_id": job.job_id(),
        "arch_meta": (meta.get("arch") if meta else None),
        "predictions": rows,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
