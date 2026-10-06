#!/usr/bin/env python3
"""MGQC-BT (bond-type-conditioned coupling) 학습 → IonQ용 bundle 저장.

v6.9 논문 (MGQC_BT_NatComm_main_v6.9.docx) 방법론:
  - 회로: Coulomb-Laplacian spectral RY encoding → ZZ bond coupling (3 rounds)
          → learnable RY/RZ + CNOT chain → PauliZ 측정
  - 결합 각도: θ_ij = J_ij × α_τ(i,j), J_ij = Z_i Z_j / ((Z_i+Z_j)√m)
    α_τ는 bond type τ (예: C-C single, C=O double, ...)별 학습 가능 스칼라
  - 훈련: Adam + adjoint gradient (SPSA 아님)
  - Split: molecule-level (이성질체 누출 없음)

이 스크립트는 code_release/mgqc 공식 패키지를 사용합니다 (v6.9 published arch).
  arm:  zz_bytype_r3 (accuracy.py Headline Arm)
  정합: `python -m mgqc run accuracy` 와 동일한 Arch 설정

학습 후 IonQ 하드웨어 실행에 필요한 모든 아티팩트를 하나의 .npz로 저장 →
`mgqc_ionq_batch.py` 가 그 bundle만으로 IonQ에서 추론 가능.

사용 (local, 학습 서버):
  cd code_release/hardware
  python train_mgqc_bt.py --config A1 --seed 42 --epochs 100
  python train_mgqc_bt.py --config B5 --seed 42        # 가장 작은 config (204 분자)

학습 결과물 (IonQ 서버로 복사할 파일):
  mgqc_bt_bundle_<cfg>.npz        — 아키텍처 + 학습된 var/alpha + readout + composition
  mgqc_bt_test_<cfg>.npz          — 테스트 분자별 encoding/bond 데이터 (IonQ용)
  mgqc_bt_meta_<cfg>.json         — 메타데이터 (config, bond-type 어휘 등)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# code_release/ 경로 삽입 (공식 mgqc 패키지 import용)
_HERE = os.path.dirname(os.path.abspath(__file__))
CODE_RELEASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if CODE_RELEASE not in sys.path:
    sys.path.insert(0, CODE_RELEASE)

try:
    from mgqc import core as C
except ImportError as e:
    sys.exit(
        f"ERROR: mgqc 패키지를 import 할 수 없습니다.\n"
        f"       예상 위치: {CODE_RELEASE}/mgqc/\n"
        f"       사유: {e}"
    )


# ============================================================================
# v6.9 Headline Arm = code_release/mgqc/experiments/accuracy.py 의 zz_bytype_r3
# ============================================================================
def make_v69_arch(n_qubits: int, n_rounds: int = 3,
                  epochs: int = 100, lr: float = 0.05,
                  target: str = "eAT", readout_scaler: bool = True):
    """code_release/mgqc/experiments/accuracy.py 의 headline arm `zz_bytype_r3`.

    인용 (accuracy.py:43-47 + :147-149):
        ("zz_bytype_r3", dict(entangler="zz", alpha=0.1, alpha_mode="bytype")),
        arch = C.Arch(n_qubits=nq, n_rounds=rounds, epochs=epochs,
                      readout_scaler=True, canonical_order=canonical,
                      types_from_train=tft, **kw)
    (canonical=True, tft=True가 기본 호출. v6.9 published 그대로.)
    """
    return C.Arch(
        n_qubits=n_qubits,
        n_rounds=n_rounds,
        entangler="zz",
        alpha=0.1,
        alpha_mode="bytype",
        readout_scaler=readout_scaler,   # True=논문 그대로(noiseless 최적) / False=shot-robust
        canonical_order=True,
        types_from_train=True,
        epochs=epochs,
        lr=lr,
        target=target,
    )


# ============================================================================
# 학습된 상태 → IonQ bundle로 추출
# ============================================================================
def export_bundle(ds, trainer, arch, out_bundle, out_test, out_meta,
                   compile_test_only=True, n_test_cap: int = 0):
    """Trainer 상태 + Dataset 상태 → IonQ 쪽에서 재현 가능한 bundle."""
    comp = trainer.comp

    # 테스트 분자 선정
    test_idx = ds.test_idx
    if n_test_cap and n_test_cap < len(test_idx):
        # 원자 수 작은 순 (하드웨어 비용/depth 최소화)
        sizes = np.array([int(ds.mols[i]["n_atoms"]) for i in test_idx])
        order = np.argsort(sizes)
        test_idx = test_idx[order[:n_test_cap]]
        print(f"  n_test_cap={n_test_cap} → 원자 수 작은 순으로 {n_test_cap}개 선택")

    # ---- 글로벌 아티팩트 ----
    alpha_vec = comp.alpha_vec(trainer.alpha)         # per-type alpha vector (학습 완료)
    var_weights = trainer.var.astype(np.float64)

    readout = trainer.readout
    readout_mu = readout.mu.astype(np.float64) if readout.mu is not None else np.zeros(arch.n_qubits)
    readout_sd = readout.sd.astype(np.float64) if readout.sd is not None else np.ones(arch.n_qubits)
    readout_w = readout.w.astype(np.float64)
    readout_b = float(readout.b)

    bundle = {
        # 아키텍처
        "n_qubits": np.array(arch.n_qubits, dtype=np.int32),
        "n_rounds": np.array(arch.n_rounds, dtype=np.int32),
        "n_reupload": np.array(arch.n_reupload, dtype=np.int32),
        "theta_scale": np.array(arch.theta_scale, dtype=np.float64),
        "pad_entangle": np.array(int(arch.pad_entangle), dtype=np.int32),
        "use_cnot": np.array(int(arch.use_cnot), dtype=np.int32),
        "entangler": arch.entangler,
        # 학습된 가중치
        "var_weights": var_weights,            # (n_var_params,)
        "alpha_vec": alpha_vec.astype(np.float64),    # (n_types,)
        # Readout
        "scaler_mu": readout_mu,
        "scaler_sd": readout_sd,
        "readout_w": readout_w,
        "readout_b": np.array(readout_b, dtype=np.float64),
        # Composition / 정규화
        "res_mean": np.array(ds.res_mean, dtype=np.float64),
        "res_std": np.array(ds.res_std, dtype=np.float64),
    }
    np.savez(out_bundle, **bundle)
    print(f"  bundle saved: {out_bundle}  "
          f"(var {var_weights.shape}, alpha {alpha_vec.shape}, "
          f"readout_w {readout_w.shape})")

    # ---- 테스트 분자별 입력 데이터 ----
    MAX_BONDS = 30
    N = len(test_idx)
    enc_ang = np.zeros((N, arch.n_qubits), dtype=np.float64)
    zang = np.zeros((N, arch.n_qubits), dtype=np.float64)
    n_active = np.zeros(N, dtype=np.int32)
    bonds_arr = np.full((N, MAX_BONDS, 4), -1, dtype=np.float64)  # (i, j, J, type_id)
    y_true = np.zeros(N, dtype=np.float64)
    y_comp_test = np.zeros(N, dtype=np.float64)
    orig_idx = np.array([int(i) for i in test_idx], dtype=np.int32)
    n_atoms_arr = np.zeros(N, dtype=np.int32)

    for k, i in enumerate(test_idx):
        enc_ang[k] = ds.ang[i]
        zang[k] = ds.zang[i]
        n_active[k] = ds.n_active[i]
        for b_idx, (bi, bj, J, t, t_sh) in enumerate(ds.bonds[i][:MAX_BONDS]):
            bonds_arr[k, b_idx] = [bi, bj, J, t]
        y_true[k] = ds.y_raw[i]
        y_comp_test[k] = ds.y_comp[i]
        n_atoms_arr[k] = int(ds.mols[i]["n_atoms"])

    test_pack = {
        "orig_idx": orig_idx,          # 원본 cache의 분자 index
        "enc_ang": enc_ang,            # (N, n_qubits) Layer A RY 각도
        "zang": zang,                  # (N, n_qubits) Layer A RZ 각도 (rz_source=="none"이면 0)
        "n_active": n_active,          # (N,) 실제 원자 수 (padding용)
        "bonds": bonds_arr,            # (N, MAX_BONDS, 4) [i, j, J, type_id]
        "y_true": y_true,              # (N,) 정답 eAT
        "y_comp": y_comp_test,         # (N,) 조성 모델 예측
        "n_atoms": n_atoms_arr,
    }
    np.savez(out_test, **test_pack)
    print(f"  test data saved: {out_test}  "
          f"(N_test={N}, max_bonds={MAX_BONDS})")

    # ---- 메타데이터 ----
    meta = {
        "arch": {
            "n_qubits": arch.n_qubits,
            "n_rounds": arch.n_rounds,
            "n_reupload": arch.n_reupload,
            "entangler": arch.entangler,
            "alpha_mode": arch.alpha_mode,
            "bond_type_scheme": arch.bond_type_scheme,
            "pad_entangle": arch.pad_entangle,
            "use_cnot": arch.use_cnot,
            "theta_scale": arch.theta_scale,
            "encode_source": arch.encode_source,
            "target": arch.target,
            "residual": arch.residual,
            "epochs": arch.epochs,
            "optimizer": arch.optimizer,
        },
        "training": {
            "config": ds.config,
            "seed": ds.seed,
            "n_train": int(len(ds.train_idx)),
            "n_test": int(len(ds.test_idx)),
            "n_test_bundled": int(N),
            "n_types": int(ds.n_types_train),
            "comp_r2": float(ds.comp_r2),
            "res_mean": float(ds.res_mean),
            "res_std": float(ds.res_std),
            "gmax": float(ds.gmax),
        },
        "bond_types": [
            {"id": idx, "label": C.type_label(key), "count": ds.type_count.get(key, 0)}
            for idx, key in enumerate(ds.type_keys)
        ],
        "alpha_learned": [
            {"id": t, "label": C.type_label(ds.type_keys[t]),
             "alpha": float(alpha_vec[t]) if t < len(alpha_vec) else None}
            for t in range(min(len(ds.type_keys), len(alpha_vec)))
        ],
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta saved: {out_meta}")

    return meta


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--config", default="A1",
                    choices=list(C.CONFIG_ATOMS.keys()),
                    help="QM7-X config (A1=11qubits, 나머지 13qubits)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n-rounds", type=int, default=3,
                    help="MGQC-BT message-passing rounds (v6.9: 3)")
    ap.add_argument("--target", default="eAT",
                    choices=["eAT", "hlgap", "dip"])
    ap.add_argument("--split-level", default="molecule",
                    choices=["molecule", "isomer", "structure"],
                    help="v6.9: molecule-level (이성질체 누출 없음)")
    ap.add_argument("--n-test-cap", type=int, default=0,
                    help="테스트 bundle 크기 상한 (0=전체). 하드웨어 비용 제한 시.")
    ap.add_argument("--shot-robust", action="store_true",
                    help="readout_scaler=False 로 학습 → shot-based 환경(하드웨어)에서 더 강건. "
                         "noiseless MAE는 약간 ↓ 되지만 shot overhead가 1/3로 감소. "
                         "IonQ 실행에 권장.")
    ap.add_argument("--out-dir", default=_HERE,
                    help="아티팩트 저장 디렉토리")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    nq = C.CONFIG_ATOMS[args.config]
    print("=" * 70)
    print(f" MGQC-BT 학습 (v6.9 아키텍처)")
    print("=" * 70)
    print(f" config: {args.config}  (n_qubits={nq})")
    print(f" seed:   {args.seed}")
    print(f" target: {args.target}")
    print(f" split:  {args.split_level}  (v6.9: molecule)")
    print(f" rounds: {args.n_rounds}")
    print(f" epochs: {args.epochs}  lr={args.lr}  optimizer=adam")

    # ---- 1. Arch 구성 ----
    if args.shot_robust:
        print(f" mode:   shot-robust (readout_scaler=False)  ← IonQ 하드웨어 실행 권장")
    arch = make_v69_arch(n_qubits=nq, n_rounds=args.n_rounds,
                          epochs=args.epochs, lr=args.lr, target=args.target,
                          readout_scaler=not args.shot_robust)
    print(f"\n[arch] n_var_params = {arch.n_var_params()}  "
          f"(1 reupload × {arch.n_rounds} rounds × {arch.n_qubits} qubits × 2)")

    # ---- 2. Dataset 로드 (molecule-level split) ----
    t0 = time.time()
    ds = C.Dataset(args.config, arch, split_level=args.split_level, seed=args.seed)
    print(f"[data] N={len(ds.mols)}  "
          f"train={len(ds.train_idx)}  test={len(ds.test_idx)}")
    print(f"[data] n_bond_types = {ds.n_types_train}  comp_R² = {ds.comp_r2:.4f}")
    print(f"[data] y_res range: {ds.resid.min():.2f} ~ {ds.resid.max():.2f} eV")

    # ---- 3. 학습 ----
    print(f"\n[train] 시작 ...")
    trainer = C.Trainer(ds, arch, seed=args.seed, device="lightning.qubit")
    trainer.fit(verbose=args.verbose)
    print(f"[train] 완료. loss {trainer.loss_first:.4f} → {trainer.loss_last:.4f}")

    # ---- 4. 평가 ----
    metrics, pred = trainer.evaluate()
    print(f"\n[eval] test MAE = {metrics['mae']:.4f} eV")
    print(f"[eval] test R²  = {metrics['r2']:.4f}")
    if not np.isnan(metrics.get('isomer_mae', np.nan)):
        print(f"[eval] isomer MAE = {metrics['isomer_mae']:.4f} eV  "
              f"(n_groups={metrics['n_isomer_groups']})")

    print(f"\n[eval] 학습된 α (bond type별):")
    alpha_vec = trainer.comp.alpha_vec(trainer.alpha)
    for t, key in enumerate(ds.type_keys):
        if t < len(alpha_vec):
            print(f"       [{t:2d}] {C.type_label(key):<10}  "
                  f"n_bonds={ds.type_count[key]:>5}  α={alpha_vec[t]:+.4f}")

    # ---- 5. IonQ bundle 저장 ----
    print(f"\n[export] IonQ 아티팩트 저장 ...")
    suffix = f"_{args.config}" + ("_shotrobust" if args.shot_robust else "")
    out_bundle = os.path.join(args.out_dir, f"mgqc_bt_bundle{suffix}.npz")
    out_test = os.path.join(args.out_dir, f"mgqc_bt_test{suffix}.npz")
    out_meta = os.path.join(args.out_dir, f"mgqc_bt_meta{suffix}.json")
    export_bundle(ds, trainer, arch, out_bundle, out_test, out_meta,
                   n_test_cap=args.n_test_cap)

    elapsed = time.time() - t0
    print(f"\n[done] 총 {elapsed:.1f}s")
    print(f"\n다음 단계: 이 3개 파일을 IonQ 실행 환경으로 복사 후")
    print(f"  python mgqc_ionq_batch.py --config {args.config} --shots 1024")


if __name__ == "__main__":
    main()
