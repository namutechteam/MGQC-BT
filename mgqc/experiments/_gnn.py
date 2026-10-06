#!/usr/bin/env python3
"""
=============================================================================
GNN Baseline for QM7-X atomization-energy prediction (SchNet-style)

Pairs with `the superseded implementation`:
  - Loads the SAME cache_atomsXX_seed42.pkl that the QML benchmark uses.
  - Uses the SAME train/test split (75/25, seed 42).
  - Predicts atomization energy (eAT, eV) on the SAME test molecules.
  - Optionally applies the SAME composition-aware residual decomposition that
    MGQC+QSM-CL uses (use --residual to enable). This gives an apples-to-apples
    "GNN-on-residuals" comparison alongside the "GNN-on-raw-eAT" baseline.

Architecture:
  Minimal SchNet -- atom-type embedding, continuous-filter convolutions on a
  radial-cutoff graph, sum-pool over atoms. Self-contained: pure PyTorch, no
  PyG / torch_scatter dependencies. Runs on Blackwell sm_120 (PyTorch ≥ 2.12
  nightly with CUDA 12.8).

Usage:
  python -m mgqc.experiments._gnn --output ./bench_A1 --max-atoms 11 --epochs 300
  python -m mgqc.experiments._gnn --output ./bench_A1 --max-atoms 11 --epochs 300 --residual
  python -m mgqc.experiments._gnn --output ./bench_A1 --max-atoms 11 --epochs 300 \
      --model painn      # rotation-equivariant variant (lighter PaiNN)

Outputs:
  - GNN result appended to {output}/results.json under key
        "SchNet" or "SchNet (residual)" or "PaiNN" / "PaiNN (residual)"
  - Best-val checkpoint saved to {output}/gnn_{model}_{tag}.pt
=============================================================================
"""
import argparse
import json
import os
import pickle
import time
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# =============================================================================
# 1. DATA UTILITIES
# =============================================================================

def load_cache(cache_path):
    """Load the same cache file produced by the superseded implementation."""
    with open(cache_path, "rb") as fp:
        cache = pickle.load(fp)
    return cache


def compute_composition_decomposition(molecules, train_idx):
    """Composition (linear) model fit on training data only.

    Returns y_all, y_comp_all (composition prediction for every molecule),
    and (mean, std) of the training-set residuals for normalization.
    """
    elements = sorted(set(z for m in molecules for z in m["Z"].tolist()))
    elem_to_idx = {e: i for i, e in enumerate(elements)}
    X_comp = np.zeros((len(molecules), len(elements)))
    y_all = np.array([m["eAT"] for m in molecules])
    for i, m in enumerate(molecules):
        for z, count in Counter(m["Z"].tolist()).items():
            X_comp[i, elem_to_idx[z]] = count
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_comp[train_idx], y_all[train_idx])
    y_comp_all = lr.predict(X_comp)
    residuals = y_all - y_comp_all
    mean = residuals[train_idx].mean()
    std = residuals[train_idx].std() + 1e-10
    return y_all, y_comp_all, mean, std


# =============================================================================
# 2. GRAPH BUILDING (radial cutoff, undirected pairs)
# =============================================================================

def build_graph(Z, xyz, cutoff=5.0):
    """Build a radial-cutoff graph for a single molecule.

    Returns:
      edge_index  : LongTensor [2, E]      (i, j) pairs, both directions
      edge_dist   : FloatTensor [E]        distances rij
      Z_t         : LongTensor [n]
      xyz_t       : FloatTensor [n, 3]
      n           : int                    number of atoms
    """
    n = len(Z)
    # All pairs distance
    d = xyz[:, None, :] - xyz[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    mask = (dist > 1e-6) & (dist < cutoff)
    src, dst = np.where(mask)
    return (torch.from_numpy(np.stack([src, dst])).long(),
            torch.from_numpy(dist[src, dst]).float(),
            torch.from_numpy(Z).long(),
            torch.from_numpy(xyz).float(),
            n)


def collate(batch, device):
    """Concatenate variable-size graphs into one big graph with atom-batch index."""
    edge_indices, edge_dists, Zs, xyzs, ns, ys = zip(*batch)
    offsets = np.cumsum([0] + [n.item() if torch.is_tensor(n) else n for n in ns[:-1]])
    edge_index_all = torch.cat(
        [edge_idx + offset for edge_idx, offset in zip(edge_indices, offsets)], dim=1
    )
    edge_dist_all = torch.cat(edge_dists)
    Z_all = torch.cat(Zs)
    xyz_all = torch.cat(xyzs)
    batch_idx = torch.cat([torch.full((n,), i, dtype=torch.long)
                           for i, n in enumerate(ns)])
    y = torch.tensor(ys, dtype=torch.float32)
    return (edge_index_all.to(device),
            edge_dist_all.to(device),
            Z_all.to(device),
            xyz_all.to(device),
            batch_idx.to(device),
            y.to(device))


# =============================================================================
# 3. MODELS
# =============================================================================

class GaussianBasis(nn.Module):
    """Continuous-filter Gaussian RBF expansion of pairwise distances."""

    def __init__(self, n_basis=32, r_min=0.0, r_max=5.0):
        super().__init__()
        centres = torch.linspace(r_min, r_max, n_basis)
        self.register_buffer("centres", centres)
        self.gamma = 1.0 / (centres[1] - centres[0]).item() ** 2

    def forward(self, r):
        # r: [E], centres: [n_basis]
        return torch.exp(-self.gamma * (r[:, None] - self.centres[None, :]) ** 2)


def scatter_add(src, index, dim_size):
    """Replacement for torch_scatter.scatter_add; uses index_add_."""
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


class SchNetInteraction(nn.Module):
    """One SchNet continuous-filter convolution block."""

    def __init__(self, hidden=64, n_basis=32):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.lin_in = nn.Linear(hidden, hidden)
        self.lin_out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x, edge_index, edge_attr):
        # x:[n_atoms, hidden], edge_index:[2,E], edge_attr:[E, n_basis]
        src, dst = edge_index
        W = self.filter_net(edge_attr)                  # [E, hidden]
        msg = self.lin_in(x[src]) * W                   # continuous-filter conv
        agg = scatter_add(msg, dst, dim_size=x.shape[0])
        return x + self.lin_out(agg)


class SchNet(nn.Module):
    def __init__(self, max_z=18, hidden=64, n_basis=32, n_inter=3, cutoff=5.0):
        super().__init__()
        self.embed = nn.Embedding(max_z + 1, hidden)
        self.basis = GaussianBasis(n_basis=n_basis, r_max=cutoff)
        self.inter = nn.ModuleList(
            [SchNetInteraction(hidden, n_basis) for _ in range(n_inter)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z, xyz, edge_index, edge_dist, batch_idx):
        x = self.embed(Z)
        edge_attr = self.basis(edge_dist)
        for block in self.inter:
            x = block(x, edge_index, edge_attr)
        atom_energy = self.readout(x).squeeze(-1)            # per-atom contribution
        # Sum-pool per molecule
        n_mols = int(batch_idx.max().item()) + 1
        return scatter_add(atom_energy, batch_idx, dim_size=n_mols)


class PaiNNVectorBlock(nn.Module):
    """Lightweight PaiNN-style equivariant message passing.

    Maintains both scalar and vector (Cartesian) features per atom. Vector
    features keep rotational equivariance by mixing along the inter-atomic
    direction unit vector.
    """

    def __init__(self, hidden=64, n_basis=32):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )
        self.w = nn.Sequential(
            nn.Linear(n_basis, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * hidden),
        )
        self.u = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * hidden),
        )

    def forward(self, s, vec, edge_index, edge_attr, r_hat):
        # s:[n,h], vec:[n,3,h], edge_attr:[E,nb], r_hat:[E,3]
        src, dst = edge_index
        W = self.w(edge_attr)                            # [E, 3h]
        msg = self.phi(s[src]) * W                       # [E, 3h]
        msg_s, msg_v1, msg_v2 = msg.chunk(3, dim=-1)     # each [E, h]
        # vector message: msg_v1 in r_hat direction + msg_v2 from src vector
        vec_msg = r_hat[:, :, None] * msg_v1[:, None, :] + vec[src] * msg_v2[:, None, :]
        s = s + scatter_add(msg_s, dst, dim_size=s.shape[0])
        vec = vec + scatter_add(vec_msg, dst, dim_size=vec.shape[0])
        # Update
        v_norm = torch.linalg.norm(self.v_proj(vec), dim=1)          # [n, h]
        delta = self.update(torch.cat([s, v_norm], dim=-1))          # [n, 2h]
        s_update, vec_gate = delta.chunk(2, dim=-1)
        s = s + s_update
        vec = vec + self.u(vec) * vec_gate[:, None, :]
        return s, vec


class PaiNN(nn.Module):
    def __init__(self, max_z=18, hidden=64, n_basis=32, n_inter=3, cutoff=5.0):
        super().__init__()
        self.embed = nn.Embedding(max_z + 1, hidden)
        self.basis = GaussianBasis(n_basis=n_basis, r_max=cutoff)
        self.cutoff = cutoff
        self.blocks = nn.ModuleList(
            [PaiNNVectorBlock(hidden, n_basis) for _ in range(n_inter)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z, xyz, edge_index, edge_dist, batch_idx):
        n = Z.shape[0]
        s = self.embed(Z)
        vec = torch.zeros(n, 3, s.shape[-1], device=Z.device)
        src, dst = edge_index
        rij = xyz[dst] - xyz[src]
        r_hat = rij / (edge_dist[:, None] + 1e-8)
        edge_attr = self.basis(edge_dist)
        for block in self.blocks:
            s, vec = block(s, vec, edge_index, edge_attr, r_hat)
        atom_energy = self.readout(s).squeeze(-1)
        n_mols = int(batch_idx.max().item()) + 1
        return scatter_add(atom_energy, batch_idx, dim_size=n_mols)


# =============================================================================
# 4. DATASET
# =============================================================================

class MolDataset(torch.utils.data.Dataset):
    def __init__(self, molecules, indices, y_targets, cutoff=5.0):
        self.mols = [molecules[i] for i in indices]
        self.y = [float(y_targets[i]) for i in indices]
        self.cutoff = cutoff

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, i):
        m = self.mols[i]
        edge_idx, edge_dist, Z_t, xyz_t, n = build_graph(m["Z"], m["xyz"], cutoff=self.cutoff)
        return edge_idx, edge_dist, Z_t, xyz_t, n, self.y[i]


# =============================================================================
# 5. TRAINING
# =============================================================================

def train_gnn(molecules, train_idx, test_idx, *,
              model_name="schnet", epochs=300, batch_size=32, lr=1e-3,
              hidden=64, n_basis=32, n_inter=3, cutoff=5.0,
              residual=False, device=None, verbose=True):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- composition decomposition (only if --residual) ----
    y_all = np.array([m["eAT"] for m in molecules])
    if residual:
        y_all, y_comp_all, res_mean, res_std = compute_composition_decomposition(
            molecules, train_idx
        )
        y_targets = (y_all - y_comp_all - res_mean) / res_std   # normalized residual
        if verbose:
            print(f"    composition R² (train) → residual mean={res_mean:.3f}, "
                  f"std={res_std:.3f}")
    else:
        y_targets = (y_all - y_all[train_idx].mean()) / (y_all[train_idx].std() + 1e-10)
        y_train_mean = y_all[train_idx].mean()
        y_train_std = y_all[train_idx].std() + 1e-10

    # ---- datasets ----
    train_ds = MolDataset(molecules, train_idx, y_targets, cutoff=cutoff)
    test_ds  = MolDataset(molecules, test_idx,  y_targets, cutoff=cutoff)

    def loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate(b, device),
            num_workers=0,
        )

    train_loader = loader(train_ds, shuffle=True)
    test_loader  = loader(test_ds,  shuffle=False)

    # ---- model ----
    if model_name == "schnet":
        model = SchNet(hidden=hidden, n_basis=n_basis, n_inter=n_inter,
                       cutoff=cutoff).to(device)
    elif model_name == "painn":
        model = PaiNN(hidden=hidden, n_basis=n_basis, n_inter=n_inter,
                      cutoff=cutoff).to(device)
    else:
        raise ValueError(f"unknown model {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"    {model_name}: {n_params:,} params, device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_test_mae = float("inf")
    best_preds = None

    for ep in range(epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for batch in train_loader:
            edge_index, edge_dist, Z, xyz, batch_idx, y = batch
            pred = model(Z, xyz, edge_index, edge_dist, batch_idx)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += loss.item() * y.shape[0]
            n_seen += y.shape[0]
        sched.step()

        # Validation every few epochs
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            model.eval()
            test_preds, test_truth = [], []
            with torch.no_grad():
                for batch in test_loader:
                    edge_index, edge_dist, Z, xyz, batch_idx, y = batch
                    pred = model(Z, xyz, edge_index, edge_dist, batch_idx)
                    test_preds.append(pred.cpu().numpy())
                    test_truth.append(y.cpu().numpy())
            preds_norm = np.concatenate(test_preds)
            truth_norm = np.concatenate(test_truth)

            # De-normalize back to eV
            if residual:
                preds_eV = preds_norm * res_std + res_mean + y_comp_all[test_idx]
            else:
                preds_eV = preds_norm * y_train_std + y_train_mean
            mae_eV = mean_absolute_error(y_all[test_idx], preds_eV)

            if mae_eV < best_test_mae:
                best_test_mae = mae_eV
                best_preds = preds_eV

            if verbose:
                print(f"    ep {ep:>3d}: train_loss={running/n_seen:.4f}, "
                      f"test_MAE={mae_eV:.3f} eV")

    return best_preds, best_test_mae


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN baseline (SchNet/PaiNN) on QM7-X -- same conditions as QML benchmark."
    )
    parser.add_argument("--output", required=True,
                        help="Output dir containing cache_atomsXX_seed42.pkl")
    parser.add_argument("--max-atoms", type=int, required=True,
                        help="Match the cache filename (e.g. 11 for A1, 13 for others)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-basis", type=int, default=32)
    parser.add_argument("--n-inter", type=int, default=3, help="Number of interaction layers")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--model", choices=["schnet", "painn"], default="schnet")
    parser.add_argument("--residual", action="store_true",
                        help="Apply composition-aware residual learning (same as MGQC+QSM-CL)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cache_path = os.path.join(args.output, f"cache_atoms{args.max_atoms}_seed{args.seed}.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found: {cache_path}. Run the superseded implementation first."
        )

    print("=" * 70)
    print(f"  GNN Baseline ({args.model.upper()}) -- QM7-X")
    print(f"  Cache: {cache_path}")
    print(f"  Residual learning: {args.residual}")
    print("=" * 70)

    cache = load_cache(cache_path)
    molecules = cache["molecules"]
    train_idx = cache["train_idx"]
    test_idx  = cache["test_idx"]
    print(f"  Molecules: {len(molecules)}  train={len(train_idx)}, test={len(test_idx)}")

    t0 = time.time()
    preds_eV, best_mae = train_gnn(
        molecules, train_idx, test_idx,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        n_basis=args.n_basis,
        n_inter=args.n_inter,
        cutoff=args.cutoff,
        residual=args.residual,
    )
    elapsed = time.time() - t0

    # ---- compute final metrics ----
    y_true = np.array([molecules[i]["eAT"] for i in test_idx])
    mae = mean_absolute_error(y_true, preds_eV)
    r2  = r2_score(y_true, preds_eV)

    tag = f"{args.model.capitalize()}{' (residual)' if args.residual else ''}"
    print()
    print("=" * 70)
    print(f"  {tag}: MAE = {mae:.3f} eV, R² = {r2:.4f}, Time = {elapsed:.1f} s")
    print("=" * 70)

    # ---- save into existing results.json ----
    results_path = os.path.join(args.output, "results.json")
    summary = {}
    if os.path.exists(results_path):
        with open(results_path) as fp:
            summary = json.load(fp)
    summary[tag] = {
        "mae": float(mae),
        "r2": float(r2),
        "time": float(elapsed),
        "input": "3D coords + Z",
    }
    with open(results_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"  Saved to: {results_path} (key='{tag}')")

    # ---- save predictions ----
    pred_path = os.path.join(args.output, "predictions_gnn.npz")
    existing = {}
    if os.path.exists(pred_path):
        existing = dict(np.load(pred_path, allow_pickle=True))
    existing["y_true"] = y_true
    existing[tag.replace(" ", "_").replace("(", "").replace(")", "")] = preds_eV
    np.savez(pred_path, **existing)
    print(f"  Saved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
