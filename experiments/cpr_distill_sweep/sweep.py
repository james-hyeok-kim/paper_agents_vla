"""
CPR-Distill Sweep: find reweight factor sweet spot to minimize free-phase tradeoff.

M0 showed CPR-3x reduces contact MSE by 17% but free MSE degrades 19.6%.
Hypothesis: smaller factor (1.5x-2x) maintains contact gain with less free-phase damage.

Conditions:
- Reweight factors: {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
- Plus: Adaptive learnable weight α_t = 1 + 2*sigmoid(MLP(state)) * contact_mask
       (only boost when contact_mask=1, learns boost magnitude per state)

Output:
- /experiments/cpr_distill_sweep/: scripts, results.json, sweep.png
- /data/jameskimh/cpr_distill_sweep/: checkpoints
"""
import os
import sys
import json
import time
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse M0 dataset class
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import LiberoBCDataset, TinyBC

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_sweep"
OUT_BIG = "/data/jameskimh/cpr_distill_sweep"
os.makedirs(OUT_USER, exist_ok=True)
os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")

torch.manual_seed(42)
np.random.seed(42)


class TinyBCWithAdaptive(nn.Module):
    """Same as TinyBC but with an extra adaptive-weight head."""
    def __init__(self, img_size=128, state_dim=8, action_dim=7, hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        # Adaptive weight head: predicts boost factor ∈ [0, max_boost]
        self.weight_head = nn.Sequential(
            nn.Linear(256 + hidden, 128), nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, img, state):
        h_img = self.cnn(img)
        h_state = self.state_enc(state)
        h = torch.cat([h_img, h_state], dim=-1)
        action = self.head(h)
        boost = self.weight_head(h).squeeze(-1)  # (B,) raw logit
        return action, boost


def eval_model(model, val_loader, device, adaptive=False):
    model.eval()
    err_sum_contact = 0.0
    err_sum_free = 0.0
    err_sum_all = 0.0
    n_contact = 0
    n_free = 0
    n_all = 0
    with torch.no_grad():
        for batch in val_loader:
            img = batch["img"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            contact = batch["contact"].to(device, non_blocking=True)
            if adaptive:
                pred, _ = model(img, state)
            else:
                pred = model(img, state)
            per_sample = ((pred - target) ** 2).sum(-1)
            cm = (contact > 0.5)
            err_sum_contact += per_sample[cm].sum().item()
            err_sum_free += per_sample[~cm].sum().item()
            err_sum_all += per_sample.sum().item()
            n_contact += cm.sum().item()
            n_free += (~cm).sum().item()
            n_all += img.size(0)
    return {
        "contact": err_sum_contact / max(n_contact, 1),
        "free": err_sum_free / max(n_free, 1),
        "all": err_sum_all / max(n_all, 1),
    }


def train_fixed_factor(factor, train_loader, val_loader, n_epochs=3, lr=1e-3):
    """Train TinyBC with fixed reweight factor."""
    torch.manual_seed(42)
    np.random.seed(42)
    model = TinyBC().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(train_loader))
    for ep in range(n_epochs):
        model.train()
        for batch in train_loader:
            img = batch["img"].to(DEVICE, non_blocking=True)
            state = batch["state"].to(DEVICE, non_blocking=True)
            target = batch["action"].to(DEVICE, non_blocking=True)
            contact = batch["contact"].to(DEVICE, non_blocking=True)
            pred = model(img, state)
            err_per = ((pred - target) ** 2).sum(-1)
            w = 1.0 + (factor - 1.0) * contact  # 1x in free, factor in contact
            loss = (err_per * w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    return model


def train_adaptive(train_loader, val_loader, n_epochs=3, lr=1e-3, max_boost=5.0,
                   weight_reg=1e-3):
    """
    Train TinyBC with adaptive learnable boost.
    boost α_t = max_boost * sigmoid(boost_logit), but gated by contact_mask.
    Final weight: w_t = 1 + α_t * contact_mask
    Loss: (err * w).mean() + weight_reg * E[α^2] (don't blow up boost)
    """
    torch.manual_seed(42)
    np.random.seed(42)
    model = TinyBCWithAdaptive().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(train_loader))
    avg_boost_history = []
    for ep in range(n_epochs):
        model.train()
        ep_boost_sum = 0.0
        ep_boost_n = 0
        for batch in train_loader:
            img = batch["img"].to(DEVICE, non_blocking=True)
            state = batch["state"].to(DEVICE, non_blocking=True)
            target = batch["action"].to(DEVICE, non_blocking=True)
            contact = batch["contact"].to(DEVICE, non_blocking=True)
            pred, boost_logit = model(img, state)
            alpha = max_boost * torch.sigmoid(boost_logit)  # (B,) ∈ [0, max_boost]
            err_per = ((pred - target) ** 2).sum(-1)
            w = 1.0 + alpha * contact
            loss = (err_per * w).mean() + weight_reg * (alpha ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                contact_mask_bool = contact > 0.5
                if contact_mask_bool.any():
                    ep_boost_sum += alpha[contact_mask_bool].sum().item()
                    ep_boost_n += contact_mask_bool.sum().item()
        avg_boost = ep_boost_sum / max(ep_boost_n, 1)
        avg_boost_history.append(avg_boost)
    return model, avg_boost_history


def main():
    print("=" * 60)
    print("CPR-Distill Sweep: Reweight Factor + Adaptive")
    print(f"Device: {DEVICE}")
    print(f"Output (user): {OUT_USER}")
    print(f"Output (big):  {OUT_BIG}")
    print("=" * 60)
    t_start = time.time()

    print("\n[1/3] Loading data (cached)...")
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    train_ds = LiberoBCDataset(hdf5_files, train=True, train_frac=0.9, seed=0)
    val_ds = LiberoBCDataset(hdf5_files, train=False, train_frac=0.9, seed=0)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    print("\n[2/3] Sweeping reweight factors...")
    factors = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    sweep_results = {}
    for f in factors:
        t0 = time.time()
        model = train_fixed_factor(f, train_loader, val_loader, n_epochs=3)
        eval_stats = eval_model(model, val_loader, DEVICE)
        sweep_results[f"factor_{f}"] = eval_stats
        torch.save(model.state_dict(), os.path.join(OUT_BIG, f"factor_{f}.pt"))
        elapsed = time.time() - t0
        print(f"  factor={f}: contact={eval_stats['contact']:.5f}, "
              f"free={eval_stats['free']:.5f}, all={eval_stats['all']:.5f} ({elapsed:.1f}s)")

    print("\n[3/3] Training adaptive learnable boost...")
    t0 = time.time()
    adaptive_model, boost_hist = train_adaptive(train_loader, val_loader, n_epochs=3, max_boost=5.0)
    adaptive_stats = eval_model(adaptive_model, val_loader, DEVICE, adaptive=True)
    sweep_results["adaptive"] = adaptive_stats
    sweep_results["adaptive"]["boost_history"] = boost_hist
    torch.save(adaptive_model.state_dict(), os.path.join(OUT_BIG, "adaptive.pt"))
    elapsed = time.time() - t0
    print(f"  adaptive: contact={adaptive_stats['contact']:.5f}, "
          f"free={adaptive_stats['free']:.5f}, all={adaptive_stats['all']:.5f} ({elapsed:.1f}s)")
    print(f"  Adaptive boost history (mean α over contact samples per epoch): {boost_hist}")

    elapsed_total = time.time() - t_start
    print(f"\n=== Total elapsed: {elapsed_total:.1f}s ===")

    # ===== Analysis =====
    print("\n" + "=" * 60)
    print("SWEEP ANALYSIS")
    print("=" * 60)
    baseline = sweep_results["factor_1.0"]  # uniform (factor=1)
    print(f"\nBaseline (factor=1.0): contact={baseline['contact']:.5f}, free={baseline['free']:.5f}")
    print(f"\n{'Factor':<12} {'Contact MSE':<13} {'Free MSE':<11} {'Overall':<10} "
          f"{'Δ Contact %':<13} {'Δ Free %':<11} {'Δ Overall %':<13}")
    print("-" * 90)
    summary = []
    for f in factors:
        r = sweep_results[f"factor_{f}"]
        dc = (baseline['contact'] - r['contact']) / baseline['contact'] * 100
        df = (baseline['free'] - r['free']) / baseline['free'] * 100
        da = (baseline['all'] - r['all']) / baseline['all'] * 100
        print(f"{f:<12} {r['contact']:<13.5f} {r['free']:<11.5f} {r['all']:<10.5f} "
              f"{dc:>+10.2f}   {df:>+8.2f}   {da:>+10.2f}")
        summary.append({"factor": f, "contact_mse": r['contact'], "free_mse": r['free'],
                        "overall_mse": r['all'],
                        "contact_gain_pct": dc, "free_gain_pct": df, "overall_gain_pct": da})
    # adaptive
    r = adaptive_stats
    dc = (baseline['contact'] - r['contact']) / baseline['contact'] * 100
    df = (baseline['free'] - r['free']) / baseline['free'] * 100
    da = (baseline['all'] - r['all']) / baseline['all'] * 100
    print(f"{'adaptive':<12} {r['contact']:<13.5f} {r['free']:<11.5f} {r['all']:<10.5f} "
          f"{dc:>+10.2f}   {df:>+8.2f}   {da:>+10.2f}")
    summary.append({"factor": "adaptive", "contact_mse": r['contact'], "free_mse": r['free'],
                    "overall_mse": r['all'],
                    "contact_gain_pct": dc, "free_gain_pct": df, "overall_gain_pct": da,
                    "mean_boost_final_ep": float(boost_hist[-1])})

    # ===== Best factor selection =====
    print("\n--- Sweet Spot Analysis ---")
    # Want: contact_gain > 0, free_gain ≥ 0 (or close), maximize overall_gain
    best_overall = max(summary, key=lambda x: x["overall_gain_pct"])
    best_pareto = max(
        (s for s in summary if s["contact_gain_pct"] > 0 and s["free_gain_pct"] >= -2.0),
        key=lambda x: x["contact_gain_pct"], default=None)
    print(f"  Best overall MSE gain: factor={best_overall['factor']} ({best_overall['overall_gain_pct']:+.2f}%)")
    if best_pareto:
        print(f"  Pareto-best (contact↑ + free not worse than -2%): "
              f"factor={best_pareto['factor']} (contact +{best_pareto['contact_gain_pct']:.2f}%, "
              f"free {best_pareto['free_gain_pct']:+.2f}%)")
    else:
        print(f"  No Pareto-best found (all factors degrade free MSE > 2%)")

    # Save
    out = {
        "elapsed_sec": elapsed_total,
        "device": str(DEVICE),
        "config": {"n_epochs": 3, "batch_size": bs, "max_boost_adaptive": 5.0,
                   "weight_reg": 1e-3, "factors": factors},
        "results": {k: {kk: (float(vv) if isinstance(vv, (int,float,np.floating)) else vv)
                        for kk, vv in v.items()} for k, v in sweep_results.items()},
        "summary": summary,
        "best_overall": best_overall,
        "best_pareto": best_pareto,
        "baseline_factor1.0": {kk: float(vv) for kk, vv in baseline.items() if isinstance(vv,(int,float,np.floating))},
    }
    json_path = os.path.join(OUT_USER, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults JSON: {json_path}")

    # ===== Plot =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Plot 1: contact / free MSE vs factor
        c = [sweep_results[f"factor_{f}"]['contact'] for f in factors]
        fr = [sweep_results[f"factor_{f}"]['free'] for f in factors]
        ov = [sweep_results[f"factor_{f}"]['all'] for f in factors]
        axes[0].plot(factors, c, 'o-', label='Contact MSE', color='C0', linewidth=2)
        axes[0].plot(factors, fr, 's-', label='Free MSE', color='C1', linewidth=2)
        axes[0].plot(factors, ov, '^--', label='Overall MSE', color='C2', linewidth=1.5)
        axes[0].axhline(adaptive_stats['contact'], color='C0', linestyle=':', alpha=0.6, label='Adaptive contact')
        axes[0].axhline(adaptive_stats['free'], color='C1', linestyle=':', alpha=0.6, label='Adaptive free')
        axes[0].set_xlabel('Reweight factor')
        axes[0].set_ylabel('MSE (val)')
        axes[0].set_title('CPR-Distill: MSE vs Reweight Factor')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Pareto curve (contact gain vs free gain)
        cgs = [s['contact_gain_pct'] for s in summary[:-1]]  # exclude adaptive
        fgs = [s['free_gain_pct'] for s in summary[:-1]]
        axes[1].plot(fgs, cgs, 'o-', linewidth=2, color='C3')
        for i, f in enumerate(factors):
            axes[1].annotate(f"{f}x", (fgs[i], cgs[i]), xytext=(5,5), textcoords='offset points', fontsize=9)
        # Adaptive
        axes[1].scatter([summary[-1]['free_gain_pct']], [summary[-1]['contact_gain_pct']],
                        s=120, marker='*', color='gold', edgecolor='black', label='Adaptive', zorder=5)
        axes[1].axhline(0, color='gray', linewidth=0.5)
        axes[1].axvline(0, color='gray', linewidth=0.5)
        axes[1].set_xlabel('Free-phase MSE gain (%) — higher is better')
        axes[1].set_ylabel('Contact-phase MSE gain (%) — higher is better')
        axes[1].set_title('Pareto: Contact gain vs Free gain')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(OUT_USER, "sweep_results.png")
        plt.savefig(plot_path, dpi=140)
        print(f"Plot: {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
