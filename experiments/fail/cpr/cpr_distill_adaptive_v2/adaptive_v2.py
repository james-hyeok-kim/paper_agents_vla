"""
CPR-Distill Adaptive v2: fix the regularization collapse from Experiment 5.

Sweep showed adaptive boost α collapsed to ~0 because weight_reg=1e-3 penalty
pulled α toward 0 stronger than the loss signal pushed it toward useful values.

Fix:
- weight_reg = 0 (remove penalty)
- boost head bias initialized to logit(0.3) ≈ -0.85 so initial sigmoid ≈ 0.3,
  giving alpha = max_boost * 0.3 = 1.5 (matches sweet spot start)
- max_boost = 5.0 (room to grow)

Compare to: fixed factor=1.5 (sweet spot from sweep) and factor=1.0 (baseline).
"""
import os, sys, json, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_sweep")
from m0_smoke import LiberoBCDataset, TinyBC
from sweep import TinyBCWithAdaptive, eval_model, train_fixed_factor

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_adaptive_v2"
OUT_BIG = "/data/jameskimh/cpr_distill_adaptive_v2"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")
torch.manual_seed(42); np.random.seed(42)


def train_adaptive_v2(train_loader, val_loader, n_epochs=3, lr=1e-3,
                       max_boost=5.0, weight_reg=0.0, bias_init_logit=-0.85):
    torch.manual_seed(42); np.random.seed(42)
    model = TinyBCWithAdaptive().to(DEVICE)
    # Initialize boost head bias to give starting α ≈ 1.5
    with torch.no_grad():
        for m in model.weight_head:
            if isinstance(m, nn.Linear) and m.out_features == 1:
                m.bias.fill_(bias_init_logit)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(train_loader))
    boost_hist = []
    for ep in range(n_epochs):
        model.train()
        ep_boost_sum = 0.0; ep_boost_n = 0
        for batch in train_loader:
            img = batch["img"].to(DEVICE, non_blocking=True)
            state = batch["state"].to(DEVICE, non_blocking=True)
            target = batch["action"].to(DEVICE, non_blocking=True)
            contact = batch["contact"].to(DEVICE, non_blocking=True)
            pred, boost_logit = model(img, state)
            alpha = max_boost * torch.sigmoid(boost_logit)
            err_per = ((pred - target) ** 2).sum(-1)
            w = 1.0 + alpha * contact
            loss = (err_per * w).mean()
            if weight_reg > 0:
                loss = loss + weight_reg * (alpha ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            with torch.no_grad():
                cmb = contact > 0.5
                if cmb.any():
                    ep_boost_sum += alpha[cmb].sum().item()
                    ep_boost_n += cmb.sum().item()
        boost_hist.append(ep_boost_sum / max(ep_boost_n, 1))
    return model, boost_hist


def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Adaptive v2 — fixed regularization collapse")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    train_ds = LiberoBCDataset(hdf5, train=True, train_frac=0.9, seed=0)
    val_ds = LiberoBCDataset(hdf5, train=False, train_frac=0.9, seed=0)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    results = {}
    print("\n[1/3] Baseline factor=1.0 ...")
    m = train_fixed_factor(1.0, train_loader, val_loader, n_epochs=3)
    results["baseline_1.0"] = eval_model(m, val_loader, DEVICE)
    print(f"  contact={results['baseline_1.0']['contact']:.5f} free={results['baseline_1.0']['free']:.5f} all={results['baseline_1.0']['all']:.5f}")

    print("\n[2/3] Sweet spot factor=1.5 ...")
    m = train_fixed_factor(1.5, train_loader, val_loader, n_epochs=3)
    results["fixed_1.5"] = eval_model(m, val_loader, DEVICE)
    print(f"  contact={results['fixed_1.5']['contact']:.5f} free={results['fixed_1.5']['free']:.5f} all={results['fixed_1.5']['all']:.5f}")

    print("\n[3/3] Adaptive v2 (reg=0, bias init=+1.5) ...")
    m, boost_hist = train_adaptive_v2(train_loader, val_loader, n_epochs=3, max_boost=5.0, weight_reg=0.0, bias_init_logit=-0.85)
    results["adaptive_v2"] = eval_model(m, val_loader, DEVICE, adaptive=True)
    results["adaptive_v2"]["boost_history"] = boost_hist
    torch.save(m.state_dict(), os.path.join(OUT_BIG, "adaptive_v2.pt"))
    print(f"  contact={results['adaptive_v2']['contact']:.5f} free={results['adaptive_v2']['free']:.5f} all={results['adaptive_v2']['all']:.5f}")
    print(f"  Boost α history (mean over contact samples): {boost_hist}")

    # Analysis
    b = results["baseline_1.0"]
    print("\n--- Gains vs baseline (factor=1.0) ---")
    for k in ["fixed_1.5", "adaptive_v2"]:
        r = results[k]
        dc = (b['contact'] - r['contact']) / b['contact'] * 100
        df = (b['free'] - r['free']) / b['free'] * 100
        da = (b['all'] - r['all']) / b['all'] * 100
        print(f"  {k}: contact {dc:+.2f}%, free {df:+.2f}%, all {da:+.2f}%")
        results[k]["gain_contact_pct"] = dc
        results[k]["gain_free_pct"] = df
        results[k]["gain_overall_pct"] = da

    adaptive_wins = results["adaptive_v2"]["gain_overall_pct"] > results["fixed_1.5"]["gain_overall_pct"]
    verdict = "ADAPTIVE BEATS FIXED 1.5×" if adaptive_wins else "Adaptive ≤ Fixed 1.5× — fixed remains primary"
    print(f"\n>>> {verdict} <<<")

    out = {
        "elapsed_sec": time.time() - t0, "device": str(DEVICE),
        "results": {k: {kk: float(vv) if isinstance(vv,(int,float,np.floating)) else vv for kk,vv in v.items()} for k,v in results.items()},
        "verdict": verdict, "adaptive_wins_overall": bool(adaptive_wins),
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        labels = ["baseline_1.0", "fixed_1.5", "adaptive_v2"]
        cm = [results[k]["contact"] for k in labels]
        fm = [results[k]["free"] for k in labels]
        x = np.arange(3); w = 0.35
        ax[0].bar(x - w/2, cm, w, label="contact MSE"); ax[0].bar(x + w/2, fm, w, label="free MSE")
        ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=15); ax[0].legend(); ax[0].grid(alpha=0.3, axis='y')
        ax[0].set_title("Adaptive v2 vs Fixed 1.5× vs Baseline")
        ax[1].plot(boost_hist, marker='o'); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("mean α on contact samples")
        ax[1].set_title("Adaptive boost α evolution"); ax[1].grid(alpha=0.3); ax[1].axhline(1.5, color='r', linestyle='--', label='target = 1.5')
        ax[1].legend()
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "adaptive_v2.png"), dpi=140)
        print(f"Plot: {os.path.join(OUT_USER, 'adaptive_v2.png')}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
