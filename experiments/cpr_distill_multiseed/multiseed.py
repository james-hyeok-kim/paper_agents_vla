"""
CPR-Distill Multi-Seed: statistical significance check.

Run 4 key conditions × 5 seeds each = 20 runs.
Report mean ± std and 95% CI for contact/free/overall MSE.

Conditions:
  - factor=1.0 (baseline)
  - factor=1.5 (sweet spot)
  - factor=3.0 (aggressive)
  - sham_3x (uniform 3x reweight — control)
"""
import os, sys, json, time, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import LiberoBCDataset, TinyBC, eval_model

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_multiseed"
OUT_BIG = "/data/jameskimh/cpr_distill_multiseed"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


def train_with_seed(train_loader, val_loader, mode, factor, seed, n_epochs=3, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
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
            err = ((pred - target) ** 2).sum(-1)
            if mode == "cpr":
                w = 1.0 + (factor - 1.0) * contact
            elif mode == "sham":
                w = torch.full_like(err, factor)
            else:
                raise ValueError(mode)
            loss = (err * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model


def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Multi-Seed (4 conditions × 5 seeds = 20 runs)")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    train_ds = LiberoBCDataset(hdf5, train=True, train_frac=0.9, seed=0)
    val_ds = LiberoBCDataset(hdf5, train=False, train_frac=0.9, seed=0)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    seeds = [0, 1, 2, 3, 4]
    conditions = [
        ("factor_1.0", "cpr", 1.0),
        ("factor_1.5", "cpr", 1.5),
        ("factor_3.0", "cpr", 3.0),
        ("sham_3.0",   "sham", 3.0),
    ]
    raw = {}  # tag -> list of dicts per seed
    for tag, mode, factor in conditions:
        print(f"\n[{tag}] mode={mode}, factor={factor}")
        raw[tag] = []
        for s in seeds:
            t1 = time.time()
            m = train_with_seed(train_loader, val_loader, mode, factor, seed=s)
            ev = eval_model(m, val_loader, DEVICE)
            raw[tag].append({"seed": s, **{k: float(v) for k, v in ev.items()}})
            print(f"  seed {s}: contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} ({time.time()-t1:.1f}s)")

    # Aggregate
    print("\n--- Mean ± Std across 5 seeds ---")
    print(f"{'Condition':<14} {'Contact':>15} {'Free':>15} {'Overall':>15}")
    print("-" * 60)
    agg = {}
    for tag, _, _ in conditions:
        c = np.array([r["contact"] for r in raw[tag]])
        f_ = np.array([r["free"] for r in raw[tag]])
        a = np.array([r["all"] for r in raw[tag]])
        agg[tag] = {
            "contact_mean": float(c.mean()), "contact_std": float(c.std()),
            "free_mean": float(f_.mean()), "free_std": float(f_.std()),
            "all_mean": float(a.mean()), "all_std": float(a.std()),
            "n_seeds": len(seeds),
        }
        print(f"{tag:<14} {c.mean():.5f}±{c.std():.5f}  {f_.mean():.5f}±{f_.std():.5f}  {a.mean():.5f}±{a.std():.5f}")

    # Significance: does factor_1.5 contact mean differ from factor_1.0 by more than 2*sum of stds?
    b_c = agg["factor_1.0"]["contact_mean"]
    b_s = agg["factor_1.0"]["contact_std"]
    cpr_c = agg["factor_1.5"]["contact_mean"]
    cpr_s = agg["factor_1.5"]["contact_std"]
    sham_c = agg["sham_3.0"]["contact_mean"]
    sham_s = agg["sham_3.0"]["contact_std"]

    delta_cpr = b_c - cpr_c
    sigma_combined_cpr = np.sqrt(b_s**2 + cpr_s**2)
    z_cpr = delta_cpr / (sigma_combined_cpr + 1e-9)
    delta_cpr_vs_sham = sham_c - cpr_c
    z_cpr_vs_sham = delta_cpr_vs_sham / np.sqrt(sham_s**2 + cpr_s**2 + 1e-9)
    print(f"\n--- Significance ---")
    print(f"  CPR-1.5× vs factor-1.0 contact MSE: Δ={delta_cpr:.5f}, z={z_cpr:.2f}σ")
    print(f"  CPR-1.5× vs Sham 3× contact MSE:    Δ={delta_cpr_vs_sham:.5f}, z={z_cpr_vs_sham:.2f}σ")
    significant_vs_baseline = abs(z_cpr) > 2.0
    significant_vs_sham = abs(z_cpr_vs_sham) > 2.0
    if significant_vs_baseline and significant_vs_sham:
        verdict = "✅ Statistically significant: 1.5× beats both baseline AND sham (>2σ each)"
    elif significant_vs_baseline:
        verdict = "🟡 Significant vs baseline only — sham margin within noise"
    else:
        verdict = "❌ Not statistically significant under current seed count"
    print(f"\n>>> {verdict} <<<")

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "seeds": seeds, "raw_per_seed": raw, "aggregated": agg,
        "delta_cpr_vs_baseline": float(delta_cpr),
        "z_cpr_vs_baseline_sigma": float(z_cpr),
        "delta_cpr_vs_sham": float(delta_cpr_vs_sham),
        "z_cpr_vs_sham_sigma": float(z_cpr_vs_sham),
        "verdict": verdict,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        tags = [t[0] for t in conditions]
        means_c = [agg[t]["contact_mean"] for t in tags]
        stds_c = [agg[t]["contact_std"] for t in tags]
        means_f = [agg[t]["free_mean"] for t in tags]
        stds_f = [agg[t]["free_std"] for t in tags]
        x = np.arange(len(tags)); w = 0.35
        ax[0].bar(x - w/2, means_c, w, yerr=stds_c, capsize=4, label="contact MSE")
        ax[0].bar(x + w/2, means_f, w, yerr=stds_f, capsize=4, label="free MSE")
        ax[0].set_xticks(x); ax[0].set_xticklabels(tags); ax[0].legend(); ax[0].grid(alpha=0.3, axis='y')
        ax[0].set_title(f"Multi-seed mean ± std (n={len(seeds)})")
        # Per-seed scatter
        for ti, tag in enumerate(tags):
            cs = [r["contact"] for r in raw[tag]]
            ax[1].scatter([ti] * len(cs), cs, alpha=0.7)
        ax[1].set_xticks(np.arange(len(tags))); ax[1].set_xticklabels(tags)
        ax[1].set_ylabel("contact MSE per seed"); ax[1].grid(alpha=0.3); ax[1].set_title("Per-seed contact MSE")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "multiseed.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nElapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
