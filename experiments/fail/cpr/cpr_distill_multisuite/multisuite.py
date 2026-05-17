"""
CPR-Distill Multi-Suite Validation.

Replicate factor sweep on 4 LIBERO suites: libero_spatial, libero_object,
libero_goal, libero_10. Does the 1.5× sweet spot generalize?

For each suite × factor ∈ {1.0, 1.5, 3.0}, train and eval.
"""
import os, sys, json, time, glob, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import LiberoBCDataset, TinyBC, eval_model

OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_multisuite"
OUT_BIG = "/data/jameskimh/cpr_distill_multisuite"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


def train_one(train_loader, val_loader, factor, n_epochs=3):
    torch.manual_seed(42); np.random.seed(42)
    model = TinyBC().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            w = 1.0 + (factor - 1.0) * contact
            loss = (err * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default=None, help="single suite to run (else all)")
    args = parser.parse_args()

    t0 = time.time()
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    if args.suite:
        suites = [args.suite]
    factors = [1.0, 1.5, 3.0]
    all_results = {}

    for suite in suites:
        print(f"\n{'='*60}\nSuite: {suite}\n{'='*60}")
        DATA_DIR = f"/data/jameskimh/james_libero_datasets/{suite}"
        hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
        if not hdf5:
            print(f"  WARNING: No files in {DATA_DIR}, skipping")
            continue
        print(f"  Found {len(hdf5)} task files in {suite}")
        train_ds = LiberoBCDataset(hdf5, train=True, train_frac=0.9, seed=0)
        val_ds = LiberoBCDataset(hdf5, train=False, train_frac=0.9, seed=0)
        bs = 256
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        suite_results = {}
        for f in factors:
            t1 = time.time()
            m = train_one(train_loader, val_loader, factor=f)
            ev = eval_model(m, val_loader, DEVICE)
            suite_results[f"factor_{f}"] = ev
            print(f"  factor={f}: contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} ({time.time()-t1:.1f}s)")
        # Compute sweet spot for this suite
        base = suite_results["factor_1.0"]
        for f in factors:
            r = suite_results[f"factor_{f}"]
            r["contact_gain_pct"] = (base["contact"] - r["contact"]) / base["contact"] * 100
            r["free_gain_pct"] = (base["free"] - r["free"]) / base["free"] * 100
            r["overall_gain_pct"] = (base["all"] - r["all"]) / base["all"] * 100
        all_results[suite] = suite_results

    # Cross-suite analysis
    print("\n" + "=" * 60)
    print("CROSS-SUITE ANALYSIS")
    print("=" * 60)
    print(f"{'Suite':<16} {'Factor':<10} {'Δ Contact %':>13} {'Δ Free %':>11} {'Δ Overall %':>13}")
    print("-" * 70)
    sweet_spot_counts = {1.0: 0, 1.5: 0, 3.0: 0}
    for suite, results in all_results.items():
        for f in factors:
            r = results[f"factor_{f}"]
            marker = ""
            # best overall in this suite
            best_f = max(factors, key=lambda x: results[f"factor_{x}"]["overall_gain_pct"])
            if f == best_f:
                marker = "  ← best overall"
                sweet_spot_counts[f] += 1
            print(f"{suite:<16} {f:<10} {r['contact_gain_pct']:>+12.2f}% {r['free_gain_pct']:>+10.2f}% {r['overall_gain_pct']:>+12.2f}%{marker}")
        print()

    print(f"\nSweet-spot factor counts across suites: {sweet_spot_counts}")
    if sweet_spot_counts.get(1.5, 0) >= 3:
        verdict = "✅ 1.5× generalizes (sweet spot in ≥3/4 suites)"
    elif sweet_spot_counts.get(1.5, 0) >= 2:
        verdict = "🟡 1.5× partial generalization (2/4 suites)"
    else:
        verdict = f"❌ 1.5× does not generalize — sweet spot shifts per suite"
    print(f">>> {verdict} <<<")

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "suites_tested": suites, "factors_tested": factors,
        "results": all_results,
        "sweet_spot_counts": {str(k): v for k, v in sweet_spot_counts.items()},
        "verdict": verdict,
    }
    # Convert numpy
    def conv(o):
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(conv(out), f, indent=2)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n_suites = len(suites)
        fig, axes = plt.subplots(1, n_suites, figsize=(4 * n_suites, 4), sharey=True)
        if n_suites == 1: axes = [axes]
        for ai, (suite, r) in enumerate(all_results.items()):
            cgs = [r[f"factor_{f}"]["contact_gain_pct"] for f in factors]
            fgs = [r[f"factor_{f}"]["free_gain_pct"] for f in factors]
            ogs = [r[f"factor_{f}"]["overall_gain_pct"] for f in factors]
            axes[ai].plot(factors, cgs, 'o-', label='Contact')
            axes[ai].plot(factors, fgs, 's-', label='Free')
            axes[ai].plot(factors, ogs, '^--', label='Overall')
            axes[ai].axhline(0, color='gray', linewidth=0.5)
            axes[ai].set_title(suite); axes[ai].set_xlabel("Factor"); axes[ai].grid(alpha=0.3)
            if ai == 0:
                axes[ai].set_ylabel("MSE gain (%)")
            axes[ai].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "multisuite.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
