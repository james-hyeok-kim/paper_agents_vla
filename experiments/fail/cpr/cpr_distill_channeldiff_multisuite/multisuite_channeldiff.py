"""
Multi-suite generalization check with the new gripper_channel_diff detector.

Question: does the +10.26% contact gain (seen on libero_spatial) generalize to
libero_object, libero_goal, libero_10 with proper contact detection?

4 suites × 3 factors {1.0, 1.5, 3.0} = 12 runs.
"""
import os, sys, json, time, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_contact_diagnostic")
from m0_smoke import TinyBC, eval_model
from diagnostic import LiberoDetectorDataset, DETECTORS

DATA_ROOT = "/data/jameskimh/james_libero_datasets"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_channeldiff_multisuite"
OUT_BIG = "/data/jameskimh/cpr_distill_channeldiff_multisuite"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


def train_for_suite(suite, detector_name, factor, n_epochs=3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, suite)
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    detector_fn = DETECTORS[detector_name]
    train_ds = LiberoDetectorDataset(files, detector_fn, train=True)
    val_ds = LiberoDetectorDataset(files, detector_fn, train=False)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    densities = [d["contact"].mean() for d in train_ds.cache.values()]
    density = float(np.mean(densities))
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
    ev = eval_model(model, val_loader, DEVICE)
    ev["mask_density"] = density
    return ev


def main():
    t0 = time.time()
    print("=" * 60)
    print("Multi-suite × gripper_channel_diff detector")
    print("=" * 60)
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    factors = [1.0, 1.5, 3.0]
    all_results = {}

    for suite in suites:
        print(f"\n=== Suite: {suite} ===")
        suite_res = {}
        for f in factors:
            t1 = time.time()
            ev = train_for_suite(suite, "gripper_channel_diff", factor=f)
            suite_res[f"factor_{f}"] = ev
            print(f"  factor={f}: contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} (density={ev['mask_density']:.3f}, {time.time()-t1:.1f}s)")
        base = suite_res["factor_1.0"]
        for f in factors:
            r = suite_res[f"factor_{f}"]
            r["contact_gain_pct"] = (base["contact"] - r["contact"]) / base["contact"] * 100
            r["free_gain_pct"] = (base["free"] - r["free"]) / base["free"] * 100
            r["overall_gain_pct"] = (base["all"] - r["all"]) / base["all"] * 100
        all_results[suite] = suite_res

    # Analysis
    print("\n--- Suite generalization with gripper_channel_diff detector ---")
    print(f"{'Suite':<16} {'Factor':<8} {'Δ Contact %':>13} {'Δ Free %':>11} {'Δ Overall %':>13} {'Density':>9}")
    print("-" * 80)
    suite_best_factors = {}
    for suite, results in all_results.items():
        for f in factors:
            r = results[f"factor_{f}"]
            marker = ""
            best_f = max(factors, key=lambda x: results[f"factor_{x}"]["overall_gain_pct"])
            if f == best_f:
                marker = "  ← best overall"
                suite_best_factors[suite] = f
            print(f"{suite:<16} {f:<8} {r['contact_gain_pct']:>+12.2f}% {r['free_gain_pct']:>+10.2f}% {r['overall_gain_pct']:>+12.2f}% {r['mask_density']:>9.3f}{marker}")
        print()

    counts = {1.0: 0, 1.5: 0, 3.0: 0}
    for s, f in suite_best_factors.items():
        counts[f] += 1
    print(f"\nBest factor across suites: {counts}")
    if counts[1.5] >= 3:
        verdict = "✅ 1.5× generalizes with new detector (≥3/4 suites)"
    elif counts[1.5] >= 2:
        verdict = "🟡 1.5× partial generalization (2/4)"
    elif sum([counts[1.5], counts[3.0]]) >= 3:
        verdict = "🟡 reweight helps, sweet spot differs per suite"
    else:
        verdict = "❌ factor=1.0 still wins in most suites — overall MSE not improved"
    print(f">>> {verdict} <<<")

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed,
        "detector": "gripper_channel_diff",
        "suites_tested": suites, "factors_tested": factors,
        "results": all_results,
        "suite_best_factor": suite_best_factors,
        "verdict": verdict,
    }
    def conv(o):
        if isinstance(o, dict): return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list): return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        return o
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(conv(out), f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(suites)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), sharey=True)
        for ai, (suite, r) in enumerate(all_results.items()):
            cgs = [r[f"factor_{f}"]["contact_gain_pct"] for f in factors]
            fgs = [r[f"factor_{f}"]["free_gain_pct"] for f in factors]
            ogs = [r[f"factor_{f}"]["overall_gain_pct"] for f in factors]
            axes[ai].plot(factors, cgs, 'o-', label='Contact')
            axes[ai].plot(factors, fgs, 's-', label='Free')
            axes[ai].plot(factors, ogs, '^--', label='Overall')
            axes[ai].axhline(0, color='gray', linewidth=0.5)
            axes[ai].set_title(suite); axes[ai].set_xlabel("Factor"); axes[ai].grid(alpha=0.3)
            if ai == 0: axes[ai].set_ylabel("MSE gain (%)")
            axes[ai].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "multisuite_channeldiff.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
