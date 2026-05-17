"""
Combined detector: union of gripper_channel_diff and ee_velocity_drop.

Hypothesis: two complementary signals (action-level + proprio-level) catch
different aspects of contact. Their union should yield strictly better coverage
without false positives accumulating.

Test: baseline (factor=1.0) vs {channel_diff alone, velocity_drop alone, union}
all at factor=1.5.
"""
import os, sys, json, time, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_contact_diagnostic")
from m0_smoke import TinyBC, eval_model
from diagnostic import LiberoDetectorDataset, DETECTORS, detect_gripper_channel_diff, detect_ee_velocity_drop

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_combined_detector"
OUT_BIG = "/data/jameskimh/cpr_distill_combined_detector"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


def detect_union(d, T):
    """Union mask: 1 if either channel_diff or velocity_drop says contact."""
    m1 = detect_gripper_channel_diff(d, T)
    m2 = detect_ee_velocity_drop(d, T)
    return np.maximum(m1, m2).astype(np.float32)


def detect_intersection(d, T):
    """Intersection mask: 1 only if both say contact."""
    m1 = detect_gripper_channel_diff(d, T)
    m2 = detect_ee_velocity_drop(d, T)
    return np.minimum(m1, m2).astype(np.float32)


DETECTORS["union"] = detect_union
DETECTORS["intersection"] = detect_intersection


def train_with_detector(hdf5_files, detector_name, factor, n_epochs=3):
    torch.manual_seed(42); np.random.seed(42)
    detector_fn = DETECTORS[detector_name]
    train_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=True)
    val_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=False)
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
    print("Combined detector: union/intersection of channel_diff + velocity_drop")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    results = {}
    # Baseline
    print("\n[baseline factor=1.0, last_30]")
    ev = train_with_detector(hdf5, "last_30", factor=1.0)
    results["baseline"] = ev
    print(f"  contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f}")

    for det in ["gripper_channel_diff", "ee_velocity_drop", "union", "intersection"]:
        print(f"\n[detector={det}, factor=1.5]")
        ev = train_with_detector(hdf5, det, factor=1.5)
        results[det] = ev
        print(f"  contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} density={ev['mask_density']:.3f}")

    base = results["baseline"]
    print("\n--- Comparison vs baseline ---")
    print(f"{'Detector':<26} {'Contact':>10} {'Free':>10} {'Overall':>10} {'Density':>9} {'Δ Cont %':>10} {'Δ All %':>10}")
    print("-" * 90)
    for det in ["gripper_channel_diff", "ee_velocity_drop", "union", "intersection"]:
        r = results[det]
        dc = (base["contact"] - r["contact"]) / base["contact"] * 100
        df = (base["free"] - r["free"]) / base["free"] * 100
        da = (base["all"] - r["all"]) / base["all"] * 100
        r["contact_gain_pct"] = dc
        r["free_gain_pct"] = df
        r["overall_gain_pct"] = da
        print(f"{det:<26} {r['contact']:>10.5f} {r['free']:>10.5f} {r['all']:>10.5f} {r['mask_density']:>9.3f} {dc:>+9.2f}% {da:>+9.2f}%")

    best = max(["gripper_channel_diff", "ee_velocity_drop", "union", "intersection"],
               key=lambda d: results[d]["overall_gain_pct"])
    print(f"\n>>> Best detector by overall gain: {best} ({results[best]['overall_gain_pct']:+.2f}%) <<<")

    # Is the combined better than the best individual?
    best_indiv = max(["gripper_channel_diff", "ee_velocity_drop"],
                     key=lambda d: results[d]["overall_gain_pct"])
    union_better = results["union"]["overall_gain_pct"] > results[best_indiv]["overall_gain_pct"]
    intersection_better = results["intersection"]["overall_gain_pct"] > results[best_indiv]["overall_gain_pct"]
    if union_better and not intersection_better:
        verdict = "✅ Union beats individuals — complementary signals"
    elif intersection_better and not union_better:
        verdict = "🟡 Intersection beats individuals — overlap is the signal"
    elif union_better and intersection_better:
        verdict = "✅ Both union and intersection beat individuals — strong evidence of contact-specific structure"
    else:
        verdict = "🟡 Neither combination beats best individual — signals are redundant or noisy"
    print(f">>> {verdict} <<<")

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed,
        "results": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                        for kk, vv in v.items()} for k, v in results.items()},
        "best_detector": best,
        "verdict": verdict,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        names = ["gripper_channel_diff", "ee_velocity_drop", "union", "intersection"]
        cgs = [results[d]["contact_gain_pct"] for d in names]
        fgs = [results[d]["free_gain_pct"] for d in names]
        ogs = [results[d]["overall_gain_pct"] for d in names]
        x = np.arange(len(names)); w = 0.27
        ax.bar(x - w, cgs, w, label="Δ Contact %")
        ax.bar(x, fgs, w, label="Δ Free %")
        ax.bar(x + w, ogs, w, label="Δ Overall %")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.legend(); ax.grid(alpha=0.3, axis='y')
        ax.set_title("Combined detectors: gain vs baseline (factor=1.5)")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "combined.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
