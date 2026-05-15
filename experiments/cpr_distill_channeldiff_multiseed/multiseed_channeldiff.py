"""
Multi-seed verification of the new gripper_channel_diff detector.

5 seeds × 3 conditions:
  - baseline (factor=1.0, last_30 mask, equivalent to no reweight)
  - channel_diff_1.5 (factor=1.5 with gripper_channel_diff detector)
  - sham_3.0 (uniform 3x reweight, no contact awareness)

Confirms: is the +10.26% contact gain statistically significant?
"""
import os, sys, json, time, glob
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_contact_diagnostic")
from m0_smoke import TinyBC, eval_model
from diagnostic import LiberoDetectorDataset, DETECTORS

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_channeldiff_multiseed"
OUT_BIG = "/data/jameskimh/cpr_distill_channeldiff_multiseed"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


def train_with_seed(hdf5_files, detector_name, mode, factor, seed, n_epochs=3):
    torch.manual_seed(seed); np.random.seed(seed)
    detector_fn = DETECTORS[detector_name]
    train_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=True)
    val_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=False)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
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
            if mode == "cpr":
                w = 1.0 + (factor - 1.0) * contact
            elif mode == "sham":
                w = torch.full_like(err, factor)
            else:
                raise ValueError(mode)
            loss = (err * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return eval_model(model, val_loader, DEVICE)


def main():
    t0 = time.time()
    print("=" * 60)
    print("Multi-seed verification: gripper_channel_diff detector")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    seeds = [0, 1, 2, 3, 4]
    conditions = [
        ("baseline",       "last_30", "cpr", 1.0),  # factor 1.0 → no reweight
        ("channel_diff_1.5", "gripper_channel_diff", "cpr", 1.5),
        ("sham_3.0",       "last_30", "sham", 3.0),
    ]
    raw = {}
    for tag, det, mode, factor in conditions:
        print(f"\n[{tag}] detector={det}, mode={mode}, factor={factor}")
        raw[tag] = []
        for s in seeds:
            t1 = time.time()
            ev = train_with_seed(hdf5, det, mode, factor, seed=s)
            raw[tag].append({"seed": s, **{k: float(v) for k, v in ev.items()}})
            print(f"  seed {s}: contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} ({time.time()-t1:.1f}s)")

    # Aggregate
    print("\n--- Mean ± Std across 5 seeds ---")
    print(f"{'Condition':<22} {'Contact':>20} {'Free':>20} {'Overall':>20}")
    print("-" * 90)
    agg = {}
    for tag, _, _, _ in conditions:
        c = np.array([r["contact"] for r in raw[tag]])
        f_ = np.array([r["free"] for r in raw[tag]])
        a = np.array([r["all"] for r in raw[tag]])
        agg[tag] = {
            "contact_mean": float(c.mean()), "contact_std": float(c.std()),
            "free_mean": float(f_.mean()), "free_std": float(f_.std()),
            "all_mean": float(a.mean()), "all_std": float(a.std()),
        }
        print(f"{tag:<22} {c.mean():.5f}±{c.std():.5f}   {f_.mean():.5f}±{f_.std():.5f}   {a.mean():.5f}±{a.std():.5f}")

    # Significance: channel_diff vs baseline contact, channel_diff vs sham
    b_c, b_s = agg["baseline"]["contact_mean"], agg["baseline"]["contact_std"]
    cd_c, cd_s = agg["channel_diff_1.5"]["contact_mean"], agg["channel_diff_1.5"]["contact_std"]
    sh_c, sh_s = agg["sham_3.0"]["contact_mean"], agg["sham_3.0"]["contact_std"]
    delta_cd = b_c - cd_c
    sigma_cd = np.sqrt(b_s**2 + cd_s**2)
    z_cd = delta_cd / (sigma_cd + 1e-9)
    delta_vs_sham = sh_c - cd_c
    z_vs_sham = delta_vs_sham / (np.sqrt(sh_s**2 + cd_s**2) + 1e-9)
    # Overall significance
    b_o, b_os = agg["baseline"]["all_mean"], agg["baseline"]["all_std"]
    cd_o, cd_os = agg["channel_diff_1.5"]["all_mean"], agg["channel_diff_1.5"]["all_std"]
    delta_overall = b_o - cd_o
    z_overall = delta_overall / (np.sqrt(b_os**2 + cd_os**2) + 1e-9)

    print(f"\n--- Significance ---")
    print(f"  channel_diff_1.5 vs baseline contact: Δ={delta_cd:+.5f}, z={z_cd:.2f}σ")
    print(f"  channel_diff_1.5 vs sham_3.0 contact: Δ={delta_vs_sham:+.5f}, z={z_vs_sham:.2f}σ")
    print(f"  channel_diff_1.5 vs baseline OVERALL: Δ={delta_overall:+.5f}, z={z_overall:.2f}σ")

    contact_sig = abs(z_cd) > 2.0 and abs(z_vs_sham) > 2.0
    overall_sig = abs(z_overall) > 2.0
    if contact_sig and overall_sig and delta_overall > 0:
        verdict = "✅ DOUBLE WIN: contact AND overall statistically significant + net positive overall"
    elif contact_sig and overall_sig:
        verdict = "🟡 Significant but overall positive/negative needs check"
    elif contact_sig:
        verdict = "🟡 Contact significant, overall within noise"
    else:
        verdict = "❌ Not significant"
    print(f"\n>>> {verdict} <<<")

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed, "seeds": seeds,
        "raw_per_seed": raw, "aggregated": agg,
        "z_channeldiff_vs_baseline_contact": float(z_cd),
        "z_channeldiff_vs_sham_contact": float(z_vs_sham),
        "z_channeldiff_vs_baseline_overall": float(z_overall),
        "delta_overall": float(delta_overall),
        "verdict": verdict,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        tags = [t[0] for t in conditions]
        means_c = [agg[t]["contact_mean"] for t in tags]
        stds_c = [agg[t]["contact_std"] for t in tags]
        means_a = [agg[t]["all_mean"] for t in tags]
        stds_a = [agg[t]["all_std"] for t in tags]
        x = np.arange(len(tags)); w = 0.35
        ax[0].bar(x - w/2, means_c, w, yerr=stds_c, capsize=4, label="contact MSE")
        ax[0].bar(x + w/2, means_a, w, yerr=stds_a, capsize=4, label="overall MSE")
        ax[0].set_xticks(x); ax[0].set_xticklabels(tags, rotation=10); ax[0].legend(); ax[0].grid(alpha=0.3, axis='y')
        ax[0].set_title(f"Multi-seed mean ± std (n=5)")
        for ti, tag in enumerate(tags):
            cs = [r["contact"] for r in raw[tag]]
            ax[1].scatter([ti] * len(cs), cs, alpha=0.7)
        ax[1].set_xticks(np.arange(len(tags))); ax[1].set_xticklabels(tags)
        ax[1].set_ylabel("contact MSE per seed"); ax[1].grid(alpha=0.3); ax[1].set_title("Per-seed contact MSE")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "multiseed_channeldiff.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
