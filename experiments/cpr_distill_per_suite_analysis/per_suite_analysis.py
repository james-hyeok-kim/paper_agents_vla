"""
CPR-Distill Per-Suite Deep Dive (Option C).

Multi-suite experiment showed CPR-1.5× effect ranges from +6.4% (libero_spatial)
to +0.05% (libero_10). Why?

This experiment:
  1. Descriptive statistics per suite:
     - Contact mask density (how much of trajectory is "contact-marked")
     - Gripper transition rate (mean # of transitions per demo)
     - Trajectory length distribution
     - Action variance in contact vs free phases
     - Action magnitude (translation/rotation/gripper components)
  2. Fine-grained factor sweep on weakest suites (libero_10, libero_object):
     factor ∈ {1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0} to find suite-specific sweet spot.

Output:
  - experiments/cpr_distill_per_suite_analysis/: stats, plots, results.json
  - /data/jameskimh/cpr_distill_per_suite_analysis/: checkpoints
"""
import os, sys, json, time, glob
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import LiberoBCDataset, TinyBC, eval_model

DATA_ROOT = "/data/jameskimh/james_libero_datasets"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_per_suite_analysis"
OUT_BIG = "/data/jameskimh/cpr_distill_per_suite_analysis"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")
torch.manual_seed(42); np.random.seed(42)


def collect_suite_stats(suite, contact_window=3):
    """Per-suite descriptive statistics from raw HDF5 files."""
    data_dir = os.path.join(DATA_ROOT, suite)
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    print(f"  [{suite}] {len(files)} task files")
    traj_lens = []
    contact_densities = []
    transition_counts = []
    action_contact_var = []
    action_free_var = []
    action_translation_mag_contact = []
    action_translation_mag_free = []
    action_rotation_mag_contact = []
    action_rotation_mag_free = []
    total_demos = 0
    for fp in files:
        with h5py.File(fp, "r") as f:
            for dk in f["data"].keys():
                d = f["data"][dk]
                actions = d["actions"][:]  # (T, 7) — translation(3) + axis-angle(3) + gripper(1)
                grip = d["obs/gripper_states"][:]
                T = actions.shape[0]
                traj_lens.append(T)
                total_demos += 1
                # Contact mask via gripper transitions
                gs = grip.sum(axis=-1)
                diff = np.abs(np.diff(gs, prepend=gs[0]))
                trans_idx = np.where(diff > 0.02)[0]
                transition_counts.append(len(trans_idx))
                m = np.zeros(T, dtype=np.float32)
                for ci in trans_idx:
                    m[max(0, ci - contact_window): min(T, ci + contact_window + 1)] = 1.0
                if m.sum() == 0:
                    m[int(T * 0.7):] = 1.0
                contact_densities.append(m.mean())
                cm = m > 0.5
                if cm.any():
                    ac = actions[cm]
                    action_contact_var.append(ac.var(axis=0).mean())
                    action_translation_mag_contact.append(np.linalg.norm(ac[:, :3], axis=-1).mean())
                    action_rotation_mag_contact.append(np.linalg.norm(ac[:, 3:6], axis=-1).mean())
                if (~cm).any():
                    af = actions[~cm]
                    action_free_var.append(af.var(axis=0).mean())
                    action_translation_mag_free.append(np.linalg.norm(af[:, :3], axis=-1).mean())
                    action_rotation_mag_free.append(np.linalg.norm(af[:, 3:6], axis=-1).mean())
    return {
        "n_demos": total_demos,
        "n_files": len(files),
        "traj_len_mean": float(np.mean(traj_lens)),
        "traj_len_std": float(np.std(traj_lens)),
        "traj_len_min": int(np.min(traj_lens)),
        "traj_len_max": int(np.max(traj_lens)),
        "contact_density_mean": float(np.mean(contact_densities)),
        "contact_density_std": float(np.std(contact_densities)),
        "transitions_per_demo_mean": float(np.mean(transition_counts)),
        "transitions_per_demo_std": float(np.std(transition_counts)),
        "frac_demos_with_no_transition": float(np.mean([1 if x == 0 else 0 for x in transition_counts])),
        "action_var_contact_mean": float(np.mean(action_contact_var)),
        "action_var_free_mean": float(np.mean(action_free_var)),
        "action_var_ratio_contact_over_free": float(np.mean(action_contact_var) / max(np.mean(action_free_var), 1e-9)),
        "translation_mag_contact": float(np.mean(action_translation_mag_contact)),
        "translation_mag_free": float(np.mean(action_translation_mag_free)),
        "rotation_mag_contact": float(np.mean(action_rotation_mag_contact)),
        "rotation_mag_free": float(np.mean(action_rotation_mag_free)),
    }


def train_for_suite(suite, factor, n_epochs=3, lr=1e-3, seed=42):
    """Train TinyBC on one suite with given factor."""
    torch.manual_seed(seed); np.random.seed(seed)
    data_dir = os.path.join(DATA_ROOT, suite)
    files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    train_ds = LiberoBCDataset(files, train=True, train_frac=0.9, seed=0)
    val_ds = LiberoBCDataset(files, train=False, train_frac=0.9, seed=0)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
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
            w = 1.0 + (factor - 1.0) * contact
            loss = (err * w).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return eval_model(model, val_loader, DEVICE)


def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Per-Suite Deep Dive")
    print("=" * 60)
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

    # ===== Part 1: Descriptive statistics =====
    print("\n[1/2] Collecting descriptive statistics per suite...")
    stats = {}
    for s in suites:
        stats[s] = collect_suite_stats(s)
        print(f"\n  {s}:")
        for k, v in stats[s].items():
            print(f"    {k}: {v}")

    # Display comparison table
    print("\n--- Suite comparison ---")
    keys = ["n_demos", "traj_len_mean", "contact_density_mean",
            "transitions_per_demo_mean", "frac_demos_with_no_transition",
            "action_var_contact_mean", "action_var_free_mean",
            "action_var_ratio_contact_over_free",
            "translation_mag_contact", "rotation_mag_contact"]
    header = f"{'metric':<40}" + " ".join(f"{s[:14]:>16}" for s in suites)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = f"{k:<40}"
        for s in suites:
            v = stats[s][k]
            if isinstance(v, float):
                row += f" {v:>16.4f}"
            else:
                row += f" {v:>16}"
        print(row)

    # ===== Part 2: Fine factor sweep on weakest suites =====
    print("\n[2/2] Fine-grained factor sweep on weakest suites...")
    # libero_10 had +0.05% gain at 1.5x → try other factors
    # libero_object had +1.23% gain at 1.5x → also worth retry
    fine_factors = [1.0, 1.1, 1.2, 1.5, 2.0]
    sweep_results = {}
    for s in ["libero_10", "libero_object"]:
        print(f"\n  Sweep on {s}...")
        sweep_results[s] = {}
        for f in fine_factors:
            t1 = time.time()
            ev = train_for_suite(s, factor=f, n_epochs=3)
            sweep_results[s][f"factor_{f}"] = ev
            print(f"    factor={f}: contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} ({time.time()-t1:.1f}s)")
        # gains
        base = sweep_results[s]["factor_1.0"]
        for f in fine_factors:
            r = sweep_results[s][f"factor_{f}"]
            r["contact_gain_pct"] = (base["contact"] - r["contact"]) / base["contact"] * 100
            r["free_gain_pct"] = (base["free"] - r["free"]) / base["free"] * 100
            r["overall_gain_pct"] = (base["all"] - r["all"]) / base["all"] * 100

    # ===== Cross-suite correlation: does effect track stats? =====
    print("\n--- Correlation analysis: effect size vs suite statistics ---")
    # Multi-suite contact gain at factor=1.5 (from previous experiment, hardcoded for reference)
    multisuite_gain_at_1_5 = {
        "libero_spatial": 6.43,
        "libero_object": 1.23,
        "libero_goal": 4.69,
        "libero_10": 0.05,
    }
    correlation_features = ["contact_density_mean", "transitions_per_demo_mean",
                            "action_var_ratio_contact_over_free", "traj_len_mean",
                            "frac_demos_with_no_transition"]
    print(f"{'feature':<45} {'spatial':>10} {'object':>10} {'goal':>10} {'libero_10':>10}  corr")
    print("-" * 100)
    suite_gains = [multisuite_gain_at_1_5[s] for s in suites]
    for feat in correlation_features:
        vals = [stats[s][feat] for s in suites]
        row = f"{feat:<45}"
        for v in vals:
            row += f" {v:>10.4f}"
        # Correlation
        try:
            corr = float(np.corrcoef(vals, suite_gains)[0, 1])
        except Exception:
            corr = float('nan')
        row += f"   {corr:+.3f}"
        print(row)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "suite_descriptive_stats": stats,
        "fine_factor_sweep": {s: {k: {kk: float(vv) for kk, vv in v.items()} for k, v in d.items()}
                              for s, d in sweep_results.items()},
        "multisuite_gain_at_1.5x_for_correlation": multisuite_gain_at_1_5,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results: {os.path.join(OUT_USER, 'results.json')}")

    # ===== Plot =====
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # Plot 1: Per-suite stats vs effect size
        feat_to_plot = ["contact_density_mean", "transitions_per_demo_mean",
                        "action_var_ratio_contact_over_free", "frac_demos_with_no_transition"]
        for ai, feat in enumerate(feat_to_plot):
            r, c = ai // 2, ai % 2
            xs = [stats[s][feat] for s in suites]
            ys = [multisuite_gain_at_1_5[s] for s in suites]
            axes[r, c].scatter(xs, ys, s=120, c=['C0','C1','C2','C3'])
            for i, sn in enumerate(suites):
                axes[r, c].annotate(sn.replace("libero_",""), (xs[i], ys[i]), xytext=(5,5),
                                     textcoords='offset points', fontsize=10)
            try:
                corr = float(np.corrcoef(xs, ys)[0, 1])
            except Exception:
                corr = float('nan')
            axes[r, c].set_xlabel(feat); axes[r, c].set_ylabel("Contact gain @ 1.5× (%)")
            axes[r, c].set_title(f"{feat}\n(corr = {corr:+.3f})")
            axes[r, c].grid(alpha=0.3); axes[r, c].axhline(0, color='gray', linewidth=0.5)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "stats_vs_effect.png"), dpi=140)

        # Plot 2: Fine factor sweep on libero_10 and libero_object
        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
        for ai, s in enumerate(["libero_10", "libero_object"]):
            r = sweep_results[s]
            cgs = [r[f"factor_{f}"]["contact_gain_pct"] for f in fine_factors]
            fgs = [r[f"factor_{f}"]["free_gain_pct"] for f in fine_factors]
            ogs = [r[f"factor_{f}"]["overall_gain_pct"] for f in fine_factors]
            axes2[ai].plot(fine_factors, cgs, 'o-', label='Contact', linewidth=2)
            axes2[ai].plot(fine_factors, fgs, 's-', label='Free', linewidth=2)
            axes2[ai].plot(fine_factors, ogs, '^--', label='Overall', linewidth=1.5)
            axes2[ai].axhline(0, color='gray', linewidth=0.5)
            axes2[ai].set_title(s); axes2[ai].set_xlabel("Factor")
            axes2[ai].set_ylabel("MSE gain (%)"); axes2[ai].legend(); axes2[ai].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "fine_sweep.png"), dpi=140)
        print(f"Plots saved.")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
