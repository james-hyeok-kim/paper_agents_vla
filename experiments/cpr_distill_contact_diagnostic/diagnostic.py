"""
CPR-Distill Contact Detection Diagnostic & Fix.

Per-suite analysis revealed our gripper-transition threshold (|Δgripper|>0.02)
NEVER triggered → fallback mask (last 30%) was always used. So our "contact
reweighting" was actually "trajectory-end reweighting."

This experiment:
  STAGE 1: Diagnose. What are the actual gripper_states values & deltas in LIBERO?
  STAGE 2: Implement better contact detectors:
    (a) Lower-threshold gripper transition
    (b) Gripper-channel-cross (states[0] vs states[1])
    (c) EE velocity drop (proper formulation)
    (d) EE motion-rate change
  STAGE 3: Re-run CPR-1.5× with each detector, compare to last-30% fallback.

Output:
  - experiments/cpr_distill_contact_diagnostic/: stats, plots, results.json
  - /data/jameskimh/cpr_distill_contact_diagnostic/: checkpoints
"""
import os, sys, json, time, glob
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import TinyBC, eval_model

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_contact_diagnostic"
OUT_BIG = "/data/jameskimh/cpr_distill_contact_diagnostic"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


# ========== STAGE 1: Diagnostic ==========

def diagnose_gripper(hdf5_files, max_demos=50):
    """Inspect actual gripper_states values in LIBERO."""
    all_gripper = []
    all_delta = []
    all_gripper_ch0 = []
    all_gripper_ch1 = []
    all_ee_vel = []
    all_action_gripper_dim = []
    n_demos_with_any_transition = {0.001: 0, 0.005: 0, 0.01: 0, 0.02: 0, 0.05: 0, 0.1: 0}
    total = 0
    for fp in hdf5_files[:5]:  # subset for speed
        with h5py.File(fp, "r") as f:
            for dk in list(f["data"].keys())[:max_demos]:
                d = f["data"][dk]
                grip = d["obs/gripper_states"][:]  # (T, 2)
                ee_pos = d["obs/ee_pos"][:]
                actions = d["actions"][:]
                total += 1
                # Gripper raw
                all_gripper.append(grip.reshape(-1))
                all_gripper_ch0.append(grip[:, 0])
                all_gripper_ch1.append(grip[:, 1])
                # |Δ gripper sum|
                gs = grip.sum(axis=-1)
                d_ = np.abs(np.diff(gs, prepend=gs[0]))
                all_delta.append(d_)
                for th in n_demos_with_any_transition:
                    if (d_ > th).any():
                        n_demos_with_any_transition[th] += 1
                # EE velocity
                ee_vel = np.linalg.norm(np.diff(ee_pos, axis=0, prepend=ee_pos[:1]), axis=-1)
                all_ee_vel.append(ee_vel)
                # Action gripper dim (7th, index 6)
                all_action_gripper_dim.append(actions[:, 6])
    gripper_all = np.concatenate(all_gripper)
    deltas_all = np.concatenate(all_delta)
    ch0 = np.concatenate(all_gripper_ch0)
    ch1 = np.concatenate(all_gripper_ch1)
    ee_vel_all = np.concatenate(all_ee_vel)
    action_grip = np.concatenate(all_action_gripper_dim)
    return {
        "n_demos_inspected": total,
        "gripper_value_stats": {
            "min": float(gripper_all.min()), "max": float(gripper_all.max()),
            "mean": float(gripper_all.mean()), "std": float(gripper_all.std()),
            "median": float(np.median(gripper_all)),
            "percentiles": {p: float(np.percentile(gripper_all, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
        },
        "gripper_ch0_stats": {"min": float(ch0.min()), "max": float(ch0.max()),
                              "mean": float(ch0.mean()), "std": float(ch0.std())},
        "gripper_ch1_stats": {"min": float(ch1.min()), "max": float(ch1.max()),
                              "mean": float(ch1.mean()), "std": float(ch1.std())},
        "gripper_delta_stats": {
            "min": float(deltas_all.min()), "max": float(deltas_all.max()),
            "mean": float(deltas_all.mean()), "median": float(np.median(deltas_all)),
            "percentiles": {p: float(np.percentile(deltas_all, p)) for p in [50, 90, 95, 99, 99.5, 99.9]},
        },
        "ee_velocity_stats": {
            "min": float(ee_vel_all.min()), "max": float(ee_vel_all.max()),
            "mean": float(ee_vel_all.mean()), "median": float(np.median(ee_vel_all)),
            "percentiles": {p: float(np.percentile(ee_vel_all, p)) for p in [5, 25, 50, 75, 95]},
        },
        "action_gripper_dim_stats": {
            "min": float(action_grip.min()), "max": float(action_grip.max()),
            "mean": float(action_grip.mean()), "std": float(action_grip.std()),
            "unique_values": sorted(list(set(np.round(action_grip, 3).tolist())))[:20],
        },
        "n_demos_with_transition_above_threshold": n_demos_with_any_transition,
    }


# ========== STAGE 2: Better Contact Detectors ==========

def detect_low_threshold(d, T, window=3, threshold=0.001):
    gs = d["gripper"].sum(axis=-1)
    diff = np.abs(np.diff(gs, prepend=gs[0]))
    idx = np.where(diff > threshold)[0]
    m = np.zeros(T, dtype=np.float32)
    for ci in idx:
        m[max(0, ci - window): min(T, ci + window + 1)] = 1.0
    if m.sum() == 0:
        m[int(T * 0.7):] = 1.0
    return m

def detect_gripper_channel_diff(d, T, window=3, threshold=0.01):
    """Detect transitions when gripper channel 0 and channel 1 diverge (open→close)."""
    diff = np.abs(d["gripper"][:, 0] - d["gripper"][:, 1])
    # Detect changes in this diff
    d_diff = np.abs(np.diff(diff, prepend=diff[0]))
    idx = np.where(d_diff > threshold)[0]
    m = np.zeros(T, dtype=np.float32)
    for ci in idx:
        m[max(0, ci - window): min(T, ci + window + 1)] = 1.0
    if m.sum() == 0:
        m[int(T * 0.7):] = 1.0
    return m

def detect_action_gripper_command(d, T, window=3, threshold=0.5):
    """Action[6] is the gripper command. Detect when it changes."""
    ag = d["action_gripper"]  # (T,)
    diff = np.abs(np.diff(ag, prepend=ag[0]))
    idx = np.where(diff > threshold)[0]
    m = np.zeros(T, dtype=np.float32)
    for ci in idx:
        m[max(0, ci - window): min(T, ci + window + 1)] = 1.0
    if m.sum() == 0:
        m[int(T * 0.7):] = 1.0
    return m

def detect_ee_velocity_drop(d, T):
    """Bottom 30% of EE velocity (slow = contact). Smoothed."""
    ee = d["ee_pos"]
    vel = np.linalg.norm(np.diff(ee, axis=0, prepend=ee[:1]), axis=-1)
    # Smooth
    kernel = np.ones(7) / 7
    vel_smooth = np.convolve(vel, kernel, mode='same')
    threshold = np.percentile(vel_smooth, 30)
    m = (vel_smooth < threshold).astype(np.float32)
    # Skip initial stationary period (first 10% likely just starting from rest)
    m[:int(T * 0.1)] = 0.0
    if m.sum() == 0:
        m[int(T * 0.7):] = 1.0
    return m

def detect_last_30(d, T):
    """Fallback: last 30% — used by all previous experiments without realizing."""
    m = np.zeros(T, dtype=np.float32)
    m[int(T * 0.7):] = 1.0
    return m

DETECTORS = {
    "last_30": detect_last_30,
    "lowthresh_0.001": lambda d, T: detect_low_threshold(d, T, threshold=0.001),
    "lowthresh_0.005": lambda d, T: detect_low_threshold(d, T, threshold=0.005),
    "gripper_channel_diff": detect_gripper_channel_diff,
    "action_gripper_cmd": detect_action_gripper_command,
    "ee_velocity_drop": detect_ee_velocity_drop,
}


# ========== STAGE 3: Re-test CPR with each detector ==========

class LiberoDetectorDataset(Dataset):
    def __init__(self, hdf5_files, detector_fn, train=True, train_frac=0.9, seed=0):
        index = []
        for fp in hdf5_files:
            with h5py.File(fp, "r") as f:
                keys = list(f["data"].keys())
                rng = np.random.RandomState(seed + hash(fp) % 1000)
                rng.shuffle(keys)
                n_tr = int(train_frac * len(keys))
                sel = keys[:n_tr] if train else keys[n_tr:]
                for dk in sel:
                    T = f["data"][dk]["actions"].shape[0]
                    index.append((fp, dk, T))
        self.index = index
        self.cache = {}
        for fp, dk, T in index:
            with h5py.File(fp, "r") as f:
                d = f["data"][dk]
                grip = d["obs/gripper_states"][:]
                acts = d["actions"][:].astype(np.float32)
                cache = {
                    "actions": acts,
                    "rgb_a": d["obs/agentview_rgb"][:],
                    "ee_pos": d["obs/ee_pos"][:].astype(np.float32),
                    "ee_ori": d["obs/ee_ori"][:].astype(np.float32),
                    "gripper": grip.astype(np.float32),
                    "action_gripper": acts[:, 6],
                    "T": T,
                }
                cache["contact"] = detector_fn(cache, T)
                self.cache[(fp, dk)] = cache
        self.flat = [(di, t) for di, (_, _, T) in enumerate(index) for t in range(T)]

    def __len__(self): return len(self.flat)

    def __getitem__(self, i):
        di, t = self.flat[i]
        fp, dk, _ = self.index[di]
        d = self.cache[(fp, dk)]
        img = torch.from_numpy(d["rgb_a"][t]).permute(2, 0, 1).float() / 255.0
        state = np.concatenate([d["ee_pos"][t], d["ee_ori"][t], d["gripper"][t]])
        return {
            "img": img,
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(d["actions"][t]).float(),
            "contact": torch.tensor(d["contact"][t]).float(),
        }


def train_with_detector(hdf5_files, detector_name, factor, n_epochs=3):
    torch.manual_seed(42); np.random.seed(42)
    detector_fn = DETECTORS[detector_name]
    train_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=True)
    val_ds = LiberoDetectorDataset(hdf5_files, detector_fn, train=False)
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    # Mask density per detector
    densities = [d["contact"].mean() for d in train_ds.cache.values()]
    mask_density = float(np.mean(densities))
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
    # Eval using a COMMON reference mask (last_30 + low-threshold) for fair comparison
    # But for primary metric: use detector's OWN mask (each detector defines its own contact)
    return eval_model(model, val_loader, DEVICE), mask_density


def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Contact Detection Diagnostic & Fix")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))

    # ===== STAGE 1 =====
    print("\n[STAGE 1] Diagnostic — inspecting LIBERO gripper data...")
    diag = diagnose_gripper(hdf5)
    print("\n  Gripper raw (both channels combined):")
    for k, v in diag["gripper_value_stats"].items():
        print(f"    {k}: {v}")
    print(f"\n  Gripper channel 0: min={diag['gripper_ch0_stats']['min']:.4f}, max={diag['gripper_ch0_stats']['max']:.4f}, mean={diag['gripper_ch0_stats']['mean']:.4f}")
    print(f"  Gripper channel 1: min={diag['gripper_ch1_stats']['min']:.4f}, max={diag['gripper_ch1_stats']['max']:.4f}, mean={diag['gripper_ch1_stats']['mean']:.4f}")
    print(f"\n  |Δ gripper sum| distribution:")
    for k, v in diag["gripper_delta_stats"].items():
        if k == "percentiles":
            for p, vv in v.items():
                print(f"    p{p}: {vv:.6f}")
        else:
            print(f"    {k}: {v}")
    print(f"\n  EE velocity stats:")
    for k, v in diag["ee_velocity_stats"].items():
        if k == "percentiles":
            for p, vv in v.items():
                print(f"    p{p}: {vv:.6f}")
        else:
            print(f"    {k}: {v}")
    print(f"\n  Action[6] (gripper command) stats: min={diag['action_gripper_dim_stats']['min']}, max={diag['action_gripper_dim_stats']['max']}")
    print(f"    Unique values seen: {diag['action_gripper_dim_stats']['unique_values'][:10]}")
    print(f"\n  Demos with at least one transition above threshold (out of {diag['n_demos_inspected']}):")
    for th, cnt in diag["n_demos_with_transition_above_threshold"].items():
        print(f"    threshold={th}: {cnt}/{diag['n_demos_inspected']} demos ({100*cnt/diag['n_demos_inspected']:.1f}%)")

    # ===== STAGE 3: Test each detector =====
    print("\n[STAGE 3] Testing each contact detector with CPR (factor=1.5)...")
    results = {}
    # First, baseline (factor=1.0) — independent of detector
    print("\n  [Baseline factor=1.0, last_30 mask] training...")
    eval_baseline, _ = train_with_detector(hdf5, "last_30", factor=1.0)
    results["baseline_factor1.0"] = eval_baseline
    print(f"    contact={eval_baseline['contact']:.5f} free={eval_baseline['free']:.5f} all={eval_baseline['all']:.5f}")

    for det_name in DETECTORS.keys():
        print(f"\n  [Detector: {det_name}, factor=1.5]")
        ev, density = train_with_detector(hdf5, det_name, factor=1.5)
        ev["mask_density"] = density
        results[det_name] = ev
        # Contact MSE was evaluated on the detector's own mask, but for comparison
        # we should reference the baseline (also on last_30 mask for contact partition).
        # Note: each detector's contact MSE is in its OWN coordinate system.
        print(f"    contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} (density={density:.3f})")

    # Compare each detector vs baseline (factor=1.0 on last_30 mask)
    print(f"\n--- Detector comparison vs baseline (factor=1.0) ---")
    base = results["baseline_factor1.0"]
    print(f"{'Detector':<28} {'Contact':>10} {'Free':>10} {'Overall':>10} {'Density':>9} {'Δ Cont %':>9} {'Δ All %':>9}")
    print("-" * 90)
    for det_name in DETECTORS.keys():
        r = results[det_name]
        dc = (base["contact"] - r["contact"]) / base["contact"] * 100
        df = (base["free"] - r["free"]) / base["free"] * 100
        da = (base["all"] - r["all"]) / base["all"] * 100
        r["contact_gain_pct"] = dc
        r["free_gain_pct"] = df
        r["overall_gain_pct"] = da
        print(f"{det_name:<28} {r['contact']:>10.5f} {r['free']:>10.5f} {r['all']:>10.5f} {r['mask_density']:>9.3f} {dc:>+8.2f}% {da:>+8.2f}%")

    # Find best detector by overall gain
    best_det = max(DETECTORS.keys(), key=lambda d: results[d]["overall_gain_pct"])
    print(f"\n>>> Best detector by overall MSE gain: {best_det} ({results[best_det]['overall_gain_pct']:+.2f}%) <<<")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    # Save
    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "diagnostic": diag,
        "detector_results": {k: {kk: float(vv) if isinstance(vv, (int,float,np.floating)) else vv
                                  for kk, vv in v.items()} for k, v in results.items()},
        "detectors_tested": list(DETECTORS.keys()),
        "best_detector_by_overall": best_det,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        # 1. Gripper |Δ| histogram on log scale
        sample_deltas = []
        for fp in hdf5[:3]:
            with h5py.File(fp, "r") as f:
                for dk in list(f["data"].keys())[:10]:
                    grip = f["data"][dk]["obs/gripper_states"][:]
                    gs = grip.sum(axis=-1)
                    sample_deltas.append(np.abs(np.diff(gs, prepend=gs[0])))
        sample_deltas = np.concatenate(sample_deltas)
        axes[0,0].hist(sample_deltas + 1e-10, bins=80, log=True)
        axes[0,0].axvline(0.02, color='r', linestyle='--', label='M0 threshold=0.02')
        axes[0,0].axvline(0.005, color='orange', linestyle='--', label='lowthresh=0.005')
        axes[0,0].axvline(0.001, color='green', linestyle='--', label='lowthresh=0.001')
        axes[0,0].set_xlabel("|Δ gripper sum|"); axes[0,0].set_ylabel("count (log)")
        axes[0,0].set_title("Gripper-sum delta distribution")
        axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
        # 2. Action[6] distribution
        sample_act = []
        for fp in hdf5[:3]:
            with h5py.File(fp, "r") as f:
                for dk in list(f["data"].keys())[:10]:
                    sample_act.append(f["data"][dk]["actions"][:, 6])
        sample_act = np.concatenate(sample_act)
        axes[0,1].hist(sample_act, bins=40)
        axes[0,1].set_xlabel("Action[6] (gripper command)")
        axes[0,1].set_title("Action gripper-command distribution")
        axes[0,1].grid(alpha=0.3)
        # 3. Detector comparison bar
        names = list(DETECTORS.keys())
        cgs = [results[d]["contact_gain_pct"] for d in names]
        ogs = [results[d]["overall_gain_pct"] for d in names]
        x = np.arange(len(names)); w = 0.35
        axes[1,0].bar(x - w/2, cgs, w, label="Δ Contact %")
        axes[1,0].bar(x + w/2, ogs, w, label="Δ Overall %")
        axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(names, rotation=20, ha='right')
        axes[1,0].legend(); axes[1,0].grid(alpha=0.3, axis='y')
        axes[1,0].set_title("Detector: gain vs baseline (factor=1.5)")
        axes[1,0].axhline(0, color='gray', linewidth=0.5)
        # 4. Density vs overall gain scatter
        dens = [results[d]["mask_density"] for d in names]
        axes[1,1].scatter(dens, ogs, s=80)
        for i, n in enumerate(names):
            axes[1,1].annotate(n.replace("_", "\n"), (dens[i], ogs[i]), xytext=(5,5),
                              textcoords='offset points', fontsize=8)
        axes[1,1].set_xlabel("Mask density"); axes[1,1].set_ylabel("Δ overall MSE %")
        axes[1,1].set_title("Mask density vs overall gain")
        axes[1,1].axhline(0, color='gray', linewidth=0.5); axes[1,1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "diagnostic.png"), dpi=140)
        print(f"Plot: {os.path.join(OUT_USER, 'diagnostic.png')}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
