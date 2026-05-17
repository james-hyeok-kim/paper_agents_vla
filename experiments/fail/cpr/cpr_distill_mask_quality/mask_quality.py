"""
CPR-Distill M3.5: Contact Mask Quality Ablation.

Test whether the gripper-transition proxy used in M0 holds up against
better contact-phase definitions. If a more accurate mask gives much stronger
CPR effects, then gripper-transition is a bottleneck for the paper claim.

Mask variants:
  (a) gripper_transition: |Δgripper| > 0.02, window ±3 (M0 default)
  (b) gt_ee_velocity_drop: low end-effector velocity (proxy for contact)
                          since contact slows down the EE.
  (c) window_last30: last 30% of trajectory marked contact, transitions ignored
  (d) gaussian_smooth: Gaussian decay around gripper transitions (σ=2)

Fixed: factor=1.5 (sweet spot from sweep).
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
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_mask_quality"
OUT_BIG = "/data/jameskimh/cpr_distill_mask_quality"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")
torch.manual_seed(42); np.random.seed(42)


def build_mask(d, T, variant, window=3):
    """Construct contact mask from demo data dict."""
    if variant == "gripper_transition":
        grip_sig = d["gripper"].sum(axis=-1)
        diff = np.abs(np.diff(grip_sig, prepend=grip_sig[0]))
        idx = np.where(diff > 0.02)[0]
        m = np.zeros(T, dtype=np.float32)
        for ci in idx:
            m[max(0, ci - window): min(T, ci + window + 1)] = 1.0
        if m.sum() == 0:
            m[int(T * 0.7):] = 1.0
        return m
    elif variant == "gt_ee_velocity_drop":
        # EE velocity from ee_pos finite differences
        ee = d["ee_pos"]  # (T, 3)
        vel = np.linalg.norm(np.diff(ee, axis=0, prepend=ee[:1]), axis=-1)  # (T,)
        # Smooth
        kernel = np.ones(5) / 5
        vel_smooth = np.convolve(vel, kernel, mode='same')
        # Contact = velocity low AND not at start (initial position holding)
        threshold = np.percentile(vel_smooth, 30)  # bottom 30% of velocity
        m = (vel_smooth < threshold).astype(np.float32)
        m[:5] = 0  # exclude initial static
        if m.sum() == 0:
            m[int(T * 0.7):] = 1.0
        return m
    elif variant == "window_last30":
        m = np.zeros(T, dtype=np.float32)
        m[int(T * 0.7):] = 1.0
        return m
    elif variant == "gaussian_smooth":
        grip_sig = d["gripper"].sum(axis=-1)
        diff = np.abs(np.diff(grip_sig, prepend=grip_sig[0]))
        idx = np.where(diff > 0.02)[0]
        m = np.zeros(T, dtype=np.float32)
        sigma = 2.0
        for ci in idx:
            for t in range(T):
                m[t] = max(m[t], np.exp(-((t - ci) ** 2) / (2 * sigma ** 2)))
        if m.sum() == 0:
            m[int(T * 0.7):] = 1.0
        return m
    else:
        raise ValueError(variant)


class LiberoMaskDataset(Dataset):
    def __init__(self, hdf5_files, variant, train=True, train_frac=0.9, seed=0):
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
                dd = f["data"][dk]
                cache = {
                    "actions": dd["actions"][:].astype(np.float32),
                    "rgb_a": dd["obs/agentview_rgb"][:],
                    "ee_pos": dd["obs/ee_pos"][:].astype(np.float32),
                    "ee_ori": dd["obs/ee_ori"][:].astype(np.float32),
                    "gripper": dd["obs/gripper_states"][:].astype(np.float32),
                    "T": T,
                }
                # Compute mask via the requested variant
                cache["contact"] = build_mask(cache, T, variant)
                # ALSO compute gripper_transition mask for evaluation alignment
                cache["contact_eval_proxy"] = build_mask(cache, T, "gripper_transition")
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
            "contact_eval_proxy": torch.tensor(d["contact_eval_proxy"][t]).float(),
        }


def train_one(train_loader, val_loader, factor, n_epochs=3, lr=1e-3):
    torch.manual_seed(42); np.random.seed(42)
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
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model


def eval_with_proxy_contact(model, val_loader):
    """Eval contact MSE using GRIPPER-TRANSITION proxy (common reference frame)."""
    model.eval()
    err_sum_c = 0.0; err_sum_f = 0.0; err_sum_a = 0.0
    n_c = 0; n_f = 0; n_a = 0
    with torch.no_grad():
        for batch in val_loader:
            img = batch["img"].to(DEVICE, non_blocking=True)
            state = batch["state"].to(DEVICE, non_blocking=True)
            target = batch["action"].to(DEVICE, non_blocking=True)
            # Use the gripper-transition proxy for eval to make all variants comparable
            contact = batch["contact_eval_proxy"].to(DEVICE, non_blocking=True)
            pred = model(img, state)
            per = ((pred - target) ** 2).sum(-1)
            cm = contact > 0.5
            err_sum_c += per[cm].sum().item(); n_c += cm.sum().item()
            err_sum_f += per[~cm].sum().item(); n_f += (~cm).sum().item()
            err_sum_a += per.sum().item(); n_a += img.size(0)
    return {"contact": err_sum_c/max(n_c,1), "free": err_sum_f/max(n_f,1), "all": err_sum_a/max(n_a,1)}


def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Mask Quality Ablation (factor=1.5 fixed)")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    variants = ["gripper_transition", "gt_ee_velocity_drop", "window_last30", "gaussian_smooth"]
    # Also baseline (no reweight)
    results = {}
    # Baseline: factor=1.0 (no reweight) — same regardless of mask
    print("\n[Baseline] factor=1.0 ...")
    train_ds_b = LiberoMaskDataset(hdf5, "gripper_transition", train=True)
    val_ds_b = LiberoMaskDataset(hdf5, "gripper_transition", train=False)
    bs = 256
    train_loader_b = DataLoader(train_ds_b, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_b = DataLoader(val_ds_b, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    m_base = train_one(train_loader_b, val_loader_b, factor=1.0)
    results["baseline_factor1.0"] = eval_with_proxy_contact(m_base, val_loader_b)
    print(f"  contact={results['baseline_factor1.0']['contact']:.5f} free={results['baseline_factor1.0']['free']:.5f}")

    for v in variants:
        print(f"\n[Variant: {v}] factor=1.5 ...")
        train_ds = LiberoMaskDataset(hdf5, v, train=True)
        val_ds = LiberoMaskDataset(hdf5, v, train=False)
        # Mask sanity
        sample_masks = [train_ds.cache[k]["contact"] for k in list(train_ds.cache.keys())[:5]]
        mask_density = np.mean([m.mean() for m in sample_masks])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        m = train_one(train_loader, val_loader, factor=1.5)
        ev = eval_with_proxy_contact(m, val_loader)
        results[v] = ev
        results[v]["mask_density"] = float(mask_density)
        print(f"  contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f}  (mask density={mask_density:.3f})")

    # Analysis: compare to baseline (factor=1.0, gripper_transition data)
    b = results["baseline_factor1.0"]
    print("\n--- Contact MSE reduction vs baseline (factor=1.0) ---")
    print(f"{'Mask variant':<28} {'Contact':>10} {'Δ %':>9} {'Density':>9}")
    print("-" * 60)
    for v in variants:
        r = results[v]
        dc = (b["contact"] - r["contact"]) / b["contact"] * 100
        print(f"{v:<28} {r['contact']:>10.5f} {dc:>+8.2f}% {r['mask_density']:>9.3f}")
        r["contact_gain_pct"] = dc

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        "variants_tested": variants,
        "factor": 1.5,
        "notes": "All variants evaluated with the SAME gripper-transition proxy contact mask for comparability."
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        names = variants
        gains = [results[v]["contact_gain_pct"] for v in variants]
        densities = [results[v]["mask_density"] for v in variants]
        ax[0].bar(names, gains, color=['C0','C1','C2','C3'])
        ax[0].axhline(0, color='gray', linewidth=0.5); ax[0].set_ylabel("Δ contact MSE vs baseline (%)")
        ax[0].set_title("Mask variant: contact gain (factor=1.5)")
        ax[0].grid(alpha=0.3, axis='y')
        for x, g in enumerate(gains):
            ax[0].text(x, g + (0.3 if g > 0 else -0.5), f"{g:+.1f}%", ha='center', fontsize=9)
        plt.setp(ax[0].get_xticklabels(), rotation=15, ha='right')

        ax[1].scatter(densities, gains, s=80)
        for i, n in enumerate(names):
            ax[1].annotate(n, (densities[i], gains[i]), xytext=(5,5), textcoords='offset points', fontsize=9)
        ax[1].set_xlabel("Mask density (fraction of timesteps marked contact)")
        ax[1].set_ylabel("Δ contact MSE %"); ax[1].grid(alpha=0.3)
        ax[1].set_title("Mask density vs effect")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "mask_quality.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nElapsed: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
