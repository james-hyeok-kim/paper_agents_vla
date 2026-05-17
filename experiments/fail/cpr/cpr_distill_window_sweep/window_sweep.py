"""
CPR-Distill Contact Window Sensitivity.

Vary the window size around gripper transitions: ±1, ±3 (M0 default), ±5, ±7.
Fixed factor=1.5 (sweet spot).

If results are robust across window sizes, gripper-transition proxy is stable.
If only ±3 works, the proxy is fragile.
"""
import os, sys, json, time, glob
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
from m0_smoke import TinyBC, eval_model

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_window_sweep"
OUT_BIG = "/data/jameskimh/cpr_distill_window_sweep"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")


class LiberoWindowDataset(Dataset):
    def __init__(self, hdf5_files, window, train=True, train_frac=0.9, seed=0):
        self.window = window
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
                gs = grip.sum(axis=-1)
                diff = np.abs(np.diff(gs, prepend=gs[0]))
                idx = np.where(diff > 0.02)[0]
                m = np.zeros(T, dtype=np.float32)
                for ci in idx:
                    m[max(0, ci - window): min(T, ci + window + 1)] = 1.0
                if m.sum() == 0:
                    m[int(T * 0.7):] = 1.0
                self.cache[(fp, dk)] = {
                    "actions": d["actions"][:].astype(np.float32),
                    "rgb_a": d["obs/agentview_rgb"][:],
                    "ee_pos": d["obs/ee_pos"][:].astype(np.float32),
                    "ee_ori": d["obs/ee_ori"][:].astype(np.float32),
                    "gripper": grip.astype(np.float32),
                    "contact": m,
                }
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
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Window Sensitivity (factor=1.5 fixed)")
    print("=" * 60)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    windows = [1, 3, 5, 7]
    results = {}
    # Baseline factor=1.0 (any window gives same baseline)
    print("\n[Baseline factor=1.0, window=3]")
    train_ds_b = LiberoWindowDataset(hdf5, window=3, train=True)
    val_ds_b = LiberoWindowDataset(hdf5, window=3, train=False)
    bs = 256
    train_loader_b = DataLoader(train_ds_b, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_b = DataLoader(val_ds_b, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    m = train_one(train_loader_b, val_loader_b, factor=1.0)
    results["baseline_factor1.0"] = eval_model(m, val_loader_b, DEVICE)
    print(f"  contact={results['baseline_factor1.0']['contact']:.5f}")

    for w in windows:
        print(f"\n[Window ±{w}, factor=1.5]")
        train_ds = LiberoWindowDataset(hdf5, window=w, train=True)
        val_ds = LiberoWindowDataset(hdf5, window=w, train=False)
        # Mask density check
        densities = [d["contact"].mean() for d in train_ds.cache.values()]
        mean_dens = float(np.mean(densities))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
        m = train_one(train_loader, val_loader, factor=1.5)
        ev = eval_model(m, val_loader, DEVICE)
        ev["mask_density"] = mean_dens
        results[f"window_{w}"] = ev
        print(f"  contact={ev['contact']:.5f} free={ev['free']:.5f} all={ev['all']:.5f} (mask density={mean_dens:.3f})")

    # Compare
    b = results["baseline_factor1.0"]
    print("\n--- Contact gain vs baseline (factor=1.0) ---")
    print(f"{'Window':<10} {'Contact':>11} {'Δ %':>8} {'Density':>9}")
    for w in windows:
        r = results[f"window_{w}"]
        dc = (b["contact"] - r["contact"]) / b["contact"] * 100
        print(f"±{w:<8} {r['contact']:>11.5f} {dc:>+7.2f}% {r['mask_density']:>9.3f}")
        r["contact_gain_pct"] = dc

    elapsed = time.time() - t0
    out = {
        "elapsed_sec": elapsed, "device": str(DEVICE),
        "windows": windows, "factor": 1.5,
        "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        gains = [results[f"window_{w}"]["contact_gain_pct"] for w in windows]
        densities = [results[f"window_{w}"]["mask_density"] for w in windows]
        ax[0].plot(windows, gains, 'o-', linewidth=2, markersize=10)
        ax[0].axhline(0, color='gray', linewidth=0.5)
        ax[0].set_xlabel("Window size (±N timesteps)"); ax[0].set_ylabel("Contact MSE gain (%)")
        ax[0].set_title("Window size vs CPR effect"); ax[0].grid(alpha=0.3)
        ax[1].plot(windows, densities, 's-', linewidth=2, color='C1', markersize=10)
        ax[1].set_xlabel("Window size (±N)"); ax[1].set_ylabel("Mask density (fraction)")
        ax[1].set_title("Mask density vs window size"); ax[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "window_sweep.png"), dpi=140)
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
