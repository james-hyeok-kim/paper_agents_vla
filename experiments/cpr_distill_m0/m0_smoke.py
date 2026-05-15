"""
CPR-Distill M0 Smoke Test on real LIBERO-spatial data.

Goal: validate "contact-phase reweighted distillation" on REAL LIBERO actions,
not synthetic SE(3). This is the critical gate the validator demanded after PoC.

Setup:
- 10 LIBERO-spatial tasks × 50 demos = 500 demos.
- Contact phase defined by gripper-state transitions (proxy, since LIBERO has no F/T).
- Minimal BC model (CNN + Transformer + action MLP, ~20M params).
  Faster than loading SmolVLA-450M but with same loss-function-ablation logic.
- 4 training conditions:
  A. L2-naive (uniform)
  B. L2-fixed + contact reweight 3x (PROPOSED)
  C. Uniform 3x reweight (SHAM CONTROL — does reweight ALONE help, or specificity to contact?)
  D. No-reweight L2-fixed (rotation-loss-only improvement)
- Metric: action MSE on held-out demos, split contact vs free.

Outputs:
- Plots/results/scripts: /home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0/
- Large data (checkpoints, raw tensors): /data/jameskimh/cpr_distill_m0/

Targets ~2h on B200.
"""
import os
import sys
import json
import time
import glob
import math
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ===== Paths =====
DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0"
OUT_BIG = "/data/jameskimh/cpr_distill_m0"
os.makedirs(OUT_USER, exist_ok=True)
os.makedirs(OUT_BIG, exist_ok=True)

DEVICE = torch.device("cuda:0")
torch.manual_seed(42)
np.random.seed(42)


# ===== Dataset =====

class LiberoBCDataset(Dataset):
    """
    Loads (obs, action, contact_mask, gripper_state) flattened across demos.
    contact_mask = 1 in window around gripper state transitions.
    """
    def __init__(self, hdf5_files, contact_window=3, train=True, train_frac=0.9, seed=0):
        self.contact_window = contact_window
        # Index: (file_path, demo_key, timestep)
        index = []
        for fp in hdf5_files:
            with h5py.File(fp, "r") as f:
                demo_keys = list(f["data"].keys())
                # split per-file for stable split
                rng = np.random.RandomState(seed + hash(fp) % 1000)
                rng.shuffle(demo_keys)
                n_train = int(train_frac * len(demo_keys))
                if train:
                    selected = demo_keys[:n_train]
                else:
                    selected = demo_keys[n_train:]
                for dk in selected:
                    T = f["data"][dk]["actions"].shape[0]
                    # store demo-level entry
                    index.append((fp, dk, T))
        self.index = index
        # Pre-cache demos into memory for speed (small dataset)
        print(f"  Caching {len(index)} demos into memory...")
        self.cache = {}
        for fp, dk, T in index:
            with h5py.File(fp, "r") as f:
                d = f["data"][dk]
                acts = d["actions"][:]  # (T, 7)
                grip = d["obs/gripper_states"][:]  # (T, 2)
                rgb_a = d["obs/agentview_rgb"][:]  # (T, 128, 128, 3)
                rgb_w = d["obs/eye_in_hand_rgb"][:]
                ee_pos = d["obs/ee_pos"][:]
                ee_ori = d["obs/ee_ori"][:]
                # Contact mask via gripper transitions
                gripper_signal = grip.sum(axis=-1)  # (T,)
                grip_diff = np.abs(np.diff(gripper_signal, prepend=gripper_signal[0]))
                grip_change_idx = np.where(grip_diff > 0.02)[0]
                contact = np.zeros(T, dtype=np.float32)
                for ci in grip_change_idx:
                    lo = max(0, ci - contact_window)
                    hi = min(T, ci + contact_window + 1)
                    contact[lo:hi] = 1.0
                # If no transitions detected, mark last 30% as contact (safety fallback)
                if contact.sum() == 0:
                    contact[int(T * 0.7):] = 1.0
                self.cache[(fp, dk)] = {
                    "actions": acts.astype(np.float32),
                    "rgb_a": rgb_a,
                    "rgb_w": rgb_w,
                    "ee_pos": ee_pos.astype(np.float32),
                    "ee_ori": ee_ori.astype(np.float32),
                    "gripper": grip.astype(np.float32),
                    "contact": contact,
                    "T": T,
                }
        # Flatten to (demo_idx, timestep)
        self.flat = []
        for di, (fp, dk, T) in enumerate(self.index):
            for t in range(T):
                self.flat.append((di, t))
        print(f"  Total timesteps: {len(self.flat)}")

    def __len__(self):
        return len(self.flat)

    def __getitem__(self, i):
        di, t = self.flat[i]
        fp, dk, _ = self.index[di]
        d = self.cache[(fp, dk)]
        # Image: agentview only (cut wrist for compactness; both is heavy)
        rgb = d["rgb_a"][t]  # (128, 128, 3) uint8
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # (3, 128, 128)
        state = np.concatenate([d["ee_pos"][t], d["ee_ori"][t], d["gripper"][t]])  # (8,)
        return {
            "img": img,
            "state": torch.from_numpy(state).float(),
            "action": torch.from_numpy(d["actions"][t]).float(),  # (7,)
            "contact": torch.tensor(d["contact"][t]).float(),
        }


# ===== Model =====

class TinyBC(nn.Module):
    """Minimal BC: small CNN + state MLP + action MLP. ~5M params."""
    def __init__(self, img_size=128, state_dim=8, action_dim=7, hidden=256):
        super().__init__()
        # Tiny CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, img, state):
        h_img = self.cnn(img)  # (B, 256)
        h_state = self.state_enc(state)
        return self.head(torch.cat([h_img, h_state], dim=-1))


# ===== Loss =====

def action_loss(pred, target, contact_mask, mode):
    """
    pred, target: (B, 7) — first 3 = translation, next 3 = rotation (axis-angle in LIBERO),
                          last 1 = gripper
    contact_mask: (B,)
    mode:
      'l2_naive'        — uniform L2 (ActDistill style baseline)
      'cpr_3x'          — contact reweight 3x (PROPOSED)
      'sham_uniform_3x' — all timesteps weighted 3x (sham control)
      'l2_fixed_only'   — uniform L2 (same as l2_naive in this minimal BC; double-cover not relevant for axis-angle)
    """
    # Per-sample loss
    err = (pred - target) ** 2  # (B, 7)
    err_per = err.sum(-1)  # (B,)
    if mode in ("l2_naive", "l2_fixed_only"):
        return err_per.mean()
    elif mode == "cpr_3x":
        # Contact-phase rotation/translation loss weighted 3x
        w = 1.0 + 2.0 * contact_mask  # 1x in free, 3x in contact
        return (err_per * w).mean()
    elif mode == "sham_uniform_3x":
        return (err_per * 3.0).mean()
    else:
        raise ValueError(mode)


# ===== Training =====

def train_model(mode, train_loader, val_loader, n_epochs=2, lr=1e-3, device=DEVICE, log_prefix=""):
    model = TinyBC().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * len(train_loader))
    history = {"train_loss": [], "val_action_mse": [], "val_action_mse_contact": [], "val_action_mse_free": []}
    for ep in range(n_epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for batch in train_loader:
            img = batch["img"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            contact = batch["contact"].to(device, non_blocking=True)
            pred = model(img, state)
            loss = action_loss(pred, target, contact, mode)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item() * img.size(0)
            n += img.size(0)
        train_loss = running / n
        # Eval
        val_stats = eval_model(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_action_mse"].append(val_stats["all"])
        history["val_action_mse_contact"].append(val_stats["contact"])
        history["val_action_mse_free"].append(val_stats["free"])
        elapsed = time.time() - t0
        print(f"  [{log_prefix}] Epoch {ep+1}/{n_epochs}: train={train_loss:.5f}, "
              f"val_all={val_stats['all']:.5f}, val_contact={val_stats['contact']:.5f}, "
              f"val_free={val_stats['free']:.5f} ({elapsed:.1f}s)")
    return model, history


def eval_model(model, val_loader, device):
    model.eval()
    err_sum_all = 0.0
    err_sum_contact = 0.0
    err_sum_free = 0.0
    n_all = 0
    n_contact = 0
    n_free = 0
    rot_err_contact = 0.0
    rot_err_free = 0.0
    trans_err_contact = 0.0
    trans_err_free = 0.0
    with torch.no_grad():
        for batch in val_loader:
            img = batch["img"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["action"].to(device, non_blocking=True)
            contact = batch["contact"].to(device, non_blocking=True)
            pred = model(img, state)
            per_sample = ((pred - target) ** 2).sum(-1)  # (B,)
            # Decompose: translation (0:3), rotation (3:6), gripper (6)
            trans_err = ((pred[:, :3] - target[:, :3]) ** 2).sum(-1)
            rot_err = ((pred[:, 3:6] - target[:, 3:6]) ** 2).sum(-1)
            contact_mask = (contact > 0.5)
            err_sum_all += per_sample.sum().item()
            n_all += img.size(0)
            err_sum_contact += per_sample[contact_mask].sum().item()
            err_sum_free += per_sample[~contact_mask].sum().item()
            n_contact += contact_mask.sum().item()
            n_free += (~contact_mask).sum().item()
            rot_err_contact += rot_err[contact_mask].sum().item()
            rot_err_free += rot_err[~contact_mask].sum().item()
            trans_err_contact += trans_err[contact_mask].sum().item()
            trans_err_free += trans_err[~contact_mask].sum().item()
    return {
        "all": err_sum_all / max(n_all, 1),
        "contact": err_sum_contact / max(n_contact, 1),
        "free": err_sum_free / max(n_free, 1),
        "rot_contact": rot_err_contact / max(n_contact, 1),
        "rot_free": rot_err_free / max(n_free, 1),
        "trans_contact": trans_err_contact / max(n_contact, 1),
        "trans_free": trans_err_free / max(n_free, 1),
        "n_contact": n_contact,
        "n_free": n_free,
    }


def main():
    print("=" * 60)
    print("CPR-Distill M0 Smoke Test on LIBERO-spatial")
    print(f"Device: {DEVICE}, output_user: {OUT_USER}, output_big: {OUT_BIG}")
    print("=" * 60)
    t_start = time.time()

    # ===== Data =====
    print("\n[1/3] Loading LIBERO-spatial dataset...")
    hdf5_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    print(f"  Found {len(hdf5_files)} task files")
    # Limit for smoke (10 tasks already, take all)
    train_ds = LiberoBCDataset(hdf5_files, train=True, train_frac=0.9, seed=0)
    val_ds = LiberoBCDataset(hdf5_files, train=False, train_frac=0.9, seed=0)
    print(f"  Train size: {len(train_ds)} timesteps, Val size: {len(val_ds)} timesteps")
    bs = 256
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    # ===== Train 4 conditions =====
    print("\n[2/3] Training 4 conditions...")
    n_epochs = 3
    conditions = {
        "A_l2_naive":            "l2_naive",
        "B_cpr_3x":              "cpr_3x",
        "C_sham_uniform_3x":     "sham_uniform_3x",
        "D_l2_fixed_only":       "l2_fixed_only",
    }
    final_results = {}
    histories = {}
    final_models_path = {}
    for tag, mode in conditions.items():
        print(f"\n  === Training {tag} (mode={mode}) ===")
        # Same seed for fair comparison
        torch.manual_seed(42)
        np.random.seed(42)
        model, hist = train_model(mode, train_loader, val_loader, n_epochs=n_epochs, log_prefix=tag)
        # Save model weights to /data
        ckpt_path = os.path.join(OUT_BIG, f"{tag}.pt")
        torch.save(model.state_dict(), ckpt_path)
        final_models_path[tag] = ckpt_path
        # Final eval
        final_eval = eval_model(model, val_loader, DEVICE)
        final_results[tag] = final_eval
        histories[tag] = hist
        print(f"  Saved {ckpt_path}")

    # ===== Analysis =====
    print("\n[3/3] Analysis...")
    print("\n--- Final validation MSE (overall / contact / free) ---")
    for tag in conditions:
        r = final_results[tag]
        print(f"  {tag}: all={r['all']:.5f}, contact={r['contact']:.5f}, free={r['free']:.5f}")

    # Compute gains vs A (L2-naive)
    a = final_results["A_l2_naive"]
    print("\n--- Improvement vs A (L2-naive) ---")
    rows = []
    for tag in ["B_cpr_3x", "C_sham_uniform_3x", "D_l2_fixed_only"]:
        r = final_results[tag]
        gain_all = (a["all"] - r["all"]) / a["all"] * 100
        gain_contact = (a["contact"] - r["contact"]) / a["contact"] * 100
        gain_free = (a["free"] - r["free"]) / a["free"] * 100
        spec = gain_contact / (abs(gain_free) + 1e-6)
        print(f"  {tag}: all={gain_all:+.2f}%, contact={gain_contact:+.2f}%, free={gain_free:+.2f}%, "
              f"specificity={spec:.2f}x")
        rows.append({"tag": tag, "gain_all_pct": gain_all, "gain_contact_pct": gain_contact,
                     "gain_free_pct": gain_free, "specificity_x": spec})

    # ===== Gate evaluation =====
    print("\n" + "=" * 60)
    print("GATE EVALUATION (≥2pp success rate gain proxy via contact MSE reduction)")
    print("=" * 60)
    b_contact_gain = (a["contact"] - final_results["B_cpr_3x"]["contact"]) / a["contact"] * 100
    c_contact_gain = (a["contact"] - final_results["C_sham_uniform_3x"]["contact"]) / a["contact"] * 100
    b_free_gain = (a["free"] - final_results["B_cpr_3x"]["free"]) / a["free"] * 100
    # Specificity: B should be much better than sham C
    delta_b_vs_c = b_contact_gain - c_contact_gain
    print(f"  B (CPR) contact-phase MSE reduction: {b_contact_gain:.2f}%")
    print(f"  C (sham 3x) contact-phase MSE reduction: {c_contact_gain:.2f}%")
    print(f"  Δ(B - C) contact gain: {delta_b_vs_c:+.2f}pp  ← contact-specificity signal")
    print(f"  B free-phase MSE change: {b_free_gain:+.2f}%  (should be ≤ 0 or near 0)")

    gate_a = b_contact_gain > 0
    gate_b = delta_b_vs_c > 2.0  # ≥2pp better than sham (proxy for >2pp SR)
    gate_c = b_free_gain < b_contact_gain  # specificity to contact

    verdict_lines = []
    if gate_a:
        verdict_lines.append("✅ Gate A: CPR reduces contact-phase MSE vs L2-naive")
    else:
        verdict_lines.append("❌ Gate A FAIL: CPR did NOT reduce contact MSE")
    if gate_b:
        verdict_lines.append(f"✅ Gate B: CPR beats sham by {delta_b_vs_c:.1f}pp on contact phase (≥2pp threshold)")
    else:
        verdict_lines.append(f"❌ Gate B FAIL: CPR did not beat sham by ≥2pp (got {delta_b_vs_c:.2f}pp)")
    if gate_c:
        verdict_lines.append("✅ Gate C: Contact-specific (gain in contact > gain in free)")
    else:
        verdict_lines.append("❌ Gate C FAIL: No contact-specificity")

    overall_pass = gate_a and gate_b and gate_c
    if overall_pass:
        verdict = "M0 SMOKE TEST: PASS — proceed to W1 (full ActDistill setup)"
    elif gate_a and gate_c:
        verdict = "M0 SMOKE TEST: PARTIAL — directional support but specificity vs sham not strong enough"
    else:
        verdict = "M0 SMOKE TEST: FAIL — fallback framing or pivot needed"

    for line in verdict_lines:
        print(f"  {line}")
    print(f"\n>>> {verdict} <<<")

    # ===== Save =====
    out = {
        "elapsed_sec": time.time() - t_start,
        "device": str(DEVICE),
        "config": {
            "n_epochs": n_epochs,
            "batch_size": bs,
            "data_dir": DATA_DIR,
            "n_task_files": len(hdf5_files),
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "contact_window": 3,
            "reweight_factor": 3.0,
        },
        "final_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in final_results.items()},
        "improvement_rows": rows,
        "gates": {
            "A_cpr_reduces_contact_mse": bool(gate_a),
            "B_cpr_beats_sham_by_2pp": bool(gate_b),
            "C_contact_specific": bool(gate_c),
            "delta_B_vs_C_contact_gain_pp": float(delta_b_vs_c),
            "B_contact_gain_pct": float(b_contact_gain),
            "C_contact_gain_pct": float(c_contact_gain),
            "B_free_gain_pct": float(b_free_gain),
        },
        "verdict": verdict,
        "checkpoint_paths": final_models_path,
        "histories": {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in histories.items()},
    }
    json_path = os.path.join(OUT_USER, "results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults JSON: {json_path}")
    print(f"Model checkpoints: {OUT_BIG}")

    # ===== Plot =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for tag, h in histories.items():
            axes[0].plot(h["val_action_mse_contact"], label=f"{tag} contact", marker='o')
        axes[0].set_title("Validation MSE (contact phase) over epochs")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("contact-phase MSE")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        tags = list(conditions.keys())
        contact_mse = [final_results[t]["contact"] for t in tags]
        free_mse = [final_results[t]["free"] for t in tags]
        x = np.arange(len(tags))
        w = 0.35
        axes[1].bar(x - w/2, contact_mse, w, label="contact MSE")
        axes[1].bar(x + w/2, free_mse, w, label="free MSE")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tags, rotation=15)
        axes[1].set_title("Final MSE by condition and phase")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = os.path.join(OUT_USER, "m0_results.png")
        plt.savefig(plot_path, dpi=140)
        print(f"Plot: {plot_path}")
    except Exception as e:
        print(f"Plot failed: {e}")

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
