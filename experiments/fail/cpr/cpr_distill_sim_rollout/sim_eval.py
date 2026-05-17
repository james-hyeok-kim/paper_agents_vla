"""
CPR-Distill Sim Rollout: MSE → SR conversion check.

This is the critical experiment that decides CPR-Distill's fate:
  Does contact MSE 6.8% reduction translate to task success rate improvement?

Pipeline:
  1. Train 3 fresh TinyBC models (single seed):
       - baseline (factor=1.0)
       - channel_diff_1.5 (factor=1.5 with gripper_channel_diff detector)
       - sham_3.0 (uniform 3× reweight)
  2. Sim rollout each model on libero_spatial (10 tasks × 3 episodes = 30 rollouts/model)
  3. Compute SR per condition, compare.

Environment setup (set externally):
  LD_LIBRARY_PATH=/tmp/egl_extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
  PYTHONPATH=/home/jovyan/workspace/Workspace_Lerobot/lerobot/src

Outputs:
  - /home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_sim_rollout/: results.json, plots
  - /data/jameskimh/cpr_distill_sim_rollout/: trained model checkpoints
"""
import os, sys, json, time, glob
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_m0")
sys.path.insert(0, "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_contact_diagnostic")
from m0_smoke import TinyBC, eval_model
from diagnostic import LiberoDetectorDataset, DETECTORS

DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_USER = "/home/jovyan/workspace/paper_agents_vla/experiments/cpr_distill_sim_rollout"
OUT_BIG = "/data/jameskimh/cpr_distill_sim_rollout"
os.makedirs(OUT_USER, exist_ok=True); os.makedirs(OUT_BIG, exist_ok=True)
DEVICE = torch.device("cuda:0")
SUITE = "libero_spatial"
EPISODES_PER_TASK = 3
MAX_STEPS = 200


# ========== Training ==========

def train_model(detector_name, mode, factor, n_epochs=3, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    detector_fn = DETECTORS[detector_name]
    train_ds = LiberoDetectorDataset(hdf5, detector_fn, train=True)
    val_ds = LiberoDetectorDataset(hdf5, detector_fn, train=False)
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
    return model, eval_model(model, val_loader, DEVICE)


# ========== Sim rollout ==========

def obs_to_model_input(obs):
    """Convert LIBERO env obs dict → (img_tensor, state_tensor) for TinyBC."""
    # Pixels: (128, 128, 3) uint8
    img_np = obs["pixels"]["image"] if isinstance(obs["pixels"], dict) else obs["pixels"]
    img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # (3, 128, 128)
    # State: ee_pos (3) + ee_rotvec (3, from quat) + gripper.qpos (2) = 8
    rs = obs["robot_state"]
    ee_pos = np.asarray(rs["eef"]["pos"], dtype=np.float32)
    ee_quat = np.asarray(rs["eef"]["quat"], dtype=np.float32)  # xyzw
    # scipy expects xyzw
    ee_rotvec = R.from_quat(ee_quat).as_rotvec().astype(np.float32)
    gripper = np.asarray(rs["gripper"]["qpos"], dtype=np.float32)
    state = np.concatenate([ee_pos, ee_rotvec, gripper])  # (8,)
    return img.unsqueeze(0).to(DEVICE), torch.from_numpy(state).unsqueeze(0).to(DEVICE)


def run_episode(env, model, max_steps=MAX_STEPS):
    obs, info = env.reset()
    model.eval()
    success = False
    with torch.no_grad():
        for step in range(max_steps):
            img, state = obs_to_model_input(obs)
            action_tensor = model(img, state)
            action = action_tensor[0].cpu().numpy().astype(np.float32)
            # Clip to action space [-1, 1]
            action = np.clip(action, -1.0, 1.0)
            obs, rew, term, trunc, info = env.step(action)
            if info.get("is_success", False) or term:
                success = True
                break
            if trunc:
                break
    return success, step + 1


def evaluate_model_on_suite(model, suite, suite_name, n_episodes_per_task=EPISODES_PER_TASK, max_steps=MAX_STEPS):
    from lerobot.envs.libero import LiberoEnv
    n_tasks = len(suite.tasks)
    results = []
    total_steps = 0
    for task_id in range(n_tasks):
        task_desc = suite.get_task(task_id).language
        env = LiberoEnv(
            task_suite=suite, task_id=task_id, task_suite_name=suite_name,
            episode_length=max_steps,
            camera_name="agentview_image",
            obs_type="pixels_agent_pos",
            observation_width=128, observation_height=128,
            init_states=True, episode_index=0, n_envs=1,
        )
        task_successes = 0
        task_steps = 0
        for ep in range(n_episodes_per_task):
            # Set episode_index to vary init state
            env.episode_index = ep
            env.init_state_id = ep
            success, steps = run_episode(env, model, max_steps=max_steps)
            task_successes += int(success)
            task_steps += steps
            print(f"      task {task_id} ep {ep}: {'✓ SUCCESS' if success else '✗ fail'} ({steps} steps)")
        sr = task_successes / n_episodes_per_task
        results.append({
            "task_id": task_id, "task_desc": task_desc,
            "n_episodes": n_episodes_per_task,
            "successes": task_successes, "success_rate": sr,
            "avg_steps": task_steps / n_episodes_per_task,
        })
        print(f"    task {task_id} ({task_desc[:40]}...): SR={sr:.3f} ({task_successes}/{n_episodes_per_task})")
        env.close()
        total_steps += task_steps
    overall_sr = float(np.mean([r["success_rate"] for r in results]))
    return {"per_task": results, "overall_sr": overall_sr, "total_steps": total_steps}


# ========== Main ==========

def main():
    t0 = time.time()
    print("=" * 60)
    print("CPR-Distill Sim Rollout — MSE → SR Translation Check")
    print(f"Suite: {SUITE}, episodes/task={EPISODES_PER_TASK}, max_steps={MAX_STEPS}")
    print("=" * 60)

    # ===== Train 3 conditions =====
    conditions = [
        ("baseline_factor1.0",    "last_30",              "cpr",  1.0),
        ("channel_diff_1.5",      "gripper_channel_diff", "cpr",  1.5),
        ("sham_3.0",              "last_30",              "sham", 3.0),
    ]
    models = {}
    eval_results = {}
    print("\n[1/2] Training 3 models...")
    for tag, det, mode, factor in conditions:
        print(f"\n  Training {tag} (det={det}, mode={mode}, factor={factor})...")
        t1 = time.time()
        m, ev = train_model(det, mode, factor)
        models[tag] = m
        eval_results[tag] = ev
        ckpt = os.path.join(OUT_BIG, f"{tag}.pt")
        torch.save(m.state_dict(), ckpt)
        print(f"  {tag}: contact MSE={ev['contact']:.5f}, free={ev['free']:.5f}, all={ev['all']:.5f} ({time.time()-t1:.1f}s)")

    # ===== Sim rollout =====
    print("\n[2/2] Sim rollout per condition...")
    from lerobot.envs.libero import _get_suite
    suite = _get_suite(SUITE)
    sr_results = {}
    for tag, _, _, _ in conditions:
        print(f"\n  === Rolling out {tag} ===")
        t1 = time.time()
        sr_results[tag] = evaluate_model_on_suite(models[tag], suite, SUITE,
                                                   n_episodes_per_task=EPISODES_PER_TASK,
                                                   max_steps=MAX_STEPS)
        print(f"  {tag} overall SR: {sr_results[tag]['overall_sr']:.3f} ({time.time()-t1:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Elapsed: {elapsed:.1f}s ===")

    # ===== Analysis =====
    print("\n--- MSE vs SR Summary ---")
    print(f"{'Condition':<24} {'Contact MSE':>13} {'Overall MSE':>13} {'SR':>8}")
    print("-" * 65)
    for tag, _, _, _ in conditions:
        e = eval_results[tag]
        s = sr_results[tag]
        print(f"{tag:<24} {e['contact']:>13.5f} {e['all']:>13.5f} {s['overall_sr']:>7.3f}")

    # Critical comparison: does channel_diff_1.5 SR > baseline SR?
    b_sr = sr_results["baseline_factor1.0"]["overall_sr"]
    cd_sr = sr_results["channel_diff_1.5"]["overall_sr"]
    sh_sr = sr_results["sham_3.0"]["overall_sr"]
    delta_sr = (cd_sr - b_sr) * 100
    delta_sham_sr = (cd_sr - sh_sr) * 100
    print(f"\nΔ SR (CPR_1.5 vs baseline): {delta_sr:+.1f}pp")
    print(f"Δ SR (CPR_1.5 vs sham_3.0):  {delta_sham_sr:+.1f}pp")
    if delta_sr >= 5.0:
        verdict = "✅ STRONG WIN: CPR significantly improves task SR — paper alive"
    elif delta_sr >= 2.0:
        verdict = "✅ Positive: CPR shows meaningful SR improvement"
    elif delta_sr >= 0.0:
        verdict = "🟡 Marginal: SR slightly positive but within noise"
    else:
        verdict = "❌ Negative: contact MSE gain does NOT translate to SR"
    print(f"\n>>> {verdict} <<<")

    out = {
        "elapsed_sec": elapsed,
        "config": {
            "suite": SUITE, "episodes_per_task": EPISODES_PER_TASK,
            "max_steps": MAX_STEPS, "n_tasks": 10,
        },
        "mse_results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in eval_results.items()},
        "sr_results": sr_results,
        "delta_sr_cpr_vs_baseline_pp": float(delta_sr),
        "delta_sr_cpr_vs_sham_pp": float(delta_sham_sr),
        "verdict": verdict,
    }
    with open(os.path.join(OUT_USER, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults: {os.path.join(OUT_USER, 'results.json')}")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        tags = [t[0] for t in conditions]
        srs = [sr_results[t]["overall_sr"] for t in tags]
        contact_mses = [eval_results[t]["contact"] for t in tags]
        # 1. Bar chart: SR per condition
        axes[0].bar(tags, srs, color=['C0', 'C1', 'C2'])
        for x, sr in enumerate(srs):
            axes[0].text(x, sr + 0.01, f"{sr:.3f}", ha='center', fontsize=11, fontweight='bold')
        axes[0].set_ylabel("Task Success Rate"); axes[0].set_ylim(0, max(srs + [0.1]) * 1.2)
        axes[0].set_title(f"SR on {SUITE} (10 tasks × {EPISODES_PER_TASK} ep)")
        axes[0].grid(alpha=0.3, axis='y'); plt.setp(axes[0].get_xticklabels(), rotation=15)
        # 2. MSE vs SR scatter
        axes[1].scatter(contact_mses, srs, s=200, c=['C0','C1','C2'])
        for i, t in enumerate(tags):
            axes[1].annotate(t, (contact_mses[i], srs[i]), xytext=(5, 5),
                             textcoords='offset points', fontsize=10)
        axes[1].set_xlabel("Contact MSE (val)"); axes[1].set_ylabel("Task SR")
        axes[1].set_title("MSE → SR translation"); axes[1].grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_USER, "sim_eval.png"), dpi=140)
        print(f"Plot: {os.path.join(OUT_USER, 'sim_eval.png')}")
    except Exception as e:
        print(f"Plot failed: {e}")


if __name__ == "__main__":
    main()
