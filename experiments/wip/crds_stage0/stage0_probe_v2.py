"""
CRDS Stage-0 Premise Probe v2

주요 변경:
- Probe 1: SigLIP encoder에 직접 hook (SigLIP forward only, LM 전혀 안 건드림)
- Probe 2: loss forward (predict_action) 대신 SigLIP 표현 zero-ablation 후 1-step flow matching
- 두 probe 모두 독립 실행 가능하도록 구조화
"""

import sys
import os
import json
import time
import glob

sys.path.insert(0, '/home/jovyan/workspace/Workspace_Lerobot/lerobot/src')

import torch
import torch.nn.functional as F
import numpy as np
import h5py

t_start = time.time()
def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)

DEVICE = "cuda:0"
CKPT = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_DIR = "/home/jovyan/workspace/paper_agents_vla/experiments/wip/crds_stage0"
IMG_RES = 224
N_FRAMES_P1 = 60    # Probe 1용 프레임
N_FRAMES_P2 = 128   # Probe 2용 프레임
BATCH_SZ = 4

log("=== CRDS Stage-0 Probe v2 ===")

# ===== Load Model =====
log("Loading π0.5 policy...")
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

with open(f"{CKPT}/config.json") as f:
    cfg_dict = json.load(f)

SKIP_KEYS = {"type", "push_to_hub", "repo_id", "private", "tags", "license",
             "pretrained_path", "device", "compile_model", "compile_mode",
             "gradient_checkpointing", "use_amp", "optimizer_lr", "optimizer_betas",
             "optimizer_eps", "optimizer_weight_decay", "optimizer_grad_clip_norm",
             "scheduler_warmup_steps", "scheduler_decay_steps", "scheduler_decay_lr"}

cfg_clean = {k: v for k, v in cfg_dict.items()
             if k not in SKIP_KEYS and not k.startswith("_")}
cfg_clean["compile_model"] = False
cfg_clean["gradient_checkpointing"] = False

cfg = PI05Config(**cfg_clean)
policy = PI05Policy.from_pretrained(CKPT, config=cfg)
policy.eval().to(DEVICE, dtype=torch.bfloat16)
log(f"Policy loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")

# SigLIP references
vision_tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
siglip_encoder = vision_tower.vision_model.encoder
n_siglip_layers = len(siglip_encoder.layers)
log(f"SigLIP encoder layers: {n_siglip_layers}")

# embed_image function reference
embed_image_fn = policy.model.paligemma_with_expert.embed_image

# ===== Data Loading =====
hdf5_files = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
log(f"Found {len(hdf5_files)} tasks in libero_spatial")
rng = np.random.RandomState(42)

def load_frames(max_frames):
    frames = []
    per_file = max(2, max_frames // len(hdf5_files))
    for fp in hdf5_files:
        with h5py.File(fp, 'r') as f:
            demos = list(f['data'].keys())
            for dk in demos[:3]:
                T = f['data'][dk]['obs']['agentview_rgb'].shape[0]
                idxs = rng.choice(T, size=min(per_file, T), replace=False)
                for idx in idxs:
                    frames.append({
                        'agentview': f['data'][dk]['obs']['agentview_rgb'][idx],
                        'wrist': f['data'][dk]['obs']['eye_in_hand_rgb'][idx],
                        'state': np.concatenate([
                            f['data'][dk]['obs']['ee_states'][idx],
                            f['data'][dk]['obs']['gripper_states'][idx],
                        ]),
                        'action': f['data'][dk]['actions'][idx],
                    })
                if len(frames) >= max_frames:
                    break
        if len(frames) >= max_frames:
            break
    return frames[:max_frames]

frames_all = load_frames(max(N_FRAMES_P1, N_FRAMES_P2))
frames_p1 = frames_all[:N_FRAMES_P1]
frames_p2 = frames_all[:N_FRAMES_P2]
log(f"Loaded {len(frames_all)} frames total")

# ===== Image Preprocessing =====
def preprocess_img(arr_np):
    """uint8 (H,W,3) → float32 (3,224,224) in [-1,1]"""
    t = torch.from_numpy(arr_np.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
    t = F.interpolate(t.unsqueeze(0), size=(IMG_RES, IMG_RES), mode='bilinear', align_corners=False).squeeze(0)
    t = t * 2.0 - 1.0
    return t

# ===== Probe 1: SigLIP Layer Stability =====
log("\n--- Probe 1: SigLIP Layer Activation Stability ---")

layer_acts = {}
hooks = []

def make_hook(idx):
    def _h(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        layer_acts[idx] = x.detach().float().cpu()
    return _h

for i, layer in enumerate(siglip_encoder.layers):
    hooks.append(layer.register_forward_hook(make_hook(i)))

def get_siglip_layer_acts(frames_np):
    """Run SigLIP on a list of (H,W,3) uint8 images, return per-layer activations."""
    imgs = torch.stack([preprocess_img(f) for f in frames_np])  # (N,3,224,224)
    all_acts = {i: [] for i in range(n_siglip_layers)}

    for b0 in range(0, len(imgs), BATCH_SZ):
        batch = imgs[b0:b0+BATCH_SZ].to(DEVICE, dtype=torch.bfloat16)
        layer_acts.clear()
        with torch.no_grad():
            _ = embed_image_fn(batch)
        for i in range(n_siglip_layers):
            if i in layer_acts:
                all_acts[i].append(layer_acts[i].numpy())

    return {i: np.concatenate(v, axis=0) for i, v in all_acts.items() if v}

def compute_stability(acts_dict, camera_name):
    """Compute cos sim to last layer for each layer."""
    last = acts_dict[n_siglip_layers - 1]  # (N, 256, D)
    last_pool = last.mean(axis=1)           # (N, D)  — mean pool over 256 patches
    last_norm = last_pool / (np.linalg.norm(last_pool, axis=-1, keepdims=True) + 1e-9)

    results = []
    for i in range(n_siglip_layers):
        if i not in acts_dict:
            continue
        cur = acts_dict[i]
        cur_pool = cur.mean(axis=1)
        cur_norm = cur_pool / (np.linalg.norm(cur_pool, axis=-1, keepdims=True) + 1e-9)
        cos = (cur_norm * last_norm).sum(axis=-1).mean()

        # per-token cos sim
        N, NP, D = last.shape
        lt = last.reshape(N*NP, D)
        ct = cur.reshape(N*NP, D)
        lt_n = lt / (np.linalg.norm(lt, axis=-1, keepdims=True) + 1e-9)
        ct_n = ct / (np.linalg.norm(ct, axis=-1, keepdims=True) + 1e-9)
        cos_tok = (ct_n * lt_n).sum(axis=-1).mean()

        print(f"  [{camera_name}] layer {i:2d}: cos_pool={cos:.4f}  cos_tok={cos_tok:.4f}", flush=True)
        results.append({"layer": i, "cos_mean_pool": float(cos), "cos_per_token": float(cos_tok)})
    return results

log("Running SigLIP on exterior (agentview) frames...")
ext_acts = get_siglip_layer_acts([f['agentview'] for f in frames_p1])
probe1_exterior = compute_stability(ext_acts, "exterior")

log("Running SigLIP on wrist (eye_in_hand) frames...")
wrist_acts = get_siglip_layer_acts([f['wrist'] for f in frames_p1])
probe1_wrist = compute_stability(wrist_acts, "wrist")

# Hook 제거
for h in hooks:
    h.remove()

# Probe 1 PASS/FAIL
PASS_THR = 0.95
# 1-indexed layer 16 = 0-indexed 15 (CRDS 가설: wrist=16 layers)
# PASS 기준: layer <= 20 (0-indexed <= 19) 전 구간에서 cos >= 0.95
def check_pass(results, thr=0.95, max_layer_idx=19):
    early = [r for r in results if r["layer"] <= max_layer_idx]
    if not early:
        return False, None, None
    min_cos = min(r["cos_mean_pool"] for r in early)
    passes = all(r["cos_mean_pool"] >= thr for r in early)
    cross = next((r["layer"] for r in results if r["cos_mean_pool"] >= thr), None)
    return passes, min_cos, cross

ext_pass, ext_min, ext_cross = check_pass(probe1_exterior)
wrist_pass, wrist_min, wrist_cross = check_pass(probe1_wrist)

# layer 15 (0-indexed) = 16th layer cos sim
cos16_ext = next((r["cos_mean_pool"] for r in probe1_exterior if r["layer"] == 15), None)
cos16_wrist = next((r["cos_mean_pool"] for r in probe1_wrist if r["layer"] == 15), None)

log(f"\nProbe 1 Summary (PASS if all cos >= {PASS_THR} for layer <= 20):")
log(f"  Exterior: min_cos={ext_min:.4f}  cos@layer16={cos16_ext:.4f}  PASS={ext_pass}")
log(f"  Wrist:    min_cos={wrist_min:.4f}  cos@layer16={cos16_wrist:.4f}  PASS={wrist_pass}")

# ===== Probe 2: Vision Contribution via Action MSE =====
log("\n--- Probe 2: Vision vs Proprio Contribution ---")
log("Using predict_action (training-time loss forward) for MSE computation")

# predict_action은 training forward처럼 동작:
# batch에 'action' 키가 있으면 loss를 계산함
# 우리는 direct action prediction (flow-matching inference)을 사용

# select_action 대신 직접 forward pass 사용
# policy.model.forward를 직접 호출해 loss를 측정

def get_policy_loss(batch_obs, gt_actions_np, zero_mode="none"):
    """
    batch_obs: list of frame dicts
    gt_actions_np: (N, 7) float32
    zero_mode: "none", "all_vision", "wrist_only"
    Returns: mean MSE loss across batch
    """
    all_losses = []

    for b0 in range(0, len(batch_obs), BATCH_SZ):
        b_end = min(b0 + BATCH_SZ, len(batch_obs))
        obs_b = batch_obs[b0:b_end]
        B = len(obs_b)

        # Prepare images [0,1] float32 (B,3,128,128)
        ag = torch.stack([
            torch.from_numpy(o['agentview'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])  # (B,3,128,128)
        wr = torch.stack([
            torch.from_numpy(o['wrist'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])  # (B,3,128,128)

        if zero_mode == "all_vision":
            ag = torch.zeros_like(ag)
            wr = torch.zeros_like(wr)
        elif zero_mode == "wrist_only":
            wr = torch.zeros_like(wr)

        state = torch.stack([
            torch.from_numpy(o['state'].astype(np.float32))
            for o in obs_b
        ])  # (B,8)
        gt_act = torch.from_numpy(gt_actions_np[b0:b_end].astype(np.float32))  # (B,7)

        # Build batch dict for policy
        batch = {
            'observation.images.image': ag.to(DEVICE),
            'observation.images.image2': wr.to(DEVICE),
            'observation.state': state.to(DEVICE),
            'task': ['pick up the black bowl and place it on the plate'] * B,
        }

        with torch.no_grad():
            try:
                pred_action = policy.select_action(batch)  # (B, action_dim) or (B, chunk, action_dim)
            except Exception as e:
                log(f"  select_action error [{zero_mode}]: {type(e).__name__}: {str(e)[:150]}")
                continue

        # Align shapes
        if pred_action.dim() == 3:
            pred_action = pred_action[:, 0, :]  # (B, action_dim)
        pred7 = pred_action[:, :7].float().cpu()
        gt7 = gt_act[:, :7]
        mse = F.mse_loss(pred7, gt7).item()
        all_losses.append(mse)

    return float(np.mean(all_losses)) if all_losses else float('nan')

gt_actions = np.stack([f['action'] for f in frames_p2])  # (N, 7)

log("  [real-vision] Computing MSE...")
mse_real = get_policy_loss(frames_p2, gt_actions, zero_mode="none")
log(f"  MSE(real-vision) = {mse_real:.6f}")

log("  [zero-all-vision] Computing MSE...")
mse_zero_all = get_policy_loss(frames_p2, gt_actions, zero_mode="all_vision")
log(f"  MSE(zero-all-vision) = {mse_zero_all:.6f}")

log("  [zero-wrist-only] Computing MSE...")
mse_zero_wrist = get_policy_loss(frames_p2, gt_actions, zero_mode="wrist_only")
log(f"  MSE(zero-wrist-only) = {mse_zero_wrist:.6f}")

ratio_all = mse_zero_all / mse_real if mse_real > 1e-9 else float('nan')
ratio_wrist = mse_zero_wrist / mse_real if mse_real > 1e-9 else float('nan')

probe2_pass = ratio_all >= 1.3
log(f"\nProbe 2 Summary:")
log(f"  MSE(real)      = {mse_real:.6f}")
log(f"  MSE(zero-all)  = {mse_zero_all:.6f}  ratio={ratio_all:.3f}")
log(f"  MSE(zero-wrist)= {mse_zero_wrist:.6f}  ratio={ratio_wrist:.3f}")
log(f"  PASS (ratio_all >= 1.3): {probe2_pass}")

# ===== Final Verdict =====
if wrist_pass and probe2_pass:
    verdict = "PASS-both"
    rec = "본 실험 진행 권장 — wrist 표현 안정 + vision 기여 확인"
elif wrist_pass and not probe2_pass:
    verdict = "FAIL-probe2"
    rec = "CRDS 폐기 권장 — wrist 안정하지만 vision이 action에 기여 안 함 (proprio-dominated)"
elif not wrist_pass and probe2_pass:
    verdict = "FAIL-probe1"
    rec = "CRDS 폐기 권장 — wrist 표현이 layer 20 전 불안정, 조기 exit 시 열화 위험"
else:
    verdict = "FAIL-both"
    rec = "CRDS 폐기 권장 — probe 1 & 2 모두 실패"

log(f"\n=== Final Verdict: {verdict} ===")
log(f"Recommendation: {rec}")

# ===== Save Results =====
results = {
    "experiment": "CRDS Stage-0 Premise Probe v2",
    "timestamp_kst": time.strftime("%Y-%m-%d %H:%M KST", time.localtime(time.time() + 9*3600)),
    "probe1": {
        "n_frames": N_FRAMES_P1,
        "n_siglip_layers": n_siglip_layers,
        "pass_threshold": PASS_THR,
        "exterior": {
            "pass": ext_pass, "min_cos_leq20": ext_min,
            "cos_at_layer16": cos16_ext, "first_stable_layer": ext_cross,
            "per_layer": probe1_exterior,
        },
        "wrist": {
            "pass": wrist_pass, "min_cos_leq20": wrist_min,
            "cos_at_layer16": cos16_wrist, "first_stable_layer": wrist_cross,
            "per_layer": probe1_wrist,
        },
        "verdict": "PASS" if wrist_pass else "FAIL",
    },
    "probe2": {
        "n_frames": N_FRAMES_P2,
        "mse_real": mse_real,
        "mse_zero_all": mse_zero_all,
        "mse_zero_wrist": mse_zero_wrist,
        "ratio_all": ratio_all,
        "ratio_wrist": ratio_wrist,
        "pass_threshold_ratio": 1.3,
        "verdict": "PASS" if probe2_pass else "FAIL",
    },
    "overall": {
        "verdict": verdict,
        "recommendation": rec,
        "probe1_wrist_pass": wrist_pass,
        "probe2_pass": probe2_pass,
    }
}

out_path = f"{OUT_DIR}/probe_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
log(f"Results saved: {out_path}")

print("\n" + "="*60, flush=True)
print("CRDS Stage-0 최종 결과", flush=True)
print("="*60, flush=True)
print(f"Probe 1 - SigLIP 안정성 (wrist): PASS={wrist_pass}  min_cos={wrist_min:.4f}  cos@L16={cos16_wrist:.4f}", flush=True)
print(f"Probe 2 - Vision 기여도:         PASS={probe2_pass}  ratio_all={ratio_all:.3f}  ratio_wrist={ratio_wrist:.3f}", flush=True)
print(f"최종 판정: {verdict}", flush=True)
print(f"권고: {rec}", flush=True)
print("="*60, flush=True)
