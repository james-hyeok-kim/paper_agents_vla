"""
CRDS Stage-0 Probe 2 Only: Vision vs Proprio Contribution

policy.forward (training loss)를 사용해서 MSE 비교
- tokenizer를 직접 로드하여 OBS_LANGUAGE_TOKENS 생성
- real-vision / zero-all-vision / zero-wrist-only 비교
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
from transformers import AutoTokenizer

t_start = time.time()
def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)

DEVICE = "cuda:0"
CKPT = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_DIR = "/home/jovyan/workspace/paper_agents_vla/experiments/wip/crds_stage0"
IMG_RES = 224
N_FRAMES = 128
BATCH_SZ = 4

log("=== CRDS Probe 2: Vision Contribution ===")

# ===== Load Model =====
log("Loading π0.5 policy...")
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, ACTION

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

# Convert input_features/output_features dict-of-dicts → dict-of-PolicyFeature
from lerobot.configs.types import FeatureType, PolicyFeature
TYPE_MAP = {"VISUAL": FeatureType.VISUAL, "STATE": FeatureType.STATE, "ACTION": FeatureType.ACTION}
if "input_features" in cfg_clean:
    cfg_clean["input_features"] = {
        k: PolicyFeature(type=TYPE_MAP[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_clean["input_features"].items()
    }
if "output_features" in cfg_clean:
    cfg_clean["output_features"] = {
        k: PolicyFeature(type=TYPE_MAP[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_clean["output_features"].items()
    }

cfg = PI05Config(**cfg_clean)
policy = PI05Policy.from_pretrained(CKPT, config=cfg)
# Use float32 to avoid dtype mismatch: sample_noise() returns float32,
# which conflicts with bfloat16 model weights in action_in_proj.
# float32 uses more memory but avoids the mismatch without code surgery.
policy.eval().to(DEVICE)  # float32
log(f"Policy loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params (float32)")

# ===== Load Tokenizer =====
log("Loading PaliGemma tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
log("Tokenizer loaded")

# Load normalizer stats from checkpoint
log("Loading normalizer stats...")
from safetensors.torch import load_file as safe_load_file
normalizer_weights = safe_load_file(
    f"{CKPT}/policy_preprocessor_step_2_normalizer_processor.safetensors"
)
log(f"Normalizer keys (first 5): {list(normalizer_weights.keys())[:5]}")

# State normalizer (QUANTILES 방식)
# key: observation.state.q01, observation.state.q99
state_q01 = normalizer_weights.get("observation.state.q01", None)
state_q99 = normalizer_weights.get("observation.state.q99", None)
if state_q01 is not None:
    log(f"State q01 shape: {state_q01.shape}  q99 shape: {state_q99.shape}")

def normalize_state_quantile(state_np, q01, q99):
    """QUANTILES normalization: maps [q01, q99] → [-1, 1]"""
    q01_np = q01.numpy().astype(np.float32)
    q99_np = q99.numpy().astype(np.float32)
    # state_np shape: (B, D) or (D,)
    normed = 2.0 * (state_np - q01_np) / (q99_np - q01_np + 1e-8) - 1.0
    return np.clip(normed, -1.0, 1.0)

def build_prompt(task_str, state_norm_np_1d):
    """state_norm_np_1d: (8,) normalized float32"""
    discretized = np.digitize(state_norm_np_1d, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, discretized.tolist()))
    cleaned = task_str.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {state_str};\nAction: "

TASK_STR = "pick up the black bowl and place it on the plate"
MAX_LEN = cfg.tokenizer_max_length  # 200

def build_lang_tokens(batch_state_norm):
    """batch_state_norm: (B, 8) float32 normalized"""
    prompts = [build_prompt(TASK_STR, batch_state_norm[i]) for i in range(len(batch_state_norm))]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )
    return encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)

# ===== Load Data =====
hdf5_files = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
log(f"Found {len(hdf5_files)} tasks")
rng = np.random.RandomState(42)

frames = []
for fp in hdf5_files:
    with h5py.File(fp, 'r') as f:
        demos = list(f['data'].keys())
        for dk in demos[:3]:
            T = f['data'][dk]['obs']['agentview_rgb'].shape[0]
            idxs = rng.choice(T, size=min(3, T), replace=False)
            for idx in idxs:
                frames.append({
                    'agentview': f['data'][dk]['obs']['agentview_rgb'][idx],
                    'wrist': f['data'][dk]['obs']['eye_in_hand_rgb'][idx],
                    'state': np.concatenate([
                        f['data'][dk]['obs']['ee_states'][idx],
                        f['data'][dk]['obs']['gripper_states'][idx],
                    ]).astype(np.float32),
                    'action': f['data'][dk]['actions'][idx].astype(np.float32),
                })
        if len(frames) >= N_FRAMES:
            break

frames = frames[:N_FRAMES]
log(f"Loaded {len(frames)} frames")

# ===== Action Loss Computation =====
# policy.forward()는 flow-matching loss를 계산함
# 이 loss가 action prediction quality의 proxy

def compute_flow_loss(obs_list, zero_mode="none"):
    """
    Flow matching loss = MSE between predicted vector field and GT vector field
    이 값이 낮을수록 모델이 action을 잘 예측함
    """
    all_losses = []

    for b0 in range(0, len(obs_list), BATCH_SZ):
        b_end = min(b0 + BATCH_SZ, len(obs_list))
        obs_b = obs_list[b0:b_end]
        B = len(obs_b)

        # Images
        ag = torch.stack([
            torch.from_numpy(o['agentview'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])  # (B,3,H,W) in [0,1]
        wr = torch.stack([
            torch.from_numpy(o['wrist'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])

        if zero_mode == "all_vision":
            ag = torch.zeros_like(ag)
            wr = torch.zeros_like(wr)
        elif zero_mode == "wrist_only":
            wr = torch.zeros_like(wr)

        # State normalization
        state_raw = np.stack([o['state'] for o in obs_b])  # (B,8)
        if state_q01 is not None:
            state_norm = normalize_state_quantile(state_raw, state_q01, state_q99)
        else:
            state_norm = state_raw

        # Actions — pad to max_action_dim
        actions_raw = np.stack([o['action'] for o in obs_b])  # (B,7)
        pad_width = cfg.max_action_dim - 7
        actions_padded = np.pad(actions_raw, ((0,0),(0,pad_width)), constant_values=0.0)
        # Action normalization (IDENTITY for this checkpoint)
        actions_t = torch.from_numpy(actions_padded.astype(np.float32))  # (B, max_action_dim)
        # chunk_size=50, repeat single action across chunk
        actions_chunk = actions_t.unsqueeze(1).expand(B, cfg.chunk_size, cfg.max_action_dim)

        # State pad to max_state_dim
        pad_s = cfg.max_state_dim - state_norm.shape[1]
        state_padded = np.pad(state_norm, ((0,0),(0,pad_s)), constant_values=0.0)

        # Build lang tokens
        tokens, masks = build_lang_tokens(state_norm)

        # Build batch dict for policy.forward (float32 model)
        batch = {
            'observation.images.image': ag.to(DEVICE),
            'observation.images.image2': wr.to(DEVICE),
            'observation.state': torch.from_numpy(state_padded.astype(np.float32)).to(DEVICE),
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: masks.bool(),
            ACTION: actions_chunk.to(DEVICE),
        }

        with torch.no_grad():
            try:
                loss, loss_dict = policy(batch)
                all_losses.append(loss.item())
            except Exception as e:
                log(f"  forward error [{zero_mode}]: {type(e).__name__}: {str(e)[:200]}")
                continue

    return float(np.mean(all_losses)) if all_losses else float('nan')

# ===== Run Probe 2 =====
log("\nComputing real-vision loss...")
loss_real = compute_flow_loss(frames, zero_mode="none")
log(f"  loss(real-vision) = {loss_real:.6f}")

log("Computing zero-all-vision loss...")
loss_zero_all = compute_flow_loss(frames, zero_mode="all_vision")
log(f"  loss(zero-all-vision) = {loss_zero_all:.6f}")

log("Computing zero-wrist-only loss...")
loss_zero_wrist = compute_flow_loss(frames, zero_mode="wrist_only")
log(f"  loss(zero-wrist-only) = {loss_zero_wrist:.6f}")

ratio_all = loss_zero_all / loss_real if loss_real > 1e-9 else float('nan')
ratio_wrist = loss_zero_wrist / loss_real if loss_real > 1e-9 else float('nan')

probe2_pass = ratio_all >= 1.3

log(f"\nProbe 2 Summary (flow matching loss ratio):")
log(f"  loss(real)        = {loss_real:.6f}")
log(f"  loss(zero-all)    = {loss_zero_all:.6f}  ratio={ratio_all:.3f}")
log(f"  loss(zero-wrist)  = {loss_zero_wrist:.6f}  ratio={ratio_wrist:.3f}")
log(f"  PASS (ratio >= 1.3): {probe2_pass}")

# ===== Update probe_results.json =====
out_path = f"{OUT_DIR}/probe_results.json"
existing = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)

existing["probe2"] = {
    "description": "Vision contribution via flow-matching loss ablation",
    "n_frames": len(frames),
    "loss_real": loss_real,
    "loss_zero_all_vision": loss_zero_all,
    "loss_zero_wrist_only": loss_zero_wrist,
    "ratio_all_vision": ratio_all,
    "ratio_wrist_only": ratio_wrist,
    "pass_threshold_ratio": 1.3,
    "verdict": "PASS" if probe2_pass else "FAIL",
    "note": "Metric: flow-matching MSE loss (lower=better). ratio>1.3 means vision meaningful."
}

# Combine with probe 1 verdict if available
probe1_wrist_pass = existing.get("probe1", {}).get("wrist", {}).get("pass", False)
if probe1_wrist_pass and probe2_pass:
    verdict = "PASS-both"
    rec = "본 실험 진행 권장"
elif probe1_wrist_pass and not probe2_pass:
    verdict = "FAIL-probe2"
    rec = "CRDS 폐기 권장 — vision이 action에 기여 안 함"
elif not probe1_wrist_pass and probe2_pass:
    verdict = "FAIL-probe1"
    rec = "CRDS 폐기 권장 — wrist 표현 불안정"
else:
    verdict = "FAIL-both"
    rec = "CRDS 폐기 권장"

existing["overall"] = {
    "verdict": verdict,
    "recommendation": rec,
    "probe1_wrist_pass": probe1_wrist_pass,
    "probe2_pass": probe2_pass,
}

with open(out_path, 'w') as f:
    json.dump(existing, f, indent=2)
log(f"Results updated: {out_path}")

print("\n" + "="*60, flush=True)
print(f"Probe 2 결과:", flush=True)
print(f"  loss(real-vision)   = {loss_real:.6f}", flush=True)
print(f"  loss(zero-all)      = {loss_zero_all:.6f}  ratio={ratio_all:.3f}", flush=True)
print(f"  loss(zero-wrist)    = {loss_zero_wrist:.6f}  ratio={ratio_wrist:.3f}", flush=True)
print(f"  PASS (ratio>=1.3):  {probe2_pass}", flush=True)
print(f"전체 판정: {verdict}", flush=True)
print(f"권고: {rec}", flush=True)
print("="*60, flush=True)
