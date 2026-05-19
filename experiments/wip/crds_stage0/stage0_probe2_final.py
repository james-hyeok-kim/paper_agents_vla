"""
CRDS Stage-0 Probe 2 Final: Vision vs Proprio Contribution (inference-time MSE)

Action prediction MSE vs GT:
- real-vision baseline
- zero-all-vision (agentview + wrist 모두 0)
- zero-wrist-only (wrist만 0)

ratio = MSE(ablation) / MSE(real)
PASS: ratio_all >= 1.3

Notes:
- flow-matching inference (predict_action_chunk, 10 denoising steps)
- policy in float32 to avoid bfloat16/float32 dtype mismatch in noise sampling
- tokenizer: google/paligemma-3b-pt-224, state discretized to 256 bins
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
from safetensors.torch import load_file as safe_load_file

t_start = time.time()
def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)

DEVICE = "cuda:0"
CKPT = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_DIR = "/home/jovyan/workspace/paper_agents_vla/experiments/wip/crds_stage0"
IMG_RES = 224
N_FRAMES = 128
BATCH_SZ = 2  # smaller batch for memory with float32
TASK_STR = None  # per-frame task string derived from HDF5 filename

log("=== CRDS Probe 2 Final: Vision Contribution via Inference MSE ===")

# ===== Load Model =====
log("Loading π0.5 policy (float32)...")
from lerobot.configs.types import FeatureType, PolicyFeature
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
cfg_clean = {k: v for k, v in cfg_dict.items() if k not in SKIP_KEYS and not k.startswith("_")}
cfg_clean["compile_model"] = False
cfg_clean["gradient_checkpointing"] = False
TYPE_MAP = {"VISUAL": FeatureType.VISUAL, "STATE": FeatureType.STATE, "ACTION": FeatureType.ACTION}
if "input_features" in cfg_clean:
    cfg_clean["input_features"] = {k: PolicyFeature(type=TYPE_MAP[v["type"]], shape=tuple(v["shape"])) for k, v in cfg_clean["input_features"].items()}
if "output_features" in cfg_clean:
    cfg_clean["output_features"] = {k: PolicyFeature(type=TYPE_MAP[v["type"]], shape=tuple(v["shape"])) for k, v in cfg_clean["output_features"].items()}
cfg = PI05Config(**cfg_clean)
policy = PI05Policy.from_pretrained(CKPT, config=cfg)
policy.eval().to(DEVICE)  # float32
log(f"Policy loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params (float32)")

# ===== Tokenizer & Normalizer =====
log("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
MAX_LEN = cfg.tokenizer_max_length

log("Loading normalizer stats...")
norm_weights = safe_load_file(f"{CKPT}/policy_preprocessor_step_2_normalizer_processor.safetensors")
state_q01 = norm_weights.get("observation.state.q01").numpy().astype(np.float32)  # (8,)
state_q99 = norm_weights.get("observation.state.q99").numpy().astype(np.float32)

def normalize_state(state_np):
    return np.clip(2.0 * (state_np - state_q01) / (state_q99 - state_q01 + 1e-8) - 1.0, -1.0, 1.0)

def build_prompt(state_norm_1d, task_str):
    disc = np.digitize(state_norm_1d, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = " ".join(map(str, disc.tolist()))
    return f"Task: {task_str}, State: {state_str};\nAction: "

def tokenize_batch(state_norms, task_strs):
    prompts = [build_prompt(state_norms[i], task_strs[i]) for i in range(len(state_norms))]
    enc = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE).bool()

# ===== Load Data =====
hdf5_files = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
log(f"Found {len(hdf5_files)} tasks")
rng = np.random.RandomState(42)

frames = []
per_file = max(2, N_FRAMES // len(hdf5_files))
for fp in hdf5_files:
    # derive task string from filename: "pick_up_X_demo.hdf5" → "pick up X"
    basename = os.path.basename(fp)
    task_str = basename.replace("_demo.hdf5", "").replace(".hdf5", "").replace("_", " ")
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
                    ]).astype(np.float32),
                    'action': f['data'][dk]['actions'][idx].astype(np.float32),
                    'task_str': task_str,  # per-frame task string
                })
    if len(frames) >= N_FRAMES:
        break
frames = frames[:N_FRAMES]
log(f"Loaded {len(frames)} frames")

# ===== Inference MSE Computation =====
def compute_inference_mse(obs_list, zero_mode="none"):
    """
    obs_list: list of frame dicts
    zero_mode: "none" | "all_vision" | "wrist_only"
    Returns: mean MSE between predicted action chunk[0] and GT action
    """
    all_mse = []
    policy.eval()

    for b0 in range(0, len(obs_list), BATCH_SZ):
        b_end = min(b0 + BATCH_SZ, len(obs_list))
        obs_b = obs_list[b0:b_end]
        B = len(obs_b)

        # Images in [0,1] float32 (B,3,H,W)
        ag = torch.stack([
            torch.from_numpy(o['agentview'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])
        wr = torch.stack([
            torch.from_numpy(o['wrist'].astype(np.float32) / 255.0).permute(2,0,1)
            for o in obs_b
        ])

        if zero_mode == "all_vision":
            ag = torch.zeros_like(ag)
            wr = torch.zeros_like(wr)
        elif zero_mode == "wrist_only":
            wr = torch.zeros_like(wr)

        # State
        state_raw = np.stack([o['state'] for o in obs_b])  # (B,8)
        state_norm = normalize_state(state_raw)             # (B,8) in [-1,1]
        state_padded = np.pad(state_norm, ((0,0),(0,cfg.max_state_dim - 8)), constant_values=0.0)

        # Tokenize (per-frame task strings)
        task_strs = [o['task_str'] for o in obs_b]
        tokens, masks = tokenize_batch(state_norm, task_strs)

        # GT action (for MSE)
        gt_action = np.stack([o['action'] for o in obs_b])  # (B,7)
        gt_t = torch.from_numpy(gt_action)

        batch = {
            'observation.images.image': ag.to(DEVICE),
            'observation.images.image2': wr.to(DEVICE),
            'observation.state': torch.from_numpy(state_padded.astype(np.float32)).to(DEVICE),
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: masks,
        }

        with torch.no_grad():
            try:
                pred = policy.predict_action_chunk(batch)  # (B, chunk, action_dim)
            except Exception as e:
                log(f"  predict error [{zero_mode}]: {type(e).__name__}: {str(e)[:150]}")
                continue

        # Compare first predicted action to GT
        pred_first = pred[:, 0, :7].cpu().float()  # (B, 7)
        mse = F.mse_loss(pred_first, gt_t).item()
        all_mse.append(mse)

    return float(np.mean(all_mse)) if all_mse else float('nan')

# ===== Run Probe 2 =====
log("\nComputing MSE(real-vision)...")
mse_real = compute_inference_mse(frames, zero_mode="none")
log(f"  MSE(real-vision) = {mse_real:.6f}")

log("Computing MSE(zero-all-vision)...")
mse_zero_all = compute_inference_mse(frames, zero_mode="all_vision")
log(f"  MSE(zero-all-vision) = {mse_zero_all:.6f}")

log("Computing MSE(zero-wrist-only)...")
mse_zero_wrist = compute_inference_mse(frames, zero_mode="wrist_only")
log(f"  MSE(zero-wrist-only) = {mse_zero_wrist:.6f}")

ratio_all = mse_zero_all / mse_real if mse_real > 1e-9 else float('nan')
ratio_wrist = mse_zero_wrist / mse_real if mse_real > 1e-9 else float('nan')
probe2_pass = ratio_all >= 1.3

log(f"\nProbe 2 Summary (inference-time action MSE):")
log(f"  MSE(real)        = {mse_real:.6f}")
log(f"  MSE(zero-all)    = {mse_zero_all:.6f}  ratio={ratio_all:.3f}")
log(f"  MSE(zero-wrist)  = {mse_zero_wrist:.6f}  ratio={ratio_wrist:.3f}")
log(f"  PASS (ratio_all >= 1.3): {probe2_pass}")

# ===== Update probe_results.json =====
out_path = f"{OUT_DIR}/probe_results.json"
existing = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        existing = json.load(f)

existing["probe2"] = {
    "description": "Vision contribution: inference-time action MSE ablation",
    "n_frames": len(frames),
    "metric": "MSE between predicted first-action and GT action",
    "mse_real": mse_real,
    "mse_zero_all_vision": mse_zero_all,
    "mse_zero_wrist_only": mse_zero_wrist,
    "ratio_all_vision": ratio_all,
    "ratio_wrist_only": ratio_wrist,
    "pass_threshold_ratio": 1.3,
    "verdict": "PASS" if probe2_pass else "FAIL",
}

probe1_wrist_pass = existing.get("probe1", {}).get("wrist", {}).get("pass", False)
if probe1_wrist_pass and probe2_pass:
    verdict = "PASS-both"
    rec = "본 실험 진행 권장"
elif probe1_wrist_pass and not probe2_pass:
    verdict = "FAIL-probe2"
    rec = "CRDS 폐기 권장 — vision이 action에 기여 안 함 (proprio-dominated)"
elif not probe1_wrist_pass and probe2_pass:
    verdict = "FAIL-probe1"
    rec = "CRDS 폐기 권장 — wrist 표현이 layer 20 전 불안정"
else:
    verdict = "FAIL-both"
    rec = "CRDS 폐기 권장 — probe 1 & 2 모두 FAIL"

existing["overall"] = {
    "verdict": verdict,
    "recommendation": rec,
    "probe1_wrist_pass": probe1_wrist_pass,
    "probe2_pass": probe2_pass,
}
existing["timestamp_kst"] = time.strftime("%Y-%m-%d %H:%M KST", time.localtime(time.time() + 9*3600))

with open(out_path, 'w') as f:
    json.dump(existing, f, indent=2)
log(f"Results saved: {out_path}")

print("\n" + "="*60, flush=True)
print("CRDS Stage-0 최종 결과", flush=True)
print("="*60, flush=True)
print(f"Probe 1 (SigLIP 안정성):", flush=True)
print(f"  Exterior: cos@L16={existing.get('probe1',{}).get('exterior',{}).get('cos_at_layer16', 'N/A'):.4f}", flush=True)
print(f"  Wrist:    cos@L16={existing.get('probe1',{}).get('wrist',{}).get('cos_at_layer16', 'N/A'):.4f}  PASS={probe1_wrist_pass}", flush=True)
print(f"Probe 2 (Vision 기여도):", flush=True)
print(f"  MSE(real)={mse_real:.4f}  MSE(zero-all)={mse_zero_all:.4f}  ratio={ratio_all:.3f}", flush=True)
print(f"  MSE(zero-wrist)={mse_zero_wrist:.4f}  ratio={ratio_wrist:.3f}  PASS={probe2_pass}", flush=True)
print(f"최종 판정: {verdict}", flush=True)
print(f"권고: {rec}", flush=True)
print("="*60, flush=True)
