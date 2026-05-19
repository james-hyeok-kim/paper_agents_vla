"""
CRDS Stage-0 Premise Probe

Probe 1: SigLIP Layer Activation Stability (wrist camera)
  - SigLIP forward에 hook을 걸어 각 layer i의 patch-token 출력이 layer 27(마지막)과 얼마나 유사한지 측정
  - cos(h_i, h_27) 를 mean-pool over 256 patch tokens, 4배치 × 256샘플 프레임으로 측정
  - PASS: i <= 20 구간에서 cos >= 0.95 유지
  - 두 카메라(exterior/wrist) 각각 측정 — 카메라 매핑 확인

Probe 2: Vision vs Proprio Contribution (action MSE ablation)
  - LIBERO libero_spatial HDF5에서 256 프레임 샘플링
  - π0.5 ChunkedBC 포워드로 GT action과의 MSE 측정:
      (a) real-vision (baseline)
      (b) zero-all-vision (모든 이미지 -1로 대체)
      (c) zero-wrist-only (image2만 -1로 대체, image1 유지)
  - ratio_all = MSE(zero-all) / MSE(real)
  - ratio_wrist = MSE(zero-wrist) / MSE(real)
  - PASS: ratio_all >= 1.3 ("vision이 meaningful")
  - 부가: ratio_wrist >= 1.3이면 wrist depth reduction이 risky

저장: experiments/wip/crds_stage0/probe_results.json
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
from PIL import Image
import torchvision.transforms.functional as TF

t_start = time.time()

def log(msg):
    print(f"[{time.time()-t_start:6.1f}s] {msg}", flush=True)

# ===== Config =====
DEVICE = "cuda:0"
CKPT = "/data/jameskimh/james_lebero_pretrained/pi05_libero_finetuned"
DATA_DIR = "/data/jameskimh/james_libero_datasets/libero_spatial"
OUT_DIR = "/home/jovyan/workspace/paper_agents_vla/experiments/wip/crds_stage0"
os.makedirs(OUT_DIR, exist_ok=True)

N_FRAMES = 256   # 샘플링할 프레임 수 (Probe 2)
BATCH_SZ = 8     # action inference batch size
IMG_RES = 224    # policy image resolution

log("=== CRDS Stage-0 Premise Probe ===")

# ===== Load Model =====
log("Loading π0.5 policy...")
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

with open(f"{CKPT}/config.json") as f:
    cfg_dict = json.load(f)

# Clean config dict for PI05Config
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
cfg.compile_model = False
cfg.gradient_checkpointing = False

policy = PI05Policy.from_pretrained(CKPT, config=cfg)
policy.eval().to(DEVICE, dtype=torch.bfloat16)
log(f"Policy loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")

# 카메라 매핑 확인
image_keys = list(cfg.image_features.keys()) if hasattr(cfg, 'image_features') else []
log(f"image_features keys (order = position in SigLIP embedding): {image_keys}")

# ===== SigLIP Vision Tower Reference =====
vision_tower = policy.model.paligemma_with_expert.paligemma.model.vision_tower
siglip_encoder = vision_tower.vision_model.encoder
n_siglip_layers = len(siglip_encoder.layers)
log(f"SigLIP encoder layers: {n_siglip_layers}")

# ===== Probe 1: SigLIP Layer Activation Stability =====
log("\n--- Probe 1: SigLIP Layer Activation Stability ---")

# SigLIP 마지막 레이어에는 LayerNorm 후 처리가 있으므로
# encoder.layers[i] output을 hook으로 캡처
layer_acts = {}  # {layer_idx: tensor (B, N_patches, D)}
hooks = []

def make_siglip_hook(idx):
    def _hook(mod, inp, out):
        # out: (B, N, D)
        if isinstance(out, tuple):
            x = out[0]
        else:
            x = out
        layer_acts[idx] = x.detach().float().cpu()
    return _hook

for i, layer in enumerate(siglip_encoder.layers):
    hooks.append(layer.register_forward_hook(make_siglip_hook(i)))

# 실제 LIBERO agentview / eye_in_hand 이미지 로드
hdf5_files = sorted(glob.glob(f"{DATA_DIR}/*.hdf5"))
log(f"Found {len(hdf5_files)} HDF5 tasks")

# 각 task에서 소수의 프레임을 균등하게 샘플링
frames_exterior = []  # agentview
frames_wrist = []     # eye_in_hand
rng = np.random.RandomState(42)

frames_per_task = max(1, 20 // len(hdf5_files))  # per task
for fp in hdf5_files:
    with h5py.File(fp, 'r') as f:
        demos = list(f['data'].keys())
        for dk in demos[:3]:  # 태스크당 최대 3 demo
            T = f['data'][dk]['obs']['agentview_rgb'].shape[0]
            idxs = rng.choice(T, size=min(frames_per_task, T), replace=False)
            for idx in idxs:
                ag = f['data'][dk]['obs']['agentview_rgb'][idx]       # (128,128,3) uint8
                wrist = f['data'][dk]['obs']['eye_in_hand_rgb'][idx]  # (128,128,3) uint8
                frames_exterior.append(ag)
                frames_wrist.append(wrist)

frames_exterior = frames_exterior[:64]  # Probe 1은 64 프레임으로 충분
frames_wrist = frames_wrist[:64]
log(f"Loaded {len(frames_exterior)} frames for Probe 1")

def preprocess_images_for_siglip(frames_np):
    """
    uint8 (H,W,3) → float32 (N,3,224,224) normalized to [-1,1]
    """
    tensors = []
    for img in frames_np:
        img_t = torch.from_numpy(img.astype(np.float32)) / 255.0  # (H,W,3) [0,1]
        img_t = img_t.permute(2, 0, 1)  # (3,H,W)
        img_t = F.interpolate(img_t.unsqueeze(0), size=(IMG_RES, IMG_RES), mode='bilinear', align_corners=False).squeeze(0)
        img_t = img_t * 2.0 - 1.0  # [-1,1]
        tensors.append(img_t)
    return torch.stack(tensors)  # (N,3,224,224)

# SigLIP의 embed_image를 직접 호출 (PaliGemma get_image_features 경유)
def run_siglip_forward(img_tensor):
    """img_tensor: (B,3,224,224) float32 [-1,1], run SigLIP, return nothing (hooks capture)"""
    img_bf16 = img_tensor.to(DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        # embed_image: (B, num_patches, D)
        _ = policy.model.paligemma_with_expert.embed_image(img_bf16)

# 각 카메라를 별도로 forward
B_PROBE = 8

def compute_stability_curve(frames_np, camera_name):
    """각 layer의 cos sim to last layer 계산"""
    layer_acts.clear()
    imgs = preprocess_images_for_siglip(frames_np)

    all_layer_acts = {i: [] for i in range(n_siglip_layers)}

    for b_start in range(0, len(imgs), B_PROBE):
        b_end = min(b_start + B_PROBE, len(imgs))
        batch = imgs[b_start:b_end]
        run_siglip_forward(batch)
        for i in range(n_siglip_layers):
            if i in layer_acts:
                all_layer_acts[i].append(layer_acts[i].numpy())  # (B, N, D)
        layer_acts.clear()

    # 각 layer: (total_frames, N_patches, D)
    stacked = {}
    for i in range(n_siglip_layers):
        if all_layer_acts[i]:
            stacked[i] = np.concatenate(all_layer_acts[i], axis=0)  # (F, N, D)

    if n_siglip_layers - 1 not in stacked:
        log(f"  WARNING: last layer not captured for {camera_name}")
        return []

    last_layer = stacked[n_siglip_layers - 1]  # (F, N, D)
    # mean pool over patches: (F, D)
    last_mean = last_layer.mean(axis=1)
    last_mean_norm = last_mean / (np.linalg.norm(last_mean, axis=-1, keepdims=True) + 1e-9)

    results = []
    for i in range(n_siglip_layers):
        if i not in stacked:
            continue
        cur_layer = stacked[i]  # (F, N, D)
        # mean pool over patches
        cur_mean = cur_layer.mean(axis=1)  # (F, D)
        cur_norm = cur_mean / (np.linalg.norm(cur_mean, axis=-1, keepdims=True) + 1e-9)
        # cos sim per frame, then average
        cos_per_frame = (cur_norm * last_mean_norm).sum(axis=-1)  # (F,)
        cos_mean = float(cos_per_frame.mean())
        cos_min = float(cos_per_frame.min())

        # Also compute per-token (all 256 patches) cosine sim
        last_tok = last_layer.reshape(-1, last_layer.shape[-1])  # (F*N, D)
        cur_tok = cur_layer.reshape(-1, cur_layer.shape[-1])      # (F*N, D)
        last_tok_n = last_tok / (np.linalg.norm(last_tok, axis=-1, keepdims=True) + 1e-9)
        cur_tok_n = cur_tok / (np.linalg.norm(cur_tok, axis=-1, keepdims=True) + 1e-9)
        cos_per_token = (cur_tok_n * last_tok_n).sum(axis=-1).mean()

        results.append({
            "layer": i,
            "cos_mean_pool": cos_mean,
            "cos_mean_pool_min": cos_min,
            "cos_per_token": float(cos_per_token),
        })
        print(f"  [{camera_name}] layer {i:2d}: cos_mean_pool={cos_mean:.4f}, cos_per_token={cos_per_token:.4f}", flush=True)

    return results

log(f"\nRunning Probe 1 on EXTERIOR camera (agentview)...")
probe1_exterior = compute_stability_curve(frames_exterior, "exterior")

log(f"\nRunning Probe 1 on WRIST camera (eye_in_hand)...")
probe1_wrist = compute_stability_curve(frames_wrist, "wrist")

# Hook 제거
for h in hooks:
    h.remove()

# PASS 판정: layer 16 (0-indexed: 16번째 = 17번째 layer)에서 cos >= 0.95
# 목표: i <= 20 구간 전체에서 cos >= 0.95 유지
PASS_THRESHOLD = 0.95
PASS_LAYER = 20  # 1-indexed layer 20 = 0-indexed 19

def check_probe1_pass(results, camera_name):
    """layer <= PASS_LAYER (0-indexed: 0~19)에서 모두 cos >= PASS_THRESHOLD인지"""
    early_layers = [r for r in results if r["layer"] <= PASS_LAYER - 1]  # 0-indexed
    if not early_layers:
        return False, None
    min_cos = min(r["cos_mean_pool"] for r in early_layers)
    all_pass = all(r["cos_mean_pool"] >= PASS_THRESHOLD for r in early_layers)
    # 실제로 cos >= 0.95가 처음 달성되는 layer 찾기
    cross_layer = None
    for r in results:
        if r["cos_mean_pool"] >= PASS_THRESHOLD:
            cross_layer = r["layer"]
            break
    return all_pass, min_cos, cross_layer

ext_pass, ext_min_cos, ext_cross = check_probe1_pass(probe1_exterior, "exterior")
wrist_pass, wrist_min_cos, wrist_cross = check_probe1_pass(probe1_wrist, "wrist")

# 층 16에서의 cos sim (CRDS 가설의 핵심: layer 16 exit)
layer16_ext = next((r["cos_mean_pool"] for r in probe1_exterior if r["layer"] == 15), None)  # 0-indexed 15 = 16th layer
layer16_wrist = next((r["cos_mean_pool"] for r in probe1_wrist if r["layer"] == 15), None)

log(f"\nProbe 1 Summary:")
log(f"  Exterior (agentview): min_cos(i<=20)={ext_min_cos:.4f}, cross@layer={ext_cross}, cos@layer16={layer16_ext}")
log(f"  Wrist (eye_in_hand):  min_cos(i<=20)={wrist_min_cos:.4f}, cross@layer={wrist_cross}, cos@layer16={layer16_wrist}")
log(f"  PASS criterion: all cos >= 0.95 for layer <= 20")
log(f"  Exterior PASS: {ext_pass} | Wrist PASS: {wrist_pass}")

# ===== Probe 2: Vision vs Proprio Contribution =====
log("\n--- Probe 2: Vision vs Proprio Contribution ---")

# N_FRAMES 프레임 샘플링 (LIBERO libero_spatial)
obs_frames = []
for fp in hdf5_files:
    with h5py.File(fp, 'r') as f:
        demos = list(f['data'].keys())
        for dk in demos:
            T = f['data'][dk]['obs']['agentview_rgb'].shape[0]
            n_sample = max(1, N_FRAMES // (len(hdf5_files) * len(demos)))
            t_idxs = rng.choice(T, size=min(n_sample, T), replace=False)
            for t in t_idxs:
                ag = f['data'][dk]['obs']['agentview_rgb'][t]       # (128,128,3) uint8
                wrist = f['data'][dk]['obs']['eye_in_hand_rgb'][t]  # (128,128,3) uint8
                state = np.concatenate([
                    f['data'][dk]['obs']['ee_states'][t],       # (6,)
                    f['data'][dk]['obs']['gripper_states'][t],  # (2,)
                ], axis=0)  # (8,)
                action = f['data'][dk]['actions'][t]  # (7,)
                obs_frames.append({
                    'agentview': ag,
                    'wrist': wrist,
                    'state': state,
                    'action': action,
                })

obs_frames = obs_frames[:N_FRAMES]
log(f"Loaded {len(obs_frames)} frames for Probe 2")

# 이미지 전처리 함수 (pi05 policy가 기대하는 형식)
def prepare_batch(obs_list, zero_wrist=False, zero_all_vision=False):
    """
    obs_list: list of dicts with agentview (128,128,3), wrist (128,128,3), state (8,), action (7,)
    Returns: policy-compatible batch dict
    """
    B = len(obs_list)
    # agentview: observation.images.image
    ag_imgs = np.stack([o['agentview'] for o in obs_list])  # (B,128,128,3) uint8
    wrist_imgs = np.stack([o['wrist'] for o in obs_list])    # (B,128,128,3) uint8
    states = np.stack([o['state'] for o in obs_list])         # (B,8)
    actions = np.stack([o['action'] for o in obs_list])       # (B,7)

    # → float32 [0,1] (B,3,128,128)
    ag_t = torch.from_numpy(ag_imgs.astype(np.float32) / 255.0).permute(0,3,1,2)
    wr_t = torch.from_numpy(wrist_imgs.astype(np.float32) / 255.0).permute(0,3,1,2)

    if zero_all_vision:
        # -1로 대체 (SigLIP 빈 이미지 sentinel)
        # policy _preprocess_images가 [0,1] → [-1,1] 변환하므로 여기서는 -1 대신 0으로 넣어도 됨
        # 하지만 완전 zero ablation을 위해 직접 -1로 만들고 싶으면 0.5를 넣으면 (0.5*2-1=0)
        # 가장 확실한 방법: 0으로 설정 (preprocessor에서 *2-1 → -1로 변환됨)
        ag_t = torch.zeros_like(ag_t)
        wr_t = torch.zeros_like(wr_t)
    elif zero_wrist:
        wr_t = torch.zeros_like(wr_t)

    state_t = torch.from_numpy(states.astype(np.float32))  # (B,8)
    action_t = torch.from_numpy(actions.astype(np.float32))  # (B,7)

    # Policy expects observation.images.image (exterior) and observation.images.image2 (wrist)
    # Based on config: 'observation.images.image', 'observation.images.image2', 'observation.images.empty_camera_0'
    batch = {
        'observation.images.image': ag_t.to(DEVICE),
        'observation.images.image2': wr_t.to(DEVICE),
        'observation.state': state_t.to(DEVICE),
        'task': ['pick up the black bowl and place it on the plate'] * B,
    }
    return batch, action_t.to(DEVICE)

def compute_action_mse_batched(obs_list, zero_wrist=False, zero_all_vision=False):
    """
    Compute MSE between predicted action chunk[0] and GT action.
    (action chunk이 50인데 GT는 1-step이라 chunk[0]만 비교)
    """
    all_mse = []
    policy.eval()

    for b_start in range(0, len(obs_list), BATCH_SZ):
        b_end = min(b_start + BATCH_SZ, len(obs_list))
        obs_batch = obs_list[b_start:b_end]
        batch, gt_action = prepare_batch(obs_batch, zero_wrist=zero_wrist, zero_all_vision=zero_all_vision)

        with torch.no_grad():
            try:
                pred = policy.select_action(batch)  # (B, action_dim) — first step of chunk
            except Exception as e:
                log(f"  select_action error: {e}")
                continue

        # pred: (B, action_dim) or (B, chunk, action_dim) depending on API
        if pred.dim() == 3:
            pred = pred[:, 0, :]  # first step
        # Truncate to 7-dim GT
        pred_7 = pred[:, :7].float()
        gt_7 = gt_action[:, :7].float()
        mse = F.mse_loss(pred_7, gt_7, reduction='mean').item()
        all_mse.append(mse)

    return float(np.mean(all_mse)) if all_mse else float('nan')

log("\nRunning real-vision baseline MSE...")
mse_real = compute_action_mse_batched(obs_frames, zero_wrist=False, zero_all_vision=False)
log(f"  MSE(real-vision) = {mse_real:.6f}")

log("Running zero-all-vision MSE...")
mse_zero_all = compute_action_mse_batched(obs_frames, zero_all_vision=True)
log(f"  MSE(zero-all-vision) = {mse_zero_all:.6f}")

log("Running zero-wrist-only MSE...")
mse_zero_wrist = compute_action_mse_batched(obs_frames, zero_wrist=True, zero_all_vision=False)
log(f"  MSE(zero-wrist-only) = {mse_zero_wrist:.6f}")

ratio_all = mse_zero_all / mse_real if mse_real > 1e-9 else float('nan')
ratio_wrist = mse_zero_wrist / mse_real if mse_real > 1e-9 else float('nan')

log(f"\nProbe 2 Summary:")
log(f"  MSE(real)       = {mse_real:.6f}")
log(f"  MSE(zero-all)   = {mse_zero_all:.6f}  ratio={ratio_all:.3f}")
log(f"  MSE(zero-wrist) = {mse_zero_wrist:.6f}  ratio={ratio_wrist:.3f}")
log(f"  PASS criterion: ratio_all >= 1.3")
probe2_pass = ratio_all >= 1.3
log(f"  Probe 2 PASS: {probe2_pass}")

# ===== 종합 판정 =====
log("\n=== CRDS Stage-0 Final Verdict ===")
probe1_wrist_pass_final = wrist_pass  # 핵심: wrist camera 안정성
probe1_ext_pass_final = ext_pass

both_pass = probe1_wrist_pass_final and probe2_pass

if both_pass:
    verdict = "PASS-both"
    recommendation = "본 실험 진행 권장 — wrist camera 표현 조기 수렴 + vision 기여도 확인"
elif probe1_wrist_pass_final and not probe2_pass:
    verdict = "FAIL-probe2"
    recommendation = "CRDS 폐기 권장 — wrist 안정성은 OK지만 vision이 action에 기여 안 함 (proprio-dominated)"
elif not probe1_wrist_pass_final and probe2_pass:
    verdict = "FAIL-probe1"
    recommendation = "CRDS 폐기 권장 — wrist 표현이 layer 20 이전에 불안정, 조기 exit 시 표현 열화 위험"
else:
    verdict = "FAIL-both"
    recommendation = "CRDS 폐기 권장 — probe 1 & 2 모두 실패"

log(f"  Verdict: {verdict}")
log(f"  Recommendation: {recommendation}")

# ===== 결과 저장 =====
results = {
    "experiment": "CRDS Stage-0 Premise Probe",
    "timestamp_kst": time.strftime("%Y-%m-%d %H:%M KST", time.localtime()),
    "probe1": {
        "description": "SigLIP layer activation stability (cos sim to last layer)",
        "n_frames": len(frames_exterior),
        "n_siglip_layers": n_siglip_layers,
        "pass_threshold_cos": PASS_THRESHOLD,
        "pass_layer_max_idx": PASS_LAYER,
        "exterior_camera": {
            "pass": ext_pass,
            "min_cos_i_leq_20": ext_min_cos,
            "first_stable_layer": ext_cross,
            "cos_at_layer16": layer16_ext,
            "per_layer": probe1_exterior,
        },
        "wrist_camera": {
            "pass": wrist_pass,
            "min_cos_i_leq_20": wrist_min_cos,
            "first_stable_layer": wrist_cross,
            "cos_at_layer16": layer16_wrist,
            "per_layer": probe1_wrist,
        },
        "verdict": "PASS" if wrist_pass else "FAIL",
        "note": "CRDS wrist depth reduction은 wrist camera의 PASS 여부에 의존",
    },
    "probe2": {
        "description": "Vision vs Proprio contribution to action MSE",
        "n_frames": len(obs_frames),
        "mse_real_vision": mse_real,
        "mse_zero_all_vision": mse_zero_all,
        "mse_zero_wrist_only": mse_zero_wrist,
        "ratio_all_vision": ratio_all,
        "ratio_wrist_only": ratio_wrist,
        "pass_threshold_ratio": 1.3,
        "verdict": "PASS" if probe2_pass else "FAIL",
        "note": "ratio_all >= 1.3이면 vision이 action에 meaningful하게 기여",
    },
    "overall": {
        "verdict": verdict,
        "recommendation": recommendation,
        "probe1_wrist_pass": probe1_wrist_pass_final,
        "probe2_pass": probe2_pass,
    },
    "camera_mapping": {
        "note": "config.image_features 순서대로 SigLIP에 embed됨",
        "image_feature_keys": image_keys,
        "assumed_mapping": {
            "observation.images.image": "agentview_rgb (exterior)",
            "observation.images.image2": "eye_in_hand_rgb (wrist)",
            "observation.images.empty_camera_0": "empty (padded -1)",
        },
    },
}

out_path = f"{OUT_DIR}/probe_results.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nResults saved to: {out_path}")

# 콘솔 최종 요약
print("\n" + "="*60)
print("CRDS Stage-0 Probe 결과 요약")
print("="*60)
print(f"Probe 1 (SigLIP 안정성):")
print(f"  Wrist camera:     layer16 cos={layer16_wrist:.4f}  → {'PASS' if wrist_pass else 'FAIL'}")
print(f"  Exterior camera:  layer16 cos={layer16_ext:.4f}  → {'PASS' if ext_pass else 'FAIL'}")
print(f"Probe 2 (Vision 기여도):")
print(f"  MSE(real)={mse_real:.6f}  MSE(zero-all)={mse_zero_all:.6f}")
print(f"  ratio_all={ratio_all:.3f}  ratio_wrist={ratio_wrist:.3f}  → {'PASS' if probe2_pass else 'FAIL'}")
print(f"\nFinal Verdict: {verdict}")
print(f"Recommendation: {recommendation}")
print("="*60)
