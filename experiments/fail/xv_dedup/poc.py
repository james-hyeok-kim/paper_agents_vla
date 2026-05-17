"""
XV-Dedup PoC: cross-view token overlap measurement on real ViT.

Gates:
A. Cross-view overlap ≥30% (top-1 cosine similarity matching at threshold)
B. LSH bucketing precision vs ground-truth cosine matching
C. Prefill latency reduction from token count reduction

Uses DINOv2-base from transformers. Synthetic stereo pairs from single image augmentations.
"""
import torch
import torch.nn.functional as F
import numpy as np
import time
import json
from PIL import Image
from io import BytesIO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

def main():
    t0 = time.time()
    results = {}
    print("=" * 60)
    print("XV-Dedup PoC: cross-view token overlap")
    print(f"Device: {device}")
    print("=" * 60)

    # ===== Load DINOv2 =====
    print("\n[1/4] Loading DINOv2-base...")
    from transformers import AutoImageProcessor, AutoModel
    model_name = "facebook/dinov2-base"
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        print(f"Failed to load DINOv2: {e}")
        # Fallback: use a small custom ViT
        return

    # Get patch grid info
    patch_size = model.config.patch_size  # typically 14 for dinov2
    img_size = 224
    n_patches = (img_size // patch_size) ** 2
    print(f"  DINOv2-base: patch_size={patch_size}, n_patches={n_patches} per view")

    # ===== Generate synthetic multi-view stereo pairs =====
    print("\n[2/4] Generating synthetic stereo views...")
    # Create N synthetic "scenes" by random RGB images, then make two views per scene
    # View 1: original. View 2: small viewpoint change (shift + small color jitter).
    N_SCENES = 8
    overlap_results = []
    lsh_results = []

    # Random "scene" tensors (simulating diverse camera scenes)
    scene_tensors = torch.rand(N_SCENES, 3, img_size, img_size, device=device)
    # Make them more "image-like" by smoothing
    scene_tensors = F.avg_pool2d(scene_tensors, 7, 1, 3)
    scene_tensors = (scene_tensors - scene_tensors.amin()) / (scene_tensors.amax() - scene_tensors.amin() + 1e-8)

    # Normalize for DINOv2
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def encode(view_tensor):
        # view_tensor: (B, 3, H, W) in [0,1]
        normed = (view_tensor - MEAN) / STD
        with torch.no_grad():
            out = model(pixel_values=normed, output_hidden_states=True)
            # Patch tokens (drop CLS)
            tok = out.last_hidden_state[:, 1:, :]  # (B, N_patch, D)
        return tok

    # View 2 = same scene but shifted by 16 px (small viewpoint translation)
    SHIFT = 16
    view1 = scene_tensors
    # Pad and shift right
    pad = F.pad(scene_tensors, (SHIFT, 0, 0, 0), mode='replicate')
    view2 = pad[:, :, :, :img_size]
    # Add small color jitter
    view2 = (view2 + 0.05 * torch.randn_like(view2)).clamp(0, 1)

    tok1 = encode(view1)  # (N, P, D)
    tok2 = encode(view2)
    P = tok1.shape[1]
    D = tok1.shape[-1]
    print(f"  Token shape per view: {tok1.shape}")

    # ===== Gate A: cross-view overlap =====
    print("\n[3/4] Measuring cross-view overlap...")
    # For each token in view1, find nearest token in view2 by cosine similarity
    tok1_n = F.normalize(tok1, dim=-1)
    tok2_n = F.normalize(tok2, dim=-1)
    sim = torch.matmul(tok1_n, tok2_n.transpose(-2, -1))  # (N, P, P)
    max_sim, _ = sim.max(dim=-1)  # (N, P)

    THRESHOLDS = [0.7, 0.8, 0.85, 0.9]
    overlap_by_thresh = {}
    for th in THRESHOLDS:
        overlap_pct = (max_sim > th).float().mean().item() * 100
        overlap_by_thresh[f"th_{th}"] = overlap_pct
        print(f"  Overlap @ cos>{th}: {overlap_pct:.2f}%")
    avg_max_sim = max_sim.mean().item()
    print(f"  Avg max-sim (any-to-best): {avg_max_sim:.4f}")
    results['gate_A_overlap'] = overlap_by_thresh
    results['gate_A_avg_max_sim'] = avg_max_sim

    # Self-similarity baseline (sanity)
    sim_self = torch.matmul(tok1_n, tok1_n.transpose(-2, -1))  # (N, P, P)
    sim_self_offdiag = sim_self - 2 * torch.eye(P, device=device).unsqueeze(0)  # exclude self
    intra_overlap = (sim_self_offdiag.max(-1).values > 0.85).float().mean().item() * 100
    print(f"  [Sanity] within-view top-1 (cos>0.85): {intra_overlap:.2f}%")
    results['intra_view_overlap_pct_at_0.85'] = intra_overlap

    # ===== Gate B: LSH bucketing precision =====
    print("\n[4/4] LSH bucketing precision...")
    for hash_dim in [32, 64, 128]:
        # Random projection (no InfoNCE — just simple LSH as a lower bound)
        proj = torch.randn(D, hash_dim, device=device) / np.sqrt(hash_dim)
        h1 = (tok1_n @ proj) > 0  # (N, P, H) binary
        h2 = (tok2_n @ proj) > 0

        # Hash code matching
        # Compute hamming distance between every pair
        h1f = h1.float()
        h2f = h2.float()
        # Hamming similarity = bits agreement
        hamming_sim = (h1f.unsqueeze(2) == h2f.unsqueeze(1)).float().mean(-1)  # (N, P, P)
        max_hamming, hamming_argmax = hamming_sim.max(-1)  # (N, P)

        # Ground truth = top-1 by cosine similarity
        cos_argmax = sim.argmax(-1)  # (N, P)
        # Agreement: hash top-1 == cosine top-1
        agreement = (hamming_argmax == cos_argmax).float().mean().item() * 100

        # Precision at high-similarity bucket: when hash says match, what fraction is true (cos>0.85)?
        high_sim_mask = max_sim > 0.85
        if high_sim_mask.any():
            precision_at_high_sim = (hamming_argmax == cos_argmax)[high_sim_mask].float().mean().item() * 100
        else:
            precision_at_high_sim = float('nan')
        print(f"  H={hash_dim}: top-1 agreement={agreement:.2f}%, "
              f"precision_at_cos>0.85={precision_at_high_sim:.2f}%")
        results[f'gate_B_lsh_h{hash_dim}_agreement_pct'] = agreement
        results[f'gate_B_lsh_h{hash_dim}_precision_at_high_sim_pct'] = precision_at_high_sim

    # ===== Gate C: Prefill latency reduction =====
    print("\n[3.5/4] Token reduction prefill latency benchmark...")
    # Simulate prefill on a small transformer with N=full vs N=reduced tokens.
    from torch.nn import TransformerEncoderLayer, TransformerEncoder
    HIDDEN = 768
    N_LAYERS = 6
    encoder_layer = TransformerEncoderLayer(d_model=HIDDEN, nhead=12, dim_feedforward=2048,
                                            batch_first=True).to(device)
    encoder = TransformerEncoder(encoder_layer, num_layers=N_LAYERS).to(device).eval()

    n_views = 3
    full_tokens = n_views * P  # e.g., 3 views * 256 = 768
    # Assume 40% reduction → roughly mid of 30-50% claim
    reduction = 0.4
    reduced_tokens = int(full_tokens * (1 - reduction))

    print(f"  Full tokens (3-view): {full_tokens}, reduced (40% off): {reduced_tokens}")

    def benchmark_prefill(n_tok, n_warm=5, n_meas=20):
        x = torch.randn(1, n_tok, HIDDEN, device=device)
        # Warm-up
        for _ in range(n_warm):
            with torch.no_grad():
                _ = encoder(x)
        torch.cuda.synchronize()
        # Measure
        t = time.perf_counter()
        for _ in range(n_meas):
            with torch.no_grad():
                _ = encoder(x)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n_meas * 1000  # ms

    lat_full = benchmark_prefill(full_tokens)
    lat_reduced = benchmark_prefill(reduced_tokens)
    reduction_pct = (1 - lat_reduced / lat_full) * 100
    print(f"  Prefill latency: full={lat_full:.3f}ms, reduced={lat_reduced:.3f}ms")
    print(f"  Latency reduction: {reduction_pct:.2f}%")
    results['gate_C_prefill_latency_full_ms'] = lat_full
    results['gate_C_prefill_latency_reduced_ms'] = lat_reduced
    results['gate_C_latency_reduction_pct'] = reduction_pct

    elapsed = time.time() - t0
    print(f"\n=== Elapsed: {elapsed:.1f}s ===")

    # ===== Final gate evaluation =====
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)
    # A: overlap >= 30% at cos>0.85
    gate_a = overlap_by_thresh.get('th_0.85', 0) >= 30
    # B: best LSH should achieve >= 50% agreement with cosine (loose for random projection)
    gate_b = max(results.get(f'gate_B_lsh_h{h}_agreement_pct', 0) for h in [32,64,128]) >= 50
    # C: >= 25% latency reduction at 40% token reduction
    gate_c = reduction_pct >= 25

    print(f"\nGate A (cross-view overlap ≥30% at cos>0.85): "
          f"{overlap_by_thresh['th_0.85']:.2f}% → {'PASS' if gate_a else 'FAIL'}")
    print(f"Gate B (LSH agreement ≥50% with random projection): {'PASS' if gate_b else 'FAIL'}")
    print(f"Gate C (≥25% prefill latency reduction): {reduction_pct:.2f}% → "
          f"{'PASS' if gate_c else 'FAIL'}")

    overall = gate_a and gate_b and gate_c
    verdict = "FEASIBILITY: PASS" if overall else "FEASIBILITY: PARTIAL/FAIL"
    if gate_a and not gate_b:
        verdict += " — overlap exists but LSH needs InfoNCE training"
    elif not gate_a:
        verdict += " — overlap insufficient with naive viewpoint shift"
    print(f"\n>>> {verdict} <<<")

    out = {
        "elapsed_sec": elapsed,
        "device": str(device),
        "config": {"hash_dims": [32,64,128], "n_scenes": N_SCENES, "shift_px": SHIFT,
                   "n_patches_per_view": P, "img_size": img_size, "patch_size": patch_size},
        "results": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in results.items()},
        "verdict": verdict,
        "pass_gate_a": bool(gate_a),
        "pass_gate_b": bool(gate_b),
        "pass_gate_c": bool(gate_c),
        "pass_all": bool(overall),
    }
    with open("/home/jovyan/workspace/paper_agents_vla/experiments/xv_dedup/results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to results.json")

if __name__ == "__main__":
    main()
