---
name: temporal-delta-kv-cache-verdict
description: NO-GO verdict for "Temporal Delta KV Cache for VLA Vision Encoders" — fully pre-empted by Eventful Transformers (ICCV 2023) + VLA-Cache (NeurIPS 2025) + TTF-VLA
metadata:
  type: project
---

# Temporal Delta KV Cache for VLA Vision Encoders — NO-GO

Idea: cache patch-level ViT KV across frames in a VLA, recompute only patches whose optical-flow / patch-diff signal exceeds a threshold. Training-free plug-in, claim 40-60% ViT latency / 15-25% e2e.

## Why: NO-GO (Novelty 2/10)

The idea is covered by a stack of three published works that, together, leave no gap:

1. **Eventful Transformers (ICCV 2023, arXiv:2308.13494)** — exactly "temporal sparse update inside ViT self-attention with per-token gating." Token-difference gate, sparse Q-K and attention-value updates, 2-4× speedup on video. Pre-empts the *general* mechanism.
2. **VLA-Cache (NeurIPS 2025, arXiv:2502.02175)** — applies the same idea to VLA + robot manipulation. Patch cosine similarity on raw RGB frames, KV reuse, layer-adaptive reuse ratio, OpenVLA + CogAct, training-free, 1.7×. Pre-empts the *VLA-specific* claim.
3. **TTF-VLA (arXiv:2508.19257)** — grayscale pixel-difference + attention-based selection, hard fusion, training-free, plays nicely with VLA-Cache. Eliminates the "but my change signal is different" pivot.

Adjacent: VLN-Cache (arXiv:2603.07080) extends this further with view-aligned remapping for moving cameras; FreqCache (arXiv:2604.24391) adds frequency-guided variants for VLN.

## Why the "ViT KV vs LLM KV" pivot does not work

VLA-Cache operates at LLM cross-attention KV (after vision encoder runs fully). The agent isolated this and the user's idea targeting ViT self-attention KV as a potential differentiator. Eventful Transformers kills this differentiator — they already do temporal sparse update *inside the ViT* with per-token gating in 2023. So the remaining "VLA + ViT self-attention KV" cell is the *combination* of two existing techniques, not a new mechanism.

## How to apply

- For any future "temporal KV cache" / "patch reuse across frames" / "delta encoding ViT" VLA idea, immediately reject unless it (a) introduces a new sparsification axis (not just signal swap), (b) targets a backbone where ViT actually dominates latency, or (c) handles a setting VLN-Cache calls out as broken (moving camera, semantic shift).
- The "optical flow vs cosine vs grayscale diff" axis is a knob, not a contribution. Don't accept it as a pivot.

## Feasibility flag (separate from novelty)

On OpenVLA-7B the ViT is a small fraction of total latency (LLM dominates). The user's "ViT 40-60% → e2e 15-25%" reduction does not follow arithmetically; a perfect ViT cache yields ~5-10% e2e at most. This is *why* VLA-Cache targeted LLM KV, not ViT KV. So even if novelty existed, the impact ceiling is low on the dominant backbone.

## Pivot directions if user insists

- Chunk-position-conditioned cross-attention sparsification ([[vla-efficiency-landscape-2025-2026]]'s empty cell)
- ACT/SmolVLA where ViT *does* dominate AND VLA-Cache/TTF-VLA have not been ported (verify first)
- Otherwise route to vla-idea-generator

Related: [[vla-multiview-token-pruning-2025]] (VLA-Cache appears as competitor there too), [[vla-efficiency-landscape-2025-2026]] (Axis 2 = temporal / action reuse, already crowded).
