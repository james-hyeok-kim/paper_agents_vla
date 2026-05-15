---
name: idea-cross-view-token-dedup
description: Multi-camera VLA token deduplication via cross-view feature hashing before LLM ingestion to cut visual token count 30-50%
metadata:
  type: project
---

# Cross-View Visual Token Deduplication for Multi-Camera VLA (XV-Dedup)

## Core Hypothesis
Across the 2-4 camera views typical of LeRobot setups (wrist + 1-2 statics), 30-50% of ViT patch tokens are *visually redundant* (same object surface from different angles) and can be merged into single shared tokens before LLM ingestion without measurable action degradation.

## Technical Approach
1. After each view's SigLIP/DINOv2 ViT, project tokens into a low-dim (64) hash space using a learned linear projector trained with InfoNCE on multi-view correspondences from the training set.
2. Apply LSH bucketing across views at inference; tokens landing in the same bucket are merged (weighted average of full-dim features) and tagged with a multi-hot "origin-view" embedding so the LLM still knows which camera saw what.
3. The resulting *de-duplicated* token sequence (e.g., 256 instead of 4x196=784) is fed to the LLM backbone.

## Why VLA Specifically (unique justification)
Single-image VLM token reduction (ToMe, LLaVA-PruMerge) operates within one view — there is no cross-view redundancy to exploit. The multi-camera setup is **definitional** to manipulation VLA: wrist camera and overhead camera share most of the scene geometry, so the redundancy ratio is fundamentally higher than in single-image VLMs. Additionally, the merging budget can be **task-conditioned** — a peg-insertion task can aggressively merge non-wrist tokens, while a search task cannot. This task-conditioning has no analog in single-view VLM pruning.

## Expected Performance (concrete)
- π0 with 3 cameras, baseline: ~3 x 196 = 588 visual tokens fed to LLM
- After XV-Dedup at 40% dedup ratio: ~350 tokens
- LLM prefill latency: -35% (since prefill scales superlinearly in seq length for full attention); per-step inference: -18-22%
- Action success rate target: within 1% absolute of baseline on the multi-view subset of LIBERO and SimplerEnv

## Implementation Difficulty: 중
- LSH + hash projector training is straightforward
- Engineering the variable-length token sequence through a fixed-shape inference pipeline needs careful padding/masking design

## Venue
CoRL 2026 (multi-view manipulation is core CoRL turf) or ICRA 2027

## Potential Conflicting Papers
- **ToMe (Bolya et al., ICLR 2023)** — token merging *within* one view; orthogonal
- **LLaVA-PruMerge (2024)** — single-image pruning for VLMs; same distinction
- **RVT / RVT-2 (Goyal et al., CoRL 2023/24)** — uses multi-view but for 3D reasoning, not efficiency
- **3D Diffuser Actor (Ke et al., 2024)** — fuses multi-view into 3D scene tokens, not a dedup story
- **RoboFlamingo multi-view variant** — needs checking; uses separate cross-attn per view, no dedup
- **Concern**: any 2025 paper on multi-view token fusion for efficiency. Highest collision risk among the three ideas.

## Risk Factors
- LSH stability under viewpoint shift — if bucket assignments flicker frame-to-frame the merged token features will jitter and degrade temporal smoothness. Mitigation: temporal smoothing on bucket assignments (sticky LSH).
- Per-view embedding tag must be small enough not to bloat the merged token, but informative enough that the LLM can disambiguate. Ablation needed.
- Safety: occluded objects seen only by wrist camera must NEVER be dedup'd away. Mitigation: minimum quorum (token must appear in >=2 views before considered for merging — singletons preserved).

## Recommended Next Step
run vla-literature-checker on "Cross-View Visual Token Deduplication for Multi-Camera VLA"
