---
name: cp-sparse-verdict
description: Literature-check verdict for CP-Sparse (chunk-position-aware cross-attention sparsification for ACT-style VLA), May 2026.
metadata:
  type: project
---

**Verdict**: Conditional GO. Novelty score 5/10.

**Why**: No prior work executes cross-attention key-set sparsification *conditioned on action-chunk position index* — that specific axis is empty in the [[vla-efficiency-landscape-2025-2026]]. Closest neighbours occupy different axes:
- AC2-VLA: token pruning + layer skipping + cognition caching on CogACT, action-prior router, no position schedule, not cross-attention sparsification.
- VLA-Pruner / SP-VLA / LightVLA / SpecPrune-VLA: visual-token pruning, not cross-attention key-set sparsification.
- FlashVLA: action reuse on OpenVLA, depth-dependent sparsity not position-dependent.
- PD-VLA: parallel decoding, orthogonal mechanism.

**How to apply (gating condition before any architectural work)**: ACT's and SmolVLA's chunk-position cross-attention entropy must be measured empirically before committing. [[radial-action-sinks-finding]] reports intra-chunk attention *invariance* on π0.5/GR00T — if the same pattern holds on ACT/SmolVLA, the position-dependent schedule motivation collapses and CP-Sparse degenerates into static visual-token pruning (already saturated). Recommended Week-1 deliverable: per-position attention-entropy plot on a trained ACT and a trained SmolVLA, plus argmax-overlap between position 0 and position chunk_len-1.

**Pivot recommendations regardless of gate outcome**:
1. Move primary target from ACT (~80M, already ~10ms) to SmolVLA-450M, where acceleration motivation is stronger and LeRobot-ecosystem impact is larger. Keep ACT for ablations.
2. Frame against AC2-VLA explicitly: "different axis (position vs. spatial/depth/temporal), composable not competing."
3. Frame against AutoHorizon: "turns the same attention-structure observation from analysis-only into a kernel-level sparsification mechanism."

**If gate fails** (intra-chunk invariance reproduces on ACT/SmolVLA): pivot to vla-idea-generator rather than salvage — the alternative branches (two-anchor static keep-set, encoder-side image-token sparsification) collapse into existing axes.
