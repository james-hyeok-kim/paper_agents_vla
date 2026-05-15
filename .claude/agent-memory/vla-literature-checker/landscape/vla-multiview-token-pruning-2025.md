---
name: vla-multiview-token-pruning-2025
description: Competitive landscape of multi-view/multi-camera token pruning and compression methods for VLA models (2025-2026) — relevant for any cross-view dedup or pruning idea
metadata:
  type: project
---

# VLA Multi-View Token Pruning Landscape (2025-2026)

Tracked while validating XV-Dedup (Cross-View LSH dedup) novelty.

## Closest competitors

- **BFA++ (arXiv:2602.20566)** — Hierarchical two-level pruning. **Inter-view predictor only does view-level binary selection** (entire wrist/head camera dropped per manipulation phase). Reports 1.8x / 1.5x speedup on pi0 / RDT. **No token-level cross-view merging, no LSH/hashing.** This is the most-cited "cross-view redundancy" work and the closest direct neighbor — XV-Dedup must position against it as finer-grained (token-level vs view-level).

- **VLA-Pruner (arXiv:2511.16449)** — Temporal-aware dual-level pruning, uses prefill attention + EMA on action-decode attention + greedy max-min redundancy filtering. Operates in unified token space, no explicit cross-view mechanism.

- **Compressor-VLA (arXiv:2511.18950)** — Instruction-guided STC (semantic) + SRC (spatial) compressors. 59% FLOPs reduction. No multi-camera cross-view focus.

- **TEAM-VLA (arXiv:2512.09927)** — Token Expand-and-Merge, training-free, language-similarity-based bipartite merging within unified token sequence. Reduces OpenVLA-OFT 109ms -> 72.1ms. Batches views in convolution but no cross-view dedup semantics.

- **VLA-Cache (arXiv:2502.02175)** — Temporal (frame-to-frame) caching of minimally-changed tokens. Orthogonal axis (time, not view).

- **VLA-IAP (arXiv:2603.22991)** — Interaction Alignment, training-free, "Interaction-First" pruning. Not cross-view focused.

- **EfficientVLA (OpenReview SELYlDHZk2)** — Training-free acceleration. Did not confirm cross-view dedup mechanism (PDF render failed).

- **Action-aware Dynamic Pruning (arXiv:2509.22093)** — Action-driven importance pruning.

- **Differentiable Token Pruning for VLA (arXiv:2509.12594)** — Learned pruning gates, not cross-view.

## Key gap remaining for XV-Dedup

After surveying 8+ direct competitors: **no published VLA work performs token-level cross-camera deduplication using similarity hashing (LSH) with merge + origin-view embedding.** BFA++ is closest but operates at view-granularity (binary). All others are within-view or temporal.

## Why: 
The why for the gap: most VLA pruning papers were written when 1-2 cameras were standard; the field is just now hitting 3-4 camera setups (pi0 with wrist+overhead+side) where cross-view redundancy becomes the dominant axis.

## How to apply:
If validating any future "cross-view token dedup" or "multi-camera pruning" idea, the differentiator must be (a) token-granularity not view-granularity, (b) explicit similarity matching across cameras (LSH/projection/clustering), and (c) preservation of view-origin info. BFA++ is the must-cite differentiation target.

Related: [[3d-vision-token-merging-vggt]]
