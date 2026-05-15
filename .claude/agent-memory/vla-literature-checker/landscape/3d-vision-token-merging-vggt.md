---
name: 3d-vision-token-merging-vggt
description: Multi-view token merging methods from 3D reconstruction (VGGT family) — adjacent but non-VLA; useful as prior-art context but not direct conflicts for VLA cross-view dedup
metadata:
  type: reference
---

# 3D Vision Multi-View Token Merging (VGGT family)

- **HTTM (arXiv:2511.21317)** — Head-wise Temporal Token Merging for VGGT global attention.
- **Co-Me (arXiv:2511.14751)** — Confidence-Guided Token Merging for visual geometric transformers.
- **LiteVGGT (arXiv:2512.04939)** — Geometry-aware cached token merging, classifies tokens into GA/dst/src buckets. Achieves up to 11.3x speedup on VGGT, 7.2x on MapAnything.

## Why these matter
These do operate cross-view, but for **3D reconstruction transformers** (multi-view geometry), not VLA robot policies. They merge tokens based on geometric similarity or confidence — not for downstream action generation, and not on the LLM-input visual tokens of a VLM-backbone VLA.

## How to apply
Cite as adjacent prior art when proposing VLA cross-view dedup — establishes that "cross-view token redundancy is real" in multi-view transformers generally. Use to justify the existence of redundancy without conflicting on the VLA-specific instantiation (LLM input compression for action policy under real-time constraint).

Related: [[vla-multiview-token-pruning-2025]]
