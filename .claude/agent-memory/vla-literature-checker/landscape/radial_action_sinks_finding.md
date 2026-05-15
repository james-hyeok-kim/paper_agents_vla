---
name: radial-action-sinks-finding
description: "VLA Knows Its Limits" finding that intra-chunk actions attend invariantly to the same VL tokens — threatens position-dependent sparsity hypotheses.
metadata:
  type: reference
---

"VLA Knows Its Limits" (arXiv:2602.21445, 2026) analyzes cross- and self-attention patterns in flow-based VLAs (π0.5 with p=10/50, GR00T N1.5 with p=16) and reports two findings load-bearing for downstream acceleration work:

1. **Intra-chunk invariance**: "actions within the same chunk consistently attend to the same vision–language tokens" — predicted actions rely on static perceptual context, not position-varying conditioning.
2. **Radial action sinks**: initial and terminal action tokens receive disproportionately high attention, acting as stable anchors; intermediate positions organize radially around them, with a low-attention plateau in between.

The paper turns these into AutoHorizon — a test-time horizon estimator using a bidirectional soft-pointer on attention mass — but does **not** propose any sparsification mechanism, position-dependent keep-set algorithm, or kernel-level acceleration.

**Why this matters for novelty triage**: any new idea whose motivation is "later chunk positions need fewer conditioning tokens than earlier ones" runs head-on into (1). If the same invariance holds on the target backbone, then a single shared keep-set per chunk suffices — which collapses to ordinary visual-token pruning (already saturated by [[vla-efficiency-landscape-2025-2026]] Axis 1). Such ideas should be gated by an empirical attention-entropy pre-check on the actual target backbone (ACT, SmolVLA) before the architectural work begins.

**Backbones not yet checked**: ACT (Tonyzhao 80M aloha policy), SmolVLA-450M, OpenVLA action head. The π0.5/GR00T finding may or may not transfer — flow-based decoders behave differently from ACT's CVAE decoder and SmolVLA's flow-matching action expert.
