---
name: vla-quant-landscape
description: Map of 2025-2026 VLA quantization papers - QuantVLA, QVLA, DyQ-VLA, EaqVLA, SQAP-VLA, SQIL, QAIL
metadata:
  type: reference
---

VLA / robot-policy quantization papers found as of 2026-05.

1. **QuantVLA** (CVPR 2026, AIoT-MLSys-Lab) - PTQ for VLA + DiT action head. Uniform W4A8, attention kept FP. Scale calibration, attention temp matching, output head balancing. Project: quantvla.github.io
2. **QVLA / AutoQVLA** (ICLR 2026, arxiv 2602.03782) - Channel-wise action-sensitivity bit allocation, first-order Taylor, unifies pruning (0-bit) + quant {0,2,4,8,16}. Greedy demotion. OpenVLA-OFT, LIBERO. 29.2% VRAM, 98.9% perf.
3. **DyQ-VLA** (arxiv 2603.07904) - Temporal-dynamic activation bit-switching {2,4,8,16} per control step. Uses kinematic proxies (Motion Fineness, Angular Jerk), not gradient. 30.9% mem, 99.5% perf.
4. **EaqVLA** (arxiv 2505.21567) - Encoding-aligned mixed precision. ViT INT4, LLaMA INT8, projector FP16, action head minor. Module-level, not layer/channel.
5. **SQAP-VLA** (arxiv 2509.09090) - Synergistic quantization-aware pruning for VLA.
6. **Spec-VLA** (EMNLP 2025) - Speculative decoding for VLA (orthogonal).
7. **SQIL** (Saliency-Aware Quantized IL, arxiv 2505.15304) - State-importance saliency + distillation; UNIFORM bit-width (not mixed precision).
8. **QAIL** (arxiv 2412.01034) - QAT for imitation learning; uniform bit-width.
9. **ActionFlow** (arxiv 2512.20276) - Pipelined action acceleration for VLM on edge.

Non-VLA but related mixed-precision sensitivity baselines:
- HAWQ (Hessian-based), APTQ (attention-aware), Mix-QViT, TACQ (task-circuit), HAQ (RL-driven hardware-aware).

**Why:** When a new VLA quantization idea arrives, this is the starting checklist.
**How to apply:** For any "action-aware quantization" claim, check QVLA + DyQ-VLA + EaqVLA first. For "layer-wise mixed precision," check HAWQ/APTQ/Mix-QViT to ensure non-trivial extension beyond standard sensitivity scoring.

Related: [[qvla-conflict]]
