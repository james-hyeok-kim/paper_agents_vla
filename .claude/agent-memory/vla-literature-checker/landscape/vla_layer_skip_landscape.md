---
name: vla-layer-skip-landscape
description: Map of 2024-2026 VLA layer-skipping / early-exit papers - DeeR-VLA, MoLe-VLA, DySL-VLA, ActDistill, DeeAD
metadata:
  type: reference
---

VLA / robot-policy adaptive layer skipping & early exit papers as of 2026-05.

1. **DeeR-VLA** (NeurIPS 2024, arxiv 2411.02359) - Dynamic Early-Exit MLLM for robot. Gate criterion: L2 norm of action prediction *consistency* between adjacent intermediate features against threshold eta_i. NOT softmax confidence - action-quality based. Explicitly contrasts with confidence-based criteria. Calibrates threshold to compute/memory budgets. 5.2-6.5x LLM compute reduction on CALVIN. Multi-exit architecture, not per-layer router. Hardware: standard GPU, no Jetson focus.
2. **MoLe-VLA** (arxiv 2503.20384, Apr 2025) - Mixture-of-Layers via Spatial-Temporal Aware Router (STAR). Selectively activates layers based on robot state. Uses Cognition Self-Knowledge Distillation (CogKD). 5.6x compute reduction, +8% success. Gating signal: spatial-temporal state (proprio + visual). NOT hardware-aware quantization. No Jetson eval.
3. **DySL-VLA** (arxiv 2602.22896, OpenReview ICLR 2026) - Dynamic-Static Layer Skipping. Static layers always run; dynamic layers conditionally skipped via feedforward controllers. Gating signal: trajectory continuity C_t = -1/k * sum||dA_j||_2 (kinematic action differences). Two-stage knowledge distillation. **Tested on Jetson Orin: 23.2 Hz for OpenVLA-OFT.** 27.4ms A6000 vs 53ms baseline. 3.75x on RoboFlamingo. NO Lagrangian formulation. NO INT4/INT8 throughput modeling. NO CUDA Graph.
4. **ActDistill** (arxiv 2511.18082, Nov 2025) - Action-guided self-derived distillation. Student has dynamic router selecting computation paths by action-prediction demand. Graph-structured teacher encapsulates action hierarchy. 50%+ compute reduction, 1.67x speedup. Aligns with "action-quality-aware" framing.
5. **DeeAD** (arxiv 2511.20720, Nov 2025) - Dynamic early exit for *autonomous driving* VLA (out of robotic manipulation scope, but adjacent).
6. **VLA-Cache** (arxiv 2502.02175) - Adaptive *token* caching with layer-adaptive reuse ratio. 1.7x CUDA latency, +15% control freq. Orthogonal axis (token-level not layer-level).
7. **EfficientVLA** (OpenReview) - Training-free acceleration via compression.
8. **NanoVLA** (arxiv 2510.25122) - Routing decoupled VLA, edge focus.
9. **ActionFlow** (arxiv 2512.20276) - Pipelined system-level acceleration. 2.55x OpenVLA-7B on Jetson AGX Orin. Orthogonal to layer skipping.
10. **SmolVLA** (HuggingFace) - Static skip: compute halfway through VLM and use intermediate features for control.

**Critical conflict notes for "action-quality-bounded layer skipping":**

- DeeR-VLA already establishes the EXACT framing: "action-prediction consistency, not softmax confidence, gates the exit" with budget-calibrated thresholds. The "VLA-specific gate = action error, not perplexity" claim is preempted.
- DySL-VLA already establishes layer-skipping for VLA on Jetson Orin, evaluated for OpenVLA-OFT, with kinematic-based gating. The "Jetson-deployed VLA layer skip" claim is preempted at the platform level.
- ActDistill establishes "dynamic router selects layers by action demand" with action-prediction-conditioned routing. The "action-quality gates layer selection" claim is preempted in its compression form.
- MoLe-VLA establishes the router architecture with proprio/visual state input. The "small router conditioned on robot state" claim is preempted.

**Residual differentiation space (narrow):**

1. **Hardware-aware Lagrangian objective** - None of the above formulate min{expected_layers} s.t. action-MSE <= eps with explicit per-layer INT4/INT8/FP16 throughput costs from Jetson profiling. DySL-VLA reports Jetson numbers but does not put hardware throughput in the objective.
2. **Throughput cliff exploitation** - The claim that INT4 vs INT8 has 2x+ throughput gap on Orin is a real Jetson property, but no VLA layer-skip paper formulates the skip-vs-demote trade-off as a unified decision. Skip = 0-bit demotion; INT4 = 4-bit demotion; FP16 = full. This unification is novel framing.
3. **CUDA Graph pre-compilation for dynamic routing** - Dynamic layer skipping breaks CUDA Graph capture (variable graph topology). DySL-VLA does not address this. A pre-compiled set of N skip-pattern graphs selected per step would be a system contribution.
4. **Prev-action-chunk as gate input** - Most routers use visual+proprio. Conditioning on the previously-emitted action chunk (not just state) is uncommon but DeeR-VLA's consistency check already approximates this idea.

**Why:** The layer-skip-for-VLA space is now crowded; any new paper must position carefully against DeeR-VLA + DySL-VLA + MoLe-VLA + ActDistill simultaneously.
**How to apply:** For "action-aware layer skipping VLA" ideas, recommend NO-GO unless the contribution is sharply hardware/systems (Jetson kernel-level, throughput-cliff modeling, CUDA Graph routing, joint skip+quant decision). Pure ML-side framing is preempted.

Related: [[qvla-conflict]] [[vla-quant-landscape]]
