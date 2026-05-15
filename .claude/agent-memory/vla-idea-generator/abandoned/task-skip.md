---
name: idea-jetson-throughput-aware-layer-skip
description: Hardware-aware layer skipping on Jetson AGX Orin that exploits INT4 vs INT8 throughput cliff and gates skips on action-quality budget
metadata:
  type: project
---

# Jetson-Throughput-Aware Action-Quality-Bounded Layer Skipping (TASK-Skip)

## Core Hypothesis
On Jetson AGX Orin, the >2x throughput gap between INT4 (via TensorRT-LLM W4A16) and INT8/FP16 layers can be exploited by promoting "non-critical" VLA LLM layers to INT4 *and* skipping a subset entirely, with a runtime gate keyed on action-prediction-error budget rather than language perplexity.

## Technical Approach
1. Profile per-layer TensorRT kernel throughput on AGX Orin for INT4/INT8/FP16 to identify the throughput cliff layers (typically MLP-heavy ones).
2. Train a tiny (~50K param) gating MLP that takes (proprio_state, prev_action_chunk, layer_idx) and outputs skip/keep decision; trained with a Lagrangian objective: minimize mean layers executed s.t. action-MSE on a held-out manipulation set <= epsilon.
3. At inference, the gate's skip mask is fused into a static CUDA graph variant per skip-pattern (precompiled for top-K most-common masks) so dispatch overhead doesn't eat the gain.

## Why VLA Specifically (unique justification)
Generic LLM layer-skipping (LayerSkip, CALM) gates on token-level confidence — but VLA's output is a continuous action in SE(3), not a discrete token. A "wrong layer skip" in an LLM produces a slightly worse word; in a VLA it can produce a collision. The skip budget must therefore be calibrated against **action-space deviation under physical constraints**, which has no LLM analog. Additionally, the Jetson edge constraint (32GB shared memory, 275 TOPS INT8 / ~550 TOPS INT4) makes the throughput-cliff observation load-bearing — datacenter A100/H100 deployments hide this cliff.

## Expected Performance (concrete)
- OpenVLA-7B on AGX Orin baseline: ~240ms/inference (FP16)
- W4A16 baseline: ~110ms
- Target with TASK-Skip: 65-80ms (1.4-1.7x further speedup over W4A16)
- Memory: -15% (skipped layers' activations not materialized)
- Task success rate degradation budget: <=2% absolute on LIBERO-Long

## Implementation Difficulty: 중상
- TensorRT-LLM custom kernel work + CUDA Graph specialization is non-trivial
- Gating MLP training is light; data collection moderate

## Venue
CoRL 2026 (primary — robotics audience cares about Jetson) or MLSys 2027 (hardware-software co-design angle)

## Potential Conflicting Papers
- **LayerSkip (Meta, 2024)** — generic LLM early exit; differs because token-confidence-gated, not action-error-gated, and not hardware-cliff-aware
- **SkipDecode (2023)** — token-level skip; same difference
- **AWQ + TensorRT-LLM Jetson benchmarks (NVIDIA blog, 2024)** — measures W4A16 speedup but no skip layer
- **MoE-style routing in VLA (RoboMoE, late 2025?)** — needs to verify; conceptually related but routes between experts not skips depth

## Risk Factors
- Skip masks may not amortize: if the top-K precompiled CUDA graphs don't cover >80% of runtime patterns, dispatch overhead negates gains. Mitigation: cluster proprio states first, then assign each cluster a fixed mask.
- Safety: layer skipping near contact-rich phases is risky. Mitigation: force full-precision pass when contact force sensor crosses threshold.

## Recommended Next Step
run vla-literature-checker on "Jetson-Throughput-Aware Action-Quality-Bounded Layer Skipping"
