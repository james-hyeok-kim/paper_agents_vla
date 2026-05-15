---
name: idea-action-sensitivity-mixed-precision
description: Quantization — Action-Sensitivity-Guided Mixed-Precision quantization that allocates INT4/INT8 bitwidth based on per-layer gradient of action output, not language perplexity
metadata:
  type: project
---

# Action-Sensitivity-Guided Mixed-Precision Quantization (ASMP-Q)

**Approach family**: Quantization (mixed-precision allocation)
**Status**: Draft, awaits vla-literature-checker novelty verification
**Scores (1-5)**: Novelty 4, Impact 5, Feasibility 3, Safety 3, Timeline (months) 5

## Core hypothesis
Standard GPTQ/AWQ allocate bitwidth using language-modeling-loss sensitivity (Hessian / activation magnitude). For VLA, the loss that matters is *action MSE* or *task success*, not next-token perplexity. We propose a sensitivity score `∂(action_error)/∂(W)` computed via teacher rollouts, then solve a knapsack to assign INT4 to action-insensitive layers and INT8 to action-sensitive ones under a memory budget.

## Why VLA-specific (not generic LLM quantization)
Generic LLM quantization optimizes perplexity, which weakly correlates with VLA task success — a layer can be perplexity-critical but action-irrelevant, or vice versa (e.g., late LLM layers that feed the action head are action-critical even if perplexity-mild). The sensitivity signal `∂action/∂W` only exists in VLA where actions are a measurable output. The optimization target is fundamentally different.

## Safety / hardware
- Safety risk: medium-high — too-aggressive INT4 on action-head-adjacent layers can produce out-of-distribution motor commands. Mitigated by holding action-head and final 2 LLM layers at INT8 minimum.
- Hardware target: dual — Jetson (edge: memory-bound) and A100/H100 (cloud: throughput).

## Expected gains
- Memory: 3.5-4x compression vs. FP16 (better than uniform INT4 at equal task success)
- Latency: 1.8-2.5x speedup on memory-bound edge hardware
- Task success rate: target ≤2% drop where uniform INT4 drops 5-10%
