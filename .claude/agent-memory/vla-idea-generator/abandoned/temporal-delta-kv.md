---
name: idea-temporal-delta-kv-cache
description: Caching/reuse — Temporal Delta KV cache that reuses vision encoder KV across consecutive robot frames based on optical-flow-gated similarity
metadata:
  type: project
---

# Temporal Delta KV Cache for VLA Vision Encoders

**Approach family**: Caching / temporal reuse
**Status**: Draft, awaits vla-literature-checker novelty verification
**Scores (1-5)**: Novelty 4, Impact 4, Feasibility 4, Safety 4, Timeline (months) 4

## Core hypothesis
Consecutive control-loop frames in robot manipulation share ~85-95% visual content. A patch-level KV-cache in the ViT vision encoder, gated by a cheap optical-flow / patch-difference signal, can skip ViT computation for unchanged patches while recomputing only changed regions — without retraining the VLA.

## Why VLA-specific (not generic VLM)
Generic VLMs process independent images and have no temporal coherence to exploit. VLA models run at 10-30Hz on a fixed camera; the frame-to-frame delta is bounded by robot kinematics, which gives a *physically grounded* upper bound on how many patches can change. This bound does not exist for general VLM inference.

## Safety / hardware
- Safety risk: medium — stale KV on a fast-moving end-effector patch could lag the action. Mitigated by a flow-magnitude trigger that forces full recomputation when motion exceeds threshold.
- Hardware target: Jetson Orin / AGX (edge deployment is where this matters most)

## Expected gains
- ViT latency: 40-60% reduction at steady-state phases
- End-to-end VLA latency: 15-25% reduction (ViT is one of 3-4 cost centers)
- Task success rate: target <1% absolute drop on LIBERO / RoboCasa
