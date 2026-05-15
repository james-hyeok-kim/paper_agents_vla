---
name: idea-stale-vision-predict-correct
description: Pipelined VLA with intentionally stale vision + tiny visual-delta correction head at execution
metadata:
  type: project
---

# Bounded-Staleness Predict-Correct Pipelining (BSPC)

**Core hypothesis**: A VLA can run its expensive vision+LLM on frame t while the robot is still executing chunk from frame t-k, *if* a cheap correction module reconciles the stale prediction with current visual observation at execution time — yielding near-zero apparent latency.

**Technical approach**:
- Main VLA runs at frame t-k producing chunk A_{t-k}. While robot executes A_{t-k} step-by-step, the next VLA forward starts immediately on frame t-k+1 (overlapped).
- At each execution step τ, a tiny "delta-correction" head takes (a) the planned action from the stale chunk, (b) current camera frame, (c) current proprioception, and outputs a residual correction δa_τ. This head is ~50M params, runs in <5ms.
- Train the correction head end-to-end on rollouts where the main VLA is intentionally given delayed observations — student learns to compensate for staleness.
- Theoretical bound: characterize max safe staleness k as a function of scene velocity (optical flow magnitude) and task contact phase. Adaptive k.

**Closest prior**: Physical Intelligence's Real-Time Action Chunking (RTC) — overlaps inference with execution but uses *temporal smoothing/blending* of overlapping chunks, no learned correction. AsyncVLA, OpenVLA-OFT — parallel decoding but not stale-vision-aware.
**Differentiator**: Explicit *learned correction* head conditioned on visual delta + adaptive staleness budget tied to scene dynamics. RTC blends; BSPC corrects.

**Expected gains**: Effective latency from VLA forward (e.g. 80ms) drops to correction-head latency (~5ms) for steady-state execution, i.e. 10-15× apparent control rate increase, with task success within 2% of synchronous baseline.

**Safety**: HIGH RISK. Stale-vision pipelining + dynamic environments (humans, moving objects) can cause collisions. Required mitigations: (1) hard-disable pipelining when optical flow > threshold; (2) correction head outputs uncertainty → fall back to synchronous mode; (3) eval must include dynamic-scene benchmarks, not just static manipulation.

**Risk**: Correction head must generalize to staleness distributions unseen at train time. Adversarial: an unexpected object appearing during the stale window.

**Status**: GENERATED 2026-05-15, needs vla-literature-checker. CONFLICT CHECK PRIORITY: PI RTC paper, Aug 2025.
