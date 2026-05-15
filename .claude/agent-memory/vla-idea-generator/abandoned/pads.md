---
name: idea-position-adaptive-denoising
description: Per-chunk-position denoising step budget for π0 — later actions need fewer steps
metadata:
  type: project
---

# Position-Adaptive Denoising Schedule (PADS) for Action Chunks

**Core hypothesis**: Within a π0-style action chunk of H steps, later action positions can be denoised with far fewer steps than the first action, because they are strongly conditioned on earlier (already-denoised) actions in the same chunk → the conditional posterior is much sharper.

**Technical approach**:
- Profile per-position posterior entropy of the flow-matching/diffusion head: H(a_t | obs, a_0..a_{t-1}) for t=0..H-1. Hypothesis: monotonically decreasing.
- Allocate per-position step budget: a_0 gets full schedule (e.g. 10 steps), a_{H-1} gets 1-2 steps, with a learned monotone schedule.
- Train a tiny "schedule predictor" that takes obs embedding + chunk position and outputs step count, optimized for action-MSE under a total-FLOPs budget.
- Cross-position denoising parallelism: once a_0 is at step k, a_1 can start using a_0's partial denoise as conditioning — pipeline the schedule across positions.

**Closest prior**: Consistency Policy, ManiCM, OneStepDP (collapse *all* positions to few steps uniformly); Streaming Diffusion Policy (streams over time, not within-chunk position).
**Differentiator**: Exploits *intra-chunk conditional structure* rather than treating each action as i.i.d. Step budget is *position-dependent* and *input-dependent*, not a fixed global schedule.

**Expected gains**: 2-3× action-head latency reduction on π0 with <2% success-rate drop (vs uniform consistency distillation which loses 4-6%). Action head is 30-40% of π0 inference, so end-to-end ~1.4-1.7×.

**Safety**: Late-chunk actions becoming too coarse → jerky trajectories. Mitigation: residual smoothing constraint between adjacent positions in the schedule predictor's loss.

**Risk**: Entropy hypothesis may not hold for highly multi-modal tasks (bimanual coordination where later actions branch).

**Status**: GENERATED 2026-05-15, needs vla-literature-checker
