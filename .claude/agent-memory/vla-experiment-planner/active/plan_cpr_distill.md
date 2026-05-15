---
name: plan-cpr-distill
description: Main experiment plan for Contact-Phase Reweighted Distillation (CPR-Distill, formerly AMP-Distill) targeting CoRL 2027
metadata:
  type: project
---

# CPR-Distill Main Experiment Plan (created 2026-05-15)

**Core claim**: Contact-phase reweighting (3x rotation loss at contact transitions) gives ≥2pp success rate gain on contact-rich LIBERO-LONG vs ActDistill, with mechanism specificity on contact-poor tasks.

**PoC outcome (2026-05-15)**: L2-fixed+reweight ≈ SO(3)+reweight; contact-phase reweighting is the main contribution, SO(3) geodesic is secondary. Specificity ratio 67x on synthetic data.

**Why**: PoC validated on synthetic data; needs LIBERO + real-robot validation to be publishable.

**How to apply**:
- 4-pillar minimum: P1 main result on contact-rich, P2 specificity on contact-poor, P3 reweight×rotation-loss ablation, P4 real-robot insertion (1 task, 180 trials).
- Baselines: ActDistill (primary), VITA-VLA, Refined Policy Distillation, no-distill floor.
- Teacher: OpenVLA-7B (cached offline). Students: SmolVLA-2.2B primary, 0.5B/3B sweep.
- Compute: ~340 GPU-hrs must-have, ~500 with nice-to-haves on 4× B200 (~3.5-5.5 days wallclock).
- Timeline: 6 weeks. W1 reproduce ActDistill, W2 main run, W3 ablations, W4 baselines, W5 real-robot, W6 writeup.
- **Venue: CoRL 2027** (not NeurIPS 2026 — deadline already passed as of 2026-05-15).

## Key risks
- LIBERO has no F/T sensor → use gripper-transition + predicted contact mask in sim, F/T only on real robot.
- Specificity may fail → fallback framing as general action-loss improvement.

## Related
- [[poc-cpr-distill]] if PoC details get saved separately later.
