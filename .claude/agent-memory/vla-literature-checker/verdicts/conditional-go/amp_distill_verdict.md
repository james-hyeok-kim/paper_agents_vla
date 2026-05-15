---
name: amp-distill-verdict
description: AMP-Distill (SE(3) manifold-preserving VLA distillation) novelty verdict — CONDITIONAL GO 5/10; components are prior art individually but VLA-distillation composition is unclaimed
metadata:
  type: project
---

# AMP-Distill Novelty Verdict

**Date**: 2026-05-15
**Verdict**: CONDITIONAL GO — Novelty score 5/10

## TL;DR
Each ingredient of AMP-Distill exists in prior work (SE(3) geodesic loss for pose
regression, manifold-aware policy learning, uncertainty-weighted KD,
contact-phase aware imitation). The unique composition is "SE(3)-manifold-correct
loss applied to VLA *distillation*". No paper makes that specific claim, but the
gap is narrow.

**Why**: VLA distillation field is racing in 2025-2026 (ActDistill, VITA-VLA,
Refined Policy Distillation, VLA-OPD) and ALL still use Euclidean MSE/MAE for
the action regression term. This is the empirical opportunity.

**How to apply**: Recommend GO only if the proposer can show in a pilot that
manifold-correct loss yields >2pp success-rate gain on contact-rich tasks vs
Euclidean MSE under matched student capacity — and explicitly positions
against ActDistill (Nov 2025) as the strongest baseline.

## Confirmed Prior Art Map

### VLA Distillation (all Euclidean) — 🟡 Partial overlap
- **ActDistill** (arXiv 2511.18082, Nov 2025) — Self-derived distillation
  with `L_act = ‖H_stu - a‖² + ‖H_stu - H_tea‖² + ‖H_stu - sg(H_{l-1})‖²`.
  Treats 3D translation + 3D rotation + gripper as one Euclidean vector.
  **Strongest baseline to beat**. No SO(3) handling, no contact awareness.
- **VITA-VLA** (arXiv 2510.09607) — MSE during representation alignment;
  MAE on 6-DoF arm actions during fine-tuning. Justifies MAE as "more stable
  optimization" — implicit acknowledgement that L2 misbehaves but no manifold
  fix.
- **VLA-OPD** (arXiv 2603.26666) — Reverse-KL on token distribution + on-policy.
  Token-level, doesn't touch action geometry.
- **Refined Policy Distillation** (arXiv 2503.05833) — RL-based; not loss design.
- **TinyVLA / MiniVLA** — MSE/L1 on flattened action vector. Standard.

### SE(3) / Geodesic Loss (non-distillation) — 🟡 Partial overlap
- **Zhou et al. "Continuity of Rotation Representations"** (CVPR 2019) —
  Foundational: 6D rep + geodesic loss > quaternion+MSE for SO(3) regression.
  AMP-Distill's *premise* (Euclidean MSE on rotations is suboptimal) is
  established here, **not novel**.
- **RFMP / Riemannian Flow Matching Policy** (arXiv 2403.10672, 2024) —
  Flow matching directly on SE(3) manifold for visuomotor BC. Closest in spirit
  to AMP-Distill's "flow vector → tangent space projection" idea, but from
  scratch, NOT distillation. AMP-Distill's flow-matching variant is at most a
  "RFMP-style head distilled from π0" which is incremental.
- **EquiBot / EquAct / RiEMann / EquiContact / ReSeFlow** — SE(3)-equivariant
  *architectures*; orthogonal to objective design. These don't conflict but
  reduce headline novelty of "SO(3)-aware robot policy".

### Contact-Aware Loss Weighting — 🟢 Complementary
- **Feel the Force** (arXiv 2506.01944) — Tactile-conditioned BC.
- **Reactive Diffusion Policy** (arXiv 2503.02881) — Slow-fast policy with
  tactile/force feedback.
- Survey of contact-rich IL (arXiv 2506.13498) — Confirms phase-specific force
  requirements as known issue.
- **No paper uses contact phase as a loss-reweighting signal in distillation**.
  This is the cleanest standalone novelty hook of AMP-Distill.

### Uncertainty-Weighted KD — ⬜ Generic prior art
- Nature Sci Rep 2024 (logit uncertainty for KD), Uncertainty-Aware KD for
  collision identification (MDPI 2021), Variance-Weighting (arXiv 2601.18909).
  Mature in classification KD; trivially adaptable to action regression. Adds
  little novelty by itself.

### RVT/PerAct (Discrete-Action Baseline) — ⬜ No conflict
- RVT uses CE on Euler-angle bins. Different paradigm (classification not
  regression). Not a conflict but a reminder: the field has *already*
  recognized "Euclidean MSE on rotations is bad" and worked around it via
  discretization. AMP-Distill must justify why a continuous geodesic loss
  beats RVT's classification approach AND beats ActDistill's Euclidean MSE.

## Direct Conflicts: NONE 🟢
No paper claims: "SE(3)-geodesic + gripper-CE + contact-phase weighting for
VLA student distillation".

## Risks That Drag Novelty Down

1. **Composition novelty only**. Reviewers (CoRL/RSS) routinely reject papers
   that combine known ingredients without a non-obvious insight. AMP-Distill
   needs a "why does manifold-correct loss matter MORE in distillation than in
   from-scratch BC" theoretical or empirical hook.

2. **ActDistill already gets ~95%+ on LIBERO with plain MSE**. If the baseline
   is saturated, the claimed 3-5pp gain may not materialize on standard
   benchmarks. Real-world contact-rich tasks (peg-in-hole, plug insertion)
   are where it could show, but these are harder to standardize.

3. **Quaternion double-cover** is already handled by 6D rep + geodesic loss
   (Zhou CVPR'19). Claiming this as VLA-specific contribution is weak.

4. **Flow-matching tangent projection** is just RFMP applied to the distillation
   teacher's flow vector. Likely 1 paragraph of novelty, not a paper.

## Recommendation: CONDITIONAL GO

**Required pivots/sharpenings**:
1. Lead with **contact-phase-conditioned reweighting** as the headline (cleanest
   gap), not generic SO(3) geodesic loss.
2. Pre-experiment: ablate ActDistill's loss term-by-term on a contact-rich task
   subset (LIBERO-LONG insertion, peg-in-hole) and confirm Euclidean MSE
   actually loses to geodesic by >2pp.
3. Pivot the framing from "manifold-preserving distillation" → "contact-aware
   manifold-correct distillation for VLA compression". The two-axis combination
   (geometry + contact phase) is genuinely unclaimed.
4. Position against ActDistill (Nov 2025), VITA-VLA (Oct 2025), Refined Policy
   Distillation (Mar 2025) as direct baselines.

**Send to vla-idea-validator** for feasibility, especially:
- How is "contact phase" labeled by teacher? Force-torque signal vs predicted
  contact mask vs gripper-state transitions?
- π0 flow matching distillation: does projecting teacher flow to SE(3) tangent
  space change the ODE trajectory enough to require retraining the flow
  schedule?

## Relation to Existing Memory
- Complements [[vla_efficiency_landscape_2025_2026]] under the "depth/distill"
  axis (not "token" or "decoding").
- Distinct from [[cp_sparse_verdict]], [[temporal_delta_kv_cache_verdict]] —
  AMP-Distill targets training-time loss, not inference-time sparsity.
