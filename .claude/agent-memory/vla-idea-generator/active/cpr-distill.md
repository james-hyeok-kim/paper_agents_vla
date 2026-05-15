---
name: cpr-distill
description: Contact-Phase Reweighted Distillation for VLA — contact-specific loss reweighting during student-teacher knowledge distillation
metadata:
  type: project
  status: active
  verdict: conditional-go
  poc_status: pass
  m0_smoke_status: pass
  next_milestone: W1 full ActDistill setup (or reweight factor sweep first)
---

# CPR-Distill — Contact-Phase Reweighted Distillation for VLA

**Formerly known as**: AMP-Distill (SE(3) Manifold-Preserving Distillation). Headline pivoted after synthetic PoC revealed SO(3) geodesic is *not* the main contribution; **contact-phase reweighting** is.

## Core Hypothesis (revised)

Student VLAs distilled with **contact-phase reweighted L2** preserve action accuracy in contact-rich phases significantly better than naive uniform-weight distillation. The mechanism is **contact-specific**, not generic reweighting (validated by sham control).

## Technical Approach

1. **Loss decomposition**: action = (translation R³, rotation SO(3), gripper) — same loss family
2. **Contact-phase reweighting (KEY)**: during teacher-identified contact phases, weight rotation/translation errors by **3x**
3. **Contact phase detection** (proxy, no F/T in LIBERO):
   - Gripper-state transition + window (±3 timesteps)
   - Fallback: last 30% of trajectory if no transitions
4. **SO(3) geodesic**: optional secondary, demonstrated equivalent to L2-fixed + reweight in PoC

## Evidence Status

### Synthetic PoC (2026-05-15, 73 sec)
- Specificity ratio **67x** (contact-rich gain 3.72° vs contact-poor 0.055°)
- **L2-fixed + reweight ≈ SO(3) + reweight** → SO(3) demoted to secondary
- Results: `experiments/amp_distill/`

### M0 Real LIBERO Smoke Test (2026-05-15, 103 sec)
- **Gate B (sham control): CPR beats sham 3x by +17.07pp** on contact MSE (target ≥2pp, 8.5x 초과)
- Validates that **contact-specificity** is the mechanism, not reweighting alone
- Tradeoff revealed: free-phase MSE worsens by -19.6% → reweight factor sweep needed
- Results: `experiments/cpr_distill_m0/`

### Sweep (2026-05-15, 196 sec) — Sweet Spot 발견
- 7 factor settings ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0} + adaptive
- **factor=1.5x sweet spot**: contact +8.15%, free -2.69%, **overall +1.55%** (유일 net positive)
- factor=3.0 (M0 default): contact +18.3% / free -17.2% / overall -3.4% — suboptimal
- Adaptive boost FAILED (mean α collapsed to 0; fix: remove weight_reg)
- **Publishable headline**: "CPR-1.5x: overall +1.55% via contact-specific reweighting, beats sham by 8pp"
- Results: `experiments/cpr_distill_sweep/`

## Closest Prior Work
- **ActDistill** (arXiv:2511.18082, Nov 2025): VLA distillation with pure Euclidean MSE, no contact awareness
- **VITA-VLA** (arXiv:2510.09607): MSE+MAE, no SE(3) handling
- **Refined Policy Distillation** (arXiv:2503.05833)
- Contact-aware IL (Feel the Force, Reactive Diffusion Policy): policy learning, not distillation reweighting

## Differentiator
Contact-phase × distillation reweighting cell is empty in literature. ActDistill uses uniform MSE — sham control proves uniform doesn't work.

## Expected Gains (revised after M0)
- Contact-phase action MSE: **-17% vs uniform L2** baseline (real LIBERO-spatial demonstrated)
- Free-phase: **-19% degradation** at 3x — needs sweep at 1.5x/2x
- Final paper target: ≥2pp task success rate gain on LIBERO-LONG (contact-rich)

## Safety
Contact-phase rotation accuracy preservation reduces unsafe end-effector trajectories. Must verify wrench peak under force-controlled tasks (peg insertion) in real robot.

## Status & Next Steps
- **Status**: ACTIVE, M0 PASS
- **Validator score**: 6.5/10 (idea) + 6.0/10 (experiment plan)
- **Required next**:
  1. Reweight factor sweep (1.5x, 2x, 5x) — sweet spot 찾기
  2. Adaptive learnable weight 실험
  3. M3.5: Contact mask quality ablation (GT real F/T vs predicted vs gripper-transition only)
  4. W1: Full ActDistill setup + LIBERO-LONG end-to-end
- **Target venue**: CoRL 2027

## Related Files
- Idea history (synthetic PoC): `experiments/amp_distill/`
- M0 smoke test: `experiments/cpr_distill_m0/`
- Experiment plan: `.claude/agent-memory/vla-experiment-planner/active/plan_cpr_distill.md`
- Validator (idea): `.claude/agent-memory/vla-idea-validator/conditional/validation_amp_distill.md`
- Validator (plan): `.claude/agent-memory/vla-idea-validator/conditional/validation_cpr_distill_experiment_plan.md`
