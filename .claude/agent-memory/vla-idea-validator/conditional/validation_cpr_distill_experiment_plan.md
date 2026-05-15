---
name: validation-cpr-distill-experiment-plan
description: CPR-Distill (구 AMP-Distill) experiment plan 검증 - CONDITIONAL GO 6/10, PoC가 원래 gate 불충족
metadata:
  type: project
---

# CPR-Distill Experiment Plan Validation (2026-05-15)

## Context
[[validation-amp-distill]] 후속. Idea가 SE(3) geodesic + contact reweight + gripper CE + uncertainty 결합에서 **contact-phase reweighted distillation 단일 mechanism**으로 축소됨. 4-pillar + real-robot 1 task + 6주 timeline 제시.

## Final Verdict: CONDITIONAL GO (6.0/10)

### 결정적 발견
**PoC가 원래 Milestone 1 gate를 만족하지 않음**:
- 원래 gate: ActDistill에서 rotation 항만 SO(3)로 swap → contact-rich task에서 ≥2pp success rate gain
- 실제 PoC: 합성 SE(3) trajectory, success rate 아닌 rotation error metric
- 결과: L2-fixed + reweight (3.40°) ≈ SO(3) + reweight (3.52°) → **SO(3) geodesic contribution = 0**
- 67x specificity는 합성 setup의 artifact 가능성 높음

### 실험 계획 점수 (1-10)
- 실험 충분성: 6/10 (4-pillar 적절하지만 phase-aware IL baselines + sham reweight control 누락)
- Baseline 적절성: 5/10 (focal/IW-BC, BC-Z phase conditioning, uniform 3x reweight sham 누락)
- Risk 평가: 6/10 (F/T 부재의 영향 과소평가, real-robot 1 task로 specificity 입증 불가능)
- Timeline 현실성: 5/10 (6주는 optimistic, W5 real-robot은 너무 늦음)
- Compute 정확성: 6/10 (training만 추정일 가능성, eval/cache storage 별도)

### 추가 필요한 Milestone
**M0 (Pre-experiment)**: 합성 PoC 결과를 real LIBERO-LONG 1-task 24h smoke test로 재현 — ≥2pp gain 확인되면 GO. 실패 시 NO-GO 또는 idea 폐기.
**M2.5 (Sham reweight control)**: Uniform 3x action loss reweight (contact phase 아닌 모든 step). Contact-specificity가 진짜인지 입증하는 must-have ablation.
**M5.5 (Real-robot specificity)**: Contact-poor real task 1개 추가 — sim specificity가 real에서도 유지되는지 검증.

### Top 3 Reject 사유 예상
1. **"이건 그냥 contact step에 weight 3배 준 것"** — Single hyperparameter trick. 방어: contact 정의의 generality, 다른 reweighting과의 차별성 ablation, contact mechanism의 interpretability story.
2. **"LIBERO에 F/T 없으니 contact 정의가 proxy일 뿐"** — Gripper transition은 contact event의 약한 proxy. 방어: real-robot F/T validation 1 task 추가, 합성 contact mask와 ground truth 비교.
3. **"SO(3) loss가 main contribution이라 framing하더니 PoC는 SO(3) 효과 없음"** — Idea drift. 방어: framing을 "contact-phase reweighted distillation"으로 명시적 축소, SO(3)는 appendix로.

### Strongest reframe
"**Phase-aware information reweighting in VLA distillation**" — contact는 그저 information density가 높은 phase의 instance라는 framing. Force profile + reaction time + gripper trans를 통합 phase score로. CoRL main paper 가능성 6 → 7.5.

## Why
사용자가 vla-experiment-planner 출력 검증 요청. 직전 validation (CONDITIONAL GO 6.5)의 pre-experiment gate가 합성 PoC로 대체되었지만 실제로 gate 통과 못 함.

## How to apply
사용자가 이 idea로 W1 실험 시작 전: **real LIBERO smoke test가 진짜 gate**임을 다시 강조. Sham reweight control이 ablation에 없으면 framework 자체가 무너질 위험. Real-robot은 2 tasks (contact-rich + contact-poor) 권장.
