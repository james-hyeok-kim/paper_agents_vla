---
name: validation-amp-distill
description: AMP-Distill (Action-Manifold-Preserving Distillation) idea의 종합 validation 결과 - CONDITIONAL GO 6.5/10
metadata:
  type: project
---

# AMP-Distill Validation (2026-05-15)

## Idea
SE(3) geodesic + gripper CE + teacher uncertainty + contact-phase reweighting을 결합한 VLA distillation loss framework.

## Final Verdict: CONDITIONAL GO (6.5/10)

### Scores
- Novelty: 3/5 (component 개별로는 약함, contact-phase × distillation 결합이 유일한 hook)
- Feasibility: 3.5/5 (loss는 쉬움, contact-phase labeling pipeline이 진짜 작업)
- Safety: 3/5 (gripper mode collapse + approach phase degradation 위험)
- Publishability: 3/5 (CoRL 적합, ActDistill 대비 차별화는 pre-experiment 결과에 운명)
- Scope: 4/5 (AR + flow matching 2개 head로 제한 권장)

### 결정적 Gate
**Pre-experiment (Milestone 1)**: ActDistill loss의 rotation 항만 SO(3) geodesic으로 swap했을 때 contact-rich task에서 ≥2pp gain, contact-poor task에서는 marginal gain. 둘 다 만족 시 GO, 실패 시 NO-GO.

### 핵심 위험
1. RFMP (arXiv:2403.10672) + ActDistill (arXiv:2511.18082) 결합으로 보일 위험 - reviewer 공격 1순위
2. Gripper mode collapse - class-balanced CE 필수
3. Contact-phase 정의가 hyperparameter heavy - sensitivity analysis 필수

### Strongest reframe
"Contact-phase가 VLA distillation의 information bottleneck"으로 framing 변경 시 6.5 → 7.5-8/10 가능. Force profile smoothness를 safety contribution으로 확장.

## Why
사용자가 vla-literature-checker(5/10 CONDITIONAL GO)에 이어 두 번째 게이트로 종합 검증 요청. Literature checker 권고 pivot은 이미 반영된 상태.

## How to apply
이 idea에 대한 후속 작업 (vla-experiment-planner 등) 진입 시: pre-experiment 결과를 첫 번째 input으로 요구할 것. Contact-phase reweighting을 main contribution으로 다룰 것. Flow matching variant는 부록으로 강등 유지.
