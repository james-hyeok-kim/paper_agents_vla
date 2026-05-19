---
name: rgv
description: Receptacle-Goal-Vector grounding head — training-time auxiliary head predicting 3D goal vector to force visual encoder into object-centric representation
metadata:
  type: project
  status: pending
  verdict: null
  round: 6
  date_generated: 2026-05-18
---

# RGV — Receptacle-Goal-Vector Grounding Head

## Core Hypothesis

ChunkedBC v6 action head 앞에 **3D goal vector regression head** (current EEF → target object position의 unit vector + distance, R⁴) 를 multi-task aux로 학습하면, vision encoder가 object-centric feature를 학습하게 되어 libero_spatial SR이 +5pp 이상 향상된다. Inference 시 goal head는 *사용되지 않음* — architectural pressure만으로 contribution.

## Mechanism (training-time aux, sham-robust by label-source)

- Multi-head output: action chunk (기존) + goal vector `g_t ∈ R⁴` (3D unit vector + distance)
- Label: LIBERO simulator의 `env.get_target_pose()` 에서 추출 (training-only)
- Loss = action_MSE + λ · goal_vector_MSE (λ=0.3)
- Inference 시 goal head forward는 skip — 추가 latency 0

## Sham battery & separation

| Sham | Label source | Why mechanism wins |
|------|-------------|--------------------|
| Random goal vector | Random R⁴ | Noise label, encoder가 학습할 incentive 없음 |
| Proprio-derived goal | current EEF position | Vision 사용 필요 없음 (proprio로 답) |
| Constant offset | dummy 고정 vector | trivial constant 학습 |

핵심: mechanism label은 *task-specific, visual-grounded*. Sham label은 정의상 visual grounding이 없음. Sham이 reproduce하려면 supervision을 사용해야 함 → 그 순간 mechanism.

## Pre-draft empirical anchor (<30 min)

- libero_spatial demo expert trajectory에서 (EEF, target_obj) vector 추출
- 첫 chunk 평균 action direction과 goal vector cosine similarity
- Pass: cosine ≥ 0.6 (action이 goal direction과 align)
- 추가: ChunkedBC v6 vision feature → goal vector linear-probe R²
  - R² > 0.5: 이미 encoded, RGV 효과 작음
  - R² < 0.3: 새 signal 주입 여지 큼 (RGV 효과 큼)

## BLACKLIST check (explicit)

| # | Family | Status |
|---|--------|--------|
| 1-13 | 전 family | NO — efficiency literature와 직접 무관 |

가장 가까운 영역: CPR-Distill의 aux reweight. Separation: **새 head + 새 label source** (학습 신호 추가), CPR은 동일 label에 weight만 변경.

ESBL adjacency: visual masking이 아니라 *학습 시 visual representation alignment*. Proprio dominance를 정면 돌파 (proprio가 못 채우는 target-object 정보를 명시적으로 학습).

## Expected SR delta

- A (baseline) → B (RGV training, inference 동일): **+5 to +10pp**
- Sham (random goal label): baseline ±2pp
- B − Sham ≥ 5pp

## Track 1 PoC

- Implementation: MLP head (4 unit), LIBERO target pose helper, training aux term
- ~50 lines code 변경
- 3 seeds × 150 ep × A/B/Sham1/Sham2 = ~12 GPU-hrs
- Difficulty: 쉬움

## Publication target

RSS 2026 — "Implicit visual grounding via auxiliary goal regression: a free SR boost for small VLA"

## Risk note

Aux task로 SR boost는 frequent technique. Contribution selling point는:
1. *Inference free* (head는 train-only)
2. SR delta가 spatial suite에서 특히 큼 (spatial reasoning specific)
3. Proprio-dominated regime에서도 작동 (ESBL 교훈을 직접 활용)
