# CONDITIONAL GO Validation Results — 2026-05-15

세 CONDITIONAL GO 아이디어에 대한 vla-idea-validator 최종 검증 결과.

---

## 1. AMP-Distill — Action-Manifold-Preserving Distillation

**판정**: CONDITIONAL GO (**6.5/10** — 최상위)

### Pre-experiment Gate (Week-1 필수)
- ActDistill에서 rotation 항만 **SO(3) geodesic으로 swap**한 ablation
- **Contact-rich task** (peg-in-hole, insertion): ≥2pp gain
- **Contact-poor task** (pick-place): marginal gain (mechanism specificity 입증)
- 둘 다 만족해야 GO. 한쪽이라도 실패 시 NO-GO 또는 reframe.

### Strongest Reframing
"Contact-phase가 VLA distillation의 information bottleneck"으로 main thesis 재정의 + force profile smoothness를 safety contribution으로 확장 → **7.5-8/10** 가능

### Top 3 Reviewer 공격
1. "RFMP + ActDistill 결합 아닌가?"
2. "Contact-phase 정의가 hyperparameter heavy"
3. "Real robot result 없으면 marginal"

---

## 2. XV-Dedup — Cross-View Visual Token Deduplication

**판정**: CONDITIONAL GO (**6.0/10**)

### Pre-experiment Gate (Week-1 3가지 모두 통과)
- Cross-view overlap ≥30%
- Prefill latency ≥25% 감소 (단독)
- BFA++ all-views-kept phase 비율 ≥40% (stacking 의미 검증)

### Week-2 Decision
- LIBERO-Spatial SR drop ≤3% at ≥1.4x speedup
- LIBERO-Goal SR drop ≤5%

### 가장 큰 약점
**Safety (2.5/5)** — wrist camera token mis-merging이 fine-manipulation에서 catastrophic failure 위험. Graceful degradation 메커니즘 없음.

### Strongest Reframing
"Phase-aware safety-guaranteed cross-view compression" — safety를 primary contribution으로 elevation

### 가장 위험한 Reviewer 공격
BFA++ stacking 추가 speedup이 1.2x 이하면 paper-level 가치 붕괴 → Week-1에 우선 확인 필요

---

## 3. CP-Sparse — Chunk-Position-Aware Sparse Attention

**판정**: CONDITIONAL GO (**5.5/10** — 최하위)

### Pre-experiment Gate (Week-1 multi-metric, 단일 metric 거부)
- **(A) Per-position entropy H(i) slope**: > 0.3 nats/position (schedule 정당화)
- **(B) Top-k mass at k=8**: late position에서 >0.8 (k=8 feasibility)
- **(C) Top-k overlap (i=0 vs i=L-1)**:
  - <0.7 → first-position argmax keep-set OK
  - 0.7~0.9 → union-of-top-k 필요
  - >0.95 → uniform keep-set으로 **reframe 필요**

### 핵심 위협: AutoHorizon Generalization 확률 40-60%
π0.5/GR00T에서 발견된 "intra-chunk attention invariance"가 ACT/SmolVLA로 generalize되면 핵심 동기 붕괴. Week-1 실측 없이는 결정 불가.

### Minimum Bar for Publication
- SmolVLA-450M 기준 ≥1.6× action expert forward speedup on Jetson AGX
- LIBERO 4개 suite 중 ≥2개 + ≥1 contact-rich task에서 SR drop ≤5%
- ≥1 real-robot demonstration

### 가장 큰 구조적 리스크
ACT/SmolVLA 절대 latency가 작아(10-45ms) contribution scale 부족 위험 → Primary target을 SmolVLA-450M으로 명시 이동, Jetson AGX/Nano 고정

---

## 종합 우선순위

| 순위 | 아이디어 | 점수 | Pre-exp 비용 | Pre-exp 통과 시 GO 확률 |
|---|---|---|---|---|
| 1 | AMP-Distill | 6.5/10 | 낮음 (loss swap만) | **높음** — clean differentiator |
| 2 | XV-Dedup | 6.0/10 | 중간 (BFA++ 재현 필요) | 중간 — safety 보강 필수 |
| 3 | CP-Sparse | 5.5/10 | 중간 (entropy 측정) | 낮음 — AutoHorizon gate 실패 시 즉시 사망 |

---

## 권장 실행 순서

1. **AMP-Distill Week-1 ablation 먼저** — 비용 가장 낮고 GO 전환 확률 가장 높음
2. AMP-Distill 통과 시: vla-experiment-planner로 main experiment 설계 진입
3. AMP-Distill 실패 시: XV-Dedup Week-1 BFA++ stacking 확인
4. CP-Sparse는 마지막 — Week-1 entropy 측정에 AutoHorizon replicate까지 포함 (negative result도 contribution이 됨)
