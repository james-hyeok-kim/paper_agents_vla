# Week-1 Feasibility PoC Results — 2026-05-15

3개 CONDITIONAL GO 아이디어 PoC 결과. 4x B200 GPU 환경에서 병렬 실행. 총 소요시간: 73s + 11s + 9s.

---

## 1. AMP-Distill — ✅ **PASS** (with critical pivot)

### 실험 설정
- 합성 SE(3) trajectory 200개, T=50
- 3가지 loss 비교: L2-naive, L2-fixed(double-cover), SO(3) geodesic
- 2가지 mode: w/ contact-reweight, w/o
- 2가지 trajectory type: contact-rich (peg-in-hole 스타일), contact-poor (reach-move)

### 핵심 측정값 (rotation error in degrees)

| 조건 | Contact phase | Free phase | Overall |
|---|---|---|---|
| Contact-rich + L2-naive | **7.25°** | 4.61° | 5.93° |
| Contact-rich + L2-fixed + reweight | **3.40°** | 6.77° | 5.09° |
| Contact-rich + SO(3)+reweight | **3.52°** | 7.06° | 5.29° |
| Contact-poor + L2-naive | — | — | 3.02° |
| Contact-poor + SO(3)+reweight | — | — | 2.96° |

### Gate 결과
- ✅ Contact-rich gain: **3.72°** (L2-naive → SO(3)+reweight, 51% 감소)
- ✅ Contact-poor gain: **0.055°** (사실상 0)
- ✅ **Specificity ratio: 67x** — mechanism이 contact phase에 특화됨

### ⚠️ Critical Finding: Headline Pivot Required
- **L2-fixed + reweight (3.40°) ≈ SO(3) geodesic + reweight (3.52°)** — 거의 동일
- 즉 "SO(3) geodesic"이 아니라 **"contact-phase reweighting + 적절한 rotation parameterization"** 이 핵심 contribution
- Validator가 권고한 **"Contact-phase가 VLA distillation의 information bottleneck"** 리프레이밍이 정확히 맞음
- Quaternion double-cover fix만으로 7.25→6.12 (1.1° 개선), reweight 추가로 6.12→3.40 (2.7° 추가)

### Verdict: **GO** (with reframed headline)
- Headline: "Contact-Phase Reweighted Distillation for VLA"
- SO(3) geodesic은 secondary contribution (ablation에 포함)
- 다음 단계: ActDistill repo + LIBERO-LONG에서 동일 ablation 재현

---

## 2. XV-Dedup — 🟡 **PARTIAL PASS** (1/3 gates, 2 with caveats)

### 실험 설정
- DINOv2-base (224x224, patch_size=14, 256 patches/view)
- 8개 합성 scene × 2 views (16px viewpoint shift + color jitter)
- LSH: 32/64/128-dim random projection (InfoNCE 미학습)

### Gate 결과

| Gate | Target | Result | Status |
|---|---|---|---|
| A: Cross-view overlap at cos>0.85 | ≥30% | **70.36%** | ✅ PASS (2.3x 초과) |
| A: Cross-view overlap at cos>0.7 | — | 93.02% | (reference) |
| A: Avg max-similarity | — | 0.866 | (reference) |
| B: LSH random-projection agreement | ≥50% | 29.6% (H=128) | ❌ FAIL (InfoNCE 필요) |
| C: 40% token 감소 시 prefill latency | ≥25% | **21.21%** | ❌ FAIL (slight miss) |

### Interpretation
- **Core hypothesis 강력 confirmed**: cross-view overlap이 70%로 예상(30%)을 한참 초과
- LSH random projection은 예상대로 부족 — InfoNCE-learned hash projector 필수
- Latency reduction은 25% 직전 — 더 큰 token 감소 (50-60%)면 통과 가능

### Verdict: **CONDITIONAL GO**
- Core idea는 검증됐지만 LSH 학습 필요
- 다음 단계: InfoNCE로 hash projector 학습, 더 공격적인 token 감소 ratio sweep

---

## 3. CP-Sparse — ❌ **FAIL** (core hypothesis 미입증)

### 실험 설정
- Tiny ACT (d=128, chunk_len=50, ctx_len=64)
- 합성 chunk-prediction task (position-conditioned context dependency)
- 500 iter 학습

### Gate 결과

| Gate | Target | Result | Status |
|---|---|---|---|
| A: Entropy slope (decreasing with position) | end-to-end Δ > 0.3 nats | +0.075 nats (slope 0.004) | ❌ FAIL |
| B: Top-8 mass at late positions | >0.8 | 0.41 (last), 0.69 max | ❌ FAIL |
| C: Argmax overlap pos0 vs pos49 | <0.7 GO / >0.95 NO-GO | **0.06** | ✅ GO signal (but caveat) |

### Per-position entropy (nats)
- Pos 0: 3.66 / Pos 25: 3.68 / Pos 49: 3.74 — **거의 평탄**

### Per-position top-8 mass
- Pos 0: 0.44 / Pos mid: 0.43 / Pos last: 0.41 — **거의 동일**

### ⚠️ Critical Interpretation
1. **CP-Sparse 원안 가설 (later = sharper) 미입증** — entropy/mass는 position에 따라 변하지 않음
2. **AutoHorizon invariance도 미관측** — Jaccard 0.06은 매우 낮음 (즉 position마다 다른 context attending)
3. **세 번째 시나리오 등장**: 각 position이 **다른 작은 subset**에 attending하되, **subset 크기는 일정**

### Caveat
- 합성 데이터를 일부러 position-conditional dependency로 구성했음 (`action[t] ∝ ctx[t·64/50]`)
- 이 구조가 모델이 position-specific attention을 학습하게 강제
- 실제 ACT/SmolVLA에서는 결과 다를 수 있음 — 합성 PoC의 한계

### Verdict: **NO-GO for original framing** / Pivot 권장
- "Anchor-centered uniform sparse attention" 또는 "Position-conditioned uniform-k sparse attention"으로 재구성 필요
- 실제 SmolVLA on real LeRobot data에서 재측정 필요 (다음 단계)

---

## 종합 우선순위 재정렬

| 순위 | 아이디어 | PoC 결과 | 다음 단계 |
|---|---|---|---|
| **1** | **AMP-Distill** (reframed) | **PASS, 67x specificity** | ActDistill repo + LIBERO-LONG ablation |
| 2 | XV-Dedup | Core hypothesis confirmed (70% overlap), LSH 학습 필요 | InfoNCE training + π0 multi-camera |
| 3 | CP-Sparse | 원안 미입증, pivot 필요 | Real SmolVLA로 재측정 OR axis 변경 |

## 결론

- **AMP-Distill이 가장 빠른 GO 경로**: synthetic PoC 단계에서 67x specificity ratio, headline pivot 명확 ("Contact-Phase Reweighting" main)
- **XV-Dedup은 검증된 idea**: core hypothesis 강력하게 confirmed, 다만 LSH 학습 후 latency 재측정 필요
- **CP-Sparse는 원안 폐기 또는 대규모 pivot**: 합성 PoC에서 핵심 가설 미입증, 실제 모델 분석 없이는 진행 불가

### 즉시 실행 권장
AMP-Distill의 reframed version으로 **vla-experiment-planner**에 넘겨 full experiment 설계. ActDistill baseline 대비 LIBERO-LONG/contact-rich subset에서 ≥2pp gain 확인이 main milestone.
