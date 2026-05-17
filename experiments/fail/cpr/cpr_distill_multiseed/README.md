# Experiment 8 — CPR-Distill Multi-Seed Statistical Significance

## Metadata
- **날짜**: 2026-05-15
- **Tier**: 후속 ablation (significance 검증)
- **상태**: ✅ PASS (mechanism) / ⚠️ NULL (overall) — 결과 해석 주의 필요
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 630초

## 검증한 가설
원래 Sweep의 1.5× sweet spot 결과 (overall +1.55%)는 통계적으로 유의한가? CPR-1.5×의 contact gain은 sham 3×와 통계적으로 구별되는가?

## 방법
- 4 condition × 5 seed = 20 runs
- Conditions: factor_1.0 (baseline), factor_1.5, factor_3.0, sham_3.0
- Seed ∈ {0, 1, 2, 3, 4}
- 3 epoch 학습, 동일한 데이터 split

## 핵심 결과 (mean ± std, n=5)

| Condition | Contact MSE | Free MSE | **Overall MSE** |
|---|---|---|---|
| factor_1.0 (baseline) | 0.894 ± 0.004 | 0.574 ± 0.005 | **0.671 ± 0.004** |
| **factor_1.5** | **0.834 ± 0.009** | 0.600 ± 0.008 | **0.671 ± 0.008** |
| factor_3.0 | 0.739 ± 0.012 | 0.682 ± 0.012 | 0.699 ± 0.011 |
| sham_3.0 | 0.901 ± 0.007 | 0.584 ± 0.006 | 0.680 ± 0.003 |

### Statistical Significance
- CPR-1.5× vs baseline contact: Δ = 0.060, **z = 5.97σ** ✅
- CPR-1.5× vs sham 3× contact: Δ = 0.068, **z = 5.75σ** ✅

## 중요 발견

### 1. Contact-Specific Mechanism은 통계적으로 매우 유의
CPR-1.5× contact reduction이 **5σ 이상**으로 baseline과 sham 양쪽에서 분리. **메커니즘 자체는 publishable real effect.**

### 2. ⚠️ "Overall +1.55%" 헤드라인은 Single-Seed Noise였음
원래 Sweep에서: factor_1.5 overall 0.663 vs factor_1.0 0.673 → +1.55%

Multi-seed에서: factor_1.5 overall **0.671 ± 0.008** vs factor_1.0 **0.671 ± 0.004**
→ **차이 0.000, 통계적으로 동일** (Δ < σ_combined)

원 sweep 결과는 단일 seed에서 우연히 발생한 fluctuation. **"Overall 개선" 주장은 retract해야 함.**

### 3. Sham 3×는 baseline보다 overall에서 통계적으로 나쁨
sham overall 0.680 vs baseline 0.671 → +0.009 (z ≈ 2σ). 단순 3× scaling은 학습 불안정성 유발. **Sham control이 "no effect" 아닌 "negative effect"라는 정정 필요.**

### 4. Tradeoff Balance at 1.5×
Contact reduction (Δ=0.060)이 free degradation (Δ=0.026)을 양적으로 능가하지만, overall은 weighting에 따라 cancel. 만약 평가가 contact-phase 비중을 더 두는 metric이면 1.5×가 우월.

## Direction (이 실험의 의미)

### 논문 헤드라인 재정정 (필수)

**Before (Experiment 5 single-seed)**:
> "CPR-1.5×는 overall +1.55% 개선"

**After (multi-seed, 통계적 정정)**:
> "CPR-1.5×는 contact-phase MSE를 5σ 유의수준으로 감소시키며 (sham 대비도 5σ 유의), overall MSE는 baseline과 통계적으로 동등하다. Mechanism은 contact-specific이며 일반 reweighting과 명확히 다르다."

### 이게 여전히 publishable한 이유

1. **Mechanism story 무사**: 5σ 유의수준은 매우 강함
2. **Sham control이 결정적**: "reweighting이 같은 효과를 낼까?" 답은 NO (5.75σ)
3. **Contact-phase metric이 task-success에 더 중요한 경우**: contact MSE 감소가 실제로 task success rate 개선으로 이어지는지가 핵심 (→ sim rollout 필요)

### 무엇을 막아주는가
- "Free MSE에도 영향 없이 모든 게 좋아진다" 식의 주장
- Single-seed 결과를 main number로 쓰는 것

### 이 결과로부터 즉시 해야 할 것
1. Sweep README의 "1.55% overall gain" 표현 정정
2. 모든 numerical claim에 ±σ 추가
3. CoRL submission 전 sim rollout으로 task success rate 측정 필수

## 한계 / 주의사항
- 5 seed는 통계적으로 minimum (n≥10 권장)
- 동일 dataset split — 다른 split에서 안정성 미측정
- TinyBC scale; SmolVLA-base에선 σ가 다를 수 있음

## 다음 단계
→ 모든 paper claim을 multi-seed mean±std로 정정. Sim rollout으로 task success 검증 필요.

## 파일
- `multiseed.py` — 스크립트
- `results.json` — raw per-seed + aggregated
- `run.log` — 실행 로그
- `multiseed.png` — bar chart with error bars + per-seed scatter
