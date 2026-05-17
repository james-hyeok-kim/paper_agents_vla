# Experiment 13 — Multi-Seed Verification with channel_diff Detector

## Metadata
- **날짜**: 2026-05-15
- **Tier**: Statistical significance check on the new detector
- **상태**: ✅ Contact mechanism confirmed at **8σ**, ⚠️ Overall MSE null
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 372초

## 검증한 가설
Experiment 12에서 발견된 `gripper_channel_diff` detector의 +10.26% contact gain은 multi-seed에서도 유의한가? Overall +1.48% 개선은 진짜인가, single-seed noise인가?

## 방법
- 3 conditions × 5 seeds = 15 runs
- conditions:
  - baseline (last_30 mask, factor=1.0 — no reweight)
  - channel_diff_1.5 (gripper_channel_diff detector, factor=1.5)
  - sham_3.0 (last_30, uniform 3× reweight)
- 동일한 데이터 split, 동일한 학습 hyper

## 핵심 결과 (mean ± std, n=5)

| Condition | Contact MSE | Free MSE | Overall MSE |
|---|---|---|---|
| baseline | 0.866 ± 0.007 | 0.551 ± 0.005 | 0.647 ± 0.002 |
| **channel_diff_1.5** | **0.807 ± 0.008** | 0.593 ± 0.007 | **0.647 ± 0.007** |
| sham_3.0 | 0.876 ± 0.003 | 0.559 ± 0.005 | 0.656 ± 0.003 |

### Statistical Significance

| 비교 | Δ | z-score | Verdict |
|---|---|---|---|
| channel_diff vs baseline (contact) | -0.059 | **5.66σ** | ✅ 매우 유의 |
| **channel_diff vs sham (contact)** | -0.069 | **8.05σ** | ✅ ⭐ 강력한 mechanism 검증 |
| channel_diff vs baseline (overall) | -0.00004 | **-0.01σ** | ❌ NULL |

## 중요 발견

### 1. ✅ Mechanism은 8σ로 더 강하게 검증
이전 multi-seed (last_30 mask, Exp 8): 5.97σ vs sham
신규 multi-seed (channel_diff mask): **8.05σ vs sham**

→ 더 좋은 detector가 mechanism signal을 amplify. False였을 가능성 사실상 0.

### 2. ❌ "+1.48% Overall" 주장 정정 (multi-seed에서 완전 null)
Experiment 12의 single-seed: overall gain +1.48%
Multi-seed: Δ overall = **-0.00004** (literally 0), z = -0.01σ

→ +1.48%는 σ=0.007 noise 안의 fluctuation. Multi-seed가 진실: contact gain = free loss로 cancel.

### 3. Sham control 추가 검증
Sham 3×는 baseline보다 살짝 worse (overall 0.656 vs 0.647, z≈2σ negative). 단순 reweight는 도움 안 되고 약간 해로움.

### 4. Variance가 매우 낮음 (signal-to-noise 명확)
- Baseline contact: σ=0.007
- channel_diff contact: σ=0.008
- Effect size: 0.059 → SNR ≈ 8

이 정도 SNR이면 single-seed로도 비교적 신뢰 가능. 단 overall에선 σ=0.007에 effect 0.000004라 절대 single-seed로 안 됨.

## Direction (이 실험의 의미)

### Paper claims의 통계적 최종 결론

**살아남는 (multi-seed 검증된)**:
- ✅ Contact gain: 6.8% (sham 대비), 8σ
- ✅ Mechanism is contact-specific (sham control 통과)
- ✅ Detector matters: channel_diff > last_30 fallback

**Retract**:
- ❌ Overall MSE improvement (Δ ≤ 0.0001, z < 1σ — null)

### 헤드라인 (multi-seed 검증된 버전)
> "CPR-Distill: contact-phase reweighted distillation provides **6.8% contact-phase MSE reduction with 8σ significance vs sham control**, while overall MSE remains baseline-equivalent. The mechanism is genuinely contact-specific."

### 다음 단계로 답해야 할 critical 질문
"Contact MSE 6.8% 감소가 task success rate 개선으로 변환되는가?" → **Sim rollout (Option B) 만이 답을 줄 수 있음**.

## 한계 / 주의사항
- libero_spatial 단일 suite — multi-suite × multi-seed가 더 강한 evidence
- TinyBC scale; SmolVLA-base에서 결과는 다를 수 있음
- 5 seeds는 minimum, 더 안전하게 하려면 10 seeds

## 다음 단계
1. **Sim rollout (Option B)**: contact MSE → task SR translation 검증
2. **Multi-suite × multi-seed**: 4 suites × 3 factors × 5 seeds (current이 단일 suite multi-seed라 generalization 미확보)
3. **W1 SmolVLA scale**: TinyBC 결과가 real student에 transfer되는지

## 파일
- `multiseed_channeldiff.py` — 스크립트
- `results.json` — raw + aggregated + z-scores
- `run.log` — 실행 로그
- `multiseed_channeldiff.png` — error bar + per-seed scatter
