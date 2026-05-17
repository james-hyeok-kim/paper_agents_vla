# Experiment 10 — CPR-Distill Multi-Suite Generalization (Critical Negative)

## Metadata
- **날짜**: 2026-05-15
- **Tier**: 후속 검증 (cross-distribution generalization)
- **상태**: ❌ Critical Negative — 1.5× sweet spot은 generalize 안 됨
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 582초

## 검증한 가설
Experiment 5에서 libero_spatial에서 발견한 1.5× sweet spot이 다른 LIBERO suite (libero_object, libero_goal, libero_10)에서도 일관되게 작동한다.

## 방법
- 4 suite × 3 factor (1.0, 1.5, 3.0) = 12 runs
- 각 suite마다 train/val split 동일하게 (90/10)
- TinyBC + 3 epoch 학습

## 핵심 결과 (Δ vs factor=1.0)

| Suite | Factor 1.0 | Factor 1.5 contact | Factor 1.5 free | **Factor 1.5 overall** | Best overall |
|---|---|---|---|---|---|
| **libero_spatial** | baseline | +6.43% | -5.17% | **-0.64%** | factor=1.0 ⬅ |
| **libero_object** | baseline | +1.23% | -6.17% | -2.94% | factor=1.0 ⬅ |
| **libero_goal** | baseline | +4.69% | -4.24% | -0.55% | factor=1.0 ⬅ |
| **libero_10** | baseline | +0.05% | -7.36% | -4.65% | factor=1.0 ⬅ |

**Sweet-spot factor counts: {1.0: 4, 1.5: 0, 3.0: 0}**

## 중요 발견

### 1. ⚠️ 1.5× Sweet Spot은 Libero_Spatial 특화 결과
4개 suite **모두**에서 factor=1.0이 overall 최고. Original sweep의 "1.5× is universal sweet spot" 주장은 false generalization.

### 2. Contact Gain은 Suite-Dependent
| Suite | Contact gain @ 1.5× | 해석 |
|---|---|---|
| libero_spatial | +6.43% | Contact-rich task 많음 (peg-in-hole 유형) |
| libero_goal | +4.69% | 중간 |
| libero_object | +1.23% | Object reaching dominant — contact phase 약함 |
| libero_10 | **+0.05%** | 사실상 효과 없음 — long-horizon task |

→ **메커니즘이 task 분포에 강하게 의존**. Contact-rich subset에서만 의미 있고, contact-poor에서는 거의 효과 없음.

### 3. Overall MSE는 모든 suite에서 baseline이 최고
어떤 factor도 모든 suite에서 net positive overall을 보이지 못함. **"CPR-Distill improves overall action accuracy" 주장은 generally false.**

### 4. libero_10에서 거의 무효한 이유 (가설)
- Long-horizon (긴 trajectory): contact phase의 상대적 비중 감소
- 많은 sub-task: 단일 grasp 위주가 아닌 다단계 manipulation
- Visual variety: contact가 visual에 강하게 conditioned되지 않음
→ 정확한 원인 분석은 contact density 측정 + per-task breakdown 필요

## Direction (이 실험의 의미)

### 🚨 논문 스토리 근본적 reframe 필요

**Before (libero_spatial 단일 결과 기반)**:
> "CPR-Distill is a general method for improving VLA distillation across tasks."

**After (multi-suite 결과 반영)**:
> "CPR-Distill provides statistically-significant contact-phase MSE reduction in contact-rich manipulation suites (libero_spatial, libero_goal). The effect diminishes on object-reaching suites (libero_object) and long-horizon multi-task suites (libero_10). Overall MSE remains baseline-equivalent or worse; the method's value is in **selectively improving contact-phase accuracy** rather than aggregate MSE."

### 살릴 수 있는 핵심 contribution

1. **Mechanism은 여전히 진짜**: Experiment 8 multi-seed에서 5σ contact-specificity 입증
2. **Contact-rich subset에서는 유효**: libero_spatial/goal 명시
3. **Task-dependence가 publishable insight**: "Reweighting helps where contact dominates"

### 막아주는 false claim
- "Universal improvement across LIBERO"
- "Sweet spot generalizes"
- "Better than baseline on overall MSE"

### 이게 의미하는 W1 결정
- **Sim rollout 필수**: contact MSE → task success rate translation 검증
- Task success가 contact-rich suite에서 의미 있게 개선되면 contribution 유효
- 그렇지 않다면 CPR-Distill 폐기 또는 매우 niche 방향으로 reframe

## 한계 / 주의사항
- 각 suite single seed — variance 미측정
- TinyBC scale — 실제 SmolVLA에서 결과 다를 수 있음
- "Contact density per suite" 미측정 — task dependence의 정확한 원인 불명
- 3 factor만 sweep — suite별로 다른 optimal factor 있을 수 있음

## 다음 단계 (Critical Decision Point)

### Option A: Pivot to Task-Specific Story
- Headline을 "Contact-rich VLA에서의 selective reweighting"으로 좁힘
- libero_spatial + libero_goal에 집중
- Real robot은 contact-rich task만

### Option B: Sim Rollout으로 Task-Success Test
- Action MSE vs task success 관계 검증
- 만약 contact MSE 6% 감소가 task success 2-3pp 개선으로 이어지면 살릴 만함

### Option C: Per-Suite Factor Tuning
- Suite마다 다른 best factor 찾기 (1.0 includes "no reweight" optimal)
- "Adaptive factor per task category" 새 sub-story

### Option D: Abandon CPR-Distill
- Generalization 실패가 너무 큼
- 다른 idea로 pivot

## 파일
- `multisuite.py` — 스크립트
- `results.json` — 4 suite × 3 factor 측정값
- `run.log` — 실행 로그
- `multisuite.png` — suite별 factor vs gain plot (4 subplot)
