# Experiment 6 — CPR-Distill Adaptive Boost v2 (구조적 실패)

## Metadata
- **날짜**: 2026-05-15
- **Tier**: 후속 ablation (Experiment 5의 adaptive 실패 fix 시도)
- **상태**: ❌ FAIL — adaptive boost가 fixed 1.5×를 능가하지 못함
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 112초

## 검증한 가설
Experiment 5의 adaptive 실패는 `weight_reg=1e-3` 페널티 때문이었다는 가설. Reg 제거 + boost head bias 양수 초기화 (logit≈-0.85 → 시작 α≈1.5)하면 adaptive가 fixed 1.5×를 능가할 것이다.

## 방법
- `TinyBCWithAdaptive` 그대로 사용
- `weight_reg = 0` (페널티 제거)
- Bias 초기화: `m.bias.fill_(-0.85)` → 시작 α = max_boost · sigmoid(-0.85) ≈ 1.5
- 비교 대상: baseline (factor=1.0), fixed 1.5× (sweet spot), adaptive v2

## 핵심 결과

| Condition | Contact MSE | Free MSE | Overall | Δ Contact vs baseline | Δ Overall |
|---|---|---|---|---|---|
| baseline_1.0 | 0.878 | 0.571 | 0.664 | — | — |
| fixed_1.5 | 0.827 | 0.604 | 0.672 | **+5.77%** | -1.14% |
| **adaptive_v2** | 0.875 | 0.584 | 0.673 | +0.33% | -1.31% |

### α 진화 (contact sample mean)
- Epoch 1: **0.030** (bias init에도 불구하고 이미 작음)
- Epoch 2: 0.0008
- Epoch 3: 0.0004

→ Bias 초기화가 첫 epoch 안에 무력화됨. Collapse는 reg 페널티 때문이 아니었음.

## 중요 발견

### 구조적 진단
End-to-end loss `mean(err·(1+α·contact))`는 α 증가 시 **항상 커짐** (err≥0이므로). Gradient는 α를 0 쪽으로 미는 자연 force가 존재 → bias init이 강해도 학습 dynamics가 collapse 유발.

**근본 원인**: α가 자유 파라미터인 동시에 loss 가중치인 self-referential setup. α를 키울수록 loss가 커지므로 동일 loss로 동시 최적화 불가.

### Fix 방향 (현재 setup으로는 불가)
- **Bilevel optimization**: outer loop가 contact-phase eval metric으로 α 학습, inner loop는 α 고정한 채 모델 학습
- **Gradient reversal**: α-branch에서 `-grad` 적용 (GAN 스타일)
- **Detached α schedule**: α를 별도 schedule predictor로 분리, 모델 loss에서 detach

## Direction (이 실험의 의미)

- **End-to-end naive adaptive는 사망**: 같은 setup으로 hyperparameter 조정해도 안 됨. Architectural change 필요.
- **Publishable로 활용**: "naive adaptive collapses by construction" — negative result로 paper에 ablation 한 줄. Fixed factor가 왜 main method인지 정당화.
- **Bilevel/reversal 시도는 W1 이후로 deferred**: 시간 대비 효용 낮음. Fixed 1.5×로 main paper 작성 후 future work.

## 한계 / 주의사항
- α head 구조 자체를 바꾸지 않음 (linear MLP). 다른 parameterization은 미테스트.
- Bilevel/reversal 실험을 추가로 안 함.

## 다음 단계
→ Adaptive 트랙은 main paper의 future work으로 강등. Fixed factor 트랙이 primary remain.

## 파일
- `adaptive_v2.py` — 스크립트
- `results.json` — 측정값
- `run.log` — 실행 로그
- `adaptive_v2.png` — bar chart + α 진화 plot
- `/data/jameskimh/cpr_distill_adaptive_v2/adaptive_v2.pt` — 체크포인트
