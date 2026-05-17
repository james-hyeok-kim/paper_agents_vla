# Experiment 1 — AMP-Distill 합성 데이터 PoC

## Metadata
- **날짜**: 2026-05-15
- **Tier**: PoC (합성 데이터)
- **상태**: ✅ PASS — 단, headline pivot 필요
- **연결 아이디어**: `cpr-distill` (구 `amp-distill`, 참조: `.claude/agent-memory/vla-idea-generator/active/cpr-distill.md`)
- **GPU**: 1× B200, 73초
- **연관 실험**: → Experiment 4 (real LIBERO M0) → Experiment 5 (factor sweep)

## 검증한 가설
SE(3) manifold-preserving loss (SO(3) geodesic + gripper CE) **+** contact-phase 3× reweight이 naive Euclidean MSE 대비 **contact-rich** trajectory에서는 student action 정확도를 크게 개선하고 **contact-poor** trajectory에서는 영향이 거의 없다 (mechanism specificity).

## 방법
- **데이터**: 합성 SE(3) trajectory 200개 × T=50 timesteps (robot env 없음)
  - Contact-rich: peg-in-hole 스타일 (후반부 회전 크게 변화)
  - Contact-poor: reach-and-move (회전 변화 작고 smooth)
- **Loss variant**: `l2_naive`, `l2_fixed` (quaternion double-cover 보정), `so3_geodesic`
- **Reweight**: contact-phase 3× 가중 on/off
- **모델**: Tiny MLP student (~50K params)
- **Metric**: contact phase vs free phase 회전 오차 (degrees)

## 핵심 결과

| 조건 | Contact phase 회전 오차 |
|---|---|
| Contact-rich + L2-naive | **7.25°** |
| Contact-rich + L2-fixed + reweight | **3.40°** |
| Contact-rich + SO(3) + reweight | 3.52° |
| Contact-poor + L2-naive | 3.02° |
| Contact-poor + SO(3) + reweight | 2.96° |

- Contact-rich 개선: **3.72°** (51% 감소)
- Contact-poor 개선: 0.055°
- **Specificity ratio: 67×**

## 중요 발견

### 1. Headline Pivot 필요
**L2-fixed + reweight (3.40°) ≈ SO(3) + reweight (3.52°)** — SO(3) geodesic은 quaternion double-cover 보정된 L2 위에서 사실상 0 기여. 진짜 main contribution은 **contact-phase reweighting**, SE(3) manifold 보존이 아님.

→ 이름 변경: AMP-Distill → **CPR-Distill** (Contact-Phase Reweighted Distillation)

### 2. Quaternion Double-Cover 보정만으로 ~1° 효과
L2-naive (부호 정렬 없음) 7.25° → L2-fixed (가까운 hemisphere) 6.12° → contact reweight 추가 3.40°. Double-cover 보정은 baseline에 포함되어야 할 cheap engineering.

### 3. Specificity는 실재 (synthetic 한정)
Contact-rich와 contact-poor 사이 67× 비율은 메커니즘이 의도한 곳에서 작동함을 보여주는 가장 깔끔한 증거.

## Direction (이 실험의 의미)

- **논문 기여로는 부족**: 합성 데이터는 실제 LIBERO action 분포를 반영하지 않음. 다음 단계인 real data로 즉시 이동.
- **무엇을 열어주는가**: Validator의 "real benchmark에서 ≥2pp gain" gate. 비용을 들이기 전에 mechanism specificity가 진짜라는 신호.
- **무엇을 금지하는가**: "SE(3) manifold-preserving"을 main contribution으로 주장하면 안 됨. SO(3) geodesic은 ablation으로 강등.

## 한계 / 주의사항
- 합성 SE(3) trajectory는 실제 로봇 운동학 제약, contact dynamics, 시각 관측을 반영하지 않음
- Free-phase degradation 같은 tradeoff가 여기서는 보이지 않음 — Experiment 4에서 비로소 드러남
- Specificity ratio (67×)는 깨끗한 합성 setup이라 부풀려진 값. 실제 데이터에선 더 낮을 것

## 다음 단계
→ **Experiment 4** (real LIBERO에서 sham control 포함 M0 smoke test)

## 파일
- `poc.py` — 스크립트
- `results.json` — 측정값
- `run.log` — 실행 로그
- (체크포인트 없음 — 합성 실험이라 모델 폐기)
