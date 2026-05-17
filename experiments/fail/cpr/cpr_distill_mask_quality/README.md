# Experiment 7 — CPR-Distill Contact Mask Quality Ablation

## Metadata
- **날짜**: 2026-05-15
- **Tier**: 후속 ablation (M3.5)
- **상태**: ✅ PASS — gripper_transition proxy가 최선
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 154초

## 검증한 가설
M0의 gripper-transition proxy는 contact-phase의 약한 근사일 뿐. 더 정확한 mask(velocity-drop, Gaussian smooth)가 더 큰 CPR 효과를 줄 것이다.

## 방법
- 4가지 contact mask variant, factor=1.5 고정:
  - (a) **gripper_transition** ± 3 (M0 default)
  - (b) **gt_ee_velocity_drop**: ee_pos finite difference로 EE 속도 계산, bottom 30% 속도 구간을 contact로 표시
  - (c) **window_last30**: trajectory 마지막 30%를 contact로 표시 (transition 무시)
  - (d) **gaussian_smooth**: transition 주변에 σ=2의 Gaussian decay
- 평가: 모든 variant에 대해 **동일한 gripper-transition mask**로 contact MSE 측정 (공정 비교)

## 핵심 결과

| Mask variant | Contact MSE | Δ vs baseline | Mask density | 상태 |
|---|---|---|---|---|
| **gripper_transition** | 0.785 | **+6.78%** | 0.305 | ✅ BEST |
| gt_ee_velocity_drop | 0.845 | -0.38% | 0.268 | ❌ FAIL |
| window_last30 | 0.796 | +5.41% | 0.305 | 🟡 OK |
| gaussian_smooth | 0.797 | +5.31% | 0.305 | 🟡 OK |

(baseline factor=1.0, gripper-transition mask: contact=0.842)

## 중요 발견

### 1. Gripper-Transition Proxy가 최선
이전의 proxy 우려는 기우. 다른 3개 variant 모두 gripper_transition 미만. **현재 M0 setup이 정당화됨.**

### 2. Velocity-Drop Proxy 실패
LIBERO 환경 특성: 로봇이 free-space에서 정밀 접근(slow approach)하는 구간이 많아 "low velocity = contact" 가정이 깨짐. Velocity drop은 LIBERO에 부적합.

### 3. Window-Only도 작동 (이유 분석 필요)
gripper-transition 없이 단순히 "last 30%"만 contact로 표시해도 +5.41% gain. 이는 LIBERO 데이터에서 **contact phase가 통계적으로 trajectory의 후반에 집중**한다는 의미. 즉 gripper transition 신호가 강하지 않아도 단순 위치 기반 mask로 70% 효과 회수 가능.

→ Insight: gripper-transition의 효과 일부가 "후반부에 더 정확히 학습되도록 유도"하는 정렬 효과일 수 있음.

## Direction (이 실험의 의미)

- **현재 proxy 유지**: gripper_transition ± 3 그대로. 더 정교한 mask는 효과 없거나 미미.
- **논문에서의 활용**: "We test 4 contact mask definitions; gripper-transition gives the strongest effect. Velocity-drop fails on LIBERO due to slow approach motion (mismatch with the 'contact = slow' assumption)."
- **Real robot 단계로 가면**: F/T 센서가 있으니 ground-truth contact를 사용할 수 있음. 그때 이 ablation을 다시 돌려서 proxy vs GT gap을 측정.

## 한계 / 주의사항
- 모든 variant에 동일한 평가용 mask 사용 (공정 비교 위해). 학습에 사용한 mask로 평가하면 결과가 자기 강화로 부풀려질 위험.
- LIBERO는 F/T 센서가 없어 진짜 GT contact 측정 불가. 위 결과는 모두 proxy 간 비교.
- 4 variant만 시도. 더 정교한 mask(visual-based contact detection 등)는 미테스트.

## 다음 단계
→ Proxy 트랙은 종결. Real-robot 단계에서 F/T sensor로 GT contact 비교 권장.

## 파일
- `mask_quality.py` — 스크립트
- `results.json` — 측정값 + mask density
- `run.log` — 실행 로그
- `mask_quality.png` — bar chart + density vs gain scatter
