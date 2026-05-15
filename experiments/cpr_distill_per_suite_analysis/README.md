# Experiment 11 — CPR-Distill Per-Suite Deep Dive (CRITICAL FINDING)

## Metadata
- **날짜**: 2026-05-15
- **Tier**: Deep-dive descriptive analysis + suite-specific factor sweep
- **상태**: ⚠️ CRITICAL — 우리가 measuring한 게 실제로는 "contact"가 아니었음
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 401초

## 검증한 가설
Multi-suite generalization 실패 (libero_object/10에서 효과 약함)의 **원인**을 통계적으로 진단한다. Suite별 statistics와 fine-grained factor sweep으로 mechanism을 정확히 이해.

## 방법
1. **Descriptive statistics** (4 suite 전체):
   - Contact mask density, gripper transition rate, trajectory length, action variance (contact vs free)
2. **Fine factor sweep**: libero_10, libero_object에 factor ∈ {1.0, 1.1, 1.2, 1.5, 2.0}
3. **Correlation analysis**: suite statistics ↔ contact gain @ 1.5×

## 핵심 결과

### Suite Statistics

| Metric | libero_spatial | libero_object | libero_goal | libero_10 |
|---|---|---|---|---|
| n_demos | 500 | 500 | 500 | 500 |
| traj_len_mean | 124.5 | 149.0 | 127.5 | **276.2** |
| contact_density (last 30%) | 0.304 | 0.303 | 0.304 | 0.302 |
| **transitions_per_demo** | **0.0** | **0.0** | **0.0** | **0.0** |
| frac_demos_with_no_transition | **1.0** | **1.0** | **1.0** | **1.0** |
| action_var_ratio (contact/free) | 0.719 | 0.727 | 0.734 | 0.684 |

### Fine Factor Sweep on Weakest Suites

**libero_10** (multisuite에서는 +0.05% gain @ 1.5×):

| Factor | Contact | Δ |
|---|---|---|
| 1.0 | 0.588 | — |
| 1.1 | 0.577 | **+1.91%** |
| 1.5 | 0.563 | **+4.26%** |
| 2.0 | 0.528 | **+10.3%** |

→ 이전 multisuite 결과 (+0.05%)는 single-seed noise. 실제 effect는 4-10%.

**libero_object** (multisuite +1.23% gain @ 1.5×):

| Factor | Contact | Δ |
|---|---|---|
| 1.0 | 0.724 | — |
| 1.5 | 0.766 | **-5.78%** ❌ |
| 2.0 | 0.812 | **-12.1%** ❌ |

→ Reweight가 contact MSE를 **악화**시킴. libero_object는 메커니즘이 다름.

### Correlation Analysis (suite feature ↔ gain @ 1.5×)

| Feature | Correlation |
|---|---|
| traj_len_mean | **-0.79** (긴 trajectory = 약한 효과) |
| contact_density | +0.87 (단 거의 일정해서 의미 약함) |
| action_var_ratio | +0.59 |

## 중요 발견

### 1. 🚨 Gripper Transition은 0번 검출됨 (모든 suite, 모든 demo)
`|Δgripper| > 0.02` threshold가 한 번도 trigger 안 됨. 모든 실험에서 **fallback (last 30%)** 만 사용됐음.

**의미**: 우리가 "contact-phase reweighting"이라 부른 것은 사실 **"trajectory-end reweighting"**.

이전 결과 재해석:
- Window sweep robustness: window가 의미 없었음 (transition 없으니)
- Mask density 0.304 동일: 모두 fallback
- gripper_transition ≈ window_last30 결과: 둘이 같은 mask였음
- velocity_drop만 다른 결과: 유일하게 진짜 다른 mask 사용

### 2. Multi-suite Single-Seed Noise 입증
libero_10에서:
- Multisuite (single seed): contact 0.544 @ factor=1.0, 0.543 @ factor=1.5 → +0.05%
- Per-suite (single seed, 다른 split): 0.588, 0.563 → +4.26%

→ Single-seed로는 어느 쪽도 신뢰 불가. **Multi-suite도 multi-seed로 재측정 필요**.

### 3. libero_object는 진짜로 다른 분포
Per-suite와 multisuite 모두에서 reweighting이 negative or marginal. Pick-and-place dominant task structure에서 "trajectory-end"는 release/withdrawal phase일 가능성, 학습 강화가 도움 안 됨.

### 4. Trajectory 길이가 효과를 결정
libero_10 (276 timesteps) >> spatial/goal (~125). 긴 trajectory에서 last 30%는 더 다양한 sub-action을 포함 → "trajectory-end"의 의미 묽어짐.

## Direction (이 실험의 의미)

### 논문 스토리 근본적 재정의 (3번째 reframe)

**1st (originally)**: AMP-Distill — SE(3) manifold preserving
**2nd (after PoC)**: CPR-Distill — Contact-phase reweighted (sham control 통과 시점에서 정직했음)
**3rd (now)**: **TER-Distill — Trajectory-End-Reweighted distillation** (gripper transition 검출 실패 발견 이후 정직한 이름)

### 즉시 해야 할 것

1. **Gripper threshold 재검증**: LIBERO의 실제 gripper_states 값 분포 분석 → 적절한 threshold 찾기
2. **진짜 contact detection 구현**:
   - Vision-based contact detector (object proximity)
   - End-effector velocity drop (per-suite analysis에서 발견한 신호)
   - Reward signal에서 detection
3. **Multi-suite + multi-seed**: 현재 결과는 모두 single-seed에 의존, 신뢰 불가

### 살아남는 contribution

Multi-seed에서 5.97σ로 검증된 "**late-phase reweighting**" 효과는 여전히 real. 단 framing이 정직해야:
> "We find that distilling student VLAs with elevated weight on trajectory-end phases (last 30% of execution) gives statistically significant action accuracy improvement in those phases. This generalizes to spatial/goal task suites but not object-reaching or long-horizon multi-task suites."

### 죽는 contribution
- "Contact-specific reweighting" (sham 입증 결과는 진짜지만 mechanism interpretation은 wrong)
- "Gripper-transition proxy로 contact detection"
- "Universal sweet spot at 1.5×"

## 한계 / 주의사항
- Gripper threshold 0.02가 부적절한지 vs LIBERO에 actual gripper transition이 없는지 미분리
- Per-suite 결과도 single-seed
- Action variance ratio < 1 의미 unclear (late phase가 더 stereotyped이라는 가설)

## 다음 단계

### 즉시 검증 가능
1. **Diagnostic**: gripper_states 분포 plot → 적절한 threshold 또는 transition 부재 확인
2. **Better contact detection**: EE velocity drop을 contact proxy로 정상 동작하게 fix
3. **Multi-seed + multi-suite**: 4 suite × 3 factor × 5 seed = 60 runs (~1시간)

### 더 큰 스케일
- Sim rollout (Option B) — 진짜 SR 측정
- 새로운 contact detection (vision or learned)

## 파일
- `per_suite_analysis.py` — 스크립트
- `results.json` — 4 suite 통계 + libero_10/object sweep
- `run.log` — 실행 로그
- `stats_vs_effect.png` — 4 feature × effect scatter
- `fine_sweep.png` — libero_10, libero_object factor sweep
