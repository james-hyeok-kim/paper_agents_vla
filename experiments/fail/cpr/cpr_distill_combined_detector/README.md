# Experiment 15 — Combined Detector (Union/Intersection of channel_diff + velocity_drop)

## Metadata
- **날짜**: 2026-05-15
- **Tier**: Detector ablation — single signal vs combined
- **상태**: 🟡 Union slightly better than individuals, but single-seed only
- **연결 아이디어**: `cpr-distill`
- **GPU**: 1× B200, 121초

## 검증한 가설
Action-level signal (gripper_channel_diff)과 proprio-level signal (ee_velocity_drop)이 **상호 보완적**이라면, 둘의 union이 개별 detector보다 contact gain을 강화할 것이다.

## 방법
- factor=1.5 고정
- 4 detector 비교:
  - gripper_channel_diff 단독
  - ee_velocity_drop 단독
  - **union** (둘 중 하나라도 contact이라고 하면 contact)
  - **intersection** (둘 다 contact이라고 해야 contact)

## 핵심 결과

| Detector | Contact MSE | Δ Contact | Δ Overall | Density |
|---|---|---|---|---|
| baseline (factor=1.0) | 0.852 | — | — | — |
| gripper_channel_diff | 0.821 | +3.63% | -2.84% | 0.253 |
| ee_velocity_drop | 0.898 | **-5.37%** | -3.16% | 0.266 |
| **union** | **0.798** | **+6.40%** | -3.32% | 0.394 |
| intersection | 1.094 | **-28.37%** | -2.01% | 0.125 |

## 중요 발견

### 1. ⭐ Union이 individual보다 contact gain 강함
gripper_channel_diff 단독 +3.63% → union +6.40%. EE velocity와 결합 시 ~1.8× 강화.

### 2. ee_velocity_drop 단독은 이 run에서 fail
Contact gain -5.37% (악화). Single-seed로는 신뢰 불가지만, 이전 Experiment 12에서는 +1.34%였음. **Variance 큼** → multi-seed 필요.

### 3. Intersection은 실패 (-28% contact)
Density 12.5%로 너무 좁음. 좁은 mask에 학습이 집중되어 그 영역의 sample을 overfit. **너무 보수적인 contact detection은 부작용**.

### 4. ⚠️ Single-seed Variance 큼
Experiment 12 (다른 run, same setup): gripper_channel_diff +10.26%
이 run: gripper_channel_diff +3.63%
→ Single-seed로 detector 비교 신뢰 불가. Multi-seed (Exp 13)이 확정 결론을 줌.

### 5. Multi-seed (Exp 13)와 비교
Multi-seed: channel_diff contact gain mean=6.81% (= (0.866-0.807)/0.866)
이 run: channel_diff contact gain=3.63%
→ 이 run은 multi-seed mean보다 ~0.5σ 낮은 fluctuation.

## Direction (이 실험의 의미)

### Union이 진짜 더 강한가? (다음 검증 필요)
Single run에서는 union > channel_diff alone (+6.40% vs +3.63%). 하지만 single-seed라 단정 어려움.

**향후 검증 방법**:
- 5-seed multi-seed on {channel_diff, velocity_drop, union, intersection}
- 만약 union이 일관되게 best → cross-modal detection이 paper에 추가 contribution

### 즉시 시사하는 것
- Mask density가 너무 작으면 (12%) overfit 위험 (intersection fail)
- 적정 density: 25-40% 범위가 안정적
- 두 signal source를 결합하는 design space가 존재

## 한계 / 주의사항
- **Single-seed only — 결과 신뢰도 낮음**. Multi-seed 재실험 필요.
- libero_spatial only
- 두 detector만 비교 (action_gripper_cmd 등 다른 detector와 union 시도 안 함)

## 다음 단계 (deferred)
- Union vs channel_diff 비교를 multi-seed로 재검증 (5 seeds, ~5분)
- Multi-suite × union — long-horizon suite에서도 union이 강한지

## 파일
- `combined.py` — 스크립트
- `results.json` — 4 detector × baseline 비교
- `run.log` — 실행 로그
- `combined.png` — bar chart (contact/free/overall gain per detector)
