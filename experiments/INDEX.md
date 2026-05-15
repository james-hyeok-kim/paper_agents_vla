# VLA Experiments Index

VLA inference efficiency 연구 실험 모음. 4× NVIDIA B200 GPU 환경.

**Last updated**: 2026-05-15

---

## 실행된 실험 (Week-1 Feasibility PoC)

3개 CONDITIONAL GO 아이디어의 핵심 가설을 합성/소형 모델로 빠르게 검증. 총 소요시간 ~90초.

| # | Idea | 실험 | Verdict | Output |
|---|---|---|---|---|
| 1 | **CPR-Distill** (구 AMP-Distill) | SE(3) loss × contact-phase reweight 합성 SE(3) trajectory | ✅ **PASS** (67x specificity) | `amp_distill/results.json` |
| 2 | **XV-Dedup** | DINOv2 cross-view token overlap + LSH bucketing | 🟡 **PARTIAL** (1/3 gates, core hypothesis confirmed) | `xv_dedup/results.json` |
| 3 | **CP-Sparse** | Tiny ACT chunk-position attention entropy | ❌ **FAIL** (원안 가설 미입증) | `cp_sparse/results.json` |
| 4 | **CPR-Distill M0** (real LIBERO) | TinyBC + LIBERO-spatial + 4-condition ablation (sham control 포함) | ✅ **PASS** (CPR vs sham Δ +17.07pp) | `cpr_distill_m0/results.json` |
| 5 | **CPR-Distill Sweep** (reweight factor + adaptive) | Reweight factor sweep {1, 1.5, 2, 2.5, 3, 4, 5} + adaptive learnable boost | ✅ **PASS** (Sweet spot 1.5x: overall +1.55% — **multi-seed에서 null로 정정**) | `cpr_distill_sweep/results.json` |
| 6 | **CPR-Distill Adaptive v2** | Adaptive collapse fix (reg=0 + bias init) | ❌ **FAIL** — 구조적 문제, bilevel 필요 | `cpr_distill_adaptive_v2/results.json` |
| 7 | **CPR-Distill Mask Quality** | 4 contact mask variant 비교 | ✅ gripper_transition가 최선 | `cpr_distill_mask_quality/results.json` |
| 8 | **CPR-Distill Multi-Seed** | 4 conditions × 5 seeds significance | ✅ **5.97σ contact gain** / ⚠️ overall null | `cpr_distill_multiseed/results.json` |
| 9 | **CPR-Distill Window Sweep** | window ±{1,3,5,7} sensitivity | ✅ robust (5.9-7.9%) | `cpr_distill_window_sweep/results.json` |
| 10 | **CPR-Distill Multi-Suite** | 4 LIBERO suite × 3 factor generalization | ❌ **factor=1.0이 모든 suite에서 overall best** | `cpr_distill_multisuite/results.json` |
| 11 | **CPR-Distill Per-Suite Deep Dive** | Suite statistics + libero_10/object fine sweep | ⚠️ **gripper-transition 0회 검출** 발견 — 진짜 "contact"가 아니었음 | `cpr_distill_per_suite_analysis/results.json` |
| 12 | **CPR-Distill Contact Diagnostic** | LIBERO gripper 데이터 분석 + 6 detector 비교 | ✅ **BREAKTHROUGH**: gripper_channel_diff로 contact gain +10.26%, overall +1.48% | `cpr_distill_contact_diagnostic/results.json` |
| 13 | **CPR-Distill Multi-seed × channel_diff** | 3 conditions × 5 seeds (significance check) | ✅ **Contact 8σ** ⭐ / ⚠️ Overall null (-0.01σ) | `cpr_distill_channeldiff_multiseed/results.json` |
| 14 | **CPR-Distill Multi-suite × channel_diff** | 4 suites × 3 factors with channel_diff | ⚠️ Suite-dependent (libero_10 prefers factor=3.0) | `cpr_distill_channeldiff_multisuite/results.json` |
| 15 | **CPR-Distill Combined Detector** | Union/intersection of channel_diff + velocity_drop | 🟡 Union slightly best (single-seed) | `cpr_distill_combined_detector/results.json` |

---

## Experiment 1: CPR-Distill PoC

**Hypothesis**: VLA distillation 시 action loss를 contact phase에서 3x reweight하면, contact-rich task에서 student의 rotation 정확도가 baseline 대비 크게 향상되며, contact-poor task에서는 영향 없음 (mechanism specificity).

### Setup
- 합성 SE(3) trajectory 200개, T=50 timesteps
- Loss variants: L2-naive / L2-fixed (double-cover) / SO(3) geodesic
- Reweight: 1x (uniform) vs 3x (contact phase)

### Key Numbers

| Setting | Contact-phase rotation error |
|---|---|
| L2-naive | 7.25° |
| L2-fixed + reweight | **3.40°** |
| SO(3) + reweight | 3.52° |

- Contact-rich gain: **3.72°** (51% 감소)
- Contact-poor gain: 0.055°
- **Specificity ratio: 67x**

### Critical Finding (Headline Pivot)
L2-fixed + reweight ≈ SO(3) + reweight (3.40 vs 3.52) — 즉 **contact-phase reweighting이 main contribution**, SO(3) geodesic은 secondary.

Headline 재정의: "Contact-Phase Reweighted Distillation for VLA" (구 "SE(3) Manifold-Preserving Distillation"에서 축소).

### Files
- `amp_distill/poc.py` — PoC 스크립트
- `amp_distill/results.json` — 측정 데이터
- `amp_distill/run.log` — 실행 로그

---

## Experiment 2: XV-Dedup PoC

**Hypothesis**: Multi-camera VLA에서 cross-view 간 30-50%의 ViT patch token이 중복되며, LSH bucketing으로 병합 가능.

### Setup
- DINOv2-base (256 patches/view, patch_size=14)
- 8개 합성 scene × 2 views (16px viewpoint shift + jitter)
- LSH: 32/64/128-dim random projection

### Gate Results

| Gate | Target | Result | Status |
|---|---|---|---|
| A: Cross-view overlap @ cos>0.85 | ≥30% | **70.4%** | ✅ PASS |
| B: LSH random-projection agreement | ≥50% | 29.6% (H=128) | ❌ FAIL (InfoNCE 학습 필요) |
| C: 40% token 감소 → prefill latency | ≥25% | 21.2% | ❌ FAIL (직전) |

### Conclusion
Core hypothesis 강력하게 검증됨 (70.4% overlap, 예상의 2.3x). LSH는 random projection이 부족 → InfoNCE 학습이 필수 다음 단계.

### Files
- `xv_dedup/poc.py`, `xv_dedup/results.json`, `xv_dedup/run.log`

---

## Experiment 3: CP-Sparse PoC

**Hypothesis**: ACT/SmolVLA에서 후반 chunk position이 더 sharp한 cross-attention을 가져서 position-dependent sparsification 가능.

### Setup
- Tiny ACT (d_model=128, chunk_len=50, ctx_len=64)
- 합성 chunk-prediction task, 500 iter 학습

### Gate Results

| Gate | Target | Result | Status |
|---|---|---|---|
| A: Entropy slope | end-to-end Δ > 0.3 nats | +0.075 nats (slope 0.004) | ❌ FAIL |
| B: Top-8 mass at late positions | >0.8 | 0.41 | ❌ FAIL |
| C: Argmax Jaccard (pos0 vs pos49) | <0.7 → GO | **0.06** | ✅ GO signal |

### Conclusion
원안 가설 ("later position = sharper attention") 미입증. 단, AutoHorizon의 "intra-chunk invariance"도 미관측 (Jaccard 0.06이 매우 낮음).

새 관찰: 각 position이 **다른 작은 subset**에 attending하되 **subset 크기는 일정**. CP-Sparse 원안 폐기 또는 "Position-conditioned uniform-k sparse attention"로 pivot 필요.

### Caveat
합성 데이터의 position-conditional dependency 구조가 모델이 position-specific attention을 학습하도록 강제. Real SmolVLA on LeRobot data에서 재측정 없이는 결정 불가.

### Files
- `cp_sparse/poc.py`, `cp_sparse/results.json`, `cp_sparse/run.log`

---

## Experiment 4: CPR-Distill M0 Smoke Test on Real LIBERO ✅

**Hypothesis (revised after PoC)**: Contact-phase reweighting이 실제 LIBERO-spatial action distribution에서도 contact MSE를 낮추며, 단순 uniform reweight (sham control)은 같은 효과를 내지 못한다.

### Setup
- LIBERO-spatial 10 tasks × 50 demos (450 train + 50 val, 55,970 train timesteps)
- Contact phase = gripper state transitions ± 3 timesteps (LIBERO에 F/T 없어 proxy 사용)
- TinyBC (~5M params: tiny CNN + MLP) — 1.7분 학습 가능
- 3 epochs × 4 conditions on single B200

### Gate Results

| Gate | Target | Result | Status |
|---|---|---|---|
| A: CPR reduces contact MSE vs L2-naive | reduction > 0 | **17.1%** | ✅ PASS |
| **B: CPR vs Sham 3x specificity** | **Δ ≥ 2pp** | **+17.07pp** | ✅ **PASS (8.5x 초과)** |
| C: Contact-specific gain | contact > free | 17.1% > -19.6% | ✅ PASS |

### Critical Insight: Sham Control 검증

| Condition | Contact MSE | vs A: contact gain |
|---|---|---|
| A: L2-naive | 0.843 | — |
| **B: CPR-3x (PROPOSED)** | **0.699** | **+17.1%** |
| C: Sham uniform 3x | 0.843 | +0.04% (사실상 0) |
| D: L2-fixed only | 0.842 | +0.2% |

→ **Reweighting alone does NOT work. Contact-specificity is the mechanism.** Validator의 가장 중요한 challenge "이게 그냥 weight 3배 준 것 아닌가?"에 대한 명확한 답.

### 발견된 Tradeoff

- CPR-3x는 free phase에서 **-19.6% 손해** (free MSE 0.585 → 0.699)
- Overall MSE는 baseline보다 약간 나쁨 (0.663 → 0.699)
- 합성 PoC에서 안 보이던 tradeoff가 real LIBERO에서 드러남 → 합성 PoC 한계 입증

### 다음 milestone (validator 권고)
1. **Reweight factor sweep**: 1.5x / 2x / 5x 비교 → tradeoff sweet spot
2. **Adaptive weight**: learnable contact-phase weight으로 free phase에 손해 없이 contact 부스트
3. **W1 시작 가능**: ActDistill 실제 setup 또는 SmolVLA full distillation
4. **M3.5**: GT contact (real F/T) vs predicted vs gripper-only 비교 (proxy quality)

### Files
- `cpr_distill_m0/m0_smoke.py` — 스크립트
- `cpr_distill_m0/results.json` — 측정값
- `cpr_distill_m0/run.log` — 실행 로그
- `cpr_distill_m0/m0_results.png` — 시각화 (epoch curve + bar chart)
- `/data/jameskimh/cpr_distill_m0/{A,B,C,D}_*.pt` — 4 모델 checkpoint (10MB total)

---

## Experiment 5: CPR-Distill Reweight Factor Sweep ✅

**Hypothesis**: Reweight factor sweep으로 contact-phase gain과 free-phase tradeoff의 sweet spot을 찾는다. 또한 learnable adaptive weight이 fixed factor를 능가하는지 검증.

### Setup
- Same TinyBC + LIBERO-spatial 환경
- 7 factor settings: {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}
- 1 adaptive condition: learnable boost via MLP head, gated by contact_mask

### Key Findings

**Sweet Spot @ factor=1.5x** — 유일한 Net Positive 설정:
- Contact MSE: 0.868 → **0.797** (+8.15% gain)
- Free MSE: 0.588 → 0.604 (-2.69% loss)
- **Overall MSE: 0.673 → 0.663 (+1.55%)** ← 모든 metric 개선

| Factor | Contact gain | Free loss | Overall | Verdict |
|---|---|---|---|---|
| 1.5 | +8.15% | -2.69% | **+1.55%** | ⭐ Sweet spot |
| 2.0 | +13.68% | -8.17% | +0.37% | Reasonable |
| 3.0 | +18.27% | -17.22% | -3.35% | M0 default, suboptimal |
| 5.0 | +25.72% | -30.31% | -8.41% | Too aggressive |

### Linear Pareto Frontier
Contact gain과 free loss는 거의 1:1 trade. No free lunch.

### ❌ Adaptive Learnable Weight FAILED
- 학습 중 평균 boost α가 0.05 → 0.0005로 collapse
- 원인: weight regularization (1e-3)이 boost를 0으로 누르는 게 유리한 dynamic
- **Fix 가능**: regularization 제거 / EMA-stabilized boost / contact-aware initialization

### Publishable Headline Revised
**"CPR-Distill at factor=1.5x improves overall action MSE by 1.55% via contact-specific reweighting, with sham-control beating margin of 8pp."**

### Files
- `cpr_distill_sweep/sweep.py` — 스크립트
- `cpr_distill_sweep/results.json` — 측정값
- `cpr_distill_sweep/run.log` — 실행 로그
- `cpr_distill_sweep/sweep_results.png` — Pareto curve + MSE vs factor
- `/data/jameskimh/cpr_distill_sweep/{factor_*.pt, adaptive.pt}` — 8 checkpoint

---

## 다음 단계 실험 (Planned)

### CPR-Distill Main Experiment (CoRL 2027 target)

**Validator 판정**: CONDITIONAL GO 6.0/10

**필수 추가 milestone**:
- **M0** (Week 0, pre-W1): Real LIBERO-LONG 24h smoke test → ≥2pp gain 확인. 실패 시 NO-GO. 합성 PoC는 gate 충족 안 됨.
- **M2.5** (Week 3): **Sham reweight control** (uniform 3x) — contact-specificity 진짜인지 핵심 입증
- **M3.5** (Week 3-4): Contact mask quality ablation (GT vs predicted vs gripper-transition only)
- **M5.5** (Week 5-6): Contact-poor real-robot task 1개 추가 (sim artifact 아님을 실물에서 입증)

**Timeline**: 6주 (W0 smoke test 포함 시 7주, optimistic)

**Compute**: must-have 450 GPU-hrs, all-in 650 GPU-hrs (validator의 30% buffer 권고 반영)

**Venue**: CoRL 2027 (NeurIPS 2026 deadline 이미 지남)

**Detailed plan**: `.claude/agent-memory/vla-experiment-planner/active/plan_cpr_distill.md`
**Validation**: `.claude/agent-memory/vla-idea-validator/conditional/validation_cpr_distill_experiment_plan.md`

### XV-Dedup (Pending)
Core hypothesis 검증 후 InfoNCE-learned LSH projector + π0 multi-camera 실험으로 확장.

---

## Hardware Profile

- **GPUs**: 4× NVIDIA B200 (183 GB each)
- **CUDA**: 13.0
- **PyTorch**: 2.9.1+cu130
- **Storage**: 1.0 TB available

## Software Stack

- `transformers` 4.57.6 (DINOv2 등)
- `torchvision` 0.24.1
- `einops`, `numpy`, `scipy`
- (미설치) `lerobot`, `pytorch3d`, `roma` — 필요 시 manual PyTorch impl로 대체
