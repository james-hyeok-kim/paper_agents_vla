# VLA Experiments Index

VLA inference efficiency 연구 실험 모음. 4× NVIDIA B200 GPU 환경.

**Last updated**: 2026-05-17

## Folder Structure

```
experiments/
├── INDEX.md          ← this file
├── fail/             ← polished off / abandoned ideas
│   ├── cpr/          ← 14 CPR-Distill folders (Exp 1, 4-21)
│   ├── cp_sparse/    ← Exp 3 FAIL
│   └── xv_dedup/     ← Exp 2 PARTIAL → abandoned
├── wip/              ← work-in-progress experiments
└── success/          ← published / publishable experiments
```

---

## 실행된 실험 (Week-1 Feasibility PoC)

3개 CONDITIONAL GO 아이디어의 핵심 가설을 합성/소형 모델로 빠르게 검증. 총 소요시간 ~90초.

| # | Idea | 실험 | Verdict | Output |
|---|---|---|---|---|
| 1 | **CPR-Distill** (구 AMP-Distill) | SE(3) loss × contact-phase reweight 합성 SE(3) trajectory | ✅ **PASS** (67x specificity) | `fail/cpr/amp_distill/results.json` |
| 2 | **XV-Dedup** | DINOv2 cross-view token overlap + LSH bucketing | 🟡 **PARTIAL** (1/3 gates, core hypothesis confirmed) | `fail/xv_dedup/results.json` |
| 3 | **CP-Sparse** | Tiny ACT chunk-position attention entropy | ❌ **FAIL** (원안 가설 미입증) | `fail/cp_sparse/results.json` |
| 4 | **CPR-Distill M0** (real LIBERO) | TinyBC + LIBERO-spatial + 4-condition ablation (sham control 포함) | ✅ **PASS** (CPR vs sham Δ +17.07pp) | `fail/cpr/cpr_distill_m0/results.json` |
| 5 | **CPR-Distill Sweep** (reweight factor + adaptive) | Reweight factor sweep {1, 1.5, 2, 2.5, 3, 4, 5} + adaptive learnable boost | ✅ **PASS** (Sweet spot 1.5x: overall +1.55% — **multi-seed에서 null로 정정**) | `fail/cpr/cpr_distill_sweep/results.json` |
| 6 | **CPR-Distill Adaptive v2** | Adaptive collapse fix (reg=0 + bias init) | ❌ **FAIL** — 구조적 문제, bilevel 필요 | `fail/cpr/cpr_distill_adaptive_v2/results.json` |
| 7 | **CPR-Distill Mask Quality** | 4 contact mask variant 비교 | ✅ gripper_transition가 최선 | `fail/cpr/cpr_distill_mask_quality/results.json` |
| 8 | **CPR-Distill Multi-Seed** | 4 conditions × 5 seeds significance | ✅ **5.97σ contact gain** / ⚠️ overall null | `fail/cpr/cpr_distill_multiseed/results.json` |
| 9 | **CPR-Distill Window Sweep** | window ±{1,3,5,7} sensitivity | ✅ robust (5.9-7.9%) | `fail/cpr/cpr_distill_window_sweep/results.json` |
| 10 | **CPR-Distill Multi-Suite** | 4 LIBERO suite × 3 factor generalization | ❌ **factor=1.0이 모든 suite에서 overall best** | `fail/cpr/cpr_distill_multisuite/results.json` |
| 11 | **CPR-Distill Per-Suite Deep Dive** | Suite statistics + libero_10/object fine sweep | ⚠️ **gripper-transition 0회 검출** 발견 — 진짜 "contact"가 아니었음 | `fail/cpr/cpr_distill_per_suite_analysis/results.json` |
| 12 | **CPR-Distill Contact Diagnostic** | LIBERO gripper 데이터 분석 + 6 detector 비교 | ✅ **BREAKTHROUGH**: gripper_channel_diff로 contact gain +10.26%, overall +1.48% | `fail/cpr/cpr_distill_contact_diagnostic/results.json` |
| 13 | **CPR-Distill Multi-seed × channel_diff** | 3 conditions × 5 seeds (significance check) | ✅ **Contact 8σ** ⭐ / ⚠️ Overall null (-0.01σ) | `fail/cpr/cpr_distill_channeldiff_multiseed/results.json` |
| 14 | **CPR-Distill Multi-suite × channel_diff** | 4 suites × 3 factors with channel_diff | ⚠️ Suite-dependent (libero_10 prefers factor=3.0) | `fail/cpr/cpr_distill_channeldiff_multisuite/results.json` |
| 15 | **CPR-Distill Combined Detector** | Union/intersection of channel_diff + velocity_drop | 🟡 Union slightly best (single-seed) | `fail/cpr/cpr_distill_combined_detector/results.json` |
| 16 | **CPR-Distill Sim Rollout v2/v3** | MediumBC ResNet18 dual-view + state-repr fix → MSE→SR translation check | ⚠️ **Inconclusive at n=30 single-seed**: CPR 10.0%, sham 10.0%, baseline 6.7% (CPR=sham within noise; same seed=42 makes per-task identity partially expected) — multi-seed needed to discriminate | `fail/cpr/cpr_distill_sim_rollout/results_v3.json`, `README.md` |
| 17 | **CPR-Distill Sim Rollout v4 multi-seed** | 3 seeds × 3 conditions × 30 ep/cond on libero_spatial | ⚠️ **Inconclusive at n=3 seeds**: per-seed Δ(CPR-sham) `{-6.7, 0, +16.7}` pp (bimodal); paired test underpowered (~50 obs needed for d=0.4). Pooled Fisher (p=0.31) is wrong test. CPR std > baseline std (variance concern). | `fail/cpr/cpr_distill_sim_rollout/results_v4_aggregated.json` |
| 18 | **CPR-Distill Sim Rollout v5 (3 seeds × 100 ep)** | Reuse v4 ckpts; 3 seeds × 3 conditions × 100 ep/cond = pooled 300/cond | ⚠️ **Still inconclusive (n=3 paired)**: per-seed SR 안정 (CI 좁아짐); means baseline 7.3%, CPR 8.0%, sham 12.3%. Δ(CPR-sham) {-12, -7, +6} pp (Wilcoxon p=0.500). Δ(Sham-base) {+6, +9, 0} pp (paired-t p=0.199, 가장 가까움). v4→v5는 reversal 아닌 CI 좁힘. **Regime mismatch (SR 7%는 reach failure 지배, CPR이 작용할 contact 단계 미도달)** → 더 많은 episode/seed로는 답 못 함. Option C (action chunking) 필요. seed44는 clean CPR win (14% vs 8%/8%) — bimodal seed pattern 그 자체가 finding | `fail/cpr/cpr_distill_sim_rollout/results_v5_aggregated.json` |
| 19 | **CPR-Distill Sim Rollout v6 (chunked BC + BCE gripper + state norm)** | Action chunk=16, BCE gripper head, tanh arm, state z-score. 3 seeds × 3 conditions × 30 ep/task = 300 ep/cond. + per-task contact_fraction correlation analysis (advisor 권고) | ⚠️ **Regime fix 성공 but mechanism null**: baseline SR 67.4% (v5의 9.5x), CPR 67.2%, sham 68.7% — paired-t p≥0.34 모두 null. Per-task contact_frac 0.21-0.28 narrow range → contact-selectivity 깨끗하게 테스트 불가. r(contact_frac, CPR-base)=-0.47 (narrow range, p=0.17), r(contact_frac, Sham-base)=-0.68 (**p=0.03**) → sham의 음의 상관은 mechanism이 아닌 **effective-LR scaling** 시사 (validator의 원래 우려 empirically supported). **MSE 5-8σ specificity는 진짜지만 SR로 translate 안 됨 (in BC-with-chunking regime, demo target)**. Teacher-VLA distillation regime은 미테스트 — prior는 낮아지지만 refute 아님 | `fail/cpr/cpr_distill_sim_rollout/results_v6_aggregated.json`, `results_v6_contact_correlation.json`, `v6_contact_correlation.png` |
| 20 | **CPR-Distill Sim Rollout v7 (libero_10 generalization)** | v6 code on libero_10 (long-horizon, multi-stage, wider contact_frac range 0.11-0.29). 3 seeds × 3 conditions × 30 ep/task = 300 ep/cond | ⚠️ **Regime fix holds + first interpretable mechanism signal (one suite)**: baseline 76.4%, CPR 77.4%, sham 78.8% — paired-t p≥0.43 모두 null overall. **Per-task Spearman ρ(contact_density, CPR-sham) = +0.69, raw p=0.028 (n=10)** — survives baseline-SR confound (partial r=+0.50) but **not** Bonferroni-adjusted (adj p≈0.08). **Interpretation correction**: "CPR is less *harmful* than sham on contact-rich tasks" (Δ(CPR-base) 상관 = -0.03), NOT "CPR helps." Suggestive enough to require libero_goal replication before workshop write-up | `fail/cpr/cpr_distill_sim_rollout/results_v7_aggregated.json`, `results_v7_contact_correlation.json`, `v7_contact_correlation.png` |
| 21 | **CPR-Distill Sim Rollout v8 (libero_goal replication of v7)** | Same v6/v7 chunked BC setup on libero_goal (third suite; episodes 128-299, low-SR regime). 3 seeds × 3 conditions × 30 ep/task | ❌ **v7's signal does NOT replicate**: baseline mean 10.0%, CPR 8.4%, sham 8.6% — paired tests p≥0.18 all null. **Spearman ρ(contact, CPR-sham) = -0.02 (p=0.96)** vs v7's +0.69 → v7 was **one-suite fluke / outlier-driven**. Pearson +0.31 (vs v7 +0.59) is consistent in direction but not significant. **4-suite picture**: across low-SR (v5 7%, v8 10%) AND high-SR (v6 67%, v7 76%) regimes, CPR does not consistently outperform baseline or sham at task SR. **MSE-level 5-8σ specificity does NOT translate to SR in BC-with-chunking regime across any suite tested**. Negative result for CPR-Distill's main task-SR thesis is now comprehensive | `fail/cpr/cpr_distill_sim_rollout/results_v8_aggregated.json`, `results_v8_contact_correlation.json`, `v8_contact_correlation.png` |

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
- `fail/cpr/amp_distill/poc.py` — PoC 스크립트
- `fail/cpr/amp_distill/results.json` — 측정 데이터
- `fail/cpr/amp_distill/run.log` — 실행 로그

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
- `fail/xv_dedup/poc.py`, `fail/xv_dedup/results.json`, `fail/xv_dedup/run.log`

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
- `fail/cp_sparse/poc.py`, `fail/cp_sparse/results.json`, `fail/cp_sparse/run.log`

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
- `fail/cpr/cpr_distill_m0/m0_smoke.py` — 스크립트
- `fail/cpr/cpr_distill_m0/results.json` — 측정값
- `fail/cpr/cpr_distill_m0/run.log` — 실행 로그
- `fail/cpr/cpr_distill_m0/m0_results.png` — 시각화 (epoch curve + bar chart)
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
- `fail/cpr/cpr_distill_sweep/sweep.py` — 스크립트
- `fail/cpr/cpr_distill_sweep/results.json` — 측정값
- `fail/cpr/cpr_distill_sweep/run.log` — 실행 로그
- `fail/cpr/cpr_distill_sweep/sweep_results.png` — Pareto curve + MSE vs factor
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
