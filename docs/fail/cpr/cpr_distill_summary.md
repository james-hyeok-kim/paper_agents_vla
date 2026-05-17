# CPR-Distill 연구 종합 정리

**기간**: 2026-05-15 ~ 2026-05-17 (3일, 4× B200 GPU)
**최종 상태**: ⚠️ MSE-level mechanism 살아있음 / Task-SR mechanism 4 suites comprehensive negative

---

## 1. 아이디어

### 원안 (validator 승인)
> **Contact-Phase Reweighted Distillation for VLA**
> Teacher VLA (OpenVLA-7B/π0) → Student VLA distillation 시 action MSE loss에서
> **contact phase에 1.5x weight**를 줘서 student가 contact-rich 동작을 우선 학습.
> 목표: student inference 1.5-2x speedup, task SR 95%+ 유지.

### 핵심 차별점
- 단순 reweighting이 아닌 **contact-specific**임을 sham(uniform 3x reweight) control로 입증
- validator 우려: "그냥 gradient scale 키운 거 아니냐?" → sham이 답

### 실험에서의 단순화
시간/비용 절약 위해 **teacher VLA 대신 human demo action**을 target으로 사용 (BC + reweight).
가설: reweighting mechanism은 target이 teacher든 demo든 동일하게 작동해야 함.

---

## 2. 실험 전체 (21개)

### Phase 1: PoC (Exp 1-3) — Week-1 feasibility

| # | 실험 | 결과 |
|---|---|---|
| 1 | CPR-Distill 합성 SE(3) trajectory PoC | ✅ 67x specificity |
| 2 | XV-Dedup (cross-view dedup) PoC | 🟡 1/3 gates, 폐기 |
| 3 | CP-Sparse (chunk-position sparse attn) PoC | ❌ FAIL, 폐기 |

→ CPR-Distill만 살아남아 집중 검증.

### Phase 2: Real LIBERO MSE-level (Exp 4-9)

| # | 실험 | 결과 |
|---|---|---|
| 4 | M0 smoke: 4 conditions × TinyBC × libero_spatial | ✅ CPR vs sham **+17pp** contact gain |
| 5 | Reweight factor sweep {1.0, 1.5, 2, 3, 4, 5} + adaptive | ⚠️ 1.5x sweet spot (single-seed; multi-seed null) |
| 6 | Adaptive learnable weight (regularization fix) | ❌ FAIL (boost collapse) |
| 7 | Contact mask quality ablation (4 detector variants) | ✅ gripper_transition 최선 (당시) |
| 8 | Multi-seed (4 conditions × 5 seeds) | ✅ **Contact 5.97σ** / ⚠️ overall null |
| 9 | Window sensitivity ±{1,3,5,7} | ✅ robust (5.9-7.9%) |

### Phase 3: Generalization failures (Exp 10-11)

| # | 실험 | 결과 |
|---|---|---|
| 10 | Multi-suite (4 LIBERO × 3 factor) | ❌ factor=1.0이 모든 suite best — generalization 실패 |
| 11 | Per-suite deep dive (libero_10/object) | ⚠️ **gripper_transition detector가 libero_10에서 0회 검출** — 진짜 contact 못 잡고 있었음 |

### Phase 4: Diagnostic + recovery (Exp 12-15)

| # | 실험 | 결과 |
|---|---|---|
| 12 | Contact detection diagnostic (6 detector 비교) | 🎯 **BREAKTHROUGH**: gripper_channel_diff 새 detector로 contact +10.26%, overall +1.48% |
| 13 | channel_diff × multi-seed | ✅ **Contact 8σ** specificity / ⚠️ overall null |
| 14 | channel_diff × multi-suite | ⚠️ Suite-dependent (libero_10은 factor=3.0 선호) |
| 15 | Combined detector (union/intersection) | 🟡 Union slightly best (single-seed) |

### Phase 5: Sim Rollout — MSE → SR translation (Exp 16-21) ← 핵심

| # | 실험 | 결과 |
|---|---|---|
| 16 | v1-v3: TinyBC → MediumBC dual-view (single-seed, libero_spatial) | ⚠️ Inconclusive at n=30. State-repr 버그 발견/수정 (scipy as_rotvec vs robosuite quat2axisangle) |
| 17 | v4: 3 seeds × 30 ep, paired test | ⚠️ Bimodal Δ(CPR-sham) {-6.7, 0, +16.7}, p=0.31 |
| 18 | v5: 3 seeds × 100 ep (pooled 300/cond) | ⚠️ Bimodal {-12, -7, +6}, p=0.50. **Regime mismatch**: baseline SR 7%면 failures가 reach (gripper bimodality), CPR은 contact용 |
| 19 | **v6: Chunked BC + BCE gripper + state norm** | ✅ **Regime fix 성공**: baseline 67.4% (9.5x ↑). CPR=baseline=sham (p≥0.34). r(contact, Sham-base)=-0.68 (p=0.03) → reweighting = effective-LR scaling, not contact-specific |
| 20 | v7: libero_10 (long-horizon, baseline 76%) | 🟡 **첫 mechanism signal**: Spearman ρ(contact, CPR-sham) = +0.69 (p=0.028), partial r=+0.50 |
| 21 | **v8: libero_goal replication** | ❌ **REPLICATION FAIL**: Spearman -0.02 (vs v7's +0.69). v7 = one-suite fluke |

---

## 3. 핵심 결과 (Sim Rollout, regime fix 후)

### 4 suites × 3 conditions × 3 seeds (900 ep/cond)

| Suite | Regime | baseline | CPR_1.5 | sham_3.0 | Paired-t Δ(CPR-base) |
|---|---|---|---|---|---|
| **v5 spatial** | Low SR (7%) | 7.33% | 8.00% | 12.33% | p=0.87 (null) |
| **v6 spatial** | High SR (67%) | 67.44% | 67.22% | 68.67% | p=0.97 (null) |
| **v7 libero_10** | High SR (76%) | 76.44% | 77.44% | 78.78% | p=0.56 (null) |
| **v8 libero_goal** | Low SR (10%) | 10.00% | 8.44% | 8.56% | p=0.18 (null) |

→ **4 suites × 2 regimes 모두 paired test null**. CPR이 baseline 또는 sham을 task SR로 능가한다는 증거 없음.

### Per-task contact correlation (mechanism test, n=10 per suite)

| Suite | r(contact, CPR-sham) | Spearman ρ | 해석 |
|---|---|---|---|
| v6 spatial | +0.37 (p=0.30) | -0.18 | narrow range로 power 부족 |
| v7 libero_10 | **+0.59 (p=0.07)** | **+0.69 (p=0.028)** | 첫 positive signal |
| v8 libero_goal | +0.31 (p=0.38) | **-0.02 (p=0.96)** | **v7 replicate 실패** |

→ v7의 ρ=+0.69가 v8에서 ρ=-0.02로 사라짐 → **v7은 outlier-driven fluke**.

---

## 4. 결론

### ✅ 살아있는 contribution (workshop tier)
1. **MSE-level contact specificity 5-8σ** (multi-seed 확실)
2. **CPR vs sham contact MSE gap 9-10pp** at MSE level
3. **gripper_channel_diff detector** (기존 gripper_transition은 libero_10에서 fail)
4. **State-repr bug fix** (scipy as_rotvec → robosuite quat2axisangle convention)
5. **v6 chunked BC pipeline** (single-step BC 7% → chunked 67% in 1 week)

### ❌ 죽은 contribution
- "CPR이 task SR을 contact-specifically 개선" — 4 suites × 2 regimes comprehensive null
- "Reweighting의 mechanism이 contact-specific" — sham이 더 좋거나 비슷 (v6 sham=68.7 > CPR=67.2)
- "1.5x factor universal sweet spot" — multi-suite에서 무너짐
- "Adaptive learnable weight" — collapse

### 핵심 mechanism insight (validator의 원래 우려 입증)
- Sham (uniform 3x) ≈ CPR (contact-specific 1.5x) at SR level
- v6에서 sham의 contact-correlation r=-0.68 (p=0.03) → reweighting이 **effective-LR scaling**처럼 작동
- → CPR의 "contact specificity" 주장은 mechanism이 아닌 gradient-scale 효과

### 미테스트 (scope 한계)
- **Teacher-VLA distillation** (original paper plan) — BC simplification으로 우회
  - Prior는 낮아졌으나 strict refutation 아님
- libero_object suite (low-priority, contact-poor)
- Real robot

---

## 5. 다음 가능 path

| Option | 비용 | 평가 |
|---|---|---|
| (a) Negative result workshop paper | 2주 | 4-suite 종합 증거로 clean negative — 권장 |
| (b) Teacher-VLA distillation (original plan) | 3주 | BC 결과로 prior 낮음, high-risk |
| (c) Pivot to other idea (Temporal Delta KV Cache 등) | - | 정직한 선택 |
| (d) libero_object 추가 후 (c) | 4h + (c) | 완전 4-suite picture 후 폐기 |

---

## 6. Artifacts

- **Experiments**: `experiments/cpr_distill_{m0, sweep, multiseed, multisuite, contact_diagnostic, channeldiff_*, combined_detector, sim_rollout}/`
- **Sim rollout pipeline**: `experiments/cpr_distill_sim_rollout/{sim_eval_v6.py, sim_eval_v7.py, sim_eval_v8.py, aggregate_*, analyze_*_contact_corr.py}`
- **Master index**: `experiments/INDEX.md`
- **Orchestrator status**: `.claude/agent-memory/vla-experiment-orchestrator/run_status.md`
- **Plots**: `experiments/cpr_distill_sim_rollout/v{6,7,8}_contact_correlation.png`

---

*Generated 2026-05-17. 21 experiments, ~30 GPU-hours total.*
