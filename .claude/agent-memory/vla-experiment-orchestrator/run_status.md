---
name: cpr-distill-campaign-2026-05-15
description: First orchestrated CPR-Distill experiment campaign — 5 follow-up experiments after Sweep
metadata:
  type: project
  status: inconclusive_pending_multiseed_or_decision
---

# CPR-Distill Campaign — 2026-05-15

## 실행 summary

- 시작: 2026-05-15
- 실행 시간: ~10분 wall-clock (5개 실험 병렬, 4× B200)
- Halt 시점: 2026-05-15 (multi-suite + multi-seed 결과가 핵심 가설을 흔듦)

## 실행된 milestone

| Milestone | 실험 폴더 | 상태 | 결과 요약 |
|---|---|---|---|
| Adaptive boost fix | `fail/cpr/cpr_distill_adaptive_v2/` | ❌ FAIL | reg=0 + bias init도 collapse |
| M3.5 mask quality | `fail/cpr/cpr_distill_mask_quality/` | ✅ PASS | gripper_transition이 최선 |
| Multi-seed significance | `fail/cpr/cpr_distill_multiseed/` | ⚠️ MIXED | contact 5.97σ ✅, overall null ⚠️ |
| Window sensitivity | `fail/cpr/cpr_distill_window_sweep/` | ✅ PASS | window 선택에 둔감 |
| Multi-suite generalization | `fail/cpr/cpr_distill_multisuite/` | ❌ CRITICAL | 1.5× 일반화 실패 |

## Campaign 확장 (Round 2): Detection Recovery (Experiment 11-15)

5번째 reframe 후 → diagnostic 발견 → 새 detector로 재검증

| Milestone | 실험 폴더 | 상태 | 결과 |
|---|---|---|---|
| Per-suite deep dive | `fail/cpr/cpr_distill_per_suite_analysis/` | ⚠️ CRITICAL | gripper-transition 0회 검출 발견 |
| Contact detection diagnostic | `fail/cpr/cpr_distill_contact_diagnostic/` | ✅ BREAKTHROUGH | channel_diff +10.26%, threshold 0.02 = bug |
| Multi-seed × channel_diff | `fail/cpr/cpr_distill_channeldiff_multiseed/` | ✅ Mechanism 8σ / ⚠️ Overall null | |
| Multi-suite × channel_diff | `fail/cpr/cpr_distill_channeldiff_multisuite/` | ⚠️ Suite-dependent | |
| Combined detector | `fail/cpr/cpr_distill_combined_detector/` | 🟡 Union > individual (single-seed) | |

**Total 15 실험 완료, ~30분 wall-clock**

## Halt 조건

**Type**: Ambiguous fork — paper story 근본적 reframe 필요

**근거**:
1. Multi-seed: 원래 sweep의 "+1.55% overall"이 single-seed noise였음 (multi-seed Δ=0)
2. Multi-suite: factor=1.5× sweet spot이 4 suite 중 0개에서 일관 (모두 factor=1.0이 overall best)
3. 그러나 contact-specific mechanism은 5.97σ로 통계적으로 매우 유의

**자동 pivot 불가 이유**:
- Headline 변경: "Contact-rich 전용" vs "General method" 선택은 user judgment
- 다음 단계 옵션 4개 모두 valid (Option A/B/C/D in `fail/cpr/cpr_distill_multisuite/README.md`)
- 다음 단계가 ≥4 GPU-hours 잠재 비용 (W1 sim rollout) — auto launch 금지

## 다음 user decision 후 가능한 path

- **Path A**: contact-rich 전용 reframe + libero_spatial/goal 집중 → 즉시 sim rollout
- **Path B**: sim rollout으로 task success 검증 후 결정
- **Path C**: per-suite factor 분석 (factor=1.0이 best인 이유 deep dive)
- **Path D**: CPR-Distill 폐기, 다른 idea로 pivot

---

## Round 3 (2026-05-15 → 2026-05-16): Sim Rollout 실행 (Path B)

User 선택: 큰 student model로 sim rollout 재실행 (Path B의 강화판).

| Milestone | 실험 폴더 | 상태 | 결과 |
|---|---|---|---|
| State-repr audit | `fail/cpr/cpr_distill_sim_rollout/sim_eval_v2.py` | ✅ Bug found | scipy as_rotvec [0,π] canonicalisation flips sign vs robosuite quat2axisangle 2·acos(w) |
| TinyBC + state fix (v2) | `results_v2.json` | ❌ All 0 SR | TinyBC 5M params + 3ep 자체가 약함 |
| MediumBC ResNet18 dual-view (v3) | `results_v3.json` | ⚠️ **Inconclusive (n=30, single-seed)** | baseline 6.7%, **CPR 10.0%, sham 10.0%** — CPR matches sham within sample noise; multi-seed discriminator needed |

**Total 16 실험 완료, ~95분 wall-clock 추가**

## v3 Finding (단정 X — multi-seed 필요)

| Level | CPR vs baseline | Sham vs baseline | CPR vs Sham |
|---|---|---|---|
| Contact MSE | -12.2% | -2.4% (~noise) | **+9.8pp gap (specificity ✓)** |
| Task SR | +3.3pp | +3.3pp | **0.0pp at n=30** (single-seed, inconclusive) |

**중요 caveats** (advisor 권고로 추가):
- n=30/condition × single-seed → Fisher's exact baseline 2/30 vs CPR 3/30: p≈0.5 (유의 X)
- Per-task identity (task 2,3 둘 다 ↑, task 6,9 둘 다 ↓)는 **same seed=42** 영향일 수도 — CPR/sham
  weight perturbation이 같은 trajectory를 따라 학습 → 동일 task에서 success/fail 상관 큼
- Baseline SR=6.7%는 published 단순 BC (~15-30%)보다 낮음 → gripper bimodality (BC + MSE → 그리퍼
  중간값 학습) 가능성. 이 regime은 contact-precision이 결정적이지 않음 (grasp 자체가 안 됨).
- 따라서 "CPR=Sham at SR" 결론은 강하게 단정 불가; multi-seed 1h 추가 시 진짜 discriminator

## Halt 조건 (Round 3)

**Type**: Inconclusive at n=30 single-seed — multi-seed가 cheap & 결정적

**근거**:
1. CPR과 sham의 SR 차이 0pp는 결정적이지 않음 (1 episode = 3.3pp)
2. Per-task 동일 패턴은 same-seed로 부분 설명 가능
3. 1h 추가 compute로 multi-seed 결론 가능 — 다른 path보다 훨씬 저렴

## 다음 user decision 옵션 (Round 3, advisor 권고 반영)

| Option | 액션 | 비용 | 기대효과 |
|---|---|---|---|
| **A (권장)** | **Multi-seed × 3 seeds × 3 conditions = 9 trainings (각 6분) + rollouts (~10분 each)** | **~1h GPU** | **CPR vs sham 진짜 discriminator** |
| B | 큰 sample (10ep/task × 3 seeds) + 더 큰 모델 | 5h GPU | A 결과 보고 결정 |
| C | Action chunking + tanh gripper로 baseline SR 30%+ 끌어올린 뒤 재실험 | 1주 | A 결과 보고 결정 |
| D | Contact-rich-only suite로 좁힘 | 3-4일 | A 결과 보고 결정 |
| E | CPR-Distill 폐기 | - | A 결과 후 명확한 기각 시

---

## Round 4 (2026-05-16): Multi-Seed v4 결과 (Option A 실행)

User가 Option A 승인. 3 seeds (42/43/44) × 3 conditions 병렬 실행 (GPU 0/1/2), ~2h wall-clock (CPU contention으로 v3 예상보다 느림).

### v4 핵심 결과 (libero_spatial, 3 seeds × 30 ep/condition)

| Condition | Per-seed SR (verbatim) | Pooled (참고용) |
|---|---|---|
| baseline | seed42 6.7, seed43 3.3, seed44 13.3 % | 7/90 |
| **channel_diff_1.5** | seed42 6.7, seed43 6.7, seed44 **20.0** % | **10/90** |
| sham_3.0 | seed42 **13.3**, seed43 6.7, seed44 3.3 % | 7/90 |

**Per-seed Δ(CPR - sham) (verbatim, n=3)**: `{-6.7, 0.0, +16.7}` pp
- median = 0, range -6.7..+16.7. Pattern는 sub-mode 2개 (-6.7 seed42, +16.7 seed44) + tie (seed43)로
  **bimodal에 가까움** — mean ± std (3.3 ± 12.0)로 summary하는 것은 오해 소지.

**올바른 통계 framing** (advisor 지적):
- Pooled Fisher's exact (7/90 vs 10/90)은 **wrong test**: 같은 seed의 30 episode는 같은 weight 공유 →
  i.i.d. Bernoulli 가정 위반. Pooled p값 인용하지 말 것.
- 옳은 분석: paired test (per-seed Δ, n=3). 그러나 **n=3는 underpowered** (Cohen's d ~ 0.4 검출
  needed ~50 paired obs at power 0.8) — 결과는 "n=3 cannot discriminate", not "p>0.05".

### v3 → v4 변화 (중요, 단정 X)
- **v3 (seed 42 only)**: CPR 10.0%, sham 10.0% → "CPR=sham" 우려. **v4에서 뒤집힘**:
  - seed42 단독으로는 sham(13.3%) > CPR(6.7%), 즉 seed42는 sham에 유리한 outlier였음
  - seed44에서는 정반대 (CPR 20%, sham 3.3%)
  - → single-seed verdict는 어느 쪽이든 **unsafe**
- **v4 결론**: 3 seeds로는 CPR vs sham 어느 쪽이 우세한지 **결정 불가** (per-seed 결과 bimodal).
  v3 critical-negative 우려는 살아 있지도 죽지도 않음.

### 추가 관찰 (advisor 지적)
- **CPR std (7.7%) > baseline std (5.1%)** — CPR이 단순히 mean을 옮기는 게 아니라 **variance도 추가**.
  "Reliable contact behavior" pitch에 역행하는 신호. paper claim이 stability라면 더 큰 문제.

### Halt 조건 (Round 4)

**Type**: Inconclusive at n=3 seeds — per-seed pattern은 bimodal, 결정 못함

**근거**:
1. Per-seed Δ가 -6.7/0/+16.7로 spread, 모드 일관성 없음
2. n=3 paired test는 effect size 3pp 검출에 underpowered (~50 obs 필요)
3. Pooled Fisher는 부적절한 test (statistical fishing)

### 다음 user decision 옵션 (Round 4, advisor 권고 반영)

advisor 노트: paired test for d=0.4 (3pp effect / 7pp seed std) at α=0.05 power=0.8는 **~50 paired obs** 필요.
8 seeds (Option A)도 여전히 underpowered 가능성 큼. Option B (episodes 증가)가 per-seed n 자체를 늘려서
seed-internal noise를 줄임 → power 더 효율적.

| Option | 액션 | 비용 | 기대효과 | 정직한 평가 |
|---|---|---|---|---|
| A | 5 추가 seeds (총 8 seeds, 4 GPU 병렬, ~3h) | ~3h GPU | n=8 paired test | 여전히 underpowered 가능성 ↑. Bimodal pattern이 noise인지 진짜인지 봄 |
| **B (권장)** | **episodes/task 3→10, 같은 3 seeds, pooled 300/cond** | **~5-6h GPU** | per-seed n 3.3x → seed-internal CI 좁힘 | 효과 크기 정확도 ↑, paired test도 약간 강화 |
| C | Action chunking + tanh gripper로 baseline SR 30%+ → effect size 검증 더 유리 | 1주 | 다른 regime, 새 baseline | 가장 robust한 fix지만 비쌈 |
| D | Contact-rich custom suite (libero_object subset 등) | 3-4일 | mechanism 더 명확 | scope narrowing — niche claim |
| E | CPR-Distill 폐기 | - | n=3에서 mechanism이 인상적이지 않음 | 정직한 선택 |

---

## Round 5 (2026-05-16): v5 (Option B) 실행 결과

User Option B 선택. v4 ckpts reuse + episodes/task 3→10 (per-seed 100/cond, pooled 300/cond). ~4h wall-clock.

### v5 핵심 결과

| Condition | seed42 | seed43 | seed44 | Mean | Std |
|---|---|---|---|---|---|
| baseline | 9.0% | 5.0% | 8.0% | 7.33% | 2.08% |
| **channel_diff_1.5** | **3.0%** | 7.0% | **14.0%** | 8.00% | **5.57%** |
| sham_3.0 | **15.0%** | **14.0%** | 8.0% | 12.33% | 3.79% |

### Per-seed paired Δ (primary analysis)

- Δ(CPR - baseline): {-6, +2, +6} pp — median +2pp, paired-t **p=0.868**, Wilcoxon **p=0.750**
- Δ(CPR - sham):     {-12, -7, +6} pp — median -7pp, paired-t **p=0.504**, Wilcoxon **p=0.500**
- Δ(Sham - baseline): {+6, +9, 0} pp — median +6pp, paired-t **p=0.199**, Wilcoxon p=0.500 (closest to effect)

### Honest verdict (advisor 교정 반영)

1. **n=3 paired tests cannot discriminate**. p≥0.5는 **literally null result**, refutation 아님.
   "Sham > CPR" 결론은 v4의 "CPR > sham" 오류와 같은 방향 (반대 부호) — 둘 다 unsafe.
2. **v4 → v5는 reversal 아님, CI 좁아짐**. v4의 2/30 binomial CI [0%, 16%]는 v5의 3/100 = 3.0% 포함.
   같은 underlying probability에서 wider → tighter sample로 갱신.
3. **Regime mismatch가 진짜 문제**: baseline SR 7.3%면 failure 93%가 **reach failure** (gripper bimodality:
   BC+MSE → gripper command 평균값 학습 → grasp 자체가 거의 안됨). **CPR mechanism은 contact-precision용**
   → reach 단계에서는 작용할 여지 없음. 이 regime에서는 **더 많은 seed/episode로 CPR 못 테스트함**.
4. **seed44는 clean CPR-specific win** (CPR 14% > sham 8% = baseline 8%) — 1개 seed지만 보존할 finding.
   Across seeds, CPR은 **high-variance perturbation** (std 5.57% vs baseline 2.08%) — init이 mechanism보다 결과를 더 결정.

### Halt 조건 (Round 5)

**Type**: Regime mismatch — CPR mechanism이 작동할 수 있는 setup에서 테스트 필요. 추가 sample은 무의미.

**근거**:
1. Failure mode의 93%가 CPR이 도울 수 없는 reach failure (gripper bimodality)
2. Seed-init noise가 mechanism effect보다 큼 (std 5.57 vs effect 0.67)
3. **Option A/B 추가 실행 무의미** — regime 자체가 잘못됨. Option C 또는 D만 valid.

### 다음 user decision 옵션 (Round 5)

| Option | 액션 | 비용 | 기대효과 |
|---|---|---|---|
| **C (advisor 강추)** | **Action chunking (chunk=16, exec all) + tanh gripper head (BCE loss separate from MSE)로 baseline SR 30%+ 끌어올린 뒤 CPR/sham 재검증** | 1주 (구현 + 학습 + rollout) | CPR mechanism이 실제로 작용할 regime. 진짜 답 가능 |
| D | Contact-rich custom suite (libero_spatial peg-in-hole 등 contact-heavy subset) — 같은 regime 유지 but mechanism 더 명확하게 stress | 3-4일 | scope narrowing, workshop tier 가능 |
| E | CPR-Distill 폐기 (다른 idea로 pivot) | - | "현 regime에서 답 못 함" 정직한 선택. MSE-level 5-8σ만으로 workshop paper 시도는 가능 |

Option A/B는 advisor가 "wrong regime에서 더 많은 데이터는 답 못 줌"으로 **명시 제외**.
