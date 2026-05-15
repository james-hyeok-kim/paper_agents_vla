---
name: cpr-distill-campaign-2026-05-15
description: First orchestrated CPR-Distill experiment campaign — 5 follow-up experiments after Sweep
metadata:
  type: project
  status: halted_at_decision_fork
---

# CPR-Distill Campaign — 2026-05-15

## 실행 summary

- 시작: 2026-05-15
- 실행 시간: ~10분 wall-clock (5개 실험 병렬, 4× B200)
- Halt 시점: 2026-05-15 (multi-suite + multi-seed 결과가 핵심 가설을 흔듦)

## 실행된 milestone

| Milestone | 실험 폴더 | 상태 | 결과 요약 |
|---|---|---|---|
| Adaptive boost fix | `cpr_distill_adaptive_v2/` | ❌ FAIL | reg=0 + bias init도 collapse |
| M3.5 mask quality | `cpr_distill_mask_quality/` | ✅ PASS | gripper_transition이 최선 |
| Multi-seed significance | `cpr_distill_multiseed/` | ⚠️ MIXED | contact 5.97σ ✅, overall null ⚠️ |
| Window sensitivity | `cpr_distill_window_sweep/` | ✅ PASS | window 선택에 둔감 |
| Multi-suite generalization | `cpr_distill_multisuite/` | ❌ CRITICAL | 1.5× 일반화 실패 |

## Campaign 확장 (Round 2): Detection Recovery (Experiment 11-15)

5번째 reframe 후 → diagnostic 발견 → 새 detector로 재검증

| Milestone | 실험 폴더 | 상태 | 결과 |
|---|---|---|---|
| Per-suite deep dive | `cpr_distill_per_suite_analysis/` | ⚠️ CRITICAL | gripper-transition 0회 검출 발견 |
| Contact detection diagnostic | `cpr_distill_contact_diagnostic/` | ✅ BREAKTHROUGH | channel_diff +10.26%, threshold 0.02 = bug |
| Multi-seed × channel_diff | `cpr_distill_channeldiff_multiseed/` | ✅ Mechanism 8σ / ⚠️ Overall null | |
| Multi-suite × channel_diff | `cpr_distill_channeldiff_multisuite/` | ⚠️ Suite-dependent | |
| Combined detector | `cpr_distill_combined_detector/` | 🟡 Union > individual (single-seed) | |

**Total 15 실험 완료, ~30분 wall-clock**

## Halt 조건

**Type**: Ambiguous fork — paper story 근본적 reframe 필요

**근거**:
1. Multi-seed: 원래 sweep의 "+1.55% overall"이 single-seed noise였음 (multi-seed Δ=0)
2. Multi-suite: factor=1.5× sweet spot이 4 suite 중 0개에서 일관 (모두 factor=1.0이 overall best)
3. 그러나 contact-specific mechanism은 5.97σ로 통계적으로 매우 유의

**자동 pivot 불가 이유**:
- Headline 변경: "Contact-rich 전용" vs "General method" 선택은 user judgment
- 다음 단계 옵션 4개 모두 valid (Option A/B/C/D in `cpr_distill_multisuite/README.md`)
- 다음 단계가 ≥4 GPU-hours 잠재 비용 (W1 sim rollout) — auto launch 금지

## 다음 user decision 후 가능한 path

- **Path A**: contact-rich 전용 reframe + libero_spatial/goal 집중 → 즉시 sim rollout
- **Path B**: sim rollout으로 task success 검증 후 결정
- **Path C**: per-suite factor 분석 (factor=1.0이 best인 이유 deep dive)
- **Path D**: CPR-Distill 폐기, 다른 idea로 pivot
