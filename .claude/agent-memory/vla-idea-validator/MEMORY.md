# Agent Memory Index — vla-idea-validator

## Passed (GO, passed/, 0개)
- (없음 — 현재까지 모든 idea가 CONDITIONAL GO)

## Conditional (CONDITIONAL GO with pre-experiment gates, conditional/, 4개)
- [AMP-Distill](conditional/validation_amp_distill.md) — 6.5/10, contact-phase reweight ablation gate
- [XV-Dedup](conditional/validation_xv_dedup.md) — 6.0/10, BFA++ stacking + safety reframe
- [CP-Sparse](conditional/validation_cp_sparse.md) — 5.5/10, multi-metric Week-1 entropy gate
- [CPR-Distill Experiment Plan](conditional/validation_cpr_distill_experiment_plan.md) — 6.0/10, M0 LIBERO smoke test + sham control 추가 필요

## Failed (NO-GO, failed/, 0개)
- (없음 — 현재까지 모든 validation이 CONDITIONAL 이상)

## Patterns (재사용 가능한 failure heuristics, patterns/, 2개)
- [VLA Efficiency Failure Patterns](patterns/patterns_vla_efficiency.md) — 반복 등장 weakness 체크리스트
- [Validation Patterns](patterns/validation_patterns_vla_efficiency.md) — Validation 시 사용할 expanded checklist
