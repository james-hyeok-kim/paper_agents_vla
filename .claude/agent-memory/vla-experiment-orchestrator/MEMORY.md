# Agent Memory Index — vla-experiment-orchestrator

## Active / Recent Campaigns
- [CPR-Distill 5-experiment follow-up](run_status.md) — 2026-05-15, halted at multi-suite generalization fork

## Completed Campaigns
- (없음)

## Patterns Learned (재사용 가능 insight)
- 5 light experiments × 4 GPUs → wall-time ≈ longest experiment (8 min)
- libero_spatial single-seed result 신뢰하지 말 것 (multi-seed/multi-suite로 즉시 검증 권장)
- Adaptive boost via end-to-end loss는 collapse — bilevel needed
- Gripper-transition contact proxy가 velocity-drop보다 우월 (LIBERO에서)
