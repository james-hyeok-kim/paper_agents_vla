# Agent Memory Index — vla-experiment-planner

## Active Plans (실험 진행 중, active/, 2개)
- [CPR-Distill Main Experiment](active/plan_cpr_distill.md) — CoRL 2027 target, 6주 timeline, 340 GPU-hrs (validator 권고 시 450)
- [ACE Track 1 PoC](active/plan_ace_track1.md) — Asymmetric per-camera encoder (RN18 wrist + RN8 static); 55 GPU-hr, 3 gates (M-1 variance, M-2 sham 5pp/n=3, M-3 multi-suite); Track 2 gated on M-3 PASS

## Completed Plans (completed/, 0개)
- (없음)

## Reference (compute calibration, infra notes, reference/, 1개)
- [B200 Compute Calibration](reference/compute_calibration_b200.md) — 4× B200 GPU 시간 추정 (OpenVLA-7B teacher + LIBERO student)
