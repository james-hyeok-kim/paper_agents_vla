---
name: run-status-ace-track1
description: ACE Track 1 campaign live status (M-1 → M-2 → M-3)
metadata:
  type: project
  status: active
  campaign: ace-track1
  started: 2026-05-17
---

# ACE Track 1 Campaign — Live Status

## Campaign overview
- Plan: `/home/jovyan/workspace/paper_agents_vla/.claude/agent-memory/vla-experiment-planner/active/plan_ace_track1.md`
- Goal: Verify Asymmetric Camera-role Encoder (wrist=RN18, static=RN8) wins by ≥5pp over reverse (sham) on ≥2 LIBERO suites.
- Budget: 55 GPU-hr (Track 1 only)
- Folder policy: artifacts in `experiments/wip/ace_track1/`, large data in `/data/jameskimh/ace_track1/`.

## Milestone DAG

| Milestone | Status | GPU-hr | Wall time | Notes |
|---|---|---|---|---|
| M-1 premise verify | RUNNING | 1.5 (est) | ~15 min | 3 suites × 10 ep/task × 10 tasks = 300 ep, parallel on GPUs 0/1/2 |
| Calibration | PENDING | 2 | ~40 min | gated by M-1 PASS |
| M-2 sham (libero_spatial) | PENDING | 22 | ~6 h | gated by calibration + M-1 PASS |
| M-3 multi-suite (libero_object) | PENDING | 22 | ~6 h | gated by M-2 PASS |
| Latency bench | PENDING | 1 | ~20 min | end-stage |
| Aggregate + plots + README | PENDING | 0.5 | ~30 min | end-stage |

## M-1 launch (2026-05-17)

- Launcher: `experiments/wip/ace_track1/launch_m1.sh`
- Checkpoint reuse: `/data/jameskimh/cpr_distill_sim_rollout/baseline_v6_seed42.pt` (single ckpt across all 3 suites — variance is a camera-geometry property, not task-specific)
- Suites: libero_spatial (GPU 0), libero_object (GPU 1), libero_goal (GPU 2)
- Expected wall-clock: 10-15 min
- Background IDs: b0snu049n (spatial), buysnfkfa (object), bhtu9zuij (goal)

## Gate criteria (locked from plan)

### M-1
- PASS: ratio ≥ 2.0× on ≥3 suites → proceed
- SOFT FAIL: ratio 1.5-2.0× on majority → ResNet12 pivot (auto-execute)
- HARD FAIL: ratio < 1.5× on majority → KILL ACE, halt to user

### M-2
- PASS: B−C ≥ 5pp AND paired-t p ≤ 0.05 AND bootstrap 95% CI > 0 AND |B−A| ≤ 3pp → proceed M-3
- SOFT FAIL: B−C in [3pp, 5pp) → stop (default) or n=5 seed extension
- HARD FAIL: B−C < 3pp OR p > 0.10 OR |B−A| > 5pp → KILL ACE

### M-3
- PASS: combined paired-t B−C ≥ 5pp AND wrist-precision category ≥ 5pp → Track 1 SUCCESS
- SOFT FAIL: spatial PASS, object FAIL → scope-narrow paper to spatial regime
- HARD FAIL: aggregate B−C < 3pp → KILL Track 1

## Next-step rules (no user input needed)

- M-1 PASS → write calibration_flops.json (fvcore), launch calibration run on spatial
- M-1 SOFT → ResNet12 pivot in sim_eval_ace.py, restart calibration
- Calibration done + FLOPs B/C match within 2% → launch M-2 (4 conds × 3 seeds × spatial)
- M-2 PASS → launch M-3 (4 conds × 3 seeds × libero_object)

## Halt-to-user triggers

- M-1 HARD FAIL
- M-2 HARD FAIL (sham collapses)
- M-3 HARD FAIL (no generalization)
- Track 1 PASS → user approval gate for Track 2 (125 GPU-hr commitment)
- Unrecoverable error / OOM during any phase
