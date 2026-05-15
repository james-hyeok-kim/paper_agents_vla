---
name: compute-calibration-b200
description: Compute-time estimates for VLA distillation experiments on 4× B200 (calibration for future plans)
metadata:
  type: reference
---

# B200 Compute Calibration (4× B200 node)

Established 2026-05-15 from CPR-Distill plan. Use these as rough priors when sizing future VLA experiments.

- **OpenVLA-7B teacher forward** (BF16, batch 16): ~150 ms/step. Cache offline whenever possible — saves 60-70% of student training time.
- **Student training, 1.5B params, LIBERO ~500K demos, 50 epochs**: ~12 GPU-hrs on 4× B200 (DDP, BF16 + FP8 attn).
- **Student training, 3B params**: ~22 GPU-hrs.
- **LIBERO eval, full suite × 50 trials × 3 seeds**: ~3 GPU-hrs per student checkpoint.
- **Single experiment cell** (1 method × 4 task families × 3 seeds): ~36 GPU-hrs end-to-end.

## How to apply
- For a typical VLA distillation paper with ~10 method-cells: budget ~350-500 GPU-hrs total on 4× B200 (3.5-5.5 days wallclock).
- Always pre-cache teacher logits/actions once — this is the single biggest savings.
- B200 is ~2.25× H100 for FP8; halve these estimates if comparing against H100 numbers in prior papers.
