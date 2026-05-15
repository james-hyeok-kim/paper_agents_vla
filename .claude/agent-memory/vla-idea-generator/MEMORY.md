# Agent Memory Index — vla-idea-generator

**MANDATORY before generation**: read [BLACKLIST.md](BLACKLIST.md) first.

## Active Ideas (CONDITIONAL GO 또는 PoC PASS, 2개)
- [CPR-Distill](active/cpr-distill.md) — Contact-phase reweighted SE(3) distillation. PoC PASS (67x specificity). Validator 6.5/10.
- [XV-Dedup](active/xv-dedup.md) — Cross-view token deduplication (LSH-based). PoC partial pass (70% overlap). Validator 6.0/10.

## Pending Ideas (생성됨, literature-checker 대기, 0개)
- (없음)

## Abandoned Ideas (NO-GO from literature 또는 FAIL from validator, 10개)
- [ASMP-Q](abandoned/asmp-q.md) — Action-Sensitivity Mixed-Precision Quantization (QVLA에 선점)
- [TASK-Skip](abandoned/task-skip.md) — Jetson-Throughput-Aware Layer Skip (DeeR/DySL/MoLe)
- [Temporal Delta KV](abandoned/temporal-delta-kv.md) — Vision encoder KV temporal cache (VLA-Cache)
- [SACV](abandoned/sacv.md) — Speculative Action-Chunk Verification (SV-VLA)
- [PADS](abandoned/pads.md) — Position-Adaptive Denoising (FASTER/AsyncVLA)
- [BSPC](abandoned/bspc.md) — Bounded-Staleness Predict-Correct (A2C2 1:1)
- [PPC-VLA](abandoned/ppc-vla.md) — Policy-Prior Cache (AC²-VLA)
- [PUG-Vision](abandoned/pug-vision.md) — Proprio-Uncertainty Vision Gate (VLA-ADP)
- [LIC-Chunk](abandoned/lic-chunk.md) — Variable Chunk Length (AutoHorizon)
- [CP-Sparse](abandoned/cp-sparse.md) — Chunk-Position Sparse Attention (PoC 핵심 가설 미입증)

## Statistics
- 총 생성: **12개** (4 rounds)
- Active: **2** / Pending: **0** / Abandoned: **10**
- NO-GO rate: 75% — 2026년 1-4월 VLA efficiency 분야 카펫 폭격에 기인
