# Agent Memory Index — vla-literature-checker

## Verdicts: NO-GO (verdicts/no-go/, 7개)
- [QVLA conflict](verdicts/no-go/qvla_conflict.md) — ICLR 2026 QVLA preempts action-sensitivity mixed-precision (ASMP-Q)
- [Temporal Delta KV Cache](verdicts/no-go/temporal_delta_kv_cache_verdict.md) — 2/10; VLA-Cache + Eventful Transformers + TTF-VLA full coverage
- [BSPC](verdicts/no-go/bspc_verdict.md) — 2-3/10; A2C2 (arXiv:2509.23224) 1:1 match (not RTC)
- [PADS](verdicts/no-go/pads_verdict.md) — 1-2/10; FASTER's HAS occupies axis + reverses hypothesis
- [LIC-Chunk](verdicts/no-go/lic_chunk_verdict.md) — 2-3/10; AutoHorizon (arXiv:2602.21445) occupies variable-horizon axis
- [PUG-Vision](verdicts/no-go/pug_vision_verdict.md) — 3-4/10; VLA-ADP (arXiv:2509.22093) preempts proprio-derived vision gate
- [PPC-VLA](verdicts/no-go/ppc_vla_verdict.md) — 2-3/10; AC²-VLA cognition caching is direct prior art

## Verdicts: CONDITIONAL GO (verdicts/conditional-go/, 2개)
- [AMP-Distill](verdicts/conditional-go/amp_distill_verdict.md) — 5/10; contact-phase reweighting axis 미점유
- [CP-Sparse](verdicts/conditional-go/cp_sparse_verdict.md) — 5/10; AutoHorizon invariance pre-check 필요

## Landscape Files (field-level survey, landscape/, 7개)
- [VLA Efficiency Landscape 2025-2026](landscape/vla_efficiency_landscape_2025_2026.md) — 4축(token/temporal/depth/decoding) 효율화 지도
- [VLA Quantization Landscape](landscape/vla_quant_landscape.md) — QVLA/DyQ-VLA/EaqVLA/SQAP-VLA map
- [VLA Layer-Skip Landscape](landscape/vla_layer_skip_landscape.md) — DeeR-VLA/MoLe-VLA/DySL-VLA/ActDistill
- [VLA Speculative Landscape](landscape/vla_speculative_landscape_2025_2026.md) — Spec-VLA/KERV/SV-VLA/ADAHI
- [VLA Multi-view Token Pruning 2025](landscape/vla-multiview-token-pruning-2025.md) — BFA++/VLA-Pruner/TEAM-VLA competitive map
- [3D Vision Multi-view Token Merging](landscape/3d-vision-token-merging-vggt.md) — VGGT/HTTM (non-VLA reference)
- [Radial Action Sinks Finding](landscape/radial_action_sinks_finding.md) — AutoHorizon parent finding (intra-chunk attention invariance)
