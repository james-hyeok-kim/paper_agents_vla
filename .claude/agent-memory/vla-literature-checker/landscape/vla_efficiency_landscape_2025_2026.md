---
name: vla-efficiency-landscape-2025-2026
description: Map of 2025-2026 VLA inference-efficiency literature, organized by sparsification axis, used to triage novelty of new acceleration ideas.
metadata:
  type: reference
---

The VLA efficiency literature in 2025-2026 clusters along four axes. New acceleration ideas should be placed against this map before being deemed novel.

**Axis 1 — Visual token pruning** (most saturated)
- VLA-Pruner (arXiv:2511.16449) — dual-level vision/action attention scoring
- SP-VLA (arXiv:2506.12723) — spatio-semantic dual-aware on OpenVLA-7B
- LightVLA (arXiv:2509.12594) — differentiable pruning with dynamic queries
- Compressor-VLA (arXiv:2511.18950) — instruction-guided STC + SRC
- VLA-IAP (arXiv:2603.22991) — training-free interaction-alignment pruning
- EfficientVLA (arXiv:2506.10100) — training-free compression
- SpecPrune-VLA (OpenReview 2025) — two-level (action-static + layer-dynamic) with end-effector velocity classifier

**Axis 2 — Temporal / action reuse**
- FlashVLA (arXiv:2505.21200, "Think Twice, Act Once") — action reuse via similarity + visual stability triggers on OpenVLA
- AC2-VLA (arXiv:2601.19634) — cognition caching across timesteps on CogACT; combines with token pruning + layer skipping; 1.79x speedup at 29.4% FLOPs

**Axis 3 — Depth / layer skipping**
- AC2-VLA conditional layer skipping
- SmolVLA layer skipping in the VLM
- A1 (arXiv:2604.05672) — truncated VLA

**Axis 4 — Decoding / parallelism**
- PD-VLA (arXiv:2503.02310) — parallel decoding with action chunking
- FAST / FASTer (arXiv:2501.09747, OpenReview) — action tokenization for autoregressive VLA

**Empty axis (as of May 2026)**: cross-attention key-set sparsification *conditioned on action-chunk position index*. Closest is [[radial-action-sinks-finding]] which analyzes attention by position but does not turn it into a sparsification mechanism.

**Common backbones in this space**: OpenVLA-7B dominates; CogACT, π0.5, GR00T N1.5, SmolVLA appear less often. ACT (the original 80M Tonyzhao/aloha policy) is rarely targeted by acceleration work — it is already small (~10ms inference). Acceleration work on the LeRobot/ACT family specifically is sparse.

**Implication for novelty triage**: An acceleration idea is more likely novel if it (a) introduces a new sparsification axis (e.g., chunk-position), (b) targets ACT/SmolVLA rather than OpenVLA, or (c) modifies the cross-attention K/V set rather than the visual token set. Ideas overlapping with [[radial-action-sinks-finding]]'s empirical claims need a pre-check on the chosen backbone before committing.
