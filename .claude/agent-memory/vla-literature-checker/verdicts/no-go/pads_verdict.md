---
name: pads-verdict
description: PADS (Position-Adaptive Denoising Schedule) NO-GO verdict — axis saturated by FASTER/AsyncVLA and FASTER's pilot study empirically reverses the load-bearing monotone hypothesis.
metadata:
  type: project
---

PADS proposed a position-adaptive denoising schedule for π0-style flow-matching VLAs: monotone decreasing step budget across chunk positions (a_0 many → a_{H-1} few) based on a "later positions have sharper conditional posterior" hypothesis.

**Verdict: NO-GO (novelty 1-2/10).** Two independent fatal findings:

1. **Axis already occupied.** FASTER (arXiv:2603.19199, March 2026, π0.5 + X-VLA) introduces a Horizon-Aware Schedule (HAS) that "decouples the local denoising timestep for each frame within the chunk, adaptively allocating sampling steps per position." AsyncVLA (arXiv:2511.14148, Nov 2025) does per-token adaptive denoising via a confidence rater for selective re-denoising. The intra-chunk position-dependent step count axis is saturated.

2. **PADS's hypothesis is empirically reversed.** FASTER's pilot study measures a *straightness* metric across chunk positions and reports the OPPOSITE of PADS's claim: early actions are straighter (tighter coupling to current observation, narrower solution space) → need FEWER steps; long-horizon actions need MORE. FASTER quote: "compressing the denoising of the immediate reaction by tenfold into a single step, while preserving the quality of long-horizon trajectory." This is published primary-source refutation of PADS's load-bearing monotone-decreasing assumption on the same backbone family.

3. **Pipelining sub-claim** ("a_0 reaches k steps → a_1 starts early") is already Streaming Diffusion Policy (arXiv:2406.04806) — N/h denoising steps per chunk with rolling buffer.

**Why no pivot survives:**
- Reverse the direction (early=many, late=few)? That's literally not what FASTER says — FASTER is early=few, late=many. So pivoting to FASTER's direction = directly reproducing FASTER.
- Different backbone (SmolVLA, ACT)? The pre-check is the same straightness/entropy pilot FASTER already published; even if results invert, the contribution is "backbone-specific reversal of FASTER" — same axis, smaller delta.
- Per-position learned predictor with FLOPs constraint? Same as AsyncVLA's confidence-rater architecture with a different training signal.

**Adjacent (🟡 partial overlap):**
- D3P (arXiv:2508.06804) — adaptive denoising step count, but per environment-timestep (one count per chunk), not intra-chunk. Lives on a different sub-axis.
- A1 (arXiv:2604.05672) — Inter-Layer Truncated Flow Matching with warm-start, but uniform across positions.
- SnapFlow (arXiv:2604.05656) — uniform 1-step distillation, position-agnostic.

**Recommendation:** route to vla-idea-generator. Don't propose adjacent position-dependent denoising ideas — the axis is closed.

Linked: [[vla-efficiency-landscape-2025-2026]] (decoding/parallelism axis), [[radial-action-sinks-finding]] (attention-side invariance, separate but reinforcing concern).
