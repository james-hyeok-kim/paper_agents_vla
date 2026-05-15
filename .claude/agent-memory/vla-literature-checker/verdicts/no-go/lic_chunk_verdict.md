---
name: lic-chunk-verdict
description: NO-GO 2-3/10 verdict on LIC-Chunk (language-instruction-conditioned variable chunk length); AutoHorizon (arXiv:2602.21445, Feb 2026) is the direct conflict, MoH is secondary.
metadata:
  type: project
---

LIC-Chunk proposes a budget head g(instr_emb, s_0) → H ∈ {4,8,16,32,64} that predicts the outer execution chunk length from language instruction + initial state, aiming for 1.5-3x fewer VLA forwards on long-horizon LIBERO.

**Verdict: NO-GO, novelty 2-3/10.**

**Why: AutoHorizon (arXiv:2602.21445, Feb 2026) occupies the axis.**
- Same problem: dynamic execution horizon H per chunk for flow-based VLAs
- Same backbones: π0.5, GR00T N1.5 on LIBERO (incl. LIBERO-10)
- Same inference loop: 1 VLA call → H actions → next chunk
- Same per-chunk granularity, training-free, "negligible overhead"
- The parent paper "VLA Knows Its Limits" is already cataloged as [[radial_action_sinks_finding]]

**Why the "language signal is unique" differentiator collapses:**
1. The VLA backbone already conditions on the instruction. AutoHorizon's self-attention proxy is computed from a forward pass that *implicitly* encodes the instruction's motion-structure cue. "pour water slowly" vs "move cup to shelf" already modulates the attention pattern AutoHorizon reads.
2. AutoHorizon's empirical finding contradicts the LIC-Chunk thesis: optimal H *shortens during contact phases* (grasp/place) and *lengthens during free-space transport*. This is an environment/contact signal, not an instruction-derived constant. Predicting H once from (instr, s_0) cannot capture intra-episode phase shifts that AutoHorizon already handles.
3. Per-episode vs per-chunk ambiguity in the proposal: if per-episode it loses to AutoHorizon on adaptivity; if per-chunk it is AutoHorizon with a trained head replacing the attention proxy — incremental at best.

**Secondary conflicts:**
- Mixture of Horizons (arXiv:2511.19433, Nov 2025) — runs multiple horizons in parallel, picks via cross-horizon consensus. 2.5x throughput, 99% LIBERO. Different mechanism but same axis (dynamic adaptive horizons), and was the user's flagged conflict.
- Adaptive Q-Chunking (arXiv:2605.05544) — value-based per-scale chunk selection. Different signal (Q-value) but same multi-scale-prefix selection problem.

**Pattern match**: Same shape as [[pads_verdict]] and [[bspc_verdict]] — a specific paper occupies the axis AND the proposed differentiator does not survive scrutiny against the prior art's actual mechanism.

**Possible pivots (speculative)**:
- Instruction-conditioned *prediction* horizon (training-time chunk length) rather than execution horizon — different axis, less crowded.
- Combine attention proxy (AutoHorizon-style) with an explicit language phase classifier — incremental but defensible.
- Apply to backbones AutoHorizon did not test (SmolVLA, OpenVLA, ACT) — engineering scope, not conceptual novelty.

**Unchecked blocker**: whether AutoHorizon already runs an instruction-conditioned baseline. If yes and attention beats it, kill shot is documented. If no, LIC-Chunk has a narrow "explicit head" empirical contribution but conceptual novelty is gone either way.
