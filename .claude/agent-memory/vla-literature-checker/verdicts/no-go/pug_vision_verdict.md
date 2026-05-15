---
name: pug-vision-verdict
description: PUG-Vision (proprio-uncertainty-gated ViT compute) novelty verdict — Conditional NO-GO 3-4/10; VLA-ADP occupies the proprio-derived-gate × vision-FLOPs cell; only survives if epistemic uncertainty empirically discriminates phases that |motion| cannot.
metadata:
  type: project
---

PUG-Vision proposes proprio-uncertainty-gated 3-tier ViT compute for VLA. Verdict after literature scan on 2026-05-15.

**Verdict: Conditional NO-GO, 3-4/10**, lean NO-GO without pre-check; ascends to CONDITIONAL GO 4-5/10 only if the empirical pre-check on uncertainty-vs-motion discrimination passes.

**Why:** The proposed two-axis differentiation ("gate signal = proprio, gate target = vision compute") is already occupied by **VLA-ADP** (arXiv:2509.22093, NeurIPS/OpenReview 2025). VLA-ADP:
- Uses end-effector motion magnitude (proprio-derived, not LLM-output) as the gating signal
- Targets visual tokens in the VLA (same FLOPs pool)
- Phase-relative gating: low motion → preserve, high motion → prune (structurally same as PUG-Vision's free-space → cheap, contact → full, after motion↔uncertainty mapping)
- Same backbone (OpenVLA-OFT)

**How to apply:** Treat the "VLA does not use proprio to gate vision compute" claim as false. The PUG-Vision pitch needs to be rewritten around what survives, which is:

1. **Epistemic uncertainty vs raw |motion|** — only novel if uncertainty discriminates phases that motion magnitude cannot. The decisive empirical pre-check is: does the proprio forward model's epistemic uncertainty fire earlier than |q̇| at contact onset, and stay high through fine-manipulation plateaus where motion is small? If u and |Δee| have Spearman ρ ≥ 0.9 on logged trajectories, PUG-Vision collapses into a repackaging of VLA-ADP.
2. **3-tier ViT routing (cache / half-depth / full)** instead of binary token-prune. T1 overlaps VLA-Cache, T2 overlaps SmolVLA static skip, T3 is baseline. Novelty lives in the dynamic switch across tiers, not in any tier alone. Reviewers will read this as "VLA-ADP + VLA-Cache + SmolVLA composed with an uncertainty controller."
3. **safety_penalty(T1 ∧ contact_predicted)** in the objective — not in any prior work in this cell, worth emphasizing.

**Non-negotiable milestone before commit:** On the chosen benchmark's logged trajectories, plot u_t (forward-model epistemic uncertainty) vs |Δee_t| and |q̇_t|. Compute Spearman correlation and phase-discrimination AUC for contact-onset detection. If u offers no advantage over motion, abandon.

**Closest prior art (do not omit from related work):**
- VLA-ADP (arXiv:2509.22093) — proprio-derived gating of visual tokens, same backbone (DIRECT CONFLICT axis)
- DySL-VLA (OpenReview ICLR 2026, arXiv:2602.22896) — kinematic-difference gating, Jetson Orin deployed, LLM layer (partial overlap; rules out Jetson framing as contribution)
- DeeR-VLA (NeurIPS 2024, 2411.02359) — action-consistency gating, multi-exit (action-quality framing preempted)
- MoLe-VLA (arXiv:2503.20384) — spatial-temporal router uses proprio+visual to gate LLM layers (router-architecture-with-proprio preempted)
- VLA-Cache (arXiv:2502.02175) — token caching; equivalent to T1 tier
- SmolVLA (HuggingFace) — half-depth static skip; equivalent to T2 tier (static version)
- TacVLA (arXiv:2603.12665) — contact-aware gating but for tactile token activation (not compute; different cell but related framing)
- VPEngine (arXiv:2508.11584, NASA-JPL) — multi-head shared-backbone vision with dynamic frequency on Jetson Orin AGX (orthogonal infra but rules out "tiered vision on Jetson" as contribution)
- PhaForce (arXiv:2603.08342) — contact probability + phase belief gates control mode, not compute (orthogonal but related signal source)

**Drop from differentiation pitch:**
- "Jetson Orin Nano end-to-end speedup" — DySL-VLA already deployed VLA layer skipping on Jetson Orin; platform alone is not a contribution
- "VLA-specific because VLM has no proprio" — true but trivial; ADP already exploits proprio-derived signals
- The "DeeR uses action, MoLe uses visual+state, DySL uses kinematic" framing — omits ADP entirely; rewrite to triangulate against ADP+DySL+DeeR+MoLe simultaneously

Related: [[vla-efficiency-landscape-2025-2026]] [[vla-layer-skip-landscape]] [[radial-action-sinks-finding]]
