---
name: ppc-vla-verdict
description: NO-GO 2-3/10 on PPC-VLA (Policy-Prior Cache, LLM-bypass via (instruction, visual cluster) keyed post-LLM action prior cache); AC2-VLA's cognition caching is direct prior art with the same skip-VLM-forward mechanism, only the cache key differs.
metadata:
  type: project
---

PPC-VLA proposes caching a post-LLM action prior tensor keyed by (instruction_emb, coarse visual cluster). On cache hit, the LLM backbone forward is skipped entirely; only vision encoder (cheap branch) + action head + correction run. Decomposition: p(a|v,q,l) ≈ p_prior(a|l, c(v_0)) · p_correction.

**Verdict: NO-GO 2-3/10.**

**Why: AC2-VLA (arXiv:2601.19634, Jan 2026) is direct prior art.**
- AC2-VLA's "cognition caching" caches backbone (VLM) features and on cache hit "skip[s] the VLM backbone forward" — identical mechanism to PPC-VLA.
- Cache key in AC2-VLA: (Quant(‖Δa_t‖), Hash(v̄_t)) — quantized action-delta + lightweight visual hash.
- Cache key in PPC-VLA: (instruction_emb, coarse visual cluster) — different signals but same shape (a compact lookup key for a heavy-tensor cache).
- AC2-VLA also uses an "action-prior router" — even the naming overlaps. Combined with token pruning + layer skipping, AC2-VLA gets 1.79x speedup at 29.4% FLOPs on CogACT.
- Reviewers will read PPC-VLA's instruction-keying as a tweak to AC2-VLA's keying strategy, not a new method.

**LangForce / BayesianVLA (arXiv:2601.15197) covers the factorization claim — but only partially (🟡, not 🔴).**
- Performs Bayesian decomposition p(a|v,l) via latent action queries; trains a vision-language-prior branch + posteriori branch.
- BUT: caching/bypass is NOT used at inference — only the Posteriori Branch runs, with "no additional computational overhead." Training-time factorization only.
- So LangForce defeats novelty of "VLA action distribution factorizes into language-prior × visual-correction," but does NOT preempt the caching mechanism. Cite as foundation for the decomposition assumption.

**RAEA (Zhu et al., CVPR 2024, arXiv:2404.11699) covers cross-episode policy retrieval.**
- Policy retriever + policy generator on Open X-Embodiment; retrieves relevant strategies by multi-modal input keys.
- Closes the "cross-episode keying" seam that AC2-VLA leaves open (AC2-VLA is within-episode).
- Combine RAEA's retrieval-keying intent with AC2-VLA's cache-and-bypass mechanism → PPC-VLA's "novel combination" story collapses.

**Other adjacencies (not blocking but populate the same space):**
- FlashVLA (arXiv:2505.21200): action reuse skips entire inference on stable steps via FlashTrigger (angle α + visual token intersection φ). No instruction keying, in-memory buffer not retrieval-cache. 🟡 weaker.
- EfficientVLA (arXiv:2506.10100): caches intermediate features in the diffusion action head (not LLM bypass).
- VLA-Cache (NeurIPS 2025): visual token KV reuse only, never bypasses LLM forward. Author's own writeup correctly identifies this as orthogonal.
- DeeR-VLA (NeurIPS 2024): early-exit/dynamic layer skipping, conditioned on action consistency. Skips suffix layers, not the whole LLM.
- ActDistill (arXiv:2511.18082): action-guided distillation to a routing student; runs the student, not "skip LLM."
- LAWS (arXiv:2605.04069): caches parametrized experts that bypass base LLM forward on cache hit. Not robotics-specific but the bypass-on-hit pattern matches.

**The author's stated differentiators don't survive:**
- "VLM prompt cache only reuses prefill KV, VLA can cache post-LLM prior" — true vs vLLM, but AC2-VLA already does post-LLM feature caching with backbone-forward skip.
- "Bounded action output → small compressible prior" — AC2-VLA's existence proves the community already exploits this property; not a new observation.

**Recommendation:** Send back to vla-idea-generator. The "post-LLM feature cache + LLM forward bypass" cell on the VLA efficiency map ([[vla-efficiency-landscape-2025-2026]] axis 2 — temporal/action reuse) is now occupied by AC2-VLA. Any pivot keeping the same shape (cache the post-backbone state, key on something, skip on hit) will face AC2-VLA as the natural baseline. Real seams remaining: (a) cross-episode policy memory at scale (RAEA exists but is small-scale CVPR'24, not VLA-targeted), (b) instruction-only keying with no visual signal at all (worth investigating if instruction alone can predict actions well enough — likely fails on dynamic scenes), (c) compressibility of the cached tensor (kB-scale claim is testable and could be a separate contribution).

**Pattern matches**: same shape as [[bspc_verdict]] (A2C2 occupied "predict-on-stale + delta-correction"), [[pads_verdict]] (FASTER occupied per-token adaptive), [[temporal_delta_kv_cache_verdict]] (Eventful + VLA-Cache occupied delta KV). When the proposal names a single mechanism + a single key, search for that mechanism with any key — the conflict is almost always there.
