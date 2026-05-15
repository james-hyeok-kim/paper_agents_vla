---
name: idea-policy-prior-cache
description: Cache language-conditioned action prior (distribution / residual), not KV — reused across repeated instructions to skip LLM forward
metadata:
  type: project
---

# Policy-Prior Cache: Caching Action Distributions, Not Tokens (PPC-VLA)

## Core hypothesis
For VLA, the language instruction induces a *stationary* prior over action chunks that is largely invariant across executions with similar initial visual context; caching this prior as an "action prior residual" (not as LLM KV) lets the runtime skip the LLM backbone entirely for repeated instructions and only run vision + action head + correction.

## Technical approach
- During training, factorize action distribution as p(a | v, q, l) ≈ p_prior(a | l, c(v_0)) · p_correction(a | v, q, residual). The prior is an action distribution embedding (or a low-rank residual added to the action head input) keyed by (instruction_embedding, coarse visual context cluster c(v_0)).
- At inference: hash (l, c(v_0)) → look up cached prior tensor → bypass LLM backbone → run only vision encoder (or its cheap branch) + action head conditioned on prior + proprio.
- Cache is built online: first time an instruction is seen, run full VLA, distill into prior; subsequent calls hit the cache.

## Why VLA-specific (and why standard prompt caching is not enough)
- Standard LLM prompt KV caching reuses *token-level KV* but still runs every decode step through the LLM. For VLA, decode = action, and KV cache still requires running the action head through the LLM stack.
- VLAs uniquely have a *bounded action output space* with stationary conditional distribution under repeated language — the prior compresses to a small tensor (kB-scale), not GB of KV.
- For repeated household / industrial tasks (pick X, place Y, wipe Z) this hits the cache an order of magnitude more often than a typical LLM prompt cache hits.

## Distinct from excluded work
- VLA-Cache / Eventful Transformers — they cache *vision* features across frames. This caches the *language-conditioned action prior*. Orthogonal and compatible.
- LLM prompt caching (Anthropic / vLLM prefix cache) — they cache prefill KV; this caches a *post-LLM action prior* — the LLM is fully skipped on cache hit.
- Speculative decoding — different mechanism; here there is no draft model, the cached prior *is* the proposal which the action head conditions on.

## Safety
Cache hit must verify: (1) instruction match (exact or paraphrase via embedding distance), (2) coarse visual context cluster match (else fall back to full LLM). A "freshness" counter rebuilds the cache after N invocations to absorb drift.

## Target hardware
Highest payoff on Jetson where the LLM backbone is the bottleneck. On A100/H100 the gain is smaller but still meaningful for high-throughput multi-robot scheduling.

## Expected gains
On a cache hit (which we expect 40-70% on factory/household repetition workloads): 3-6× end-to-end speedup because the 7B LLM forward is removed. Even with 50% hit rate, ≥1.8× average speedup.

## Adjacent landmines
- "Task embedding" / "skill embedding" literature — distinguish by *cache-mechanism framing* (storage + lookup + invalidation) rather than just a learned embedding.
- Hypernetwork / adapter conditioning — distinguish by *cache hit bypasses LLM*, whereas hypernets still run a forward pass.
- Prompt caching in serving systems — distinguish by the cached object being a *post-LLM prior*, not prefill KV.

## Venue
NeurIPS 2026 / ICLR 2027 (systems-ML flavor) or CoRL 2026 (robotics framing). NeurIPS preferred — cleanest "amortized inference" story.

## Suggested next step
Run vla-literature-checker on "Policy-Prior Cache for VLA" + keywords {action prior caching, instruction-conditioned residual, LLM-skipping VLA inference, amortized policy inference}.
