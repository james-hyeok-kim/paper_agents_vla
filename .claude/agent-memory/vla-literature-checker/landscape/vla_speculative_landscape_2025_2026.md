---
name: vla-speculative-landscape-2025-2026
description: Speculative decoding / draft-verify / hierarchical small-large VLA inference acceleration competitive map (2025-2026); reference for chunk-level speculative novelty checks.
metadata:
  type: reference
---

Competitive map of speculative / draft-verify / hierarchical inference work for VLA, as of 2026-05. Used to triage chunk-level speculative ideas like [[cp_sparse_verdict]] siblings (e.g. SACV).

**Token-level speculative decoding (orthogonal to chunk-level)**
- Spec-VLA (arXiv:2507.22424, EMNLP 2025) — EAGLE-style draft (LLaMA block) verifies every action token within a single 7-DoF action; relaxed acceptance via bin-ID distance r; 1.42x speedup on OpenVLA. Not chunk-level; not Lipschitz-based.
- KERV / Kinematic-Rectified Speculative Decoding (arXiv:2603.01581) — adds Kalman-filter compensation + dynamic threshold via kinematic variability K_var to token-level SD. 1.48-1.57x over naive SD, 27-37% over Spec-VLA. Still token-level within single action.
- CEED-VLA (arXiv:2506.13725) — consistency distillation + early-exit decoding for Jacobi parallel decoding. 2-4.1x. Not draft-verify.
- PD-VLA (arXiv:2503.02310) — parallel Jacobi decoding for action chunks; no separate draft model.

**Single-action draft-verify (chunk granularity = 1)**
- ADAHI (arXiv:2510.02851) — small on-device drafts a single action, large remote verifies only when action-deviation Δ > Δ_th. Closest match to "sparse verification" but operates per-step, not over a multi-step chunk. Threshold motivated empirically via mean-reversion analogy, NOT Lipschitz.

**Chunk-level hierarchical (planner / verifier roles)** — most overlapping cluster
- SV-VLA (arXiv:2604.02965, "Open-Loop Planning, Closed-Loop Verification", Apr 2026) — INVERTED roles vs. SACV: heavy VLA proposes K=64 chunk, lightweight verifier reruns at EVERY control step computing L1 distance to planned action; >τ triggers replanning. OpenVLA-OFT backbone, 2.17x speedup, LIBERO 79.5%→90.9%. No Lipschitz bound; verification is dense (every t), not sparse.
- DP-VLA (arXiv:2410.15549) — System 2 (VLM) low-frequency planner + System 1 (small policy) high-frequency executor. No explicit verification; small model just executes intentions.
- RoboDual (OpenReview 3flhuT2QGB) — OpenVLA generalist conditioning a DiT specialist for multi-step rollouts; CALVIN; 15Hz. Specialist generates, generalist conditions; no verification step.
- HiRT — hierarchical low-frequency VLM + high-frequency policy; same pattern as DP-VLA.
- CoVer-VLA (arXiv:2602.12281) — test-time selection/reranking, NOT speculative. Generates K×M candidates and contrastive-scores them. Different scheme.

**Asynchronous inference (no draft-verify) — densely populated as of 2026-05**
- LeRobot async inference / SmolVLA async stack — server computes next chunk while robot executes current; ~2x. No verification, pure pipelining.
- Real-Time Chunking (RTC, pi.website, arXiv:2506.07339) — inpaints next chunk to be consistent with frozen execution prefix; not draft-verify.
- **A2C2 / Leave No Observation Behind (arXiv:2509.23224)** — dominant prior art for "predict-on-stale + cheap delta-correction head" pattern. 32M correction head (LIBERO) / 0.31M (Kinetix), 4.7ms, runs every Δt while base VLA runs async. a^exec = a^base + Δa. SmolVLA base. +23pp over RTC on Kinetix d=4. Blocks any BSPC-style idea. See [[bspc_verdict]].
- VLASH (arXiv:2512.01031) — future-state-aware async; rolls state forward to s_{t+Δ} for conditioning. NO correction head, no overhead. 2.03x on π0.5/SmolVLA.
- REMAC / Masked Action Chunking (arXiv:2601.20130) — training-time delay conditioning, uniform-sampled delays. No test-time correction.
- VLA-RAIL (arXiv:2512.24673) — quintic polynomial blending to mask latency; engineering pipeline.
- ADAHI's selective transmission is similar to async but adds rejection.

**Closely adjacent but not speculative**
- Bidirectional Decoding (arXiv:2408.17355) — backward coherence + forward contrast at chunk boundary.
- HiPolicy (arXiv:2604.06067) — hierarchical multi-frequency action chunking.

**Empty cell (as of 2026-05)**: SMALL policy proposes a multi-step chunk + LARGE VLA verifies SPARSE subset of timesteps (e.g. t=0, K/2, K-1) with a Lipschitz/dynamics-smoothness bound justifying the sparsity. SV-VLA flips the roles AND verifies densely; ADAHI is chunk-size-1; Spec-VLA/KERV are token-level. No paper found combining (a) small-proposes-chunk, (b) large-verifies-sparse-timesteps, (c) Lipschitz-bounded trajectory deviation argument.

**Risk for SACV-like ideas**:
- SV-VLA is the dominant prior art and will be the natural baseline; reviewers will ask "why not just invert SV-VLA roles?" — answer must justify why small-proposer is more compute-amortizable than small-verifier (it isn't obviously: large-as-proposer means one expensive call per chunk vs. small-as-proposer means many cheap calls + occasional verification).
- ADAHI already established "sparse-verification under action-deviation" framing without Lipschitz; SACV's Lipschitz bound is a tighter theoretical hook but the engineering trick overlaps.
- Reviewers likely to demand a formal Lipschitz constant measurement on the target backbone (SmolVLA / OpenVLA / π0); if measured Lipschitz is too loose, the "verify only 3 of 16 timesteps" claim collapses.

**Implication**: SACV is best positioned as (i) inverted-direction speculative verification (small proposes, large sparse-verifies), (ii) formalized via Lipschitz bound on action trajectory, (iii) targeting SmolVLA where the small/large gap is steep (450M vs 7B). The unsaturated cell exists; SV-VLA is the closest direct conflict.
