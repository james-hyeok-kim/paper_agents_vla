---
name: bspc-verdict
description: BSPC (Bounded-Staleness Predict-Correct Pipelining) NO-GO verdict — direct conflict with A2C2 (Leave No Observation Behind, arXiv:2509.23224); same predict-on-stale + delta-correction mechanism, same SmolVLA backbone.
metadata:
  type: project
---

**Verdict: NO-GO, 2-3 / 10 novelty (2026-05-15)**

BSPC's core mechanism — "main VLA runs on stale frame t-k during chunk execution; cheap delta-correction head at execution time produces residual δa using current frame + proprio" — is mechanically identical to A2C2 (Leave No Observation Behind, arXiv:2509.23224, Sep 2025). RTC was the wrong reference conflict in the original pitch; A2C2 is the tight one.

**Direct mechanism match with A2C2:**
| BSPC | A2C2 |
|---|---|
| Main VLA on stale frame, correction at t | Base VLA predicts chunk from outdated obs; correction head refines using latest obs |
| Tiny correction head ~50M, <5ms | 32M (LIBERO) / 0.31M (Kinetix), 4.7ms |
| Inputs: (stale planned action, current frame, current proprio) → δa | Inputs: (o_{t+k}, a^base_{t+k}, positional embed τ_k, base latent z_t) → Δa |
| Residual: a_exec = a_planned + δa | Same: a^exec_{t+k} = a^base_{t+k} + Δa_{t+k} |
| Full async overlap during execution | Same; correction head runs every Δt, 20x faster than base |
| Likely backbone: SmolVLA | SmolVLA tested as base |

**Why:** A2C2 already established the predict-on-stale + residual-correction architecture for VLA action chunks; the "VLA closed-loop staleness ↔ tracking error" framing BSPC pitched is the exact framing A2C2 used.

**How to apply:** When evaluating async-VLA ideas, A2C2 is now the dominant prior art for any "correct on the fly" mechanism; RTC is only relevant for inpainting/blending approaches. SV-VLA (sparse verification, dense rerun) and VLASH (future-state-aware, no correction head) occupy adjacent cells.

**BSPC components that DO survive against A2C2:**
1. Adaptive staleness budget k via optical-flow magnitude + contact-phase detection (A2C2 uses fixed positional embedding, no dynamic k).
2. Hard-sync fallback when threshold exceeded (A2C2 has no fallback).

Neither is paper-sized on its own. Reviewers will read these as "A2C2 + a gating heuristic on top." Not a novelty rescue.

**Other 2025-2026 papers in the same cell (all 🟡 or 🔴):**
- VLASH (arXiv:2512.01031) — future-state-aware async, NO correction head; 2.03x speedup. Conditions on s_{t+Δ} rolled forward. Distinct from BSPC but in same async family.
- REMAC / Real-Time Robot Execution with Masked Action Chunking (arXiv:2601.20130) — training-time delay conditioning, no test-time correction. Distinct mechanism.
- VLA-RAIL (arXiv:2512.24673) — quintic polynomial blending to mask latency; engineering pipeline.
- SV-VLA (arXiv:2604.02965) — dense verification with replanning trigger. Different role assignment.
- RTC (arXiv:2506.07339) — inpainting/blending only, no learned correction.

**Pivot if user insists:**
The only narrow re-pitch with a chance is to drop the predict-correct framing entirely and lead with a formal "bounded staleness ↔ tracking error" theorem + an adaptive controller that A2C2 lacks. Reframe as "Bounded-Staleness Safety Layer for Async VLA" — not as a new inference mechanism but as a safety controller bolted on top of any async VLA (A2C2, VLASH, RTC). Even then, the contribution is a controller, not an acceleration method, and would belong in a different venue.

**Recommendation:** NO-GO under current pitch. If user wants to push further, route to vla-idea-generator with the constraint "must not be predict-correct on stale frame + cheap delta head."
