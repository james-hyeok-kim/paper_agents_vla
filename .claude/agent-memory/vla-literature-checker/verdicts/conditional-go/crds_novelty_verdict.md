---
name: crds-novelty-verdict
description: CRDS (Camera-Role Depth Schedule) CONDITIONAL GO. Per-camera ViT depth with weight-sharing axis is empty in VLA literature, but per-role depth premise unverified on LIBERO and ACE M1 precedent is a warning flag.
metadata:
  type: project
  verdict: CONDITIONAL_GO
  novelty_score: 5/10
  date_kst: 2026-05-19
---

# CRDS Novelty Verdict: CONDITIONAL GO

**One-line summary**: Per-camera ViT depth with shared weights (scene=27, wrist=16, empty=4) on OpenVLA-OFT SigLIP is an empty cell in 2024-2026 VLA literature. ACE (asymmetric weights), TaskMoE (per-task experts), BFA++ (cross-view token selection), VLA-LPAF (uniform ViT + fusion), TVVE (virtual view selection) all miss the {weight-shared × per-camera-depth} axis. **However**, the load-bearing empirical claim (wrist@16 ≈ wrist@27 SR) is unverified and the recent ACE NO-GO at validator M1 ("LIBERO multi-camera asymmetry weaker than premise") is a directly applicable warning.

**Novelty score**: 5/10 (with 4 pre-experiment gates required)

---

## What overlaps (adjacent prior work)

| Paper | Venue | Overlap level | What overlaps | What does NOT overlap |
|---|---|---|---|---|
| ACE (Asymmetric Camera Encoder) | Internal validator memory + prior verdict | YELLOW | Per-camera asymmetric capacity premise (wrist=full, static=tiny) on LIBERO motion-asymmetry | Different weights per camera (encoder swap); CRDS shares weights, varies depth. **ACE failed at validator M1 — see warning below.** |
| TaskMoE | arXiv 2508.05186 | YELLOW | Routes visual features to task-specialized experts | Per-task routing, not per-camera; expert MoE, not depth schedule |
| BFA++ | arXiv 2602.20566 | YELLOW | Cross-view importance predictor + intra-view token pruning | Per-step view selection + token-level; not encoder depth |
| VLA-LPAF | arXiv 2509.18183 | GREEN | Multi-camera perspective fusion in latent space | Uniform single ViT encoder, depth-agnostic MLP fusion |
| TVVE (Task-Aware Virtual View) | arXiv 2508.05186 | GREEN | Task-relevant virtual viewpoint selection | View selection, not per-view depth |
| Compressor-VLA | arXiv 2511.18950 | GREEN | Token compression at projector | Encoder-side compression, not per-camera depth |
| VLA-Cache | arXiv 2502.02175 | GREEN | Adaptive token caching | Temporal axis, not per-camera depth |
| DepthCache | arXiv 2603.10469 | GREEN | Depth-guided token merging with spatial differentiation | Uses scene-depth as prior for token-merging; not per-camera ViT-depth |
| Visual Perception Engine | arXiv 2508.11584 | GREEN | Shared ViT backbone + multi-head outputs at intermediate layers | Multi-task head, single-camera; not per-camera |
| HiDrop / VTW / FastV / V²Drop / FREE | Various MLLM | WHITE | LLM-side vision token reduction | Different stage (LLM backbone), different axis (token vs depth) |
| Flex / NVIDIA multi-camera driving | arXiv 2512.10947 | WHITE | Per-camera scene-token compression in driving | Different domain, uses learnable scene tokens not per-camera depth |
| Cortical Policy | arXiv 2603.21051 | WHITE | Dual-stream view transformer | Dual-stream for different processing, not per-camera depth |

**Verdict on prior-art coverage**: The cell {shared SigLIP weights × per-camera depth schedule × empty-camera architectural-zero × VLA inference efficiency} is empty. No 1:1 preempt.

---

## What does NOT overlap (CRDS-specific novelty)

1. **Weight sharing with depth variation**: ACE uses different weights per camera (separate encoders). CRDS uses **one shared SigLIP weight applied to different layer counts per camera** — this is the load-bearing distinction.
2. **Empty-camera architectural zero**: Constant input → constant features after few layers. This is a structural argument (information-theoretic ceiling), not a learned criterion. No VLA paper exploits this.
3. **Per-role schedule (scene/wrist/empty) as a discrete, hard-coded prior**: not learned, not task-conditioned. This is more rigid than TaskMoE / BFA++ — and rigidity is the contribution (no router overhead, CUDA-graph friendly).

---

## Why CONDITIONAL not GO: load-bearing empirical premise unverified

### Gate G1 (pre-experiment, M1-style): wrist@K SR drop curve

The 9% e2e savings depend on **wrist@16 ≈ wrist@27 within SR-drop ≤ 1pp**. If wrist@16 collapses SR by ≥3pp, only the empty-camera trivial savings (~3% e2e) remain — below validator threshold.

**Required pre-experiment**: measure wrist-only ablation at K ∈ {4, 8, 16, 20, 27} on libero_spatial. Premise survives only if SR(K=16) ≥ SR(K=27) − 1pp.

### Gate G2 (Sham D — critical, ACE-mirror): camera-identity shuffle

Per-role identity is the load-bearing claim. If batch-wise camera identity shuffle + CRDS schedule applied → SR drop ≤ 0.5pp vs. correctly-routed CRDS, then the "per-role" claim collapses (it's just average-depth reduction, not role-specific). The mechanism becomes a generic depth reduction.

**Required pre-experiment**: Sham D (camera ID shuffle) must show SR(CRDS-correct) > SR(CRDS-shuffle) + 1pp on libero_spatial + libero_object.

### Gate G3 (Sham A): camera-role permutation

Closely related to G2: if scene gets wrist's depth and vice versa, does SR collapse? If not, the schedule is arbitrary (not role-meaningful) and the contribution narrows to "average ViT depth reduction on multi-camera VLA" — which is not novel.

### Gate G4 (ACE M1 precedent flag — see warning)

The validator memory entry `project_esbl_killed` references "static arm 81% at warmup=20 vs 55% target, libero_spatial+ViT-small is proprio-dominated; next: ACE/KRAM". The ACE PoC then failed (per BLACKLIST and existing verdicts). The implication: **LIBERO + small-vision-backbone is proprio-dominated, which weakens any vision-encoder-side asymmetry premise**.

CRDS proposes reducing wrist depth — if LIBERO is proprio-dominated, this may trivially "work" because vision matters less in general, not because wrist-specific role-based asymmetry is real. G2 (Sham D) is the discriminator that breaks this confound.

---

## Why the 9% e2e savings is small but acceptable

The validator's "savings vs novelty" trade-off: small savings (5-15% e2e) are acceptable when the mechanism opens a new design axis. CRDS opens the {per-view depth schedule with shared weights} axis, which composes with token-level (Compressor-VLA, BFA++) and temporal (VLA-Cache) reductions. Composability is the strategic value.

Validator should evaluate whether 9% + composability passes their bar (this is their judgment, not the checker's).

---

## Required pre-experiment gate package (for validator → orchestrator)

| Gate | Test | Threshold |
|---|---|---|
| G1 | Wrist-only K-sweep ablation on libero_spatial: SR(K=16) vs SR(K=27) | SR drop ≤ 1pp |
| G2 (Sham D) | Camera-ID shuffle with CRDS schedule on libero_spatial + libero_object | SR(correct) > SR(shuffle) + 1pp |
| G3 (Sham A) | Camera-role permutation (scene↔wrist) | SR drop ≥ 2pp (i.e., role matters) |
| G4 | Empty-camera ablation: with vs without constant black input | SR delta ≤ 0.2pp at K=4 |

If G1 + G2 + G3 + G4 all pass → CRDS has a load-bearing per-role asymmetry claim → proceed to full LIBERO-4 eval.
If G1 fails → CRDS reduces to empty-camera-only ≈3% e2e — likely below validator bar. **Pivot or abandon.**
If G1 passes but G2 fails → CRDS reduces to "average depth reduction" — not novel mechanism, but engineering. **Reframe or abandon.**

---

## Recommendation

**CONDITIONAL GO**. Send to `vla-idea-validator` for feasibility/score check, with G1-G4 as required pre-experiment gates. Validator should weight:

- Novelty: 5/10 (empty cell but adjacent ACE failed and the empirical premise is unverified)
- Differentiation strength: dependent entirely on G2 (Sham D) outcome
- E2E savings: 9% nominal, 3% if G1 fails — borderline validator-bar
- Composability: high (orthogonal to token-level and temporal axes)

---

## Required follow-up actions

1. Move `vla-idea-generator/pending/crds.md` (or active) → `vla-idea-generator/active/crds.md`
2. Add G1-G4 gate spec to validator handoff
3. Add CRDS to `vla-idea-generator/MEMORY.md` active list

---

Related memories: [[ace-novelty-verdict]] [[vla-multiview-token-pruning-2025]] [[vla-efficiency-landscape-2025-2026]] [[project-esbl-killed]]
