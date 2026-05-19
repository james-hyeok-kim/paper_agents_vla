---
name: svee-novelty-verdict
description: SVEE (Static Vision-token Early-Exit on LLM backbone) NO-GO verdict. VTW (AAAI 2025) is a 1:1 mechanism match. VLM→VLA renaming is insufficient differentiation under V-MIB/InfoPrune precedent.
metadata:
  type: project
  verdict: NO-GO
  novelty_score: 2/10
  date_kst: 2026-05-19
---

# SVEE Novelty Verdict: NO-GO

**One-line summary**: VTW (Visual Tokens Withdrawal, AAAI 2025) implements the exact static-K-layer-vision-token-drop mechanism SVEE proposes, on the same Llama-2-7B backbone family OpenVLA-OFT uses. SVEE = "VTW applied to OpenVLA-OFT" without an articulated VLA-specific mechanistic claim — fails the V-MIB / InfoPrune NO-GO precedent for VLM→VLA renamings.

**Novelty score**: 2/10

---

## The 1:1 Preempting Paper

**Paper**: Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference (VTW)
**Venue**: AAAI 2025
**arXiv**: 2405.05803 (v3, Jan 2025)
**Overlap level**: RED — Direct Conflict

### Mechanism alignment table

| SVEE design axis | VTW finding | Match |
|---|---|---|
| Static schedule, same K for all samples | "K is fixed per model... computed once during setup using KL divergence on size=20 subset, not per-sample" | EXACT |
| Vision tokens dropped from residual stream after layer K | "𝒳K_t = X_K_t − 𝒱K_t... only text tokens engage in subsequent layers" | EXACT |
| No future KV computation for vision tokens after K | "compatible with KV Cache because all vision tokens are preserved and removed simultaneously" | EXACT |
| Hypothesis: early layers fuse vision, late layers generate | "attention sink + information migration: visual information transferred to text tokens within first few layers, vision tokens get minimal attention in deep layers" | EXACT |
| Backbone | Tested on LLaVA-1.5-7B (Llama-2-7B base), Video-LLaVA, Qwen2-VL-2B, InternVL2-4B | OpenVLA-OFT uses Llama-2-7B — same family |
| Compute savings | "over 40% computational overhead reduction" | SVEE claims ~28% e2e — within VTW's range |
| Sham battery (random-layer, late-exit, masking) | VTW's own ablations test exactly these alternatives | NO new mechanism tested |

The mechanism is 1:1. The only structural distinction is the output token type (action tokens vs text tokens), which is purely a renaming — both are autoregressive outputs from the same LLM family.

---

## Why "VLM→VLA = partial overlap" rule does not save SVEE

The checker's standing policy allows VLM efficiency techniques to be classified as **PARTIAL OVERLAP** (not direct conflict) when transferred to VLA, because the real-time constraint and action head create distinct novelty. **SVEE does not invoke that distinction.**

Three failure modes vs. the partial-overlap rule:

1. **No action-specific mechanism claim**: SVEE's hypothesis ("LLM understands vision in early layers, generates action in late layers") is identical to VTW's information-migration hypothesis ("visual info transferred to text tokens in early layers, then text tokens generate"). The "action vs text" relabeling adds no mechanistic content because OpenVLA-OFT tokenizes actions as a vocabulary extension of text — the same autoregressive process.

2. **Sham battery doesn't discriminate**: SVEE's Sham A (random-layer), Sham B (late-exit), Sham C (random masking) are exactly the alternatives VTW already ablated. A SVEE result of "K_vis = optimal early, random ≠ optimal" would just reproduce VTW's finding on LIBERO. No new hypothesis is tested.

3. **No real-time / action-head specific contribution**: SVEE does not propose anything that breaks if VTW is applied directly to OpenVLA-OFT (no action-chunk-aware schedule, no proprio-conditioned K, no per-action-step variation). It is a benchmark transfer, not a method.

---

## Direct precedent: V-MIB NO-GO (Round 8)

The checker memory contains a directly analogous NO-GO from the same pattern:

> **V-MIB (verdicts/no-go/vmib_novelty_verdict.md, 2-3/10)**: InfoPrune (arXiv:2511.19518, 2025-11-24) bottleneck Linear(d→k)+act+Linear(k→d) on VLM vision encoder FFN ... = 3-axis 1:1 match

V-MIB also proposed a VLM-domain technique (bottleneck FFN compression) transferred to VLA. It was NO-GO'd because the VLM-paper mechanism was 1:1 and the VLA "transfer" claim was not load-bearing. **SVEE is the same shape**: VTW occupies the {static-K × vision-token-residual-drop × KV-cache-compatible × Llama-2-7B} cell completely, and "apply it to OpenVLA-OFT on LIBERO" is the entirety of the proposed contribution.

---

## Secondary close-prior papers (corroborating, not blocking)

| Paper | Venue | Overlap | What overlaps | Gap |
|---|---|---|---|---|
| HiDrop | arXiv 2602.23699 (2026) | YELLOW partial | Static "Late Injection" at fixed layer (L_inj=9 for LLaVA-1.5-7B) — inverse direction (vision tokens skipped in early layers, injected at L_inj). Concave Pyramid Pruning at fixed filter layers {10,14,16,18}. | Opposite hypothesis (vision needed late, not early) — but co-occupies static-vision-layer-schedule axis. MLLM only. |
| FastV | arXiv 2403.06764 (ECCV 2024) | YELLOW partial | Static layer-2 cutoff for vision token pruning + attention-ranked per-sample dropping. Adaptive only in which-tokens-drop, not when. | Per-sample token ranking; SVEE proposes total drop, not ranked drop. MLLM only. |
| FREE | ACL 2025 Findings | YELLOW partial | Token-difficulty-tier early exit at layers 1-12, 13-24, 25-32 boundaries. | Per-token, not "all vision after K". MLLM only. |
| DySL-VLA | arXiv 2602.22896 (ICLR 2026) | YELLOW adjacent | Static-dynamic layer-skipping ON OPENVLA-OFT, Jetson Orin 23.2 Hz. | Skips entire LLM layers per-sample (informative + incremental classification), not vision tokens only at static K. |
| MoLe-VLA | arXiv 2503.20384 | YELLOW adjacent | Mixture-of-Layers router on VLA. | Per-sample dynamic layer selection, not static vision-token drop. |
| V²Drop | arXiv 2509.01552 | YELLOW adjacent | Variation-aware vision token dropping, progressive across layers. | Per-token variation criterion, not static-K-layer. MLLM only. |
| LUVC | arXiv 2512.09010 | YELLOW adjacent | "Lossless ultimate vision tokens compression" — progressive compression to elimination at final layer. | Progressive merging schedule, not abrupt static drop. MLLM only. |

None of these alone preempt SVEE — but together with VTW they fully cover the design space for {static layer-K × all-vision-tokens × residual-stream-removal × LLM-backbone × Llama-2-7B family}.

---

## Failed-to-find escape routes

The following potential VLA-specific contributions were searched and **not** preempted, but **not** invoked by SVEE either:

1. **Per-camera static K** (different K for wrist vs scene cameras) — SVEE proposes uniform K across all cameras. (Note: this is CRDS's territory.)
2. **Action-chunk-position-dependent K** (e.g., chunk-step 0 needs full vision, step k>0 can drop earlier) — empty cell, but SVEE does not propose this.
3. **Proprio-gated K selection** (VLA-ADP-style, but for layer-K instead of token-budget) — empty cell, but SVEE does not propose this.
4. **Joint K + INT4/INT8 quantization decision** (the "throughput cliff" axis from `vla_layer_skip_landscape.md`) — empty cell, but SVEE does not propose this.

Any of (1)-(4) could rescue SVEE as a new idea. As written, SVEE = VTW.

---

## Recommendation

**NO-GO. Abandon SVEE as currently framed.**

If the user wants to salvage the direction, the recommended pivots (in order of differentiation strength):

1. **K_vis as a function of action-chunk position** (chunk[0] = full depth, chunk[k] = drop after K_early). This articulates a real VLA-specific axis: action generation has temporal structure that text generation doesn't. Send to `vla-idea-generator` as a new idea.
2. **K_vis per camera role** (CRDS already takes this — see CRDS verdict).
3. **Proprio-conditioned K_vis** (when proprio is informative, drop vision earlier). Adjacent to VLA-ADP — needs separate novelty check.

---

## Required follow-up actions

1. Move `vla-idea-generator/active/svee.md` (or `pending/svee.md`) → `vla-idea-generator/abandoned/svee.md`.
2. Add SVEE row to `vla-idea-generator/BLACKLIST.md` under "절대 금지 Mechanism Family" with preempting paper VTW (arXiv:2405.05803, AAAI 2025).
3. Add VTW to landscape file `vla_efficiency_landscape_2025_2026.md` — it is the canonical static-vision-token-withdrawal paper and was missing from the index.

---

Related memories: [[vmib-novelty-verdict]] [[vla-layer-skip-landscape]] [[vla-efficiency-landscape-2025-2026]]
