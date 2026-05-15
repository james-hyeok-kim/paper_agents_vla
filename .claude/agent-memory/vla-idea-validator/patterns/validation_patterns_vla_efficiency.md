---
name: validation-patterns-vla-efficiency
description: Recurring failure patterns in VLA inference-efficiency ideas — use as a checklist when validating new proposals
metadata:
  type: feedback
---

# Common failure patterns in VLA efficiency ideas

**Why**: Across validations, the same classes of weakness recur. Surfacing them upfront saves a full evaluation cycle.

**How to apply**: When validating any new VLA efficiency idea, run through this list explicitly.

## Pattern 1: "VLM technique applied to VLA" without robot-specific contribution
The robotics analogue of an established LLM/VLM optimization (KV cache, INT8, token pruning, distillation) is 🟡 Partial Overlap, not 🟢 Novel. Required articulation: (a) what fails when you naively port the VLM technique to a VLA, (b) what VLA-specific structure (real-time constraint, action distribution multi-modality, temporal correlation, safety) the new method exploits.

## Pattern 2: Speedup measured on wrong hardware
A100/H100 speedups do not transfer to Jetson/edge. Kernel overhead, memory bandwidth, and small-batch behavior all differ. Always require: stated target hardware, absolute latency budget, not just relative speedup ratio.

## Pattern 3: Absolute latency floor too low for contribution
ACT and other small (<100M) policies already run in 5–15ms. A 2× speedup is 5ms — below human perception threshold and below the control-loop budget. Prefer 300M+ targets (SmolVLA, π0, OpenVLA) where headroom exists.

## Pattern 4: Single-metric ablation gate
"If X < 0.8 → GO." Compound hypotheses need multi-metric gates. Common conflations:
- Entropy decay (does mass concentrate?) ≠ argmax stability (does top-1 stay put?)
- Top-1 accuracy ≠ top-k coverage
- Average attention ≠ tail/worst-case attention (which is what fails fragilely)

## Pattern 5: Contact-rich evaluation missing
LIBERO-Spatial/Object/Goal are mostly free-space. Sparsification or quantization that looks fine there can fail catastrophically in peg-insertion, threading, deformable manipulation. Require ≥1 contact-rich task.

## Pattern 6: Sim-only with no real-robot evidence
Reviewers at CoRL/RSS will hammer sim-only efficiency papers. Even a single real-robot demonstration with one task changes the review outcome.

## Pattern 7: "Position-aware" / "adaptive" framing without proof of variance
If the idea claims to exploit variation along some axis (position, time, layer, task), the Week-1 sanity check must measure: does the axis actually vary in the trained model? Many "adaptive" methods turn out to be uniform-good-enough in disguise.

## Pattern 8: Conflicting literature evidence that is *adjacent*, not *direct*
When a related paper finds an opposing phenomenon on a different architecture (e.g. flow-based VLA finding A, but the proposal is on CVAE VLA), the right move is not "they're different so it doesn't apply" — it's a pre-experiment measurement on the proposal's architecture. Generalization probability should be expressed as a range, not a point.

## Pattern 9: Hidden mechanism unstated
Many proposals bundle 2–3 mechanisms ("schedule" + "keep-set selection" + "kernel implementation"). Each can fail independently. Force each mechanism to be stated as a separable claim with its own validation.
