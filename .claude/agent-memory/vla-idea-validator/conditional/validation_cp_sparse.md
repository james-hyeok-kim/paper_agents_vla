---
name: validation-cp-sparse
description: CP-Sparse (Chunk-Position-Aware Sparse Attention for ACT/SmolVLA) idea validation — CONDITIONAL GO with tightened gates, primary threat is AutoHorizon (arXiv:2602.21445)
metadata:
  type: project
---

# CP-Sparse Validation (2026-05-15)

**Idea**: Position-dependent cross-attention sparsity for ACT/SmolVLA action chunks; later chunk positions assumed to need fewer conditioning tokens than earlier ones. Keep-set picked at position 0 (argmax) and reused.

**Verdict**: CONDITIONAL GO. Overall **5.5/10**.

## Why: Core risks
- **AutoHorizon (arXiv:2602.21445, Feb 2026) verified real.** Finds **(i)** intra-chunk actions attend *invariantly* to VL tokens in π0.5/GR00T-N1.5 (directly contradicts CP-Sparse's *position-dependent* motivation), and **(ii)** initial+terminal action tokens are stable anchors that intermediates organize around. Finding (ii) partially helps CP-Sparse (anchors are stable → keep-set candidates) but (i) attacks the entropy-decay-with-position premise head-on.
- **AutoHorizon-to-ACT/SmolVLA generalization probability: ~40–60% (range, not point estimate).** AutoHorizon is on flow-based VLAs (π0.5, GR00T-N1.5). ACT is CVAE-decoder; SmolVLA action expert is flow-matching-style. Attention sink phenomena tend to be transformer-universal (argues FOR generalization) but training objective changes what cross-attention optimizes (argues AGAINST). Don't claim false precision — Week-1 measurement resolves it.
- **Single-metric Week-1 gate is wrong.** Argmax overlap tests *keep-set reusability* only, not *entropy decay with position*. You can have HIGH overlap AND strong entropy decay simultaneously — that supports position-dependent k but kills keep-set-reuse shortcut. Must measure three quantities separately.
- **Absolute latency scope is small.** ACT ~10ms, SmolVLA-450M ~30–45ms (already optimized: skips half VLM layers, async inference). 2× speedup on ACT = 5ms. Need explicit hardware + absolute latency target stated, otherwise unreviewable.
- **Safety**: Late-chunk = late-action = contact-phase in manipulation. Sparsification errors at late positions are the highest-risk failure mode. Must include contact-rich task in eval.

## How to apply: GO milestones (tightened from user proposal)

1. **Week-0**: Verify ACT and SmolVLA architectures locally; pick concrete checkpoints (e.g. LeRobot ACT, SmolVLA-450M HuggingFace). State hardware target (e.g. Jetson Orin Nano / AGX / RTX 4090) and absolute latency budget (e.g. "<20ms end-to-end on Jetson AGX at chunk_len=50").

2. **Week-1 three-metric gate** (replace single argmax-overlap gate):
   - **(A) Per-position cross-attention entropy curve**: Does H(i) decay monotonically with chunk position i? Slope > 0.3 nats/position needed to justify schedule s(i).
   - **(B) Top-k mass concentration at k=8**: For each i, what fraction of attention mass do the top-8 tokens capture? Needs >0.8 at late positions for k=8 to be feasible.
   - **(C) Argmax / top-k overlap (position 0 vs chunk_len-1)**: If overlap >0.7, AutoHorizon-style invariance holds → pivot keep-set selection to "union of top-k across early positions" instead of "first-position only".
   - Decision rule: (A) and (B) jointly justify the schedule; (C) tells you *how* to pick the keep-set, not whether to proceed.

3. **Pivot if Week-1 fails**: If (A) fails (entropy flat across positions), the schedule motivation is dead — must pivot to a different axis (e.g. token-importance-based sparsity, like VLA-Pruner's visual-token axis but applied to all conditioning tokens). Don't try to rescue CP-Sparse if its core premise is empirically false.

4. **Primary target**: SmolVLA-450M (more headroom: 30–45ms baseline, action expert is the bottleneck). ACT as ablation only — its absolute latency is too small for standalone publication.

5. **Eval bar for publication**:
   - ≥1.5× speedup on action expert forward pass on stated hardware
   - ≤5% success rate drop on ≥2 LIBERO suites + ≥1 contact-rich task (e.g. peg-insertion or LIBERO-Long)
   - At least one real-robot demonstration (even single-task) — reviewers will hammer sim-only papers
   - Direct comparison with AC²-VLA, VLA-Pruner on same checkpoints

6. **Framing**: Position the axis distinction explicitly — "AC²-VLA (depth/spatial/temporal) and VLA-Pruner (visual token importance) sparsify *which* tokens; CP-Sparse sparsifies *how many* tokens are needed *as a function of action-chunk position*." If Week-1 (C) shows invariance, drop the keep-set-reuse claim and reframe as "uniform keep-set, position-dependent k."

## Hidden assumption to flag
The "first-position argmax → keep-set → reused" mechanism is a *separate* claim from "entropy decays with position." Even if entropy decays, argmax can shift position-by-position. The agent's design must allow:
- Plan A: first-position argmax keep-set (if (C) overlap high)
- Plan B: union-of-top-k across early positions (if (C) overlap moderate)
- Plan C: per-position keep-set with shared sparsity mask budget (if (C) overlap low but (A)(B) hold)

## What kills this idea
- (A) flat entropy across positions on ACT AND SmolVLA → NO-GO, pivot axis entirely.
- (B) <0.6 top-8 mass at late positions → k=8 infeasible, would need k=16+, speedup vanishes.
- AutoHorizon invariance holds tight on SmolVLA → keep-set is one global set, "position-aware" framing is fiction.
- FlashAttention block-sparse on small action chunks (50 queries × ~600 keys) has kernel overhead exceeding savings on Jetson — measure on target hardware, not A100.

Related: [[validation-patterns-vla-efficiency]]
