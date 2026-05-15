---
name: idea-instruction-conditioned-chunk-budget
description: Language instruction predicts variable outer chunk length, cutting total VLA forward passes (distinct from intra-chunk denoising adaptation)
metadata:
  type: project
---

# Language-Instruction-Conditioned Variable Chunk Length (LIC-Chunk)

## Core hypothesis
The natural-language task instruction carries enough information about motion structure (e.g., "pour water slowly" vs. "move cup to shelf") to predict the optimal *outer* action chunk length H, so a chunk-length predictor conditioned on instruction + initial state can reduce total VLA forward calls by 1.5-3× without intra-chunk modifications.

## Technical approach
- Add a lightweight chunk-budget head g(instruction_embedding, s_0) → H ∈ {4, 8, 16, 32, 64}; trained with a budget-aware imitation loss that penalizes wasted chunk steps (open-loop divergence from teacher) and rewards long H when feasible.
- At inference: run VLA once → produce H actions → execute → repeat. The number of full VLA inferences over an episode drops as ⌈episode_len / E[H]⌉.
- Pretrain g(·) on (instruction, optimal-open-loop-horizon) labels mined offline from existing trajectories using a closed-loop drift threshold.

## Why VLA-specific
- Pure LLMs do not have an *outer execution horizon* — they generate then stop. Robots execute, then re-perceive; this re-perception cost is the dominant term for short fixed chunks.
- Diffusion/flow policies and ACT use *fixed* chunk length (typically 8 or 16). Predicting H from language is uniquely VLA because only here does instruction language causally constrain motion smoothness/duration.

## Distinct from excluded work
- FASTER / AsyncVLA / Streaming Diffusion Policy — they adapt *denoising steps per position within a chunk*. This work fixes denoising and varies *H itself*. Different axis.
- Action chunking literature (ACT, π0) — these *fix* H. Variable-H conditioned on language is the novel axis.
- Adaptive RL horizon work — typically uses *value/state* signals, not language; framing must lead with language conditioning to avoid that bucket.

## Safety
A miss-prediction of H that is too long causes open-loop drift. Mitigation: train g(·) to be *conservative* (penalize over-prediction more), and add a state-deviation early-abort that triggers re-inference if observed state diverges from predicted.

## Target hardware
Any (Jetson and cloud both). The savings are *call-count* reductions, not per-call FLOPs.

## Expected gains
1.5-3× reduction in total VLA inferences per episode on long-horizon LIBERO / RoboCasa benchmarks; task success rate ≥ fixed-H baseline because long phases stop being interrupted by redundant re-inference.

## Adjacent landmines
- Re-check that no recent paper conditions chunk length on language. The closest miss is value-conditioned adaptive horizon (RL flavor) — emphasize language-as-signal.
- Check overlap with "skill chunking" literature (e.g., BC-Z skill tokens) — distinguish by H being a *continuous-control budget*, not a skill ID.

## Venue
RSS 2026 / CoRL 2026. RSS values the policy-architecture angle.

## Suggested next step
Run vla-literature-checker on "Language-Instruction-Conditioned Variable Chunk Length" + keywords {adaptive action horizon language, variable chunk length VLA, instruction-conditioned execution horizon}.
