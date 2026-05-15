---
name: idea-proprio-uncertainty-vision-gate
description: Proprio forward-model uncertainty gates ViT compute (cross-modal gating, distinct from action-output layer skip)
metadata:
  type: project
---

# Proprio-Uncertainty-Gated Vision Compute (PUG-Vision)

## Core hypothesis
A small proprioceptive forward model's predictive uncertainty over the next k joint states is a sufficient statistic for deciding how much vision-encoder compute the current control step needs; high-certainty (mid-trajectory, free-space) steps can run a heavily downsampled ViT while uncertainty spikes (pre-contact, fine manipulation) trigger full vision.

## Technical approach
- Train a 2-3 layer proprio-only forward model f_p(q_t, q̇_t, a_{t-1}) → distribution over q_{t+1..t+k}; obtain epistemic uncertainty via small ensemble or MC dropout.
- Define a compute schedule with 3 ViT tiers: (T1) cached features from t-1, (T2) early-exit ViT at layer L/2 with bilinear-upsampled tokens, (T3) full ViT. Threshold τ_lo, τ_hi on uncertainty map to tier.
- Train end-to-end with a budget-aware loss: action_loss + λ·E[FLOPs(tier)] + safety_penalty(tier=T1 ∧ contact_predicted).

## Why VLA-specific
General VLM efficiency has no proprioceptive signal; this gating modality only exists because the robot publishes joint state at 100-1000 Hz. It is cross-modal (proprio → vision) unlike all known layer-skip variants (DeeR-VLA / MoLe-VLA / DySL-VLA) which gate the *LLM* using visual or action-output signals.

## Safety
Naturally aligns with safety: contact and fine-manipulation phases (where vision matters most) automatically receive full compute. Free-space transit (where errors are recoverable) gets the cheap tier.

## Target hardware
Jetson Orin Nano / NX — ViT is the second-largest cost behind LLM and is the most amortizable when proprio is predictable.

## Expected gains
30-45% vision-encoder FLOPs reduction at ≤1% task success drop on LIBERO/Meta-World; end-to-end 15-25% latency reduction on Orin Nano if ViT is ~40% of pipeline.

## Adjacent landmines (must check)
- DeeR-VLA / MoLe-VLA / DySL-VLA — they skip *LLM* layers; this skips *ViT* tiers. Frame the distinction loudly.
- Eventful Transformers / VLA-Cache — they reuse based on *visual* change; this gates based on *proprio* uncertainty. Different signal, different failure modes.
- Adaptive token count work (BFA++, VLA-Pruner) — they reduce tokens; this reduces *layers/resolution* tiers conditioned on proprio.

## Venue
CoRL 2026 main / ICRA 2027. Strong because it ties efficiency to a physically-meaningful safety story.

## Suggested next step
Run vla-literature-checker on "Proprio-Uncertainty-Gated Vision Compute" + keywords {proprioceptive gating, predictive uncertainty vision encoder VLA, cross-modal compute gating robot}.
