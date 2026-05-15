---
name: qvla-conflict
description: QVLA (ICLR 2026) directly preempts action-sensitivity-guided mixed-precision quantization for VLA - first-order Taylor + bit allocation
metadata:
  type: project
---

QVLA (arxiv 2602.03782, ICLR 2026, AutoLab-SAI-SJTU) is a direct conflict for any "action-sensitivity-guided mixed-precision quantization for VLA" idea.

Core overlap:
- Uses action-space sensitivity (not perplexity) to guide bit allocation
- First-order Taylor approximation via Jacobian: Δaction ≈ J · ΔW
- Compares layer-wise vs channel-wise explicitly (Table 3); shows channel-wise wins
- Explicitly identifies projector + action head as most action-sensitive (Figure 1a) - exact same novelty claim
- Bit set {0,2,4,8,16} with greedy demotion (close cousin to knapsack)
- Tested on OpenVLA-OFT / LIBERO

What's NOT covered (residual differentiation possible):
- Uses greedy demotion, not strict knapsack/ILP
- Channel-wise granularity only; layer-wise dismissed as inferior
- No hardware-aware latency/energy term in objective
- No edge deployment (Jetson) angle

**Why:** Knowing the field state lets future novelty checks avoid recommending preempted ideas.
**How to apply:** Any VLA quantization proposal that uses action-error gradients for bit allocation must address QVLA, DyQ-VLA, EaqVLA prior art and identify a sharper differentiation than "we do it layer-wise instead of channel-wise."

Related: [[dyq-vla-conflict]] [[eaqvla-conflict]] [[quantvla-base]]
