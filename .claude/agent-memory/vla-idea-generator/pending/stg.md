---
name: stg
description: Spatial-Token Gating — object-supervised saliency head selects top-K=16 tokens, beating similarity-based pruning on spatial tasks
metadata:
  type: project
  status: pending
  verdict: null
  round: 6
  date_generated: 2026-05-18
  risk: literature-checker may reject as adjacent to #11 (token pruning family)
---

# STG — Spatial-Token Gating via Object-Supervised Saliency

## Core Hypothesis

ChunkedBC v6 vision feature map (8×8×512)에 lightweight saliency head (1×1 conv → 8×8 logits)를 학습 시 supervise (label = LIBERO ground-truth object bbox)하고, inference 시 top-K=16 token만 action head로 forward하면 SR 손실 없이 token 수 75% 감소 + spatial SR이 baseline 동등 이상. 차별성: token selection이 *visual similarity가 아니라 학습된 object-likelihood*에 기반.

## Mechanism

- Vision encoder 동일 + 1×1 conv saliency head (8ch → 1)
- Training: BC loss + λ · BCE(saliency, gt_object_mask) — bbox 8×8 downsample
- Inference: 64 token 중 saliency top-16 → action head
- 나머지 48 token: buffer mean 대체 (positional 정보 유지)

## Sham battery & separation

| Sham | Selection key | Why mechanism wins |
|------|--------------|--------------------|
| Random top-K | random 16 | object 위치 무시, SR 큰 폭 하락 |
| Fixed top-K | 학습 데이터 평균 saliency | task 따라 object 위치 다름 → 하락 |
| Similarity top-K (VLA-Pruner style) | feature 차이 (unsupervised) | ESBL 교훈: proprio dominant regime에서 arm token 무시는 SR 무관 → baseline 동등이지만 mechanism은 + |

핵심: mechanism saliency key는 **object supervision으로 학습된 quantity**. Sham이 같은 key를 reproduce하려면 supervision 사용 → mechanism이 됨.

## Pre-draft empirical anchor (<30 min)

- libero_spatial 5 demo × 10 task에서 LIBERO bbox API로 object cell fraction 측정 (64 grid)
- Pass: object cell ≤ 25% (즉 75% token이 background) → top-16 선택 의미 있음
- 추가: ChunkedBC v6 ResNet18 stage4 feature linear-probe로 "object cell?" AUC
  - AUC > 0.7: feature가 이미 object 정보 carry, STG가 amplify 가능
  - AUC < 0.5: feature 자체에 object 정보 거의 없음, STG가 새 signal 강제

## BLACKLIST check (explicit)

| # | Family | Status |
|---|--------|--------|
| 1-10, 12, 13 | 전 family | NO |
| **11** | **Similarity-based token pruning** | **인접 — 가장 위험** |
| 9 | PUG-Vision | NO — proprio gate 아님 |

### #11 separation

VLA-Pruner / TEAM-VLA / Compressor-VLA / VLA-ADP: 전부 *unsupervised* (feature similarity, attention score).

STG: **명시적 object supervision** via aux head. Selection criterion이 다른 function class.

추가로 STG의 contribution은 throughput improvement가 아니라 *"spatial task에서 random top-K vs object-top-K 5pp 차이"* 라는 **empirical 발견**.

### ESBL adjacency

같은 patch-masking이지만:
- ESBL: motion-derived static patch (unsupervised)
- STG: object-likelihood (supervised)

ESBL 결과로 "motion-based selection은 proprio dominance regime에서 SR 무관" 입증됨 → **selection criterion이 object-supervised여야 SR effect가 생긴다** 는 게 STG contribution.

## Expected SR delta

- A (baseline 64 token) → B (STG 16 token): **−0 to +3pp**
- B with aux: **+3 to +5pp**
- Sham 1 (random): −5 to −10pp
- Sham 3 (similarity): baseline ±2pp
- **B − Sham3 ≥ 5pp** (object-vs-arm token spatial-task 차이가 contribution)

## Track 1 PoC

- 1×1 conv head, bbox→mask helper, top-K gather op
- ~100 lines
- 3 seeds × 150 ep × A/B/Sham1/Sham3 = ~14 GPU-hrs
- Difficulty: 보통

## Publication target

ICRA 2026 short or workshop — "Object supervision beats similarity for VLA token gating"

## Status

**OCB와 RGV가 더 안전. STG는 backup** — literature-checker가 #11로 reject할 risk MEDIUM-HIGH. Defense는 *object supervision*과 *spatial-suite specific contribution*에 의존.
