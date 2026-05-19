---
name: ocb
description: Object-Centric Bottleneck — Slot Attention layer on vision encoder output for explicit object-binding in chunked BC policies
metadata:
  type: project
  status: pending
  verdict: null
  round: 6
  date_generated: 2026-05-18
---

# OCB — Object-Centric Bottleneck for VLA spatial reasoning

## Core Hypothesis

ChunkedBC v6 vision encoder의 출력에 K개 object-slot token (K=5) + 1 global summary로 압축하는 Slot Attention bottleneck을 삽입하면, libero_spatial SR이 baseline 대비 +5pp 이상 향상된다. 이유: spatial-reasoning 정책은 "이 object를 저 receptacle로" binding이 필요한데, 현재 ResNet18 dense feature는 binding이 implicit. Slot bottleneck은 binding을 explicit하게 만든다.

## Mechanism (architectural, sham-robust by function-class)

- ResNet18 stage4 (8×8×512) → **Slot Attention** (K=5, 3 iter) → K×D slot vector
- Policy head input: `[slot_1..slot_5, avg_pool, proprio]`
- baseline: 동일 dim의 `[avg_pool, proprio]`, slot 자리에 zero pad
- chunk=16, action head 변경 없음

## Sham battery & separation

| Sham | Output | Why mechanism wins |
|------|--------|--------------------|
| Random K spatial sample | K개 1×1 patch | Object identity ↔ slot index 학습 불가 |
| Uniform K-grid crop | K×D vector | Iterative competition 없음 |
| K=1 (no binding) | 1 slot | binding 정보 없음 |

핵심: mechanism output은 *iterative competition*으로 형성된 K vector. Sham이 reproduce하려면 attention iteration을 복원해야 함 → 그 순간 sham이 mechanism이 됨.

## Pre-draft empirical anchor (<30 min)

- libero_spatial 5 demo × 10 task의 agentview에서 SAM2 (또는 color-cluster)로 object mask 추출
- frame-to-frame object centroid variance vs task-mean centroid variance 비교
- Pass: task variation의 ≥60%가 object placement variation으로 설명됨
- 추가: ChunkedBC v6 stage4 feature에서 (object_region_pixel, action) MI vs (background_region, action) MI

## BLACKLIST check (explicit)

| # | Family | Status |
|---|--------|--------|
| 1 | Quantization | NO |
| 2 | Layer skip | NO |
| 3 | Bit switching | NO |
| 4 | Vision KV cache | NO |
| 5 | Speculative chunk | NO |
| 6 | Denoising schedule | NO |
| 7 | Stale + correct | NO |
| 8 | LLM skip | NO |
| 9 | PUG (proprio-gated vision) | NO — vision은 항상 forward, gating 없음 |
| 10 | Chunk length adapt | NO |
| 11 | Similarity token pruning | NO — pruning이 아니라 fixed-K binding |
| 12 | View selection | NO |
| 13 | Chunk-position attention | NO |

ESBL adjacency: visual masking이 아니라 *visual binding restructure*. Proprio dominance를 정면 돌파 (proprio가 못 가진 object-binding을 architectural slot으로).

## Expected SR delta

- A (baseline) → B (OCB): **+5 to +8pp**
- Sham (random K): baseline ±2pp
- B − Sham ≥ 5pp

## Track 1 PoC

- Implementation: Slot Attention ~80 lines, inject ~30 lines
- 3 seeds × 150 ep × A/B/Sham1/Sham2 = ~12 GPU-hrs
- Difficulty: 보통

## Publication target

CoRL 2026 — "Object binding via slot bottleneck enables small chunked BC to match larger VLA on spatial reasoning"
