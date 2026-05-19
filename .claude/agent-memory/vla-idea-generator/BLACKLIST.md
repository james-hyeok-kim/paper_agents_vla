# VLA Idea Blacklist — MUST READ BEFORE GENERATION

**Purpose**: 새 아이디어 생성 시 반드시 이 파일을 읽고, 아래 mechanism family에 속하는 아이디어는 **절대 생성하지 마세요**. literature-checker NO-GO와 idea-validator FAIL 양쪽 결과를 통합한 목록입니다.

**Last updated**: 2026-05-19

---

## 절대 금지 Mechanism Family (literature NO-GO)

| # | Mechanism family | Preempting paper(s) | Source file |
|---|---|---|---|
| 1 | Action sensitivity 기반 mixed-precision quantization | **QVLA** (ICLR 2026) | `abandoned/asmp-q.md` |
| 2 | Action-error/consistency 기반 layer skip/exit | **DeeR-VLA** (NeurIPS 2024), DySL-VLA, MoLe-VLA | `abandoned/task-skip.md` |
| 3 | Temporal-dynamic bit switching | DyQ-VLA (2026) | `abandoned/asmp-q.md` |
| 4 | Vision encoder KV temporal cache | **VLA-Cache** (NeurIPS 2025), Eventful Transformers, TTF-VLA | `abandoned/temporal-delta-kv.md` |
| 5 | Speculative action-chunk verification | **SV-VLA** (2604.02965), ADAHI, Spec-VLA, KERV | `abandoned/sacv.md` |
| 6 | Intra-chunk position-adaptive denoising | **FASTER** (2603.19199), AsyncVLA, Streaming Diffusion Policy | `abandoned/pads.md` |
| 7 | Stale frame + correction head async pipelining | **A2C2** (2509.23224), VLASH, REMAC, VLA-RAIL | `abandoned/bspc.md` |
| 8 | Post-LLM action prior caching / LLM backbone skip | **AC²-VLA** (2601.19634), FlashVLA, LangForce, RAEA | `abandoned/ppc-vla.md` |
| 9 | Proprioception-gated visual compute (motion-derived) | **VLA-ADP** (2509.22093) | `abandoned/pug-vision.md` |
| 10 | Variable execution horizon / chunk length adaptation | **AutoHorizon** (2602.21445), Mixture of Horizons | `abandoned/lic-chunk.md` |
| 11 | Single/multi-view visual token pruning (similarity-based) | VLA-Pruner, TEAM-VLA, Compressor-VLA, VLA-ADP, BFA++ (view-level) | `landscape/vla-multiview-token-pruning-2025.md` |
| 12 | Cross-view binary view-granularity selection | **BFA++** (2602.20566) | (literature finding) |
| 13 | Intra-chunk attention invariance reframing (decreasing entropy by position) | AutoHorizon found invariance; PoC failed to reproduce in tiny ACT | `abandoned/cp-sparse.md` |
| 14 | ViT MLP intermediate bottleneck (V-MIB family) | InfoPrune (2511.19518) + Diversity-Guided MLP Reduction (2506.08591) + Compressor-VLA (2511.18950) | `abandoned/v-mib.md` |
| 15 | **Static vision-token withdrawal from LLM residual stream at fixed layer K (SVEE family)** | **VTW** (arXiv:2405.05803, AAAI 2025) — Llama-2-7B family, KL-divergence-chosen static K, KV-cache-compatible drop. HiDrop (2602.23699), V²Drop (2509.01552), LUVC (2512.09010), FREE (ACL 2025) cover adjacent token-pruning axes | `abandoned/svee.md` (pending move) |

---

## 절대 금지 Idea (Exact Match)

다음 정확한 idea는 이미 폐기됨:

- **ASMP-Q** — Action-Sensitivity Mixed-Precision Quantization
- **TASK-Skip** — Jetson-Throughput-Aware Action-Quality-Bounded Layer Skipping
- **Temporal Delta KV Cache** for VLA Vision Encoders
- **SACV** — Speculative Action-Chunk Verification
- **PADS** — Position-Adaptive Denoising Schedule
- **BSPC** — Bounded-Staleness Predict-Correct Pipelining
- **PPC-VLA** — Policy-Prior Cache
- **PUG-Vision** — Proprio-Uncertainty-Gated Vision Compute
- **LIC-Chunk** — Language-Instruction-Conditioned Variable Chunk Length
- **CP-Sparse** — Chunk-Position-Aware Sparse Attention (PoC failed core hypothesis)
- **V-MIB** — ViT MLP Intermediate Bottleneck
- **SVEE** — Static Vision-token Early-Exit (= VTW on OpenVLA-OFT, no action-specific mechanism)

---

## Idea-Validator FAIL 기록

(현재 없음. validator는 모두 CONDITIONAL GO 판정. 향후 FAIL 발생 시 여기에 추가.)

---

## 사용 규칙

### vla-idea-generator 호출 시
1. **MUST**: 이 BLACKLIST.md를 먼저 읽고 모든 금지 항목을 회피한다
2. **MUST**: 새 idea가 위 13개 mechanism family 중 어느 것에 속하는지 명시적으로 점검한다
3. **MUST**: 해당하면 즉시 폐기하고 다른 방향 탐색

### Idea-Validator NO-GO/FAIL 판정 시
- validator는 FAIL 판정 직후 이 BLACKLIST.md에 해당 idea와 mechanism family를 추가해야 한다
- 추가 형식: `| # | Mechanism family | Preempting / Failed reason | Source file |`

### Literature-Checker NO-GO 판정 시
- checker는 NO-GO 판정 직후 이 BLACKLIST.md에 해당 mechanism family를 추가해야 한다

---

## 살아있는 방향 (참고)

CONDITIONAL GO 또는 PoC PASS 상태:
- **CPR-Distill** (구 AMP-Distill) — Contact-phase reweighted distillation. PoC PASS (specificity 67x). 다음: LIBERO 실험
- **XV-Dedup** — Cross-view token merge (token-granularity, BFA++와 stackable). PoC partial pass (overlap 70% 입증). 다음: InfoNCE LSH 학습

피해야 할 인접 영역 (확장 시 충돌 가능):
- VLA distillation에서 Euclidean MSE 단독 (ActDistill, VITA-VLA에 선점)
- π0-style flow matching SE(3) from-scratch policy (RFMP에 선점)
- ACT chunk position-attention 메커니즘 자체 (AutoHorizon, AC²-VLA가 인접)
