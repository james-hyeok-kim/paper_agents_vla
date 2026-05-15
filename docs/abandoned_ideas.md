# Abandoned Ideas — Blacklist

이 파일에 기록된 아이디어는 novelty 검증에서 NO-GO 판정을 받아 폐기된 것입니다.
**새 아이디어 생성 시 반드시 이 목록을 확인하고 중복되는 방향을 제외하세요.**

---

## [1] ASMP-Q — Action-Sensitivity Mixed-Precision Quantization
- **판정**: NO-GO (Novelty 2/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **QVLA** (ICLR 2026, arXiv:2602.03782): action sensitivity 기반 bit allocation (channel-wise). 핵심 novelty 3가지 모두 선점. Layer-wise는 inferior로 이미 반박됨.
  - DyQ-VLA (arXiv:2603.07904): temporal-dynamic bit switching
  - EaqVLA (arXiv:2505.21567): action head 인접부 민감도 (module-level)
- **폐기 이유**: "perplexity-critical ≠ action-critical" 주장, action sensitivity calibration, action head 인접 레이어 민감도 — 3가지 핵심 claim 모두 QVLA에 선점됨. Layer-wise 접근은 QVLA에서 channel-wise보다 inferior로 이미 결론남.
- **제외할 방향**: action sensitivity 기반 mixed-precision quantization 전반

---

## [2] TASK-Skip — Jetson-Throughput-Aware Action-Quality-Bounded Layer Skipping
- **판정**: NO-GO (Novelty 3/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **DeeR-VLA** (NeurIPS 2024, arXiv:2411.02359): "token confidence 대신 action consistency로 exit" — 핵심 주장과 verbatim으로 동일
  - **DySL-VLA** (ICLR 2026, arXiv:2602.22896): Jetson Orin에서 VLA layer skip 검증 완료 (23.2 Hz)
  - **MoLe-VLA** (arXiv:2503.20384): robot state 기반 ~50K param gating MLP — 구조적으로 동일
  - ActDistill (arXiv:2511.18082): action-prior 기반 dynamic routing
- **폐기 이유**: action-error budget으로 layer skip을 gate하는 VLA 고유 주장 → DeeR-VLA에 선점. Jetson 플랫폼 기여 → DySL-VLA에 선점. State-conditioned gating MLP → MoLe-VLA에 선점.
- **제외할 방향**: action-error/action-consistency 기반 VLA layer skip/exit 전반. Jetson 기반 VLA layer-level adaptive compute (단순 platform 기여로는 부족).

---

## [3] Temporal Delta KV Cache for VLA Vision Encoders
- **판정**: NO-GO (Novelty 2/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **VLA-Cache** (NeurIPS 2025, arXiv:2502.02175): 연속 프레임 간 정적 시각 토큰 KV 재사용, training-free plug-in — 핵심 메커니즘 동일
  - **Eventful Transformers** (ICCV 2023, arXiv:2308.13494): ViT 내부 self-attention token-wise sparse update의 원조 구현체
  - **TTF-VLA** (arXiv:2508.19257): grayscale pixel-diff + attention 기반 temporal token fusion
- **폐기 이유**: ViT KV 캐시 + 패치별 재사용 → Eventful Transformers. VLA 적용 → VLA-Cache. 픽셀/플로우 기반 변화 측정 → TTF-VLA. 모든 셀이 닫혀 있음. 추가로 ViT는 OpenVLA-7B 전체 latency의 작은 비중이라 e2e 임팩트 상한도 낮음.
- **제외할 방향**: 시각 encoder KV 캐싱/재사용, 프레임 간 patch-level delta caching 전반

---

## [4] SACV — Speculative Action-Chunk Verification
- **판정**: NO-GO leaning (Novelty 3-4/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **SV-VLA** (arXiv:2604.02965, 2026): "Speculative Verification for VLA" — 동일 컨셉, large=proposer/small=verifier 역할. 게다가 SV-VLA가 compute 측면에서 SACV보다 ~3배 더 효율적 (large 1 call vs SACV 3 large calls per chunk)
  - **ADAHI** (arXiv:2510.02851): small on-device draft + remote large verify, deviation threshold 기반 reject/accept
  - **Spec-VLA** (EMNLP 2025, arXiv:2507.22424): VLA speculative decoding + relaxed acceptance
  - **KERV** (arXiv:2603.01581): kinematic variability 기반 dynamic threshold
- **폐기 이유**: SV-VLA가 동일 chunk verification 컨셉을 선점하고 compute 측면에서도 더 유리한 설계. Lipschitz 논증 결함 (policy disagreement를 robot dynamics Lipschitz로 bound할 수 없음). Intra-chunk attention invariance 발견이 sparse verification 가정을 약화.
- **제외할 방향**: VLA speculative action chunk verification, small-draft/large-verify 패턴 전반

---

## [5] PADS — Position-Adaptive Denoising Schedule
- **판정**: NO-GO (Novelty 1-2/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **FASTER** (arXiv:2603.19199): Horizon-Aware Schedule(HAS) — intra-chunk position별 denoising step 차등 할당, 정확히 동일 메커니즘. 게다가 pilot study에서 PADS의 "late position = sharper = fewer steps" 가설을 **반대 방향으로 반증**
  - **AsyncVLA** (arXiv:2511.14148): per-position confidence-based denoising 할당
  - **Streaming Diffusion Policy** (arXiv:2406.04806): cross-position pipelining sub-claim 선점
- **폐기 이유**: 메커니즘 axis 전체가 FASTER+AsyncVLA에 선점됨. 핵심 가설도 FASTER에 의해 경험적으로 반증됨.
- **제외할 방향**: intra-chunk position-adaptive denoising step 수 결정 전반

---

## [6] BSPC — Bounded-Staleness Predict-Correct Pipelining
- **판정**: NO-GO (Novelty 2-3/10)
- **폐기 날짜**: 2026-05-15
- **선점 논문**:
  - **A2C2** (arXiv:2509.23224, NeurIPS Workshop 2025): stale frame으로 base VLA → tiny correction head(~32M, 4.7ms) → residual δa. 메커니즘 1:1 동일. SmolVLA + LIBERO에서 검증 완료. RTC와의 "correct vs blend" 차별점도 이미 청구함
  - VLASH (arXiv:2512.01031), REMAC (arXiv:2601.20130), VLA-RAIL (arXiv:2512.24673): 동일 cell 포화
- **폐기 이유**: 핵심 메커니즘이 A2C2와 1:1 동일. RTC 대비 차별점도 A2C2가 먼저 청구함. 2025-09~2026-02 사이 해당 cell이 급격히 포화됨.
- **제외할 방향**: stale observation + correction head 기반 VLA async pipelining 전반

---

## 살아있는 Pivot 방향 (NO-GO 아이디어에서 파생)

ASMP-Q Pivot A (미검증):
- Hardware-Action Co-Optimization: Jetson INT4/INT8 throughput 차이를 *objective 함수 안에* 넣은 unified skip+quantization allocator
- SQAP-VLA(token prune+quant)와 DySL-VLA(layer skip)를 넘어선 "layer-wise {skip, INT4, INT8, FP16} 4-way dynamic allocation"
- **주의**: QVLA의 "layer-wise < channel-wise" 결과를 hardware-aware framing으로 우회할 수 있는지 사전 검토 필요

TASK-Skip Pivot 1 (미검증):
- "Unified Skip-Demote Allocator": {skip=0bit, INT4, INT8, FP16} 4-way를 단일 Lagrangian으로 동적 결정
- Jetson per-bit throughput을 cost 항에 명시 (DySL-VLA는 사후 측정만)
- SQAP-VLA(token-level)와 달리 layer-level unified

---

*새 아이디어 생성 시 이 파일을 먼저 읽고 제외 방향을 확인할 것.*
