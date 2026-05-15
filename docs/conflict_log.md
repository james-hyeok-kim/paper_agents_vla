# Literature Conflict Log

---

## ASMP-Q (Action-Sensitivity Mixed-Precision Quantization)

### 직접 충돌

**QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization**
- Venue: ICLR 2026
- arXiv: 2602.03782
- 충돌 내용:
  - Action sensitivity 기반 bit allocation → 수학적으로 동일 (1차 Taylor: Δa ≈ J·ΔW)
  - "Perplexity-aware LLM quantization이 VLA에 부적합" — 동일한 동기
  - Projector + action head가 action에 가장 민감 — ASMP-Q novelty 주장 #3과 동일
  - Layer-wise를 baseline으로 비교하여 **inferior로 결론** → ASMP-Q 접근을 부정적으로 선점
- 열린 공간: Hardware-aware bit allocation (엣지 HW 제약 결합)

### 부분 충돌

**DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied VLA Models**
- arXiv: 2603.07904
- 충돌 내용: VLA-specific mixed-precision (temporal axis 기반)
- 열린 공간: Layer-axis + temporal-axis 결합 (Phase-conditioned layer-wise)

**EaqVLA: Encoding-aligned Quantization for VLA Models**
- arXiv: 2505.21567
- 충돌 내용: Action head 인접부 민감도 (module-level)
- 열린 공간: LeRobot / ACT / Diffusion Policy 적용 (QVLA는 OpenVLA family만 다룸)

### 보완 관계

**QuantVLA**
- Venue: CVPR 2026
- Uniform W4A8, scale calibration이 주 contribution — mixed-precision 아님
- 관련성: VLA PTQ baseline으로 활용 가능

---

*항목 추가 형식:*
```
## [아이디어 이름]
### 직접 충돌 / 부분 충돌 / 보완 관계
**[논문 제목]**
- Venue: ...
- arXiv: ...
- 충돌 내용: ...
- 열린 공간: ...
```
