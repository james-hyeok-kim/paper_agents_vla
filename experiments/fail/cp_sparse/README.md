# Experiment 3 — CP-Sparse Chunk-Position Attention PoC

## Metadata
- **날짜**: 2026-05-15
- **Tier**: PoC (합성 chunk task, tiny ACT-like 모델)
- **상태**: ❌ FAIL — 핵심 가설 미입증
- **연결 아이디어**: `cp-sparse` (현재 `.claude/agent-memory/vla-idea-generator/abandoned/cp-sparse.md`)
- **GPU**: 1× B200, 9초
- **Validator 점수 (PoC 전)**: 5.5/10 CONDITIONAL GO

## 검증한 가설
ACT/SmolVLA action-chunk decoder에서 후반 chunk position (i > L/2)이 전반보다 **더 sharp한** cross-attention을 가진다 (즉 더 작은 conditioning token subset에 집중). 이 가설이 맞다면 position-dependent sparsification이 정당화됨.

## 방법
- **모델**: Tiny ACT-like (d_model=128, chunk_len=50, ctx_len=64, 2-layer encoder + cross-attention weight 노출 2-layer decoder)
- **합성 task**: position-conditioned dependency (`action[t] = ctx[t · ctx_len/chunk_len] · proj` — 각 position이 다른 작은 context 필요)
- **학습**: 2000 episode에서 500 iter
- **Metric**:
  - A: per-position entropy slope (target end-to-end Δ > 0.3 nats)
  - B: late position에서 top-8 attention mass (target > 0.8)
  - C: pos 0과 pos L-1 argmax overlap (Jaccard <0.7 GO / >0.95 NO-GO)

## 핵심 결과

| Position | Entropy (nats) | Top-8 mass |
|---|---|---|
| 0 | 3.66 | 0.44 |
| 25 (중간) | 3.68 | 0.43 |
| 49 (마지막) | 3.74 | 0.41 |

| Gate | Target | Result | 상태 |
|---|---|---|---|
| A: entropy 감소 | Δ > 0.3 nats | +0.075 nats (slope 0.004) | ❌ FAIL |
| B: top-8 mass late ≥ 0.8 | >0.8 | 0.41 | ❌ FAIL |
| C: argmax Jaccard pos0 vs last | <0.7 → GO | 0.06 | ✅ Signal (단 caveat 있음) |

## 중요 발견

### 1. 원안 가설 반박됨
Entropy와 top-k mass가 chunk position에 따라 **거의 일정**. 후반 position이 sharper하지 않음. "후반 position은 적은 context 필요"라는 framing은 이 setup에서는 틀림.

### 2. AutoHorizon Invariance도 관측 안 됨
Jaccard 0.06 (매우 낮음)은 각 position이 **다른** context에 attending함을 의미 — sharper도 invariant도 아님. 제3의 시나리오 등장: position마다 다른 작은 subset, subset 크기는 일정.

### 3. 합성 데이터 Artifact 경고
합성 task가 hard-coded position→context 매핑 (`action[t] ∝ ctx[t · 64/50]`)으로 구성됨. 모델이 정확히 그것을 학습 → position-specific attention이 생성됨. **Gate C 결과를 편향시킴.** 실제 ACT/SmolVLA는 다른 패턴 가능.

## Direction (이 실험의 의미)

- **현재 형태로는 dead idea**. 원안 "later = sharper" 주장이 best-case 합성 setup에서도 holds되지 않음.
- **가능한 pivot** (미진행): "Position-conditioned uniform-k sparse attention" — 각 position이 비슷한 크기의 keep-set을 사용하되 위치별로 다르게. Headline 훨씬 약하고 작은 chunk_len에서 engineering speedup도 의심스러움.
- **AutoHorizon caveat**: AutoHorizon의 invariance 주장은 실제 π0.5/GR00T flow 모델 위. 여기서 검증 불가.

## 한계 / 주의사항
- 합성 task가 position-specific dependency를 명시적으로 유도; 실제 chunked action dynamics는 더 smooth할 수 있음
- Tiny model (d=128); 큰 모델은 다른 attention 패턴 보일 수 있음
- 500 iter는 underfit 가능; 수렴 후 거동 다를 수 있음
- ACT (CVAE-decoder) vs SmolVLA (flow-matching) 아키텍처 차이 — ACT-like만 테스트됨

## 다음 단계 (재방문 시만)
- 실제 SmolVLA-450M의 LIBERO에서 attention 분석만이 결정적 테스트
- 그때까지 아이디어는 `abandoned/`에 머무름

## 파일
- `poc.py` — 스크립트
- `results.json` — 측정값
- `run.log` — 실행 로그
- (체크포인트 없음 — tiny model 폐기)
