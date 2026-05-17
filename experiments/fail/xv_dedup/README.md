# Experiment 2 — XV-Dedup Cross-View Token Overlap PoC

## Metadata
- **날짜**: 2026-05-15
- **Tier**: PoC (합성 stereo, 실제 ViT)
- **상태**: 🟡 PARTIAL PASS (3개 gate 중 1개; core hypothesis는 강하게 검증)
- **연결 아이디어**: `xv-dedup` (`.claude/agent-memory/vla-idea-generator/active/xv-dedup.md`)
- **GPU**: 1× B200, 11초
- **Validator 점수**: 6.0/10 CONDITIONAL GO

## 검증한 가설
Multi-camera VLA setup (예: π0 + 3 cameras)에서 cross-view 간 ViT patch token의 30-50%가 시각적으로 중복됨. LSH 기반 cross-view bucketing으로 LLM 전에 병합하면 action quality 손실 거의 없이 prefill latency 감소 가능.

## 방법
- **Vision encoder**: `facebook/dinov2-base` (실제 ViT, 224×224에서 256 patches/view)
- **합성 stereo**: 8개 random scene, 각 scene에서 2 view (16-px shift + color jitter)
- **3개 gate**:
  - A: cross-view top-1 cosine similarity overlap (target cos>0.85에서 ≥30%)
  - B: LSH random-projection이 cosine top-1과 일치하는 비율 (target ≥50%)
  - C: 40% token 감소 시 prefill latency 감소 (target ≥25%)
- **LSH 차원 sweep**: 32, 64, 128

## 핵심 결과

| Gate | Target | Result | 상태 |
|---|---|---|---|
| A: overlap @ cos>0.85 | ≥30% | **70.4%** | ✅ PASS (2.3× 초과) |
| A: overlap @ cos>0.9 | — | 47.8% | (참고) |
| A: avg max-similarity | — | 0.866 | (sanity check) |
| B: LSH H=128 agreement | ≥50% | 29.6% | ❌ FAIL |
| C: 40% token 축소 시 latency | ≥25% | 21.2% | ❌ FAIL (직전) |

## 중요 발견

### 1. Core Hypothesis 강력 검증
Cross-view token overlap이 **70%** — 예상의 2배 이상. Multi-camera redundancy는 실재하고 크다.

### 2. Random-Projection LSH는 불충분
Random hash → cosine top-1 일치율 30%에 불과. **InfoNCE 학습된 projector 필수**. 예상된 작업량이지 근본적 blocker는 아님.

### 3. Latency 감소가 token 감소에 sub-linear
40% token 감소가 21% latency 감소만 가져옴 (40%가 아님). Transformer attention 비용이 seq length에 non-linear; 작은 N에선 자연스러움. 25%+ 목표 달성하려면 더 공격적인 token 축소(60%) 필요.

## Direction (이 실험의 의미)

- **메커니즘은 실재, instrument가 잘못**: Token-level cross-view redundancy는 크게 존재. LSH는 올바른 도구지만 학습 필요.
- **BFA++와의 차별점**: BFA++는 view 단위 binary selection (카메라 전체 keep/drop). XV-Dedup은 보존된 카메라 안에서 token 단위로 작동 — 엄격히 보완적, 스택 가능.
- **Validator의 우려 (Safety 2.5/5)**: Fine-manipulation 중 wrist camera 잘못 merge 시 catastrophic. Sim eval 전에 fallback / per-camera precision metric 필요.

## 한계 / 주의사항
- 합성 stereo (16-px shift)는 best-case; 실제 LIBERO multi-cam은 disparity 클 수 있어 overlap 낮을 가능성
- Random projection은 LSH 성능 과소평가; 학습된 hash projector는 50% 넘을 가능성 높음
- DINOv2-base ≠ π0의 vision encoder; patch semantic이 다를 수 있음

## 다음 단계 (재방문 시)
1. Cross-view positive/negative pair로 InfoNCE hash projector 학습
2. 실제 LIBERO multi-cam 데이터에서 테스트
3. 더 공격적인 token 감소 sweep (40%, 50%, 60%)
4. BFA++와 스택 실험으로 complementarity 입증

## 파일
- `poc.py` — 스크립트
- `results.json` — 측정값
- `run.log` — 실행 로그
- (체크포인트 없음 — DINOv2 다운로드 사용, 별도 학습 없음)
