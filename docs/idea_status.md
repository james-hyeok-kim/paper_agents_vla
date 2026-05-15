# VLA Idea Status — Master Table

Last updated: 2026-05-15 (Round 4 + Validation 완료)

## 활성 아이디어 (CONDITIONAL GO, 우선순위 순)

| 순위 | 아이디어 | 접근 | Novelty | Validator | 다음 단계 |
|---|---|---|---|---|---|
| 1 | AMP-Distill | Distillation (SE(3) + contact) | 5/10 | **6.5/10** | Week-1: ActDistill rotation→SO(3) swap ablation |
| 2 | XV-Dedup | Cross-view token merge (LSH) | 6.5/10 | **6.0/10** | Week-1: cross-view overlap + BFA++ stacking |
| 3 | CP-Sparse | Chunk-position sparse attention | 5/10 | **5.5/10** | Week-1: entropy slope + top-k overlap (AutoHorizon gate) |

## 폐기 아이디어 (NO-GO, 6개)

| # | 아이디어 | 폐기 라운드 | 선점 논문 |
|---|---|---|---|
| 1 | ASMP-Q | R1 | QVLA (ICLR 2026) |
| 2 | TASK-Skip | R2 | DeeR-VLA, DySL-VLA, MoLe-VLA |
| 3 | Temporal Delta KV | R1→R3 | VLA-Cache (NeurIPS 2025), Eventful Transformers |
| 4 | SACV | R1→R3 | SV-VLA (arXiv:2604.02965) |
| 5 | PADS | R3 | FASTER, AsyncVLA, Streaming DP |
| 6 | BSPC | R3 | A2C2 (1:1 match) |
| 7 | PPC-VLA | R4 | AC2-VLA |
| 8 | PUG-Vision | R4 | VLA-ADP |
| 9 | LIC-Chunk | R4 | AutoHorizon |

전체 9개 NO-GO 상세는 `abandoned_ideas.md` 참조.

## 통계

- 총 생성: **12개** (Round 1: 3, Round 2: 3, Round 3: 3, Round 4: 3)
- NO-GO: **9개** (75%)
- CONDITIONAL GO: **3개** (25%)
- GO: **0개**

## 상태 정의

- **Draft**: 생성됨, novelty 검증 미수행
- **Novelty-Checked**: vla-literature-checker 통과
- **CONDITIONAL GO (Validated)**: pre-experiment 통과 시 GO 가능
- **In-Experiment**: vla-experiment-runner 실행 중
- **Submitted**: 논문 제출 완료
- **Abandoned (reason)**: NO-GO 판정

## 결론

2026년 1-4월 ICLR/CoRL submission cycle에서 대부분의 obvious VLA efficiency axis가 선점됨.
True GO는 단일 mechanism level에서 거의 불가능. **CONDITIONAL GO + pre-experiment gate**가 현실적 최선.
