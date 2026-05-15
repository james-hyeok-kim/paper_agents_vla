# Agent Memory Index — VLA Inference Efficiency Research

전체 agent-memory의 마스터 인덱스. 각 sub-agent는 자신의 폴더 안에서 status별 sub-folder 구조를 유지합니다.

## Folder Structure

```
agent-memory/
├── MEMORY.md                   ← this file
│
├── vla-idea-generator/
│   ├── MEMORY.md
│   ├── BLACKLIST.md            ← MUST READ before idea generation
│   ├── pending/                ← 새로 생성, literature-check 대기
│   ├── active/                 ← CONDITIONAL GO 또는 PoC PASS
│   └── abandoned/              ← NO-GO 또는 FAIL
│
├── vla-literature-checker/
│   ├── MEMORY.md
│   ├── verdicts/
│   │   ├── no-go/
│   │   └── conditional-go/
│   └── landscape/              ← 분야 전체 survey 메모
│
├── vla-idea-validator/
│   ├── MEMORY.md
│   ├── passed/                 ← GO
│   ├── conditional/            ← CONDITIONAL GO
│   ├── failed/                 ← FAIL
│   └── patterns/               ← 재사용 가능한 failure heuristics
│
├── vla-experiment-planner/
│   ├── MEMORY.md
│   ├── active/
│   ├── completed/
│   └── reference/              ← compute calibration 등
│
└── vla-experiment-runner/
    └── (실험은 `experiments/`에 직접 저장)
```

## Current Status (2026-05-15)

| Agent | Total artifacts | Status distribution |
|---|---|---|
| vla-idea-generator | 12 ideas | 2 active / 0 pending / 10 abandoned |
| vla-literature-checker | 16 files | 7 no-go / 2 conditional / 7 landscape |
| vla-idea-validator | 6 files | 0 passed / 4 conditional / 0 failed / 2 patterns |
| vla-experiment-planner | 2 files | 1 active / 0 completed / 1 reference |

## Cross-Agent Workflow

```
[user] → vla-idea-generator (pending/에 저장, BLACKLIST.md 의무 참조)
       → vla-literature-checker (novelty 검증)
              ├─ NO-GO  → pending → abandoned/  +  BLACKLIST.md 업데이트
              └─ Other → pending → active/  +  verdicts/conditional-go/
       → vla-idea-validator (publishability)
              ├─ FAIL → active → abandoned/  +  BLACKLIST.md 업데이트
              ├─ COND → conditional/, pre-experiment gates 명시
              └─ GO   → passed/
       → vla-experiment-planner → active/<idea>_plan.md
       → vla-experiment-runner → experiments/<idea-slug>/
              완료 시 → planner의 plan을 active/ → completed/로 이동
```

## Key Documents

- **Blacklist** (idea-generator 의무 참조): `vla-idea-generator/BLACKLIST.md`
- **Experiment overview**: `/home/jovyan/workspace/paper_agents_vla/experiments/INDEX.md`
- **Session/status docs**: `/home/jovyan/workspace/paper_agents_vla/docs/`
- **CPR-Distill experiment plan validation**: `vla-idea-validator/conditional/validation_cpr_distill_experiment_plan.md`
