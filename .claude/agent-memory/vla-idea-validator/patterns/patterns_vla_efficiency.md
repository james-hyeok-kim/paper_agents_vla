---
name: patterns-vla-efficiency
description: VLA efficiency idea validation 시 반복적으로 등장하는 failure patterns와 evaluation heuristics
metadata:
  type: feedback
---

# VLA Efficiency Idea — Common Failure Patterns

## Pattern 1: "VLM efficiency technique + applied to VLA"
대부분의 VLA efficiency 아이디어는 LLM/VLM에서 이미 연구된 기법의 robotics 적용이다. 이 경우 novelty는 자동으로 🟡 Partial Overlap. VLA-specific contribution이 명시되어야 함:
- Real-time constraint (control deadline)
- Action head structure (diffusion/AR/flow)
- Temporal correlation
- Safety implication

**Why**: Reviewer가 가장 자주 꺼내는 카드. "이건 [LLM 기법]의 robotics 버전인데, [LLM 기법]은 이미 있다."
**How to apply**: 모든 VLA efficiency idea에서 "이 기법이 VLM/LLM에 적용된 사례가 있는가?"를 먼저 묻는다. 있다면 novelty 자동 -1점.

## Pattern 2: Component 결합 vs Single mechanism
여러 loss/component를 결합한 idea (AMP-Distill처럼)는 reviewer가 "단순 결합" 카드를 꺼낸다. 방어법은:
- 각 component의 단독 ablation
- Component 간 상호작용을 증명 (e.g., A는 B 있을 때만 효과)
- 통합 framework가 individual sum보다 큰 효과

**Why**: Pure novelty가 부족한 idea의 마지막 방어선이 "상호작용". 이게 증명 안 되면 incremental로 분류됨.
**How to apply**: 결합형 idea에는 반드시 "어느 component가 어느 task에서 main effect를 내는지" pre-experiment 요구.

## Pattern 3: Sim-only result는 publication weak
LIBERO/CALVIN/ManiSkill만으로 끝나는 paper는 CoRL/RSS에서 점점 약해진다. 최소 1개 real robot task (특히 contact-rich) 필수.

**Why**: Sim-to-real gap이 VLA 분야 핵심 의제. Sim-only는 community에서 점점 valueless로 취급.
**How to apply**: Publishability 평가 시 real-robot result 없으면 점수 -1.

## Pattern 4: Latency target 명시 안 함
"빠르다"만 주장하고 구체적 target hardware (Jetson Orin, RTX 4090, etc.) + control frequency (10Hz, 20Hz, 50Hz)를 명시 안 하는 idea는 marginal로 분류.

**Why**: Real-time robotics는 단순 throughput이 아니라 deadline miss rate가 핵심.
**How to apply**: "어느 hardware에서 몇 Hz를 hit해야 하는가?"를 모든 idea에서 명시적으로 묻는다.

## Pattern 5: Safety degradation 분석 누락
대부분의 efficiency idea는 "success rate가 X% 보존"만 보고하고 **failure mode가 어떻게 변하는가**는 분석하지 않는다. 진짜 safety story는:
- Catastrophic failure rate (collision, drop) 분리 측정
- Force/wrench peak 변화
- Mode collapse 여부
- Recovery behavior

**Why**: VLA는 image generation과 달리 approximation error가 물리적 위험으로 직결. Reviewer/community가 점점 더 요구.
**How to apply**: Safety 점수 매길 때 "최악의 failure mode가 무엇이고 어떻게 측정했는가"를 항상 묻는다.
