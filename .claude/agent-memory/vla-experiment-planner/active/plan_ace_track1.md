---
name: plan-ace-track1
description: ACE Track 1 (Asymmetric Camera-Role Encoder Allocation) PoC plan — 55 GPU-hr v6 chunked-BC drop-in with 3 mandatory pre-experiment gates (M-1 variance, M-2 sham-discriminate at 5pp/n=3, M-3 multi-suite replication)
metadata:
  type: project
  status: active
  idea_file: ../../vla-idea-generator/active/ace.md
  lit_verdict: ../../vla-literature-checker/verdicts/conditional-go/ace_novelty_verdict.md
  validator_verdict: ../../vla-idea-validator/conditional/validation_ace.md
  round: 5
  date: 2026-05-17
  budget_gpu_hr: 55
---

# Experiment Plan: ACE Track 1 — Asymmetric Camera-Role Encoder Allocation (PoC)

## Core Claim to Prove

On the v6 chunked-BC pipeline, a per-camera-role asymmetric vision-encoder allocation (`wrist=ResNet18, static=ResNet8`) achieves **≥1.4× end-to-end speedup** with **task SR within ±3pp of symmetric baseline**, AND the **camera-role direction is load-bearing**: a reverse-allocated control with identical total FLOPs (`wrist=ResNet8, static=ResNet18`) loses by **≥5pp paired across ≥2 LIBERO suites**.

If the 5pp/multi-suite sham margin fails, the idea is killed before any Track-2 spend.

---

## File / Disk Layout (mandatory)

- **User-facing artifacts** (scripts, `results.json`, plots, README, ablation tables, run logs):
  `/home/jovyan/workspace/paper_agents_vla/experiments/ace_track1/`
- **Large/binary artifacts** (model checkpoints, rollout videos, cached features, per-frame variance dumps):
  `/data/jameskimh/ace_track1/`
- **Plan file** (this document): `/home/jovyan/workspace/paper_agents_vla/.claude/agent-memory/vla-experiment-planner/active/plan_ace_track1.md`
- **Existing assets to reuse (do NOT re-train/re-download)**:
  - LIBERO suites: `/data/jameskimh/james_libero_datasets/{libero_spatial,libero_object,libero_goal,libero_10}/*.hdf5`
  - v6 baseline ckpt (libero_spatial, seed 42): `/data/jameskimh/cpr_distill_sim_rollout/baseline_v6_seed42.pt`
  - v7 baseline ckpt (libero_10, seed 42): `/data/jameskimh/cpr_distill_sim_rollout/baseline_v7_seed42.pt`
  - v6 source: `/home/jovyan/workspace/paper_agents_vla/.claude/worktrees/cpr-sim-rollout-v2/experiments/fail/cpr/cpr_distill_sim_rollout/sim_eval_v6.py` (fork as `sim_eval_ace.py` into the new ace_track1 dir)
  - lerobot (editable, for `LiberoEnv`): `/home/jovyan/workspace/Workspace_Lerobot/lerobot/src` (set `PYTHONPATH`)

**Mandatory output**: `experiments/ace_track1/README.md` (overview + findings + next-step) alongside `results.json`, `run.log`, and the trade-off plot. README is the user's primary entry point.

---

## Per-Gate Gantt (GPU-hrs on 1× B200, wall-clock on 4× B200 node)

| Gate | What runs | Conditions × seeds × suites | GPU-hr | Wall-clock |
|---|---|---|---|---|
| **M-1** (premise verify) | Rollout existing v6/v7 ckpts, log per-frame wrist/static pixel variance | 1 ckpt × 3 suites × 10 ep/task × 10 tasks ≈ 300 ep | **1.5** | ~25 min on 1 GPU (parallel-suite on 3 GPUs → ~10 min) |
| **calibration** (1 condition × 1 seed) | confirm asymmetric ChunkedBC trains end-to-end, time-per-epoch | 1 × 1 × libero_spatial, 30 epochs + 300 rollout ep | **2** | ~40 min |
| **M-2** (sham discriminate, libero_spatial) | Conditions A/B/C/D × 3 seeds × libero_spatial × 30 epochs train + 30 ep/task × 10 tasks rollout | 4 × 3 × 1 = 12 runs | **22** | ~6h wall on 4 GPUs |
| **M-3** (multi-suite replication, libero_object) | Conditions A/B/C/D × 3 seeds × libero_object | 4 × 3 × 1 = 12 runs | **22** | ~6h wall on 4 GPUs |
| **Latency bench** | Wall-clock `ms/step` measure for A/B/C/D on B200 + Jetson Orin (if available) | 4 conds × 1 ckpt × 1000 iters | **1** | ~20 min |
| **Analysis + plots + README** | Paired t + bootstrap, taxonomy resolution, trade-off plot | n/a | **0.5** | ~30 min (CPU) |
| **Buffer** | re-runs on flaky seeds, OOM debug | n/a | **6** | as needed |
| **Total Track 1** | | | **~55 GPU-hr** | ~3-4 days wall on 4× B200 |

**Critical**: M-2 must complete and PASS the 5pp gate **before** M-3 spend. M-3 must PASS the multi-suite gate **before** any Track-2 conversation.

---

## Exact Code Reuse & Forks

Fork these v6 modules into `/home/jovyan/workspace/paper_agents_vla/experiments/ace_track1/`:

1. `sim_eval_v6.py` → `sim_eval_ace.py` — modify `ChunkedBC` (see PyTorch spec below); add `CPR_VARIANT={A_sym_full,B_ace,C_reverse,D_sym_tiny}` env var; keep the rest of the v6 pipeline (BCE gripper, tanh arm, state z-score, chunk=16, AdamW 3e-4, cosine LR, 30 epochs) unchanged.
2. New: `measure_variance.py` — M-1 rollout + per-frame wrist/static pixel-variance computation (uses existing baseline ckpts, no training).
3. New: `bench_latency.py` — `torch.cuda.Event` timing for each condition's vision branch + full forward, batch=1, 1000 iterations.
4. New: `aggregate.py` — paired t-test (scipy) + 10k bootstrap CI on `A`, `B-C`, `B-A` deltas across seeds × suites; per-task taxonomy resolution.
5. New: `plot_tradeoff.py` — SR vs latency scatter, 1 dot per (condition, suite, seed) + condition-level mean.

**No changes** to `diagnostic.py`, `LiberoEnv`, dataset chunking logic, state normalization, optimizer, scheduler, gripper BCE weight (0.5), chunk size (16), or batch size (128). Only the `ChunkedBC` model class.

---

## PyTorch Code Spec — Asymmetric ChunkedBC

### ResNet8 design (pre-registered, locked before M-2)

`torchvision` has no ResNet8; use this exact spec:

```python
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class ResNet8(nn.Module):
    """
    Stripped ResNet: 1 BasicBlock per stage × 4 stages, channels=[32,64,128,256].
    Input 3×128×128 → output 256-d.

    Param count: ~2.8M (verify with sum(p.numel() for p in m.parameters()) before M-2 lock).
    FLOPs at 128×128: ~310M (verify with fvcore.nn.FlopCountAnalysis).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = BasicBlock(32, 32)
        self.layer2 = self._make_down(32, 64)
        self.layer3 = self._make_down(64, 128)
        self.layer4 = self._make_down(128, 256)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def _make_down(in_c, out_c):
        ds = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
        )
        return BasicBlock(in_c, out_c, stride=2, downsample=ds)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.avgpool(x).flatten(1)   # (B, 256)
```

### Asymmetric ChunkedBC (replaces v6 lines 162-196)

```python
import torchvision.models as tvm

def _make_resnet18():
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    bb = tvm.resnet18(weights=weights)
    return nn.Sequential(*list(bb.children())[:-1])   # (B, 512, 1, 1)

VARIANTS = {
    # (wrist_factory, wrist_dim, static_factory, static_dim)
    "A_sym_full":  (_make_resnet18, 512, _make_resnet18, 512),
    "B_ace":       (_make_resnet18, 512, ResNet8,        256),
    "C_reverse":   (ResNet8,        256, _make_resnet18, 512),
    "D_sym_tiny":  (ResNet8,        256, ResNet8,        256),
}

class ChunkedBC_ACE(nn.Module):
    def __init__(self, variant, state_dim=8, hidden=512, chunk=16):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant = variant
        self.chunk = chunk
        wf, wd, sf, sd = VARIANTS[variant]
        self.bb_wrist  = wf()
        self.bb_static = sf()
        self.feat_w, self.feat_s = wd, sd
        self.state_enc = nn.Sequential(
            nn.Linear(state_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(wd + sd + 256, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.arm_head  = nn.Linear(hidden, chunk * 6)
        self.grip_head = nn.Linear(hidden, chunk)

    @staticmethod
    def _enc(bb, x):
        # v6 normalize_img (ImageNet mean/std)
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x = (x - mean) / std
        return bb(x).flatten(1)

    def forward(self, img_a, img_w, state):
        # NOTE: v6 names: img_a = agentview (static), img_w = eye_in_hand (wrist)
        zw = self._enc(self.bb_wrist,  img_w)   # always wrist→bb_wrist
        zs = self._enc(self.bb_static, img_a)   # always static→bb_static
        zst = self.state_enc(state)
        h = self.trunk(torch.cat([zw, zs, zst], dim=-1))
        arm = torch.tanh(self.arm_head(h)).reshape(-1, self.chunk, 6)
        grip_logit = self.grip_head(h).reshape(-1, self.chunk)
        return arm, grip_logit
```

Key change from v6: per-camera backbone instead of shared `self.backbone`. Trunk input width changes per variant. Everything else (heads, normalization, IO contract) is byte-identical to v6.

### Sham FLOPs/param table (compute once during calibration, report in README)

| Variant | Wrist BB | Static BB | Wrist params | Static params | Total vision params | Vision FLOPs (128²) | Total trainable |
|---|---|---|---|---|---|---|---|
| A_sym_full | ResNet18 | ResNet18 | 11.2M | 11.2M | 22.4M | ~3.6G | ~22.4M + heads |
| **B_ace** | ResNet18 | ResNet8  | 11.2M | 2.8M  | **14.0M** | **~2.1G** | smaller |
| **C_reverse** | ResNet8 | ResNet18 | 2.8M | 11.2M | **14.0M** | **~2.1G** | matches B exactly |
| D_sym_tiny | ResNet8 | ResNet8 | 2.8M | 2.8M | 5.6M | ~0.6G | smallest |

**Sham invariant for B vs C**: same architecture set, same total vision FLOPs, only the camera→backbone mapping differs. Confirm numerically with `fvcore.nn.FlopCountAnalysis` during calibration and lock the numbers in `experiments/ace_track1/calibration_flops.json` BEFORE M-2.

D's lower FLOPs is intentional (lower bound); call this out in the README so reviewers don't read it as a bug.

---

## Sham/Control Condition Spec (locked)

| Cond | Spec | Role | Pre-registered prediction |
|---|---|---|---|
| **A** | sym full (RN18+RN18) | upper baseline | SR floor reference |
| **B** | ACE (wrist=RN18, static=RN8) | proposed | SR ≈ A ± 3pp, latency < A |
| **C** | reverse (wrist=RN8, static=RN18) | **critical sham** — same total FLOPs as B | SR ≤ B − 5pp (paired, n=3 seeds × 2 suites) |
| **D** | sym tiny (RN8+RN8) | lower bound — is small enough for both? | SR < B, by how much shows wrist-capacity value |

**Excluded from Track 1** (defer to Track 2 if it happens): the "E: sym medium" condition from the idea file — not informative at v6 ResNet scale.

**Pre-registered ordering**: `B > D` AND `B − C ≥ 5pp` paired across ≥2 suites. Aggregate-only; per-task ordering only resolved after taxonomy (below).

---

## Statistical Analysis Plan (matches v6 protocol)

- **Unit**: per-(suite, seed) overall SR (mean over 10 tasks × 30 episodes = 300 episodes per cell).
- **Primary test**: paired t-test on `SR_B − SR_C`, paired by (suite, seed) → n_pairs = 2 suites × 3 seeds = 6 pairs. Two-sided α=0.05.
- **Effect-size CI**: 10k-resample bootstrap on `SR_B − SR_C`, report 95% CI.
- **Secondary**: same paired t on `SR_B − SR_A` (target: CI overlaps [−3pp, +3pp] → "within noise of baseline").
- **Pre-registered kill rule**: if bootstrap 95% CI on `(B − C)` lower bound < 0, OR if paired-mean `(B − C)` < 5pp → MECHANISM NULL, archive and stop.
- **Per-task taxonomy** (see below): category-resolved means with bootstrap CI; only reported after all 4 conditions × all seeds × both suites have completed (no-peek protocol).

**No-peek protocol**: per-(seed, condition) raw `results.json` written to disk as runs finish, but `aggregate.py` runs ONCE at end. Per-task SR breakdown not viewed until all 24 runs (M-2 + M-3) complete. This prevents the CPR drawer-outlier failure mode.

---

## Per-Task Taxonomy (PRE-REGISTERED, DO NOT EDIT AFTER 2026-05-17)

Locked before any M-2 run. LIBERO task indices (0-9) per suite, with hypothesis-driven category assignment based on task description text and known v6 baseline SR.

### libero_spatial (10 tasks)

| task_id | description (paraphrase) | category | rationale |
|---|---|---|---|
| 0 | pick up the black bowl between plates → place on plate | static-context | scene disambiguation, multi-object |
| 1 | pick up the black bowl from table center → place on plate | ceiling | simple pick-place, baseline SR high |
| 2 | pick up the black bowl next to the cookie box → place on plate | static-context | reference-object disambiguation |
| 3 | pick up the black bowl next to the plate → place on plate | static-context | similar to 2 |
| 4 | pick up the black bowl on the cookie box → place on plate | wrist-precision | small target, close-range placement |
| 5 | pick up the black bowl on the ramekin → place on plate | wrist-precision | small target on top of object |
| 6 | pick up the black bowl on the stove → place on plate | wrist-precision | close-range, stove obstruction |
| 7 | pick up the black bowl on the wooden cabinet → place on plate | wrist-precision | close-range, cabinet edge |
| 8 | pick up the black bowl in the top drawer → place on plate | wrist-precision | drawer interior, severe wrist-view need |
| 9 | pick up the black bowl on the wooden tray → place on plate | ceiling | open-surface pickup |

### libero_object (10 tasks, by index — VERIFY at calibration that the description→category mapping matches the actual suite ordering; if mismatch, re-anchor by task DESCRIPTION not index)

All libero_object tasks are "pick up the X and place it in the basket" with varied X. Classify by target object visual size and stack context:

| task_id | object | category |
|---|---|---|
| 0 | alphabet soup | ceiling |
| 1 | bbq sauce | ceiling |
| 2 | butter | wrist-precision (small) |
| 3 | chocolate pudding | ceiling |
| 4 | cream cheese | ceiling |
| 5 | ketchup | ceiling |
| 6 | milk | static-context (taller, basket-aim) |
| 7 | orange juice | static-context (taller) |
| 8 | salad dressing | ceiling |
| 9 | tomato sauce | ceiling |

### Category-resolved success criteria (mandatory)

| Category | Tasks (count across both suites) | Criterion |
|---|---|---|
| **wrist-precision** | ~6 (mostly libero_spatial) | `B − C ≥ 5pp` paired (this is the load-bearing margin) |
| **ceiling** | ~9 (mostly libero_object) | `B ≈ D ≈ A` within ±3pp; no degradation |
| **static-context** | ~5 | `B ≥ D − 3pp` (ACE must not catastrophically fail) |

If aggregate `B − C` passes but **wrist-precision category alone** fails the 5pp gate, the mechanism is null (the asymmetry hypothesis specifically predicts wrist-precision benefit). If aggregate fails but wrist-precision passes AND we can show ceiling tasks washed it out, the paper is scope-narrowed to wrist-precision regime with that as an explicit finding.

**LOCK DATE: 2026-05-17.** Re-classification of any task_id after this date invalidates the experiment.

---

## Abort Decision Tree

```
M-1 (premise verify, 1.5 GPU-hr)
├── PASS: wrist:static temporal variance ratio ≥ 2.0× on ≥3 LIBERO suites (BC rollout, not demos)
│       → proceed to calibration + M-2
├── SOFT FAIL: ratio 1.5-2.0× on majority of suites
│       → pivot static to ResNet12 (intermediate) instead of ResNet8; re-spec sham; restart M-2 plan
└── HARD FAIL: ratio < 1.5× on majority of suites
        → KILL ACE. Archive M-1 results to /data/jameskimh/ace_track1/m1_killed/.
          Update MEMORY.md, move idea to dead/. Do not run M-2.

M-2 (sham discriminate, libero_spatial, 4×3 runs, 22 GPU-hr)
├── PASS: B − C ≥ 5pp AND paired t p ≤ 0.05 AND bootstrap 95% CI on (B − C) > 0
│       AND |B − A| ≤ 3pp (no SR degradation)
│       → proceed to M-3
├── SOFT FAIL: B − C in [3pp, 5pp) with p ≤ 0.10
│       → underpowered; option to raise to n=5 seeds (adds ~15 GPU-hr) OR stop and write
│          "single-suite signal too weak for multi-suite spend" note. Default: stop.
└── HARD FAIL: B − C < 3pp OR p > 0.10 OR |B − A| > 5pp
        → KILL ACE. Sham-collapse: mechanism is not camera-role-conditioned.
          Archive results, do NOT run M-3, update MEMORY.md, move idea to dead/.

M-3 (multi-suite replication, libero_object, 4×3 runs, 22 GPU-hr)
├── PASS: combined-suite paired t on (B − C) ≥ 5pp AND CI > 0
│       AND wrist-precision category alone shows ≥ 5pp
│       → Track 1 SUCCESS. Write paper-ready summary. Gate-check Track 2 separately.
├── SOFT FAIL: libero_spatial PASS but libero_object FAIL (replication failure, CPR-style)
│       → SCOPE-NARROW: paper claims "ACE on libero_spatial-style precision-pick tasks";
│          state limitation explicitly. Do NOT escalate to Track 2.
│          Optionally try libero_goal as third suite (adds ~22 GPU-hr) before scope-narrow.
└── HARD FAIL: libero_object kills the aggregate margin (B − C < 3pp combined)
        → KILL ACE Track 1. The mechanism does not generalize beyond spatial.
          Move idea to dead/. Do not run Track 2.

Track 2 gate (NOT part of this 55 GPU-hr plan; gate-check only):
  PROCEED ONLY IF M-3 PASS (not soft fail) AND B − C ≥ 5pp on BOTH suites
  individually (not just aggregate). Plan separately under plan_ace_track2.md.
```

---

## Latency Measurement (B200 baseline, Jetson Orin if hardware available)

`bench_latency.py` measures:

- **Vision-branch only** (img_a + img_w → trunk input), batch=1, FP32, 1000 iterations after 100 warmup, `torch.cuda.Event` timing.
- **Full forward** (vision + trunk + heads), batch=1, 1000 iterations.
- Report median + p90 ms/step per condition.
- **End-to-end speedup target**: `latency(A) / latency(B) ≥ 1.4×`. If less, the idea is still PoC-positive on SR but the speedup claim must be re-stated.

Jetson Orin (if accessible): re-run the same bench script via `torch` with FP16. Report separately; not required for Track 1 PASS (Track 1 is the SR-mechanism result, latency is supportive).

---

## Metrics (always report)

- **Efficiency**: vision-branch FLOPs, total params, B200 ms/step (median + p90), speedup ratio vs A.
- **Task SR**: overall SR per (variant, suite, seed); paired Δ vs A; paired Δ vs C; bootstrap 95% CI.
- **Per-task**: SR by category (wrist-precision / ceiling / static-context).
- **Tradeoff plot**: x=ms/step, y=overall SR, one dot per (variant, suite, seed), connected by variant-mean lines.

---

## Total Track 1 Budget Breakdown

| Component | GPU-hr |
|---|---|
| M-1 premise verify | 1.5 |
| Calibration (1 cond × 1 seed) | 2.0 |
| M-2 sham (libero_spatial, 4×3) | 22 |
| M-3 multi-suite (libero_object, 4×3) | 22 |
| Latency benchmark | 1.0 |
| Aggregation + plots + README | 0.5 |
| Buffer (re-runs, debug) | 6.0 |
| **Track 1 total** | **~55 GPU-hr** |

Wall-clock on 4× B200 node: ~3-4 days end-to-end (assuming 4-way parallelism on independent seeds/conditions).

---

## What Track 1 Does NOT Cover (explicit non-goals)

- **Track 2** (ViT-pair swap on SmolVLA/OpenVLA): gated separately, ~125 GPU-hr; do NOT scope into this plan.
- **ACE variants** (ACE-distill, ACE-joint, ACE-shared-prefix): appendix-only per validator; not in Track 1 critical path.
- **Stacking with VLA-ADP**: defer to Track 2 (it's a real-VLA comparison, not v6).
- **Wrist-occlusion ablation**: required only if claiming real-robot deployment; Track 1 paper scopes to fixed-base table-top LIBERO with reliable wrist camera.
- **Real-robot evaluation**: not in Track 1 (no compute, no need for the chunked-BC-efficiency framing).
- **Token-count parity check / projection layer**: ViT-specific concern, not relevant to ResNet-pair Track 1.

---

## Risks & Contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| M-1 variance ratio < 2× on BC rollouts | Med | ResNet12 fallback (not ResNet8) for static; re-spec sham |
| ResNet8 underfits even as wrist BB in condition C | Med | This is the POINT — if C catastrophically fails, that's evidence for ACE, not against. But if C also kills A's training (e.g., bad gradient flow), need to confirm calibration trains stably |
| M-2 wins on libero_spatial but M-3 kills it on libero_object (CPR-style replication failure) | High | Scope-narrow per soft-fail branch in abort tree; do NOT push to Track 2 |
| ResNet8's tiny matmul shapes underutilize B200 → wall-clock speedup smaller than FLOPs ratio | Med | Report both FLOPs ratio and measured ms/step; Jetson Orin numbers (where vision dominates) likely show the FLOPs benefit more clearly |
| 30 epochs insufficient for asymmetric joint training to converge | Low | Calibration step catches this; extend to 40 epochs if calibration val loss not plateaued |
| LIBERO env import path breaks during 3-day run | Low | Pin `lerobot` commit in calibration's first 5 min; record SHA in README |
| Per-task taxonomy misclassifies a task | Med | LOCKED 2026-05-17; misclassification is a known limitation, not a re-do trigger. Report category sensitivity in appendix |

---

## Implementation Roadmap (sequential, each gate gates the next)

- **Day 0 (today)**: Plan committed. Create `experiments/ace_track1/` + `/data/jameskimh/ace_track1/`. Pin lerobot SHA.
- **Day 1 morning**: M-1 (1.5 GPU-hr). Read `m1_variance_ratio.json` → decide PASS/SOFT/HARD.
- **Day 1 afternoon**: Calibration run (2 GPU-hr). Lock FLOPs table to `calibration_flops.json`. Confirm ResNet8 trains.
- **Day 1-2**: M-2 (22 GPU-hr). 4 conditions × 3 seeds on libero_spatial.
- **Day 2 EOD**: M-2 aggregate. PASS gate → continue, FAIL → write up null + archive.
- **Day 3**: M-3 (22 GPU-hr). 4 conditions × 3 seeds on libero_object.
- **Day 4 morning**: Aggregate, taxonomy resolve, plots, latency bench, README.
- **Day 4 EOD**: Plan moves to `completed/` (or `dead/` if killed). Track 2 gate-check.

---

## Ready-to-Execute Checklist (for orchestrator/runner)

- [ ] `mkdir -p /home/jovyan/workspace/paper_agents_vla/experiments/ace_track1 /data/jameskimh/ace_track1`
- [ ] Fork `sim_eval_v6.py` → `experiments/ace_track1/sim_eval_ace.py`, swap `ChunkedBC` → `ChunkedBC_ACE`, add `CPR_VARIANT` env var dispatch
- [ ] Write `experiments/ace_track1/measure_variance.py` (M-1)
- [ ] Write `experiments/ace_track1/bench_latency.py`
- [ ] Write `experiments/ace_track1/aggregate.py` (paired-t + bootstrap + taxonomy)
- [ ] Write `experiments/ace_track1/plot_tradeoff.py`
- [ ] Pin lerobot SHA, record in `experiments/ace_track1/README.md` header
- [ ] Verify ResNet8 param count + FLOPs at calibration (`calibration_flops.json`)
- [ ] Lock taxonomy as `experiments/ace_track1/taxonomy.json` BEFORE M-2 launch
- [ ] Execute gates M-1 → calibration → M-2 → (gate-check) → M-3 → (gate-check) → aggregate
- [ ] Produce `experiments/ace_track1/README.md` (overview / findings / next-step) + `results.json` + plots
- [ ] Move plan to `completed/` (PASS) or annotate `dead/` outcome (FAIL)
- [ ] Update `MEMORY.md` index pointer

---

## Cross-References

- Idea: [[ace]] (`vla-idea-generator/active/ace.md`)
- Lit verdict: [[ace-novelty-verdict]]
- Validator verdict: [[validation-ace]]
- Compute calibration: [[compute-calibration-b200]]
- Related (CPR's failure-mode lessons applied here): `experiments/fail/cpr/cpr_distill_sim_rollout/`
