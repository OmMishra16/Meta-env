# Counsel-Env Training Improvements Report

## Status

Counsel-Env is locally validated and ready for the final Hugging Face deployment/training step after explicit credit approval. No Hugging Face credits were used in this pass.

Latest local preflight:

```text
python pre_hf_validate.py
[OK] PRE-HF PREFLIGHT PASSED
```

Coverage included:

- manifest and README checks
- notebook syntax and credit guards
- package-style and Docker/Space-style imports
- component validation
- full pytest suite: `20 passed`
- rollout diagnostics
- held-out baseline evaluation
- reward-hacking audit
- local FastAPI/OpenEnv WebSocket reset/step/state smoke

## What Changed

### 1. Replayable held-out cases

`reset(seed=...)` now produces stable case IDs, briefs, evidence, and contradiction structures. This enables fair baseline-vs-trained comparisons on the same case seeds.

### 2. Richer observations

Observations now expose:

- `case_id`
- `difficulty`
- `available_evidence`
- `evidence_descriptions`
- `reward_components`

This makes the agent reason from exhibit text instead of only memorizing exhibit IDs.

### 3. Transcript export

The environment can export:

- structured JSON transcript
- human-readable Markdown transcript

Each event records:

- action step
- tool
- input
- witness response
- triggered contradiction IDs
- surfaced contradiction IDs

Artifact:

```text
assets/transcripts/before_after_pairs.md
```

### 4. More diverse case templates

Added two professional-world templates:

- corporate fraud deposition
- workplace retaliation investigation

The environment now spans criminal, corporate, and HR-style adversarial questioning.

### 5. Reward shaping and timing

Primary reward remains contradiction surfacing:

```text
primary_reward = contradictions_surfaced / contradictions_total
```

Auxiliary reward:

```text
+0.2 per triggered contradiction
+0.1 per trigger-keyword question
+0.1 per correctly timed evidence presentation
-0.05 per duplicate/irrelevant question
-0.05 per blind evidence presentation
-0.1 per inadmissible action
```

Final reward:

```text
total_reward = 0.8 * primary_reward + 0.2 * auxiliary_reward
```

This gives exploration signal while keeping contradiction surfacing dominant.

### 6. Reward-hacking baselines

`counsel_env/evaluation.py` compares:

- `random`
- `keyword_spam`
- `present_all`
- `scripted_oracle`

Latest 30-seed held-out results:

| Agent | Avg Reward | Primary Reward | Trigger Rate | Surface Rate |
| --- | ---: | ---: | ---: | ---: |
| random | 0.000 | 0.000 | 0.000 | 0.000 |
| keyword_spam | 0.073 | 0.000 | 0.678 | 0.000 |
| present_all | 0.000 | 0.000 | 0.000 | 0.000 |
| scripted_oracle | 0.902 | 0.950 | 0.950 | 0.950 |

This shows the obvious hacks fail: keyword spam can trigger claims but cannot score primary reward, and blind evidence presentation scores zero.

Artifacts:

```text
assets/heldout_eval.jsonl
assets/heldout_eval_summary.json
assets/heldout_eval_summary.csv
assets/plots/baseline_vs_oracle.svg
assets/plots/rubric_breakdown.svg
```

### 7. Tiny dry-run mode

The training notebook supports a cheap sanity run after approval:

```bash
RUN_TRAINING=1 COUNSEL_MODEL=Qwen/Qwen3-0.6B COUNSEL_MAX_STEPS=5 COUNSEL_DATASET_SIZE=12
```

Remote training is still disabled by default:

- `RUN_TRAINING=0`
- `push_to_hub=False`
- `report_to="none"`

## Why This Improves the Submission

The project now demonstrates the full loop judges care about:

- The environment is replayable.
- The reward cannot be gamed by obvious baselines.
- The transcript artifacts explain agent behavior.
- The evaluation set gives measurable before/after structure.
- The local preflight catches Docker/Space import issues before HF.

The only missing evidence is the real trained-model curve, which requires approved GPU compute.

## HF Credit Note

No Hugging Face credits were used.

Estimated next costs:

- A10G dry run: about `$0.50`
- full A100 GRPO run, roughly 90 minutes: `$6-$10`
