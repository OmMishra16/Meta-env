# Run-3 Training Plan

Run 2 proves the environment works: the model beats reward-hacking baselines and surfaces contradictions on held-out cases. Run 3 targets the visible failure modes in the trained transcripts:

- copying the case brief into `ask_question`
- repeating the same question after a successful trigger
- inventing invalid exhibit IDs
- failing to rest after surfacing the available contradiction

## Code Improvements Ready for Run 3

- Start from `heavycoderhh/counsel-env-qwen3-0.6b-grpo-run2`.
- Publish to `heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3`.
- Use assistant-only SFT labels so the model learns tool calls, not user case briefs or witness responses.
- Train on one SFT row per next action instead of one long transcript row.
- Include compact state feedback after each tool response:
  - triggered count
  - surfaced count
  - questions remaining
  - valid exhibit IDs
- Increase default SFT demonstrations from 320 to 720.
- Increase default GRPO steps from 250 to 500.
- Increase `num_generations` from 4 to 6.
- Increase evidence pressure and duplicate-question penalties.

## Default Run-3 Settings

```text
COUNSEL_MODEL=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run2
COUNSEL_ARTIFACT_REPO=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3
COUNSEL_SFT_DATASET_SIZE=720
COUNSEL_MAX_STEPS=500
COUNSEL_DATASET_SIZE=320
COUNSEL_NUM_GENERATIONS=6
COUNSEL_MAX_COMPLETION_LENGTH=256
COUNSEL_LEARNING_RATE=3e-6
COUNSEL_EVIDENCE_PRESSURE=2.0
```

## Success Target

The run is worth publishing only if it improves the held-out trained row over run 2:

| Metric | Run 2 | Run 3 target |
| --- | ---: | ---: |
| Avg reward | 0.387 | >= 0.500 |
| Primary reward | 0.461 | >= 0.600 |
| Surface rate | 0.461 | >= 0.600 |
| Invalid tool calls | varies | lower than run 2 |
| Useless question ratio | 0.829 | lower than run 2 |

If run 3 misses these targets, keep run 2 as the public submission checkpoint and use the run-3 artifacts as ablation evidence only.

## Post-Run Evaluation

After training, run:

```bash
COUNSEL_EVAL_MODEL=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
COUNSEL_EVAL_LABEL=trained_sft_grpo_run3 \
COUNSEL_EVAL_UPLOAD_REPO=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
python counsel_env/scripts/evaluate_trained_checkpoint_job.py
```

Then mirror the new `eval/` files into:

```text
assets/trained_eval_run3/
```

Do not replace the run-2 metrics in the README unless run 3 beats them on the same 30 deterministic seeds.
