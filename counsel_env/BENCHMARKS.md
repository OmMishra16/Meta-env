# Counsel-Env Benchmark Card

Held-out evaluation uses 30 deterministic seeds and four fixed policies. The goal is not just to ask relevant questions; the agent must make the witness commit and then present the matching disprover exhibit.

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate | What It Shows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | 30 | 0.000 | 0.000 | 0.000 | 0.000 | Vague questions and early evidence do not score. |
| keyword_spam | 30 | 0.073 | 0.000 | 0.678 | 0.000 | Trigger words alone cannot earn primary reward. |
| present_all | 30 | 0.000 | 0.000 | 0.000 | 0.000 | Blind evidence dumping fails because timing matters. |
| trained_sft_grpo_run2 | 30 | 0.387 | 0.461 | 0.589 | 0.461 | SFT warm-start + GRPO learns the core question-then-evidence loop on held-out cases. |
| scripted_oracle | 30 | 0.902 | 0.950 | 0.950 | 0.950 | The desired strategy is trigger first, evidence second. |

The trained checkpoint is published at `heavycoderhh/counsel-env-qwen3-0.6b-grpo-run2`. Evaluation artifacts are in the model repo under `eval/`.

## Anti-Hacking Checks

- Primary reward is zero unless a contradiction is surfaced.
- Evidence only surfaces a contradiction after the witness has made the sealed claim.
- Duplicate, irrelevant, blind-evidence, and inadmissible actions are tracked separately.
- Auxiliary shaping exists only to reduce sparsity; it remains secondary to primary reward.

## Reproduce Locally

From the repository root:

```bash
python scripts/run_local_eval.py --episodes 30 --output-dir assets
python -m pytest -p no:cacheprovider -q
```

