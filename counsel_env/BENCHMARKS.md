# Counsel-Env Benchmark Card

Held-out evaluation now includes a 150-seed expansion for the current best checkpoint. The goal is not just to ask relevant questions; the agent must make the witness commit and then present the matching disprover exhibit.

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate | What It Shows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| random | 150 | 0.000 | 0.000 | 0.000 | 0.000 | Vague questions and early evidence do not score. |
| keyword_spam | 150 | 0.066 | 0.000 | 0.650 | 0.000 | Trigger words alone cannot earn primary reward. |
| present_all | 150 | 0.000 | 0.000 | 0.000 | 0.000 | Blind evidence dumping fails because timing matters. |
| trained_qwen3_8b_qlora_sft_run4b_eval150 | 150 | 0.864 | 0.943 | 0.943 | 0.943 | Qwen3-8B QLoRA SFT learns the trigger-then-evidence loop reliably. |
| scripted_oracle | 150 | 0.901 | 0.957 | 0.957 | 0.957 | The desired strategy is trigger first, evidence second. |

The trained checkpoint is published at `heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`. The original 30-seed submission eval remains under `eval/`, and the 150-seed expansion is under `eval_150/`, with a local mirror at `assets/trained_eval_run4b_8b_sft_eval150/eval_150/`.

## Difficulty Slices

| Slice | Episodes | Avg Reward | Primary Reward | Surface Rate | Invalid Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 150 | 0.864 | 0.943 | 0.943 | 0 |
| easy | 50 | 0.836 | 1.000 | 1.000 | 0 |
| medium | 67 | 0.849 | 0.903 | 0.903 | 0 |
| hard | 33 | 0.939 | 0.939 | 0.939 | 0 |

Run4c is not promoted or launched from this diagnosis. Medium is the weakest slice, but it is still above `0.90` primary/surface with zero invalid tool calls; another paid run would risk noisy, tiny gains.

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
