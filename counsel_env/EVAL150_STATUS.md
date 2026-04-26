# Run4b 150-Seed Evaluation Status

Status: complete.

Job:

- Hugging Face job: `69edd72fd70108f37acdff57`
- Model: `heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`
- Eval label: `trained_qwen3_8b_qlora_sft_run4b_eval150`
- Seeds: 150 deterministic episodes from start seed `20260425`
- Uploaded artifacts: `https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval_150`
- Local mirror: `assets/trained_eval_run4b_8b_sft_eval150/eval_150/`

## Overall Result

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate | Invalid Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| trained_qwen3_8b_qlora_sft_run4b_eval150 | 150 | 0.864 | 0.943 | 0.943 | 0.943 | 0 |

## Difficulty Slices

| Slice | Episodes | Avg Reward | Primary Reward | Surface Rate | Invalid Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 50 | 0.836 | 1.000 | 1.000 | 0 |
| medium | 67 | 0.849 | 0.903 | 0.903 | 0 |
| hard | 33 | 0.939 | 0.939 | 0.939 | 0 |

The 95% confidence intervals are included in `trained_eval_expanded_summary.json` and `trained_eval_expanded_summary.csv`.

## Run4c Decision

Do not spend on run4c yet.

The requested trigger for run4c was a clear hard/medium weakness. The expanded eval does not show one: hard is strong, medium is the softest slice but still above `0.90` primary/surface, and invalid tool calls are `0` across all 150 episodes.
