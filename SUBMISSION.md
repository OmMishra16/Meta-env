# Counsel-Env Submission Card

## Public Links

- Space: https://huggingface.co/spaces/heavycoderhh/counsel-env
- Live demo: https://heavycoderhh-counsel-env.hf.space/demo
- Trained checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3
- Trained eval artifacts: https://huggingface.co/heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3/tree/main/eval

## Verification Snapshot

- Space visibility: public
- Space runtime: `RUNNING` on `cpu-basic`
- Checkpoint visibility: public
- Checkpoint SHA: `a4b4693e0615e97e78e13318d5389c30aabddcc0`
- Space SHA: `8fb858c9f441779893e0d4383b362cfd8820bd66`

## Held-Out Results

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| keyword_spam | 30 | 0.073 | 0.000 | 0.678 | 0.000 |
| present_all | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained_sft_grpo_run3 | 30 | 0.615 | 0.689 | 0.728 | 0.689 |
| scripted_oracle | 30 | 0.902 | 0.950 | 0.950 | 0.950 |

## Local Proof Artifacts

- `assets/trained_eval/trained_eval_summary.json`
- `assets/trained_eval/trained_eval_rows.csv`
- `assets/trained_eval/trained_eval_rows.jsonl`
- `assets/trained_eval/trained_eval_transcripts.md`
- `assets/trained_eval_run3/trained_eval_summary.json`
- `assets/trained_eval_run3/trained_eval_rows.csv`
- `assets/trained_eval_run3/trained_eval_rows.jsonl`
- `assets/trained_eval_run3/trained_eval_transcripts.md`
- `assets/heldout_eval_summary.json`
- `assets/transcripts/before_after_pairs.md`
- `assets/demo/video_script.md`
- `assets/demo/blog_draft.md`

## Local Validation

```bash
python pre_hf_validate.py
python -m pytest -p no:cacheprovider -q
```
