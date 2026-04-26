# Counsel-Env Submission Card

## Public Links

- Space: https://huggingface.co/spaces/heavycoderhh/counsel-env
- Live demo: https://heavycoderhh-counsel-env.hf.space/demo
- Trained checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
- Trained eval artifacts: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval

## Verification Snapshot

- Space visibility: public
- Space runtime: `RUNNING` on `cpu-basic`
- Checkpoint visibility: public
- Checkpoint SHA: `4002e75edfd36e8fc7453dce4f8fe84eff628a76`
- Space live smoke: passed on April 26, 2026

## Held-Out Results

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| keyword_spam | 30 | 0.073 | 0.000 | 0.678 | 0.000 |
| present_all | 30 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained_qwen3_8b_qlora_sft_run4b | 30 | 0.860 | 0.928 | 0.928 | 0.928 |
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
- `assets/trained_eval_run4b_8b_sft/eval/trained_eval_summary.json`
- `assets/trained_eval_run4b_8b_sft/eval/trained_eval_rows.csv`
- `assets/trained_eval_run4b_8b_sft/eval/trained_eval_rows.jsonl`
- `assets/trained_eval_run4b_8b_sft/eval/trained_eval_transcripts.md`
- `assets/trained_eval_run4b_8b_sft/training_summary.json`
- `assets/heldout_eval_summary.json`
- `assets/transcripts/before_after_pairs.md`
- `assets/demo/video_script.md`
- `assets/demo/blog_draft.md`

## Local Validation

```bash
python pre_hf_validate.py
python -m pytest -p no:cacheprovider -q
```
