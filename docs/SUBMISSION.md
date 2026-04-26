# Counsel-Env Submission Card

## Public Links

- Space: https://huggingface.co/spaces/heavycoderhh/counsel-env
- Live demo: https://heavycoderhh-counsel-env.hf.space/demo
- Trained checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
- Trained eval artifacts: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval
- Expanded 150-seed eval: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval_150
- Mini-blog draft: `assets/demo/blog_draft.md`
- Video script: `assets/demo/video_script.md`

## Verification Snapshot

- Space visibility: public
- Space runtime: `RUNNING` on `cpu-basic`
- Checkpoint visibility: public
- Checkpoint SHA: `4002e75edfd36e8fc7453dce4f8fe84eff628a76`
- Space live smoke: passed on April 26, 2026

## Held-Out Results

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 150 | 0.000 | 0.000 | 0.000 | 0.000 |
| keyword_spam | 150 | 0.066 | 0.000 | 0.650 | 0.000 |
| present_all | 150 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained_qwen3_8b_qlora_sft_run4b_eval150 | 150 | 0.864 | 0.943 | 0.943 | 0.943 |
| scripted_oracle | 150 | 0.901 | 0.957 | 0.957 | 0.957 |

## Expanded Slice Check

| Slice | Episodes | Avg Reward | Primary Reward | Surface Rate | Invalid Tool Calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 50 | 0.836 | 1.000 | 1.000 | 0 |
| medium | 67 | 0.849 | 0.903 | 0.903 | 0 |
| hard | 33 | 0.939 | 0.939 | 0.939 | 0 |

Run4c is not needed for submission right now. The expanded eval found no hard/medium collapse and no invalid tool-call issue.

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
- `assets/trained_eval_run4b_8b_sft_eval150/eval_150/trained_eval_summary.json`
- `assets/trained_eval_run4b_8b_sft_eval150/eval_150/trained_eval_expanded_summary.json`
- `assets/trained_eval_run4b_8b_sft_eval150/eval_150/trained_eval_rows.csv`
- `assets/trained_eval_run4b_8b_sft_eval150/eval_150/trained_eval_rows.jsonl`
- `assets/trained_eval_run4b_8b_sft_eval150/eval_150/trained_eval_transcripts.md`
- `assets/training_curves/run4b_training_loss.png`
- `assets/training_curves/run4b_eval_rewards.png`
- `assets/training_curves/run4b_training_loss.csv`
- `assets/training_curves/run4b_eval_rewards.csv`
- `assets/heldout_eval_summary.json`
- `assets/transcripts/before_after_pairs.md`
- `assets/demo/video_script.md`
- `assets/demo/blog_draft.md`

## Final Human Step

Publish either the mini-blog or the sub-2-minute video, then paste its public URL into the Space README before final form submission. The draft/script are already in the repo; no large video file should be uploaded to the Hub.

## Local Validation

```bash
python scripts/pre_hf_validate.py
python -m pytest -p no:cacheprovider -q
```
