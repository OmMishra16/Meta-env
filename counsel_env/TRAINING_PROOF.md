# Tiny Training Proof Path

The deployed Space does not start paid training. Use this page to run the smallest GRPO smoke test only after compute is explicitly approved.

## Current Status

- Space deployment: complete on free `cpu-basic`.
- Training notebook: `notebooks/train_counsel.ipynb`.
- Default safety: `RUN_TRAINING=0`, `push_to_hub=False`, `report_to="none"`.
- Paid remote training has completed only after explicit approval.
- Best checkpoint: `heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`.
- Best recipe: `Qwen/Qwen3-8B` with 4-bit QLoRA, assistant-only oracle SFT next-action rows, no rest-only rows, and adapter upload.
- Held-out result: average reward `0.860`, primary reward `0.928`, trigger rate `0.928`, surface rate `0.928` across 30 deterministic seeds.
- Prior run-3 SFT+GRPO checkpoint: `heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3`, average reward `0.615`.

## Lowest-Cost Dry Run

Use this only after approving the spend:

```bash
RUN_TRAINING=1 \
COUNSEL_MODEL=Qwen/Qwen3-0.6B \
COUNSEL_MAX_STEPS=5 \
COUNSEL_DATASET_SIZE=12
```

Expected purpose:

- Prove that the GRPO loop connects to Counsel-Env.
- Produce a tiny training log/checkpoint artifact.
- Avoid claiming model quality from a five-step run.

## Submission Standard Run

Run 4b is the current submission-standard path:

1. Trained on generated curriculum cases.
2. Evaluated on the same 30-seed harness used in `BENCHMARKS.md`.
3. Published trained eval rows, summary, and transcripts under the checkpoint repo's `eval/` folder.
4. Mirrored the trained eval artifacts locally under `assets/trained_eval_run4b_8b_sft/eval/`.

Run 4b satisfies this path for a submission demo: it is close to the scripted oracle ceiling and clearly above random, keyword spam, present-all, and the previous run-3 checkpoint.
