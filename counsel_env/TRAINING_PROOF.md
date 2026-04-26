# Tiny Training Proof Path

The deployed Space does not start paid training. Use this page to run the smallest GRPO smoke test only after compute is explicitly approved.

## Current Status

- Space deployment: complete on free `cpu-basic`.
- Training notebook: `notebooks/train_counsel.ipynb`.
- Default safety: `RUN_TRAINING=0`, `push_to_hub=False`, `report_to="none"`.
- Paid remote training has completed only after explicit approval.
- Best checkpoint: `heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`.
- Best recipe: `Qwen/Qwen3-8B` with 4-bit QLoRA, assistant-only oracle SFT next-action rows, no rest-only rows, and adapter upload.
- Original held-out result: average reward `0.860`, primary reward `0.928`, trigger rate `0.928`, surface rate `0.928` across 30 deterministic seeds.
- Expanded 150-seed result: average reward `0.864`, primary reward `0.943`, trigger rate `0.943`, surface rate `0.943`, with `0` invalid tool calls.
- Run4c decision: do not launch yet. Expanded medium and hard slices remain strong, so another paid run is not justified by the current evidence.
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
2. Evaluated on the same 30-seed harness used for the original submission benchmark.
3. Re-evaluated on a 150-seed expansion with difficulty-slice summaries.
4. Published trained eval rows, summary, and transcripts under the checkpoint repo's `eval/` and `eval_150/` folders.
5. Mirrored the trained eval artifacts locally under `assets/trained_eval_run4b_8b_sft/eval/` and `assets/trained_eval_run4b_8b_sft_eval150/eval_150/`.

Run 4b satisfies this path for a submission demo: it is close to the scripted oracle ceiling and clearly above random, keyword spam, present-all, and the previous run-3 checkpoint.
