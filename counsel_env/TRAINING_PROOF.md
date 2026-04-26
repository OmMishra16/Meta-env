# Tiny Training Proof Path

The deployed Space does not start paid training. Use this page to run the smallest GRPO smoke test only after compute is explicitly approved.

## Current Status

- Space deployment: complete on free `cpu-basic`.
- Training notebook: `notebooks/train_counsel.ipynb`.
- Default safety: `RUN_TRAINING=0`, `push_to_hub=False`, `report_to="none"`.
- Paid remote training has completed only after explicit approval.
- Best checkpoint: `heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3`.
- Run-3 recipe: run-2 base checkpoint, 720 assistant-only oracle SFT next-action rows, then 500 GRPO steps with stronger evidence and duplicate-loop shaping.
- Held-out result: average reward `0.615`, primary reward `0.689`, surface rate `0.689` across 30 deterministic seeds.

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

Run 3 completes the submission-standard path:

1. Trained on generated curriculum cases.
2. Evaluated on the same 30-seed harness used in `BENCHMARKS.md`.
3. Published trained eval rows, summary, and transcripts under the checkpoint repo's `eval/` folder.
4. Mirrored the trained eval artifacts locally under `assets/trained_eval/`.

Run 3 satisfies this path for a submission demo: it is not oracle-level, but it substantially improves learned evidence timing over run 2 while staying below the scripted oracle ceiling.
