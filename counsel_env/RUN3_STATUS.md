# Run-3 Status

Run-3 is complete and has been superseded by run-4b (`heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`). Keep this file as historical proof for the SFT+GRPO attempt.

Run-3 was launched as a Hugging Face Job after explicit approval. The first A100 job remained in `SCHEDULING` with no logs, so it was cancelled and relaunched on L40S. The first L40S attempt completed SFT, then failed before GRPO because `num_generations=6` was incompatible with TRL's default `generation_batch_size=4`; run-3 was patched to use `num_generations=4`.

```text
Cancelled job ID: 69ed7e38d2c8bd8662bcef3c
Cancelled flavor: a100-large
Failed job ID: 69ed8005d2c8bd8662bcef9a
Failed flavor: l40sx1
Failed reason: `generation_batch_size (4) must be divisible by num_generations (6)`
Active job ID: 69ed848ed70108f37acdf694
Active URL: https://huggingface.co/jobs/heavycoderhh/69ed848ed70108f37acdf694
Active flavor: l40sx1
Latest status: COMPLETED
Relaunched: 2026-04-26 03:22 UTC
Target repo: heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3
Evaluation job ID: 69ed95fed70108f37acdf871
Evaluation status: COMPLETED
```

Run-3 held-out result:

```text
avg_reward=0.615
primary_reward=0.689
trigger_rate=0.728
surface_rate=0.689
avg_useless_ratio=0.550
```

Run-3 evaluation command:

```bash
COUNSEL_EVAL_MODEL=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
COUNSEL_EVAL_LABEL=trained_sft_grpo_run3 \
COUNSEL_EVAL_UPLOAD_REPO=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
python counsel_env/scripts/evaluate_trained_checkpoint_job.py
```
