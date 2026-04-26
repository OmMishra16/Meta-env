# Run-3 Status

Run-3 was launched as a Hugging Face Job after explicit approval. The first A100 job remained in `SCHEDULING` with no logs, so it was cancelled and relaunched on L40S. The first L40S attempt completed SFT, then failed before GRPO because `num_generations=6` was incompatible with TRL's default `generation_batch_size=4`; run-3 was patched to use `num_generations=4`.

```text
Cancelled job ID: 69ed7e38d2c8bd8662bcef3c
Cancelled flavor: a100-large
Failed job ID: 69ed8005d2c8bd8662bcef9a
Failed flavor: l40sx1
Failed reason: `generation_batch_size (4) must be divisible by num_generations (6)`
Active job ID: pending relaunch
Active flavor: l40sx1
Latest status: patched locally
Relaunched: pending
Target repo: heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3
```

Expected next checks:

```bash
hf jobs inspect <active-run3-job-id>
hf jobs logs <active-run3-job-id>
```

After the job succeeds, evaluate and upload run-3 artifacts:

```bash
COUNSEL_EVAL_MODEL=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
COUNSEL_EVAL_LABEL=trained_sft_grpo_run3 \
COUNSEL_EVAL_UPLOAD_REPO=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
python counsel_env/scripts/evaluate_trained_checkpoint_job.py
```
