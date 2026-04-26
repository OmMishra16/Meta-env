# Run-3 Status

Run-3 was launched as a Hugging Face Job after explicit approval. The first A100 job remained in `SCHEDULING` with no logs, so it was cancelled and relaunched on L40S.

```text
Cancelled job ID: 69ed7e38d2c8bd8662bcef3c
Cancelled flavor: a100-large
Active job ID: 69ed8005d2c8bd8662bcef9a
Active URL: https://huggingface.co/jobs/heavycoderhh/69ed8005d2c8bd8662bcef9a
Active flavor: l40sx1
Latest status: SCHEDULING
Relaunched: 2026-04-26 03:01:25 UTC
Target repo: heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3
```

Expected next checks:

```bash
hf jobs inspect 69ed8005d2c8bd8662bcef9a
hf jobs logs 69ed8005d2c8bd8662bcef9a
```

After the job succeeds, evaluate and upload run-3 artifacts:

```bash
COUNSEL_EVAL_MODEL=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
COUNSEL_EVAL_LABEL=trained_sft_grpo_run3 \
COUNSEL_EVAL_UPLOAD_REPO=heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3 \
python counsel_env/scripts/evaluate_trained_checkpoint_job.py
```
