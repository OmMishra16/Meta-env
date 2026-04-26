# Run-4b Status

Run-4b is the current official submission checkpoint.

```text
Training job ID: 69edb014d2c8bd8662bcf5ba
Evaluation job ID: 69edb609d70108f37acdfc39
Base model: Qwen/Qwen3-8B
Training method: 4-bit QLoRA SFT
SFT rows: 1460
Training runtime: 1287.7 seconds
Target repo: heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
Checkpoint SHA: 4002e75edfd36e8fc7453dce4f8fe84eff628a76
Local eval mirror: assets/trained_eval_run4b_8b_sft/eval/
Expanded eval mirror: assets/trained_eval_run4b_8b_sft_eval150/eval_150/
```

Run-4b held-out result on the same 30 deterministic seeds:

```text
avg_reward=0.860
primary_reward=0.928
trigger_rate=0.928
surface_rate=0.928
```

Expanded 150-seed eval:

```text
avg_reward=0.864
primary_reward=0.943
trigger_rate=0.943
surface_rate=0.943
invalid_tool_calls=0
```

Difficulty slices from the expanded eval:

| Slice | Episodes | Avg reward | Primary/surface | Invalid tool calls |
| --- | ---: | ---: | ---: | ---: |
| easy | 50 | 0.836 | 1.000 | 0 |
| medium | 67 | 0.849 | 0.903 | 0 |
| hard | 33 | 0.939 | 0.939 | 0 |

Run4c was not launched after this diagnosis: medium is the weakest slice, but it remains strong enough that another paid run is not justified.

The run uses assistant-only oracle next-action labels rendered through the Qwen chat template, trains only on the assistant tool-call span, excludes rest-only rows by default, and uploads a PEFT adapter. This fixed the first 8B SFT attempt, which learned poor tool-call formatting and scored zero.

Run-4b beats the previous official run-3 checkpoint:

| Metric | Run 3 | Run 4b |
| --- | ---: | ---: |
| Avg reward | 0.615 | 0.860 |
| Primary reward | 0.689 | 0.928 |
| Trigger rate | 0.728 | 0.928 |
| Surface rate | 0.689 | 0.928 |

Run-4b should remain the official checkpoint unless a later model beats it on the same held-out evaluation harness.
