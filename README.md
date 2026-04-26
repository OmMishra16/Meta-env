---
title: "Counsel Env: Cross-Examination Arena"
emoji: "⚖️"
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
 - openenv
 - rl
 - multi-agent
 - theory-of-mind
 - adversarial-dialogue
 - procgen
models:
 - heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
 - Qwen/Qwen3-8B
---

# Counsel-Env: Cross-Examination Arena

Counsel-Env is an OpenEnv courtroom environment where an LLM learns to cross-examine a deterministic witness: make the witness commit to a claim, then present the exhibit that proves the claim false.

We built it around a simple courtroom failure mode: a witness says something that does not match the evidence, but the examiner asks vague questions or shows the evidence too early. Counsel-Env trains the opposite behavior. The agent must ask with intent, track what the witness has committed to, and use the right exhibit at the right moment.

> Baseline behavior: vague questions, early evidence, zero reward.
>
> Target behavior: trigger sealed claim, present matching exhibit, surface contradiction.

## Public Links

- Hugging Face Space: https://huggingface.co/spaces/heavycoderhh/counsel-env
- Live demo: https://heavycoderhh-counsel-env.hf.space/demo
- Official checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
- 30-seed eval mirror: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval
- 150-seed eval mirror: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval_150
- Blog writeup: https://huggingface.co/spaces/heavycoderhh/counsel-env/blob/main/BLOG.md
- Run4b training notebook: https://huggingface.co/spaces/heavycoderhh/counsel-env/blob/main/notebooks/train_counsel_run4b.ipynb
- GitHub source: https://github.com/OmMishra16/Meta-env

## Try It

Open the live Space demo:

```text
https://heavycoderhh-counsel-env.hf.space/demo
```

What to try:

1. Reset an easy case.
2. Ask the witness the oracle-hint question to make them commit to a sealed claim.
3. Present the hinted exhibit.
4. Rest the case and watch primary reward appear.

The hint is intentionally exposed for the demo. The training task is to make a model learn that sequence from observations, evidence descriptions, and reward.

## Why This Is Hard

The agent must track another actor's commitments. Presenting evidence too early fails; asking generic questions fails; keyword spam can trigger a claim but does not prove anything. Reward only becomes strong when the agent sequences the cross-examination correctly.

```mermaid
flowchart LR
  A[Procgen Case Generator] --> B[Deterministic Witness]
  B --> C[Agent Tools]
  C --> D[Transcript + State Tracker]
  D --> E[Weighted Reward Rubric]
  E --> F[GRPO Training Loop]
```

Each episode is a procedurally generated case with:

- a public case brief
- a deterministic witness story
- hidden contradiction objects
- evidence exhibits visible to the agent
- a 15-question budget
- replayable seeds for fair evaluation

A contradiction is surfaced only when both steps happen in order:

1. The agent asks a trigger question and the witness gives a sealed claim.
2. The agent presents the matching disprover exhibit.

The witness is deterministic by design, so reward verification is fast, reproducible, and non-LLM-judged.

## OpenEnv Interface

Counsel-Env uses OpenEnv's standard environment shape:

- `reset`: start a new case
- `step`: execute an action
- `state`: inspect compact environment state

The environment implementation lives in `counsel_env/server/counsel_env_environment.py` and the OpenEnv manifest is in `counsel_env/openenv.yaml`.

Available actions:

| Tool | Field | Purpose |
| --- | --- | --- |
| `ask_question` | `text` | Ask the witness a question. |
| `present_evidence` | `exhibit_id` | Present an exhibit from `available_evidence`. |
| `make_objection` | `reason` | Penalized unless an objection window exists. |
| `rest_case` | none | End the episode and receive final reward. |

## Reward Design

Primary reward is binary per contradiction:

```text
primary_reward = contradictions_surfaced / contradictions_total
```

Auxiliary shaping reduces sparsity while staying secondary:

```text
auxiliary =
  +0.2 * contradictions_triggered
  +0.1 * trigger_keyword_questions
  +0.1 * correctly_timed_evidence
  -0.05 * duplicate_or_irrelevant_questions
  -0.05 * blind_evidence
  -0.1 * inadmissible_actions

total_reward = 0.8 * primary_reward + 0.2 * auxiliary
```

This makes the reward hard to game. Random questions, keyword spam, and blind evidence dumping do not earn primary reward unless a contradiction is actually surfaced.

## Reward-Hacking Audit And Results

The expanded evaluator compares four baselines plus the trained checkpoint across 150 deterministic seeds:

| Agent | Episodes | Avg Reward | Primary Reward | Trigger Rate | Surface Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| random | 150 | 0.000 | 0.000 | 0.000 | 0.000 |
| keyword_spam | 150 | 0.066 | 0.000 | 0.650 | 0.000 |
| present_all | 150 | 0.000 | 0.000 | 0.000 | 0.000 |
| trained_qwen3_8b_qlora_sft_run4b_eval150 | 150 | 0.864 | 0.943 | 0.943 | 0.943 |
| scripted_oracle | 150 | 0.901 | 0.957 | 0.957 | 0.957 |

Difficulty breakdown for the trained model:

| Slice | Episodes | Avg Reward | Primary/Surface | Invalid Tool Calls |
| --- | ---: | ---: | ---: | ---: |
| easy | 50 | 0.836 | 1.000 | 0 |
| medium | 67 | 0.849 | 0.903 | 0 |
| hard | 33 | 0.939 | 0.939 | 0 |

Run4b is the official submission checkpoint. Run4c was not launched because the expanded eval did not show a hard/medium weakness worth spending more credits on.

## Training Evidence

Run4b was a real 4-bit QLoRA SFT run on `Qwen/Qwen3-8B`, launched as Hugging Face job `69edb014d2c8bd8662bcf5ba`. It trained on 1,460 assistant-only next-action rows generated from the environment curriculum and uploaded the PEFT adapter to `heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`.

The logged SFT loss dropped quickly during the 220-step run. Final `train_loss` was `0.0565`, with runtime `1287.7s`.

![Run4b training loss](assets/training_curves/run4b_training_loss.png)

The reward plot compares the trained checkpoint against random, keyword-spam, present-all, the previous run3 checkpoint, and the scripted oracle.

![Run4b held-out reward vs baselines](assets/training_curves/run4b_eval_rewards.png)

## Training Scripts And Notebooks

Run4b training notebook (mirrors the script that produced the official checkpoint):

```text
counsel_env/notebooks/train_counsel_run4b.ipynb
```

Credit-safe GRPO demo notebook:

```text
counsel_env/notebooks/train_counsel.ipynb
```

Fast 8B QLoRA SFT command used for run4b:

```bash
COUNSEL_MODEL=Qwen/Qwen3-8B \
COUNSEL_ARTIFACT_REPO=heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b \
COUNSEL_SFT_DATASET_SIZE=480 \
COUNSEL_SFT_MAX_STEPS=220 \
COUNSEL_MAX_SFT_LENGTH=1536 \
COUNSEL_SFT_LEARNING_RATE=1e-4 \
COUNSEL_LORA_R=16 \
COUNSEL_LORA_ALPHA=32 \
COUNSEL_GRAD_ACCUM=4 \
COUNSEL_INCLUDE_REST_ROWS=0 \
python counsel_env/scripts/run_qlora_sft_training_job.py
```

TRL GRPO training paths are also included:

```text
counsel_env/scripts/run_grpo_training_job.py
counsel_env/scripts/run_sft_grpo_training_job.py
```

The notebooks and scripts are credit-safe by default and do not start paid GPU training unless explicitly configured. Estimated remote costs:

- A10G dry run: about `$0.50`
- Full A100 GRPO run: about `$6-$10`

## Run Locally

Start the OpenEnv server:

```bash
uvicorn counsel_env.server.app:app --host 0.0.0.0 --port 8000
```

Client example:

```python
from counsel_env import CounselAction, CounselEnv

with CounselEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset(curriculum_stage="easy")
    print(result.observation.case_brief)

    result = client.step(CounselAction(tool="ask_question", text="Where were you that night?"))
    print(result.observation.witness_response)
```

## Validation

Full local preflight:

```bash
python scripts/pre_hf_validate.py
```

Fast test suite:

```bash
python -m pytest -p no:cacheprovider -q
```

Latest validation result:

```text
21 passed
PRE-HF PREFLIGHT PASSED
```

## File Structure

```text
.
|-- README.md                 # this file (also the HF Space frontpage)
|-- BLOG.md                   # short human-readable project blog
|-- LICENSE                   # BSD-3-Clause license
|-- counsel_env/              # runnable OpenEnv package and HF Space source
|   |-- BENCHMARKS.md         # benchmark numbers and reward-hacking audit
|   |-- notebooks/            # train_counsel.ipynb + train_counsel_run4b.ipynb
|   |-- scripts/              # QLoRA SFT and GRPO training entry points
|   `-- server/               # FastAPI app and CounselEnvironment
|-- assets/
|   |-- training_curves/      # run4b loss + reward plots and CSVs
|   |-- trained_eval_run4b_8b_sft/        # 30-seed run4b eval mirror
|   |-- trained_eval_run4b_8b_sft_eval150/ # 150-seed run4b eval mirror
|   |-- demo/                 # video script and same-seed demo case
|   |-- diagnostics/          # rollout diagnostics jsonl
|   `-- plots/                # baseline-vs-oracle and rubric plots
|-- scripts/                  # validation, eval, and plotting utilities
`-- pytest.ini                # local test config
```

## Limitations

- The witness is rule-based so reward stays verifiable and cheap.
- Cases are template-generated rather than open-domain.
- The environment models adversarial questioning mechanics, not full legal procedure.

Future work: self-play witness training, civil deposition templates, jurisdiction-specific admissibility rules, and larger trained-vs-baseline model ablations.

## Status

Counsel-Env is submission-ready:

- HF Space is public and runnable on free `cpu-basic` hardware.
- OpenEnv API is implemented and validated locally.
- Run4b checkpoint is published and evaluated on 150 deterministic seeds.
- Training scripts, both notebooks, and run4b plots are committed.
- This README links the Space, checkpoint, eval mirrors, training curves, and blog.
