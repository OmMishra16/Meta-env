# Teaching an LLM to Cross-Examine

Modern LLMs are usually trained to be helpful conversationalists. Cross-examination asks for a different skill: strategic pressure under uncertainty.

In Counsel-Env, the model plays a trial lawyer. It receives a case brief, a list of exhibits, and a strict question budget. Its job is to surface contradictions in a deterministic witness's story. The witness is not an LLM. That choice is deliberate: it makes the reward fast, reproducible, and hard to argue with.

Each contradiction has three pieces:

- a trigger question category
- a sealed claim from the witness
- an exhibit that disproves that claim

The agent only receives primary reward when it completes the sequence: ask the right question, get the witness to commit, and present the matching evidence. Asking vaguely is not enough. Presenting evidence too early is not enough. Spamming trigger keywords is not enough.

We built a local reward-hacking audit with four baselines:

| Agent | Avg Reward | What It Tests |
| --- | ---: | --- |
| random | 0.000 | vague questions and accidental exhibits |
| keyword_spam | 0.066 | trigger words without proof |
| present_all | 0.000 | blind evidence dumping |
| trained_qwen3_8b_qlora_sft_run4b_eval150 | 0.864 | reliable question-then-evidence timing |
| scripted_oracle | 0.902 | upper-bound trigger-then-evidence strategy |

This gives the environment a clear training target. The best checkpoint is a fast Qwen3-8B QLoRA SFT run trained on assistant-only oracle next-action rows. On the expanded 150-seed evaluation it reaches 0.864 average reward and 0.943 primary/surface rate: clearly above random, keyword spam, blind evidence dumping, and the earlier run-3 SFT+GRPO checkpoint, while still leaving a small gap below the oracle ceiling.

The environment supports replayable seeds, held-out evaluation, transcript export, and per-component reward logging. That means we can compare the same case before and after training, inspect whether the model learned evidence timing, and catch cases where it only learned shallow trigger patterns.

Counsel-Env is aimed at Theme 1: multi-agent interactions. It also touches long-horizon planning because the agent must track what the witness has committed to across turns, choose among exhibits, and avoid wasting a limited question budget.

The core idea is simple: if we want LLMs that reason about other agents, we should train them in worlds where another agent's commitments matter.

Artifacts:

- Space: `https://huggingface.co/spaces/heavycoderhh/counsel-env`
- Checkpoint: `https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b`
- Local trained eval mirror: `assets/trained_eval_run4b_8b_sft/eval/`
- Expanded trained eval mirror: `assets/trained_eval_run4b_8b_sft_eval150/eval_150/`
