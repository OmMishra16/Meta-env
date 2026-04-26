# Counsel-Env

Counsel-Env is an OpenEnv courtroom environment where an LLM learns to cross-examine a deterministic witness: make the witness commit to a claim, then present the exhibit that proves the claim false.

The environment targets a capability gap that normal chat training does not emphasize: strategic questioning under partial information. The agent receives a public case brief, a list of evidence exhibits, and a strict question budget. It does not see the hidden contradiction directly. To earn primary reward, it must first ask a question that makes the witness seal a false claim, then choose the exact exhibit that disproves that claim.

This makes the task harder than ordinary QA or evidence retrieval. Asking vague questions scores poorly. Dumping evidence before the witness commits scores zero. Keyword spam can sometimes trigger a claim, but still gets no primary reward unless the agent proves the contradiction. The reward is implemented with OpenEnv rubrics: primary contradiction-surfacing reward dominates, while smaller auxiliary terms track progress, timing, duplicate questions, blind evidence use, and invalid actions.

The submitted model is `run4b`, a Qwen3-8B 4-bit QLoRA SFT checkpoint trained on assistant-only oracle next-action rows generated from the environment curriculum. It is evaluated against random, keyword-spam, present-all, prior run3, and scripted-oracle baselines on deterministic held-out seeds.

Public submission links:

- Hugging Face Space: https://huggingface.co/spaces/heavycoderhh/counsel-env
- Live demo: https://heavycoderhh-counsel-env.hf.space/demo
- Official checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b
- Expanded 150-seed eval: https://huggingface.co/heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4b/tree/main/eval_150

Start with the full Space README here:

- [counsel_env/README.md](counsel_env/README.md)

## File Structure

```text
.
├── README.md                 # this submission overview
├── LICENSE                   # BSD-3-Clause license
├── counsel_env/              # runnable OpenEnv package and HF Space source
├── assets/                   # plots, eval mirrors, transcripts, blog/video drafts
├── docs/                     # submission card and engineering report
├── scripts/                  # validation, eval, and plotting utilities
└── pytest.ini                # local test config
```

Submission evidence:

- [docs/SUBMISSION.md](docs/SUBMISSION.md)
- [counsel_env/BENCHMARKS.md](counsel_env/BENCHMARKS.md)
- [counsel_env/TRAINING_PROOF.md](counsel_env/TRAINING_PROOF.md)
- [counsel_env/EVAL150_STATUS.md](counsel_env/EVAL150_STATUS.md)
- [assets/demo/blog_draft.md](assets/demo/blog_draft.md)
- [assets/demo/video_script.md](assets/demo/video_script.md)

Key result: run4b reaches `0.864` average reward and `0.943` primary/surface rate on the expanded 150-seed evaluation, with zero invalid tool calls.

## Charts

The training/evaluation chart images are here:

- [assets/training_curves/run4b_training_loss.png](assets/training_curves/run4b_training_loss.png)
- [assets/training_curves/run4b_eval_rewards.png](assets/training_curves/run4b_eval_rewards.png)

The source CSVs used to generate those charts are in the same folder:

- [assets/training_curves/run4b_training_loss.csv](assets/training_curves/run4b_training_loss.csv)
- [assets/training_curves/run4b_eval_rewards.csv](assets/training_curves/run4b_eval_rewards.csv)

![Run4b training loss](assets/training_curves/run4b_training_loss.png)

![Run4b held-out reward vs baselines](assets/training_curves/run4b_eval_rewards.png)

Local validation:

```bash
python scripts/pre_hf_validate.py
python -m pytest -p no:cacheprovider -q
```
