# Counsel-Env

Counsel-Env is an OpenEnv courtroom environment where an LLM learns to cross-examine a deterministic witness: make the witness commit to a claim, then present the exhibit that proves the claim false.

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

Local validation:

```bash
python scripts/pre_hf_validate.py
python -m pytest -p no:cacheprovider -q
```
