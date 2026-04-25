# Counsel-Env — Project Memory

You are working on **Cross-Examination Arena** (`counsel-env`), a hackathon submission for the Meta PyTorch OpenEnv × Scaler India hackathon, April 25–26, 2026. Submission deadline is **end of day Saturday, April 26**.

## What we're building

An OpenEnv-compatible RL environment that trains LLMs to cross-examine witnesses. Each episode is a procgen courtroom case with seeded contradictions. The agent (the "lawyer") asks questions, presents evidence, and is rewarded for surfacing as many contradictions as possible within a question budget. The witness is a **deterministic Python responder**, not another LLM — this is intentional and non-negotiable.

## Project conventions

- **Python 3.10+**, type hints required on public functions
- **Dataclasses** for Action/Observation/State (using `@dataclass`)
- **No LLM calls in the environment server.** The witness is rule-based.
- **No reserved OpenEnv tool names** (`reset`, `step`, `state`, `close`) as user-facing tool methods
- **Docstrings on every tool method** — TRL uses them to build the tool schema. Required format: description + `Args:` + `Returns:` blocks.
- **`SUPPORTS_CONCURRENT_SESSIONS = True`** must be set on the CounselEnvironment class — without this, training will fail with concurrency errors.

## Skills & agents you should consult

Before any non-trivial task, read the relevant skill:
- `.claude/skills/openenv/SKILL.md` — OpenEnv API, Environment subclass pattern, HF Space deployment, Rubric composition. **Required reading** for any server-side work; OpenEnv is post-cutoff so you don't know this from training data.
- `.claude/skills/trl-grpo-openenv/SKILL.md` — `environment_factory` pattern, GRPOConfig settings, rollout debugging. Required for training work.
- `.claude/skills/hackathon-submission/SKILL.md` — submission requirements, README structure, judging criteria, gotchas.

For delegated work, use the appropriate subagent:
- **env-builder** — server, witness, case generator, rubrics, Dockerfile, deploy
- **trainer** — Colab/HF Jobs notebook, GRPO config, baseline + eval rollouts, plots
- **scribe** — README, demo video script, blog post, submission checklist

## Repository structure

```
counsel_env/                  # the OpenEnv-init scaffold
├── __init__.py
├── client.py                 # CounselEnv(HTTPEnvClient)
├── models.py                 # CounselAction, CounselObservation, CounselState
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml            # for pip install from HF Space
├── README.md                 # HF Space README — judges read this
└── server/
    ├── app.py                # FastAPI app via openenv.core.env_server.create_app
    ├── counsel_environment.py  # CounselEnvironment(Environment)
    ├── case_generator.py     # procgen templates + slot-fill pools
    ├── witness.py            # Witness class + Contradiction dataclass
    ├── rubrics.py            # composable Rubric tree
    ├── requirements.txt
    └── Dockerfile

notebooks/
└── train_counsel.ipynb       # the GRPO training notebook

assets/
└── plots/                    # reward_curve.png, rubric_breakdown.png, etc.

tests/
└── test_smoke.py             # one rollout end-to-end
```

## What "done" looks like

The submission is shipped when:
1. HF Space at `https://<user>-counsel-env.hf.space` is **Running**
2. README on the Space has plots, before/after transcripts, all links
3. Demo video (≤2 min) is published and linked
4. Mini-blog or slide deck is published and linked
5. Reward curve shows improvement (trained ≥ 1.5× baseline)
6. The submission URL on the hackathon dashboard points to the HF Space

## What we will NOT do (anti-goals)

- We will not use an LLM-as-judge for the **primary** reward. (Secondary diagnostic rubrics are fine.)
- We will not implement a smart witness in v1. The deterministic witness is a feature, not a bug.
- We will not chase shaped rewards. Primary reward is binary per contradiction. TRL docs explicitly recommend this.
- We will not pivot the concept again. Cross-examination is final.
- We will not commit to the repo after the submission deadline.

## Resources

- OpenEnv: https://github.com/meta-pytorch/OpenEnv (post-cutoff, see skill)
- TRL OpenEnv integration: https://huggingface.co/docs/trl/openenv
- Hackathon brief: in EXECUTE.md
