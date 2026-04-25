# EXECUTE.md — Cross-Examination Arena Build Playbook

**Read this first.** This is the master doc for executing the build with Claude Code. Each phase below has a copy-paste-ready prompt for Claude Code, acceptance criteria, and the commands to run. Files referenced (`models.py`, `witness.py`, etc.) are skeletons embedded inline — Claude Code will consume them as starting points and complete the implementation.

The supporting files you've already dropped into the repo:
- `CLAUDE.md` — Claude Code reads this on every session
- `.claude/skills/openenv/SKILL.md` — OpenEnv API reference (post-cutoff knowledge)
- `.claude/skills/trl-grpo-openenv/SKILL.md` — TRL GRPO + environment_factory pattern
- `.claude/skills/hackathon-submission/SKILL.md` — submission requirements & gotchas
- `.claude/agents/env-builder.md` — server-side build subagent
- `.claude/agents/trainer.md` — TRL training subagent
- `.claude/agents/scribe.md` — README/demo subagent

---

## 0. Phase 0 — Setup (15 min, everyone in parallel)

Run on each teammate's machine:

```bash
# Repo
mkdir counsel-env && cd counsel-env
git init
# Drop the bootstrap files into the project root (CLAUDE.md, .claude/, EXECUTE.md)

# Python env
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install openenv-core trl unsloth datasets transformers vllm accelerate
pip install fastapi uvicorn pytest pydantic

# HF login (each teammate does this with their own token)
pip install huggingface_hub
huggingface-cli login

# Initialize the OpenEnv scaffold
openenv init counsel_env
cd counsel_env
```

**Verify Python ≥ 3.10 and Docker is installed** (Docker is required for HF Space deploy).

**Role assignments — confirm now:**
- **Builder:** owns `server/` and HF Space deploy. Drives Phases 1–2 + 5.
- **Trainer:** owns Colab/HF Jobs notebook + reward plots. Drives Phases 3–4 + 6.
- **Scribe:** owns README, demo video, blog, submission checklist. Drives Phase 7. Tracks deadline.

---

## 1. HF Credit Budget Strategy ($30 × 3 = $90 total)

**Cost map:**
- HF Space hosting (env): **free** on CPU tier — our env is deterministic Python, no GPU needed
- HF Jobs (training): A100 ≈ $4/hr, L4 ≈ $0.80/hr, A10G ≈ $1.05/hr
- Wordle reference run was 90 min on A100 — assume same for us

**Spend plan:**
| Step | Cost |
|---|---|
| Env hosting on HF Space (CPU) | $0 |
| 1 × dry-run training: 30 min on A10G | ~$0.50 |
| 1 × baseline rollouts on A10G (no training, just inference): 30 min | ~$0.50 |
| 1 × full GRPO training: 90 min on A100 | ~$6 |
| 1 × eval rollouts (50× trained): 30 min on A10G | ~$0.50 |
| **Estimated spend: ~$8 of $90** | |

You have a 10× safety margin. If something goes wrong, retrain. Don't be cheap — the bottleneck is your time, not the budget.

**Practical guidance:**
- Use one teammate's HF account as the "primary" — pool the credits there if HF allows, otherwise split runs across accounts
- Save the bulk training run for AFTER baseline + dry run pass green
- Use Colab Pro (free tier T4 / cheap A100 PAYG) as a backup if HF Jobs has queue issues

**Why HF Jobs over Colab for the final run:**
- All artifacts stay in the HF ecosystem — your env Space, your training logs, your model checkpoints all under one account. Cleaner submission.
- No Colab idle disconnects.
- Reproducibility — judges can re-run your job spec.

---

## 2. Phase 1 — Core environment (Builder, ~3 hrs)

### Claude Code prompt to paste:

```
You are the env-builder subagent. Read .claude/skills/openenv/SKILL.md and .claude/agents/env-builder.md before starting.

TASK: Build the server-side core of the Cross-Examination Arena environment.

Create these files inside counsel_env/ (the openenv init scaffold):

1. server/witness.py — deterministic Python responder. Use the skeleton at the bottom of this prompt as starting point. Do NOT call any LLM.

2. server/case_generator.py — procgen case templates. Use the starter file I have at /path/to/starter_case_generator.py as the seed. Add 2 more templates beyond the 3 provided (timeline_shift, motive_coverup) so we have 5 total.

3. server/counsel_environment.py — Environment subclass. Use the skeleton at the bottom. Wire reset/step/state. SUPPORTS_CONCURRENT_SESSIONS must be True.

4. models.py — CounselAction, CounselObservation, CounselState dataclasses.

5. server/rubrics.py — A composable Rubric tree using openenv.core.rubrics: WeightedSum of ContradictionsSurfaced (1.0), EvidenceTiming (0.3), AdmissibilityRubric (0.3). Each rubric is a class with a forward(action, observation) method returning a float in [0,1].

ACCEPTANCE CRITERIA:
- `pytest tests/test_smoke.py -v` passes (smoke test below)
- Running `python -m server.case_generator` prints 3 valid case JSONs
- Manually constructed rollout completes: reset() → 3× ask_question with trigger keywords → 1× present_evidence → reward should be > 0.0

SMOKE TEST to write at tests/test_smoke.py:
- Import CounselEnvironment, instantiate, call reset(), assert observation has case_brief and questions_remaining=15
- step() with ask_question that contains a known trigger keyword for the alibi case; assert witness commits to sealed_claim
- step() with present_evidence using the matching disprover_id; assert reward bumps when episode ends

DO NOT touch the FastAPI server, Dockerfile, or HF Space deployment in this phase. That's Phase 2.

[paste skeletons from EXECUTE.md §5.1, §5.2, §5.4 here]
```

### Code skeletons to embed in the Claude Code prompt:

**`models.py`:**
```python
from dataclasses import dataclass
from typing import List, Optional
from openenv.core.models import Action, Observation, State

@dataclass
class CounselAction(Action):
    tool: str
    text: Optional[str] = None
    exhibit_id: Optional[str] = None
    reason: Optional[str] = None

@dataclass
class CounselObservation(Observation):
    witness_response: str = ""
    available_evidence: List[str] = None
    questions_remaining: int = 0
    transcript_tail: str = ""
    case_brief: str = ""
    done: bool = False
    reward: float = 0.0

@dataclass
class CounselState(State):
    case_id: str = ""
    contradictions_total: int = 0
    contradictions_surfaced: int = 0
    questions_used: int = 0
    inadmissible_count: int = 0
```

**`server/witness.py`:**
```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Contradiction:
    cid: str
    trigger_keywords: List[str]
    sealed_claim: str
    disprover_evidence_id: str
    triggered: bool = False
    surfaced: bool = False

class Witness:
    def __init__(self, story: Dict[str, str], contradictions: List[Contradiction]):
        self.story = story
        self.contradictions = contradictions
        self.committed_claims: List[str] = []

    def respond_to_question(self, q: str) -> str:
        q_lower = (q or "").lower()
        for c in self.contradictions:
            if any(kw in q_lower for kw in c.trigger_keywords):
                if not c.triggered:
                    c.triggered = True
                    self.committed_claims.append(c.sealed_claim)
                return c.sealed_claim
        for topic, resp in self.story.items():
            if topic in q_lower:
                return resp
        return "I don't recall."

    def react_to_evidence(self, exhibit_id: str) -> str:
        for c in self.contradictions:
            if c.disprover_evidence_id == exhibit_id and c.triggered and not c.surfaced:
                c.surfaced = True
                return "[Witness stammers] I... I'm not sure what to say."
        return "[Witness] I have no comment on that exhibit."
```

**`server/counsel_environment.py`:** (full version in cross_examination_arena_hackathon_plan.md §5.4)

---

## 3. Phase 2 — FastAPI server + HF Space deploy (Builder, ~2 hrs)

### Claude Code prompt to paste:

```
You are the env-builder subagent. Read .claude/skills/openenv/SKILL.md, especially the HF Space deployment section.

TASK: Wire the FastAPI server, write the Dockerfile, deploy to HF Spaces.

1. server/app.py — Use create_app from openenv.core.env_server. Pass max_concurrent_envs=64.

2. server/Dockerfile — base from openenv-base. Include all server requirements.

3. server/requirements.txt — fastapi, uvicorn, pydantic, openenv-core, our internal modules.

4. openenv.yaml — manifest. Set name to counsel-env. Tag rl, multi-agent, theory-of-mind.

5. client.py — CounselEnv subclass of HTTPEnvClient with Action/Observation/State types.

6. pyproject.toml — package metadata so pip install git+<space-url> works.

7. README.md — placeholder (the Scribe will rewrite this in Phase 7), but include the install command and a one-paragraph description so the Space page isn't empty when judges peek mid-development.

DEPLOYMENT:
- Run `openenv push` from the counsel_env/ directory.
- The Space will build under <your-username>/counsel-env on huggingface.co/spaces.
- After push, wait 3-5 min for build. Verify the Space status is "Running" (not Building/Error).
- Test from a separate terminal:
    pip install git+https://huggingface.co/spaces/<your-username>/counsel-env
    python -c "from counsel_env import CounselEnv; from counsel_env.models import CounselAction; \
      with CounselEnv(base_url='https://<your-username>-counsel-env.hf.space').sync() as c: \
        r = c.reset(); print(r.observation.case_brief); \
        r = c.step(CounselAction(tool='ask_question', text='where were you that night')); print(r.observation.witness_response)"

ACCEPTANCE CRITERIA:
- HF Space shows "Running" status
- The above test prints a case brief and a sealed_claim from the witness
- The Space's web UI (built into OpenEnv) renders the action form correctly

GOTCHAS to watch:
- If the build fails, check requirements.txt — common issue is openenv-core version pinning
- If Running but step() hangs, check that SUPPORTS_CONCURRENT_SESSIONS=True is set on the Environment subclass
- The Space URL pattern is https://<username>-<space-name>.hf.space (note the dash, not a slash)
```

---

## 4. Phase 3 — TRL training scaffold (Trainer, ~1.5 hrs, parallel with Phase 2)

### Claude Code prompt to paste:

```
You are the trainer subagent. Read .claude/skills/trl-grpo-openenv/SKILL.md before starting.

TASK: Set up the GRPO training notebook with environment_factory pointing at our deployed Space.

Create notebooks/train_counsel.ipynb using the structure below. Then verify the pipeline with the Echo env FIRST, before swapping to Counsel. This is critical — debug the pipeline, not the env.

NOTEBOOK STRUCTURE:
1. Cell: install (TRL, vLLM, OpenEnv core, our env client)
2. Cell: smoke test against Echo env using environment_factory pattern (5-step dry run, confirm rewards flow)
3. Cell: smoke test against our LIVE Counsel Space (10-step dry run, confirm non-zero rewards on at least 1 trajectory)
4. Cell: CounselToolEnv class (from EXECUTE.md §5.6 below)
5. Cell: dataset construction (256 prompts initially)
6. Cell: GRPOTrainer config — Qwen3-1.7B, vLLM colocate, max_completion_length=2048, num_generations=4, gradient_accumulation_steps=16
7. Cell: trainer.train() — leave commented for now; we run this in Phase 5.

ACCEPTANCE CRITERIA:
- Cell 2 (Echo dry run) completes with non-zero rewards
- Cell 3 (Counsel dry run) completes; assert at least 1 of 10 episodes returns reward > 0
- Cell 4-6 imports cleanly with no errors

[paste CounselToolEnv class skeleton from cross_examination_arena_hackathon_plan.md §5.6]
```

---

## 5. Phase 4 — Baseline measurement (Trainer, ~1 hr)

### Claude Code prompt to paste:

```
TASK: Generate the BEFORE numbers for the README.

Run 50 episodes of CounselToolEnv with untrained Qwen3-1.7B (not in training mode, just inference via vLLM serve mode or transformers).

For each episode:
- record total reward
- record number of contradictions surfaced / total contradictions
- save the full transcript

Produce:
- baseline_metrics.json: {avg_reward, std_reward, avg_contradictions_surfaced_ratio, distribution_by_case_template}
- baseline_transcripts.jsonl: one episode per line
- baseline_plot.png: histogram of episode rewards

ACCEPTANCE CRITERIA:
- baseline_metrics.json produced
- avg_reward should be in roughly [0.05, 0.40] range — if it's 0.0, the env is too hard or the witness is too strict; if it's >0.5, the env is too easy. Either way, flag the issue.
- 5 transcripts hand-picked as "good representative examples" saved separately (a mix of partial-success and full-failure)
```

---

## 6. Phase 5 — Training run (Trainer, 30 min setup + 90 min compute)

### Run on HF Jobs:

```bash
# From the counsel-env repo root
huggingface-cli jobs run \
  --gpu a100-large \
  --image huggingface/transformers-pytorch-gpu:latest \
  --secret HF_TOKEN=$HF_TOKEN \
  --command "pip install trl[vllm] unsloth openenv-core git+https://huggingface.co/spaces/<your-username>/counsel-env && \
             python notebooks/train_counsel.py 2>&1 | tee training.log"
```

### Or alternative with Colab Pro:

Open `notebooks/train_counsel.ipynb` in Colab Pro, set runtime to A100, run all cells. Save checkpoints to HF Hub via `trainer.push_to_hub()`.

### During training, monitor:
- `train/reward` — should trend upward over ~200 steps
- `train/reward_func_0` — primary reward (contradictions surfaced)
- `train/loss` — should decrease but not collapse
- `eval/reward` if you set up an eval split

**HARD RULE during training:** do NOT push commits to the HF Space. Concurrent connections will break and training will fail mid-run.

### Acceptance criteria:
- Final `train/reward` ≥ 1.5× baseline mean
- Training completes without OOM
- Final checkpoint pushed to HF Hub or saved to outputs/

---

## 7. Phase 6 — Evaluation (Trainer, ~1.5 hrs)

### Claude Code prompt to paste:

```
TASK: Generate the AFTER numbers for the README and produce the killer plots.

1. Load the trained checkpoint from Phase 5.
2. Run 50 episodes against the live Counsel Space (same setup as baseline).
3. Produce trained_metrics.json, trained_transcripts.jsonl.
4. Produce these plots:
   - reward_curve.png: x=training step, y=train/reward, with baseline mean as dashed horizontal line
   - rubric_breakdown.png: stacked bar of rubric components (contradictions, evidence_timing, admissibility) at start vs end of training
   - episode_distribution.png: side-by-side histograms of baseline vs trained episode rewards

5. Produce a "before/after" pairs file: 3 carefully selected case_ids, with both the baseline transcript and the trained transcript on that same case. Annotate where contradictions were surfaced (or missed).

ACCEPTANCE CRITERIA:
- All 4 PNGs committed to the repo at /assets/plots/
- before_after_pairs.md produced with 3 high-quality comparisons
- avg_reward_trained ≥ 1.5× avg_reward_baseline (if not, investigate and consider retraining)
```

---

## 8. Phase 7 — Storytell (Scribe, ~3 hrs, can start in parallel from Phase 1)

### Claude Code prompt to paste:

```
You are the scribe subagent. Read .claude/agents/scribe.md for voice/tone guidance.

TASK: Write the public-facing materials — README.md (on HF Space), demo video script, mini-blog.

1. Update README.md on the HF Space with the structure in .claude/skills/hackathon-submission/SKILL.md §README. Specifically:
   - Hook image (use one of the trained transcripts with contradictions highlighted)
   - One-paragraph elevator pitch
   - Problem framing — current LLMs fold under adversarial dialogue
   - Architecture diagram (Mermaid block) — case_generator → witness → rubric tree
   - Action/observation tables
   - "How a contradiction works" with the alibi case as a worked example
   - Reward rubric explanation
   - Training results (link the 4 plot PNGs)
   - Before/after transcript section (use before_after_pairs.md)
   - Try-it-yourself install command
   - Limitations and future work
   - Links: Colab notebook, demo video, blog post

2. Write a 2-minute demo video script (4 beats × 30s):
   - Hook: baseline rollout where witness escapes
   - Trained rollout: model surfaces contradiction with annotations
   - The reward plot
   - Why it matters

3. Write a 600-word blog post on HuggingFace blog format. Title: "Teaching an LLM to Cross-Examine: Building Counsel-Env on OpenEnv". Voice: confident, slightly playful, technical-but-accessible.

ACCEPTANCE CRITERIA:
- README on HF Space renders correctly (Mermaid diagrams display, plots load, links work)
- Reading time on README is 4-6 min — long enough to be substantive, short enough to actually read
- Video script length matches a 2-min recording (assume ~150 words/min spoken, so ~300 words total)
- Blog post is on huggingface.co/blog/<your-handle>/counsel-env or similar
- All 4 surfaces (Space README, video, blog, GitHub mirror) link to each other
```

---

## 9. Phase 8 — Submission (Scribe + everyone, ~30 min)

### Final checklist (do not submit before all green):

- [ ] HF Space `<username>/counsel-env` shows **Running** (not paused/sleeping/error)
- [ ] `SUPPORTS_CONCURRENT_SESSIONS=True` in CounselEnvironment
- [ ] `pip install git+https://huggingface.co/spaces/<username>/counsel-env` works on a fresh machine
- [ ] `openenv.yaml` is valid and has appropriate tags
- [ ] No reserved tool names (`reset`, `step`, `state`, `close`) used as MCP tool names
- [ ] README.md lives on the HF Space (not just in GitHub)
- [ ] README links to: Colab notebook URL, video URL (YouTube/Loom), blog URL, GitHub mirror URL
- [ ] All 4 plot PNGs are committed to the repo at /assets/plots/
- [ ] Per-rubric metrics PNG is committed
- [ ] At least 2 (ideally 3) before/after transcript pairs in the README
- [ ] Colab notebook re-runs end-to-end on a fresh A100 — actually verify by re-running once
- [ ] Demo video published (YouTube unlisted is fine) and linked
- [ ] Blog post published and linked
- [ ] Smoke test passes: a fresh teammate's machine can `pip install` your client and run reset() + step() against the live Space
- [ ] Submit URL is the **HF Space URL** (per hackathon rules), locked in
- [ ] No commits to the repo after submit time — judges only see the version at deadline

### When everything is green:

Submit via the hackathon dashboard. Then go eat. Stop touching the repo.

---

## 10. Mental model: what each Claude Code prompt looks like

The prompts above follow a consistent structure that works well with Claude Code:

```
[Subagent / context loader]
You are the X subagent. Read [skills/agents].

TASK: [one-sentence what]

[file-by-file build instructions OR sub-tasks]

ACCEPTANCE CRITERIA:
- [verifiable check 1]
- [verifiable check 2]

GOTCHAS to watch:
- [thing that will trip Claude up]

[any code skeletons or reference snippets]
```

If a prompt fails or Claude Code drifts, rewrite the prompt, don't argue with the agent. Tighten acceptance criteria. Add explicit "do not do X" lines.

---

## 11. Failure modes & escalation

| Symptom | Likely cause | Fix |
|---|---|---|
| HF Space stuck "Building" >10 min | requirements.txt has dependency conflict | Pin versions, especially openenv-core |
| HF Space "Running" but step() hangs | concurrent sessions not enabled | SUPPORTS_CONCURRENT_SESSIONS=True + max_concurrent_envs in app.py |
| Training reward stays at 0 | Witness too strict / triggers don't fire | Hand-rollout 5 episodes with GPT-4 as lawyer; if even GPT-4 fails, ease trigger keywords |
| OOM during training on A100-40GB | Qwen3-1.7B + vLLM colocate too tight | Drop max_completion_length to 1024, num_generations to 2 |
| Eval reward not significantly above baseline | Insufficient training | Check loss trajectory; if loss plateaued, KL coefficient may be too high — drop to 0.04 |
| HF Jobs queue full | Peak hackathon time | Fall back to Colab Pro A100 |

---

## 12. Mentors / Meta engineers — talking points

When mentors come by:

- "Theme 1 with theory-of-mind angle. Cross-examination is underexplored in the hub — we verified."
- "Reward rubric is composable using openenv.core.rubrics: WeightedSum over 5 components, primary signal kept binary per TRL guidance."
- "Witness is intentionally deterministic — verifiable rewards, no GPU contention with trainer, swappable for an LLM-witness in v2."
- "Procgen case templates, currently 5, easily extensible. ~30k unique slot-fill combinations."

Avoid: "we hope this will train," "we built a Wordle variant," any hedging language. You did this work — own it.
