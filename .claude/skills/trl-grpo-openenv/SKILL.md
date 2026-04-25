# TRL GRPO + OpenEnv Integration

**This skill loads when working with TRL training scripts that use OpenEnv.** The `environment_factory` pattern was added to TRL in v1.0 (post-cutoff). Read this before writing any GRPOTrainer code.

## When to use environments vs tools

`GRPOTrainer` supports two modes for agentic tasks:
- **`tools`** — model calls stateless functions; each call is independent
- **`environments`** — state is maintained across turns; what the agent sees next depends on what it did before

For Counsel-Env, **always use `environments`**. The witness's responses depend on prior commitments; that's the whole point.

## The `environment_factory` pattern (canonical)

You pass a **class** (not an instance) to `GRPOTrainer(..., environment_factory=MyEnvClass)`. The trainer creates one instance per generation in a group.

```python
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from counsel_env import CounselEnv
from counsel_env.models import CounselAction

ENV_URL = "https://<user>-counsel-env.hf.space"

class CounselToolEnv:
    def __init__(self):
        # NO arguments allowed in __init__ when using environment_factory.
        # Capture URL from enclosing scope or module-level constants.
        self.client = CounselEnv(base_url=ENV_URL)
        self.reward = 0.0
        self.done = False

    def reset(self, **kwargs) -> str | None:
        # Called at start of each episode. Receives all dataset columns as kwargs.
        # Return: a string observation that the model sees as its first message,
        # or None if you want no initial observation.
        result = self.client.reset()
        self.reward = 0.0
        self.done = False
        return f"CASE BRIEF: {result.observation.case_brief}\nYou have {result.observation.questions_remaining} questions."

    def ask_question(self, question: str) -> str:
        """Ask the witness a question.

        Args:
            question: The question to ask the witness.

        Returns:
            The witness's response.
        """
        if self.done:
            raise ValueError("Episode is over.")
        result = self.client.step(CounselAction(tool="ask_question", text=question))
        self.reward = result.observation.reward
        self.done = result.observation.done
        return f"WITNESS: {result.observation.witness_response}"

    def present_evidence(self, exhibit_id: str) -> str:
        """Present an exhibit to the witness.

        Args:
            exhibit_id: The id of the exhibit to present.

        Returns:
            The witness's reaction.
        """
        if self.done:
            raise ValueError("Episode is over.")
        result = self.client.step(CounselAction(tool="present_evidence", exhibit_id=exhibit_id))
        self.reward = result.observation.reward
        self.done = result.observation.done
        return result.observation.witness_response

    def rest_case(self) -> str:
        """End the cross-examination."""
        result = self.client.step(CounselAction(tool="rest_case"))
        self.reward = result.observation.reward
        self.done = True
        return "Case rested."
```

### Critical rules for environment classes

1. **`__init__(self)` takes no arguments.** Capture configuration from enclosing scope (module-level constants like `ENV_URL`) or env vars. The trainer instantiates `MyEnvClass()` directly with no args.

2. **`reset(self, **kwargs)`** receives all dataset columns as kwargs. Use them to route to the right case, dataset row, etc. Return a string (the initial observation) or None.

3. **Public methods (not starting with `_`) other than `reset` are auto-discovered as tools.** The trainer parses their docstrings to build the tool schema. **Docstrings must have `Args:` blocks** in the format above; without them, the tool schema is malformed and the model can't call them.

4. **State for reward.** Store anything you want on the instance (`self.reward`, `self.done`, etc.) and access in your reward function via the `environments` parameter.

5. **Errors signal episode end.** If a tool method raises (e.g., `ValueError("Game over")`), TRL catches it and feeds the error message back to the model as a tool response. Use this for "you tried to act after the episode ended" cases.

6. **Don't name a tool `step` or `state` or `close`.** Use semantic names (`ask_question`, `present_evidence`, `rest_case`). Models also learn faster from descriptive tool names.

## Reward functions

```python
def reward_func(environments, **kwargs) -> list[float]:
    """environments is a list of MyEnvClass instances, one per generation."""
    return [env.reward for env in environments]
```

For multiple reward components, pass a list to GRPOTrainer; TRL sums them:

```python
def reward_primary(environments, **kwargs):
    return [env.reward for env in environments]

def reward_admissibility(environments, **kwargs):
    return [-0.1 * env.inadmissible_count for env in environments]

trainer = GRPOTrainer(
    ...,
    reward_funcs=[reward_primary, reward_admissibility],
)
```

Each reward function shows up as `train/reward_func_0`, `train/reward_func_1`, ... in the logs. Watch these individually; the combined `train/reward` metric alternates and can look noisy.

### TRL guidance on reward shape (FROM THE OFFICIAL DOCS)

- **Simple rewards work well.** Binary (1.0 success / 0.0 fail) outperformed shaped rewards in TRL's Wordle/Sudoku experiments. GRPO compares within-group; relative ranking matters more than absolute values.
- **Check final state, not the path.** Let the env judge outcome. Don't reward "the model said the right thing in turn 3."
- **Test before training.** Hand-rollout 5 episodes with GPT-4 or Claude as the lawyer; if a strong model can't beat random, your env is too hard or your reward signal is broken. Fix that first.

## GRPOConfig settings that matter for our use case

```python
from trl import GRPOConfig

config = GRPOConfig(
    # Generation
    use_vllm=True,
    vllm_mode="colocate",           # single GPU mode; "server" if multi-GPU
    chat_template_kwargs={"enable_thinking": False},  # disable for small models
    max_completion_length=2048,     # TOTAL tokens across all turns + tool results
    num_generations=4,              # group size; more = better signal but more compute

    # Optimization
    gradient_accumulation_steps=16,
    per_device_train_batch_size=1,
    learning_rate=1e-5,
    num_train_epochs=1,
    max_steps=200,                  # cap so we know when to stop

    # Logging
    log_completions=True,
    logging_steps=5,
    report_to="wandb",              # or "trackio", or "none"

    # KL penalty (lower = more divergence allowed)
    beta=0.04,

    # Output
    output_dir="./counsel-grpo-output",
    push_to_hub=True,
    hub_model_id="<user>/counsel-qwen3-1.7b-grpo",
)
```

**`max_completion_length` is total per episode, not per turn.** Multi-turn episodes with tool calls eat through this fast — 15 questions × ~80 tokens question + ~50 tokens witness response + ~30 tokens model thinking = ~2400 tokens per episode. Set 2048 as a starting point; if episodes are getting truncated, raise to 3072.

## Server concurrency requirements

TRL opens N WebSocket connections per training step (one per generation in the group). The OpenEnv server **must** support this:

1. In the Environment subclass: `SUPPORTS_CONCURRENT_SESSIONS = True`
2. In `server/app.py`: `create_app(..., max_concurrent_envs=64)` — must be ≥ `per_device_train_batch_size × gradient_accumulation_steps × num_generations`. Our setup: 1 × 16 × 4 = 64.

**For training, duplicate the HF Space to your own account.** Shared Spaces don't support enough concurrent sessions and you'll get connection-refused errors mid-training.

## Standard training script structure

```python
# 1. Imports
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from counsel_env import CounselEnv
from counsel_env.models import CounselAction

# 2. Constants
ENV_URL = "https://<user>-counsel-env.hf.space"
MODEL = "Qwen/Qwen3-1.7B"

# 3. Environment factory class
class CounselToolEnv:
    # ... (see above)

# 4. Reward function(s)
def reward_func(environments, **kwargs):
    return [env.reward for env in environments]

# 5. Dataset
prompt = "You are a sharp prosecutor cross-examining a witness. Use ask_question and present_evidence to surface contradictions in their story. Be efficient — you have a 15-question budget."
dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * 256})

# 6. Train
trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=reward_func,
    train_dataset=dataset,
    args=GRPOConfig(...),
    environment_factory=CounselToolEnv,
)
trainer.train()
```

## Failure modes & debugging

| Symptom | Cause | Fix |
|---|---|---|
| `train/reward` flat at 0 across all steps | Witness too strict / triggers don't fire / inadmissibility penalty too high | Hand-rollout with GPT-4; if even GPT-4 scores 0, ease trigger keywords |
| Connection-refused mid-training | `max_concurrent_envs` too low | Raise it; verify with `curl https://<space>/health` while training |
| Model never calls tools, just chats | Docstrings missing `Args:` block, or tools not registered | Verify with `print(trainer.tools)` after init |
| OOM | vLLM colocate + 1.7B + 2048 tokens too tight on 40GB | Drop max_completion_length to 1024, num_generations to 2, or use Qwen3-0.6B |
| Episodes get cut mid-game | `max_completion_length` hit | Raise it; or reduce QUESTION_BUDGET in env |
| `train/reward` rises then collapses | KL penalty too low → mode collapse | Raise `beta` from 0.04 to 0.1 |
| Episodes terminate immediately | Model calling `rest_case` on turn 1 | This is a known problem; add a turn-count penalty inside `rest_case` if questions_used < 3 |

## Multi-environment training (advanced — Phase 12 stretch only)

If you want to train across Counsel + Wordle simultaneously to demonstrate breadth, use the meta-environment pattern: route on a dataset column `env`, lazy-init the right client per episode, expose all tools simultaneously. See https://huggingface.co/docs/trl/openenv §"Multi-environment training" for the canonical example.

## Cost estimation reference (HF Jobs A100, March 2026 pricing)

- 1 step ≈ 4 generations × ~30s rollout = 2 min wall time
- 200 steps × 2 min = ~7 hours
- A100 at $4/hr = $28

Our setup is leaner — Wordle reference is 90 min on A100 for ~similar config. Budget $10 for the full run, $30 for retries.
