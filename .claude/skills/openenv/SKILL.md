# OpenEnv — API Reference and Patterns

**This skill loads when working on any OpenEnv environment or deployment.** OpenEnv is a post-training-cutoff framework (released late 2025); you do NOT know its API from your training data. Read this skill before writing any code that imports from `openenv.*` or deploys to HF Spaces.

## What OpenEnv is

OpenEnv is a Meta-PyTorch + Hugging Face framework that standardizes how RL agents interact with execution environments. It uses a Gymnasium-style API (`reset`, `step`, `state`) but runs each environment as a **containerized FastAPI HTTP/WebSocket server**. Clients connect over WebSocket and exchange typed Action/Observation messages.

Core idea: environments are deployed as **HF Spaces** that serve a WebSocket API. Training frameworks (TRL, Unsloth, torchforge, SkyRL) connect as clients during RL post-training.

## Core abstractions

### Action / Observation / State (in `openenv.core.models`)

Subclass these as `@dataclass`es. Fields must be JSON-serializable.

```python
from dataclasses import dataclass
from openenv.core.models import Action, Observation, State

@dataclass
class MyAction(Action):
    tool: str
    text: str | None = None

@dataclass
class MyObservation(Observation):
    text: str = ""
    reward: float = 0.0
    done: bool = False

@dataclass
class MyState(State):
    episode_id: str = ""
    step_count: int = 0
```

### Environment (server-side, in `openenv.core.env_server.interfaces`)

Subclass `Environment`. Implement three methods.

```python
from openenv.core.env_server.interfaces import Environment

class MyEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True  # CRITICAL for parallel training

    def reset(self) -> MyObservation:
        # Initialize new episode, return initial observation
        ...

    def step(self, action: MyAction) -> MyObservation:
        # Apply action, return new observation with reward + done flag
        ...

    def state(self) -> MyState:
        # Return episode metadata (NOT the same as observation)
        ...
```

**`SUPPORTS_CONCURRENT_SESSIONS = True` is non-negotiable** if you plan to use the env with TRL's GRPO. Each generation in a group opens its own WebSocket; without this flag, the server only handles one session at a time and training will hang or crash.

### Client (`openenv.core.client.http_env_client.HTTPEnvClient`)

```python
from openenv.core.client.http_env_client import HTTPEnvClient
from .models import MyAction, MyObservation, MyState

class MyEnv(HTTPEnvClient):
    Action = MyAction
    Observation = MyObservation
    State = MyState

# Usage:
async with MyEnv(base_url="https://<user>-<space>.hf.space") as client:
    result = await client.reset()
    result = await client.step(MyAction(tool="my_tool"))

# Sync wrapper:
with MyEnv(base_url="...").sync() as client:
    result = client.reset()
```

### FastAPI server (`openenv.core.env_server.create_app`)

```python
from openenv.core.env_server import create_app
from .counsel_environment import CounselEnvironment
from ..models import CounselAction, CounselObservation

app = create_app(
    CounselEnvironment,        # the class, not an instance
    CounselAction,
    CounselObservation,
    max_concurrent_envs=64,    # match or exceed TRL's generation_batch_size
)
```

`max_concurrent_envs` must be ≥ `per_device_train_batch_size × gradient_accumulation_steps`. With our settings (1 × 16) we need ≥ 16; we set 64 for headroom.

## Rubrics (`openenv.core.rubrics`)

OpenEnv provides composable reward functions modeled on PyTorch's `nn.Module`. **Use these for the rubric tree.** Don't roll your own.

```python
from openenv.core.rubrics import Rubric
from openenv.core.rubrics.containers import WeightedSum, Sequential, Gate

class ContradictionsSurfaced(Rubric):
    def forward(self, action, observation) -> float:
        # Return reward in [0, 1]
        ...

class EvidenceTiming(Rubric):
    def forward(self, action, observation) -> float:
        ...

class Total(Rubric):
    def __init__(self):
        super().__init__()
        self.surfaced = ContradictionsSurfaced()
        self.timing = EvidenceTiming()
        self.combined = WeightedSum(
            [self.surfaced, self.timing],
            weights=[1.0, 0.3],
        )

    def forward(self, action, observation) -> float:
        return self.combined(action, observation)
```

Available containers:
- `WeightedSum(rubrics, weights)` — linear combination
- `Sequential(*rubrics)` — fail-fast: if any returns 0, return 0
- `Gate(rubric, threshold)` — pass-through if score ≥ threshold else 0
- `RubricList`, `RubricDict` — pure containers (you define aggregation)

Child rubrics auto-register when assigned as attributes (PyTorch pattern). `state_dict()`, `load_state_dict()`, `get_submodule()` all work.

**Wire the rubric to the Environment** by passing it to the constructor:
```python
class CounselEnvironment(Environment):
    def __init__(self):
        super().__init__(rubric=Total())
```
Then in `step()`, use `self.rubric(action, observation)` to compute reward.

## Manifest (`openenv.yaml`)

```yaml
name: counsel-env
description: Cross-examination arena — train LLMs to surface contradictions in witness testimony.
tags:
  - rl
  - multi-agent
  - theory-of-mind
  - dialogue
  - hackathon
license: bsd-3-clause
```

## Dockerfile (server-side)

Use the OpenEnv base image. Minimal Dockerfile:

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
COPY server/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## CLI

```bash
openenv init <name>          # scaffold a new env
openenv push                 # deploy current dir to HF Spaces
openenv push --private       # deploy as private Space
openenv push --repo-id org/name
```

After push, the Space URL pattern is `https://<username>-<env-name>.hf.space` (note: dash between user and env, not slash). Build takes 3-5 min.

## pyproject.toml (so pip install from Space works)

```toml
[project]
name = "counsel-env"
version = "0.1.0"
dependencies = [
    "openenv-core>=0.2.1",
    "fastapi",
    "uvicorn",
    "pydantic",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

This makes `pip install git+https://huggingface.co/spaces/<user>/<env>` work, which is how TRL imports your client.

## Common gotchas

1. **WebSocket vs HTTP.** The Python client uses WebSocket by default (more efficient for multi-turn). HTTP endpoints exist for debugging. If you see "connection refused" but Space is Running, you may be hitting HTTP when WebSocket is expected — make sure you're using `EnvClient(base_url=...)` not raw requests.

2. **Reserved tool names.** Don't name MCP tools `reset`, `step`, `state`, or `close`. They collide with the framework.

3. **Public methods become tools (TRL pattern).** When using `environment_factory`, every public method on your TRL-wrapper class becomes a callable tool. Prefix internal helpers with `_`. Methods MUST have docstrings with `Args:` blocks — TRL parses these to build the tool schema.

4. **Action dataclass with optional fields.** Always provide defaults (`= None`) for optional Action fields. Otherwise FastAPI rejects requests missing them.

5. **Build failures on push.** Most common cause is openenv-core version mismatch between local pyproject.toml and the base image. Keep the version pinned at `>=0.2.1`.

6. **Space sleeping.** Free CPU spaces sleep after inactivity. Mid-training, this can cause silent failures. For training runs, either upgrade to "always-on" CPU upgrade tier (cheap, ~$0.05/hr) or duplicate the Space privately and set persistence.

7. **Concurrent connections + TRL.** TRL opens N WebSocket connections (one per generation). Set `max_concurrent_envs` ≥ N or training will fail mid-rollout with a stale-connection error.

8. **`from_env` vs base_url.** `MyEnv.from_env("openenv/echo-env")` pulls and runs the env locally. `MyEnv(base_url="https://...")` connects to a remote running Space. Use `base_url` for training; `from_env` is for local development.

9. **Environment instance per session.** When SUPPORTS_CONCURRENT_SESSIONS=True, the framework creates a fresh Environment instance per session. This means `__init__` must be cheap (no model loading). Defer expensive setup to `reset()` or constructor-time module-level singletons.

## Reference environments to study

These are real OpenEnv envs in the hub — read their source for patterns:

- `meta-pytorch/OpenEnv/envs/echo_env` — minimal reference, shows MCP tool pattern
- `meta-pytorch/OpenEnv/envs/openspiel_env` — wraps an external library
- `meta-pytorch/OpenEnv/envs/coding_env` — sandboxed Python execution
- `meta-pytorch/OpenEnv/envs/chess_env` — game with configurable opponent (similar pattern to our witness)

The Wildfire env (huggingface.co/spaces/shankerram3/wildfire_env) is cited as a community reference implementation worth reading.
