# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.12.0",
#   "datasets>=4.0.0",
#   "huggingface_hub>=1.0.0",
#   "jmespath>=1.0.1",
#   "numpy>=2.0.0",
#   "openenv-core>=0.2.1",
#   "torch>=2.8.0",
#   "transformers>=5.2.0",
#   "trl>=0.29.0",
# ]
# ///
"""Headless GRPO training job for Counsel-Env.

This script is designed for Hugging Face Jobs. It downloads the latest Space
source, connects to the live Counsel-Env server, runs a bounded GRPO training
job, and uploads artifacts/checkpoints to a model repo.
"""

from __future__ import annotations

import json
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from trl import GRPOConfig, GRPOTrainer


SPACE_REPO = os.getenv("COUNSEL_SPACE_REPO", "heavycoderhh/counsel-env")
MODEL = os.getenv("COUNSEL_MODEL", "Qwen/Qwen3-0.6B")
ENV_URL = os.getenv("COUNSEL_ENV_URL", "https://heavycoderhh-counsel-env.hf.space")
OUTPUT_DIR = Path(os.getenv("COUNSEL_OUTPUT_DIR", "/tmp/counsel-grpo-output"))
ARTIFACT_REPO = os.getenv("COUNSEL_ARTIFACT_REPO", "heavycoderhh/counsel-env-qwen3-0.6b-grpo")
MAX_STEPS = int(os.getenv("COUNSEL_MAX_STEPS", "50"))
DATASET_SIZE = int(os.getenv("COUNSEL_DATASET_SIZE", "96"))
NUM_GENERATIONS = int(os.getenv("COUNSEL_NUM_GENERATIONS", "4"))
MAX_COMPLETION_LENGTH = int(os.getenv("COUNSEL_MAX_COMPLETION_LENGTH", "512"))
LEARNING_RATE = float(os.getenv("COUNSEL_LEARNING_RATE", "1e-5"))
EVIDENCE_PRESSURE = float(os.getenv("COUNSEL_EVIDENCE_PRESSURE", "1.0"))
USE_VLLM = os.getenv("COUNSEL_USE_VLLM", "0") == "1"


def prepare_imports() -> None:
    """Make the Space package importable inside HF Jobs."""
    try:
        import counsel_env  # noqa: F401

        return
    except Exception:
        pass

    source_dir = snapshot_download(repo_id=SPACE_REPO, repo_type="space")
    init_path = Path(source_dir) / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "counsel_env",
        init_path,
        submodule_search_locations=[source_dir],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Counsel-Env package from {source_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["counsel_env"] = module
    spec.loader.exec_module(module)
    print(f"Loaded Counsel-Env source from {source_dir}")


prepare_imports()

from counsel_env import CounselEnv  # noqa: E402
from counsel_env.models import CounselAction  # noqa: E402


def sample_stage_schedule(total: int) -> List[str]:
    schedule = []
    for index in range(total):
        frac = index / max(1, total - 1)
        if frac < 0.25:
            schedule.append("easy")
        elif frac < 0.70:
            schedule.append("medium")
        else:
            schedule.append("hard")
    return schedule


def create_dataset(num_samples: int) -> Dataset:
    base_prompt = (
        "You are a sharp prosecuting attorney cross-examining a deterministic witness. "
        "Your goal is to surface contradictions by first making the witness commit to a claim, "
        "then presenting the exact exhibit that disproves it. Use the limited question budget efficiently. "
        "Avoid repeated, irrelevant, leading, or compound questions. "
        "Critical sequence: ask one targeted question, wait for a committed claim, immediately present the "
        "matching exhibit, then continue only if another contradiction remains."
    )
    stages = sample_stage_schedule(num_samples)
    prompts = [[{"role": "user", "content": base_prompt}] for _ in stages]
    return Dataset.from_dict({"prompt": prompts, "curriculum_stage": stages})


def question_pattern(question: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", (question or "").lower())
    return " ".join(text.split()[:12])


class CounselToolEnv:
    """TRL tool environment wrapper. Constructor intentionally takes no args."""

    def __init__(self):
        self.client_context = CounselEnv(base_url=ENV_URL).sync()
        self.client = self.client_context.__enter__()
        self.reward = 0.0
        self.done = False
        self.stage = "medium"
        self.questions: List[str] = []
        self.actions: List[str] = []
        self.last_components: Dict[str, Any] = {}

    @staticmethod
    def _format_evidence(evidence_descriptions: Dict[str, str]) -> str:
        return "\n".join(f"- {exhibit_id}: {description}" for exhibit_id, description in evidence_descriptions.items())

    def reset(self, **kwargs: Any) -> str:
        self.stage = kwargs.get("curriculum_stage") or "medium"
        result = self.client.reset(curriculum_stage=self.stage)
        obs = result.observation
        self.reward = 0.0
        self.done = False
        self.questions = []
        self.actions = []
        self.last_components = obs.reward_components
        return (
            f"CASE BRIEF:\n{obs.case_brief}\n\n"
            f"You have {obs.questions_remaining} questions. "
            "Available exhibits with descriptions:\n"
            f"{self._format_evidence(obs.evidence_descriptions)}\n\n"
            "Use ask_question to make the witness commit. Once the witness commits to a claim, use "
            "present_evidence with the exact exhibit ID that disproves that claim. Do not rest until you "
            "have presented evidence for the committed contradiction."
        )

    def ask_question(self, question: str) -> str:
        """Ask the witness a question.

        Args:
            question: The cross-examination question to ask the witness.
        """
        if self.done:
            raise ValueError("Episode is over.")
        self.actions.append("ask_question")
        self.questions.append(question)
        result = self.client.step(CounselAction(tool="ask_question", text=question))
        obs = result.observation
        self.reward = obs.reward
        self.done = obs.done
        self.last_components = obs.reward_components
        return f"WITNESS: {obs.witness_response}\n[{obs.questions_remaining} questions left]"

    def present_evidence(self, exhibit_id: str) -> str:
        """Present an exhibit to the witness.

        Args:
            exhibit_id: The ID of the exhibit to present.
        """
        if self.done:
            raise ValueError("Episode is over.")
        self.actions.append("present_evidence")
        result = self.client.step(CounselAction(tool="present_evidence", exhibit_id=exhibit_id))
        obs = result.observation
        self.reward = obs.reward
        self.done = obs.done
        self.last_components = obs.reward_components
        return obs.witness_response

    def rest_case(self) -> str:
        """End the cross-examination."""
        self.actions.append("rest_case")
        result = self.client.step(CounselAction(tool="rest_case"))
        obs = result.observation
        self.reward = obs.reward
        self.done = True
        self.last_components = obs.reward_components
        return f"Case rested. Final reward: {self.reward:.3f}"

    @property
    def action_diversity(self) -> int:
        return len(set(self.actions))

    @property
    def unique_question_patterns_pct(self) -> float:
        patterns = {question_pattern(question) for question in self.questions}
        return len(patterns) / max(1, len(self.questions))


def reward_func(environments: List[CounselToolEnv], **_: Any) -> List[float]:
    rewards: List[float] = []
    for env in environments:
        components = env.last_components or {}
        total = max(1.0, float(components.get("contradictions_total", 1.0)))
        surfaced = float(components.get("contradictions_surfaced", 0.0))
        triggered = float(components.get("contradictions_triggered", 0.0))
        timing = float(components.get("evidence_timing_successes", 0.0))
        blind = float(components.get("blind_evidence_count", 0.0))
        inadmissible = float(components.get("inadmissible_actions", 0.0))
        duplicate = float(components.get("duplicate_questions", 0.0))
        irrelevant = float(components.get("irrelevant_questions", 0.0))

        action_count = max(1, len(env.actions))
        questions = env.actions.count("ask_question")
        evidence = env.actions.count("present_evidence")
        rested = env.actions[-1] == "rest_case" if env.actions else False

        surface_rate = surfaced / total
        trigger_rate = triggered / total
        timing_rate = timing / total

        reward = 1.35 * surface_rate
        reward += 0.35 * timing_rate
        reward += 0.04 * min(trigger_rate, surface_rate + 0.25)
        reward += 0.04 if evidence > 0 and blind == 0 else 0.0
        reward += 0.02 if rested and surfaced > 0 else 0.0

        if triggered > surfaced and evidence == 0:
            reward -= 0.30 * EVIDENCE_PRESSURE
        if evidence > 0 and surfaced == 0:
            reward -= 0.12 * EVIDENCE_PRESSURE
        if questions > 3 and surfaced == 0:
            reward -= 0.03 * (questions - 3)
        reward -= 0.08 * blind
        reward -= 0.08 * inadmissible
        reward -= 0.03 * (duplicate + irrelevant)
        reward -= 0.01 * max(0, action_count - 6)

        rewards.append(float(max(-0.5, min(1.5, reward))))
    return rewards


def wait_for_env() -> None:
    print(f"Checking environment at {ENV_URL}")
    deadline = time.time() + 120
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with CounselEnv(base_url=ENV_URL).sync() as client:
                result = client.reset(curriculum_stage="easy")
                print(f"Environment ready: {result.observation.case_id}")
                return
        except Exception as exc:  # pragma: no cover - remote readiness
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"Counsel-Env did not become ready: {last_error}")


def write_summary(path: Path, train_result: Any) -> None:
    metrics = getattr(train_result, "metrics", {}) or {}
    summary = {
        "model": MODEL,
        "env_url": ENV_URL,
        "space_repo": SPACE_REPO,
        "artifact_repo": ARTIFACT_REPO,
        "max_steps": MAX_STEPS,
        "dataset_size": DATASET_SIZE,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "learning_rate": LEARNING_RATE,
        "evidence_pressure": EVIDENCE_PRESSURE,
        "use_vllm": USE_VLLM,
        "metrics": metrics,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def upload_artifacts(output_dir: Path) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping artifact upload.")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=ARTIFACT_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=ARTIFACT_REPO,
        repo_type="model",
        folder_path=str(output_dir),
        commit_message=f"Upload Counsel-Env GRPO checkpoint ({MAX_STEPS} steps)",
    )
    print(f"Uploaded artifacts to https://huggingface.co/{ARTIFACT_REPO}")


def main() -> None:
    print(
        {
            "model": MODEL,
            "env_url": ENV_URL,
            "artifact_repo": ARTIFACT_REPO,
            "max_steps": MAX_STEPS,
            "dataset_size": DATASET_SIZE,
            "num_generations": NUM_GENERATIONS,
            "learning_rate": LEARNING_RATE,
            "evidence_pressure": EVIDENCE_PRESSURE,
            "use_vllm": USE_VLLM,
        }
    )
    wait_for_env()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = create_dataset(DATASET_SIZE)
    training_args = GRPOConfig(
        use_vllm=USE_VLLM,
        chat_template_kwargs={"enable_thinking": False},
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        temperature=0.8,
        top_p=0.95,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        beta=0.04,
        log_completions=True,
        logging_steps=5,
        save_steps=max(10, MAX_STEPS // 2),
        report_to="none",
        output_dir=str(OUTPUT_DIR),
        push_to_hub=False,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=training_args,
        environment_factory=CounselToolEnv,
    )
    train_result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    write_summary(OUTPUT_DIR / "training_summary.json", train_result)
    upload_artifacts(OUTPUT_DIR)


if __name__ == "__main__":
    main()

