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
"""Oracle warm-start plus GRPO refinement for Counsel-Env."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer


SPACE_REPO = os.getenv("COUNSEL_SPACE_REPO", "heavycoderhh/counsel-env")
MODEL = os.getenv("COUNSEL_MODEL", "heavycoderhh/counsel-env-qwen3-0.6b-grpo-run2")
ENV_URL = os.getenv("COUNSEL_ENV_URL", "https://heavycoderhh-counsel-env.hf.space")
OUTPUT_DIR = Path(os.getenv("COUNSEL_OUTPUT_DIR", "/tmp/counsel-sft-grpo-output"))
SFT_DIR = OUTPUT_DIR / "sft_warm_start"
ARTIFACT_REPO = os.getenv("COUNSEL_ARTIFACT_REPO", "heavycoderhh/counsel-env-qwen3-0.6b-grpo-run3")
SFT_DATASET_SIZE = int(os.getenv("COUNSEL_SFT_DATASET_SIZE", "720"))
SFT_EPOCHS = float(os.getenv("COUNSEL_SFT_EPOCHS", "1"))
SFT_LEARNING_RATE = float(os.getenv("COUNSEL_SFT_LEARNING_RATE", "1e-5"))
MAX_STEPS = int(os.getenv("COUNSEL_MAX_STEPS", "500"))
GRPO_DATASET_SIZE = int(os.getenv("COUNSEL_DATASET_SIZE", "320"))
NUM_GENERATIONS = int(os.getenv("COUNSEL_NUM_GENERATIONS", "4"))
MAX_COMPLETION_LENGTH = int(os.getenv("COUNSEL_MAX_COMPLETION_LENGTH", "256"))
GRPO_LEARNING_RATE = float(os.getenv("COUNSEL_LEARNING_RATE", "3e-6"))
EVIDENCE_PRESSURE = float(os.getenv("COUNSEL_EVIDENCE_PRESSURE", "2.0"))
USE_VLLM = os.getenv("COUNSEL_USE_VLLM", "0") == "1"
MAX_SFT_LENGTH = int(os.getenv("COUNSEL_MAX_SFT_LENGTH", "3072"))


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
from counsel_env.server.counsel_env_environment import CounselEnvironment  # noqa: E402


def ask_question(question: str) -> str:
    """Ask the witness a question.

    Args:
        question: The cross-examination question to ask the witness.
    """
    raise RuntimeError("Tool schema only")


def present_evidence(exhibit_id: str) -> str:
    """Present an exhibit to the witness.

    Args:
        exhibit_id: The ID of the exhibit to present.
    """
    raise RuntimeError("Tool schema only")


def rest_case() -> str:
    """End the cross-examination."""
    raise RuntimeError("Tool schema only")


TOOLS = [ask_question, present_evidence, rest_case]


BASE_PROMPT = (
    "You are a sharp prosecuting attorney cross-examining a deterministic witness. "
    "Your goal is to surface contradictions by first making the witness commit to a claim, "
    "then presenting the exact exhibit that disproves it. Use the limited question budget efficiently. "
    "Critical sequence: ask one targeted question, wait for the committed claim, immediately present "
    "the matching exhibit ID, and only then rest or pursue another contradiction."
)


def sample_stage_schedule(total: int) -> List[str]:
    schedule = []
    for index in range(total):
        frac = index / max(1, total - 1)
        if frac < 0.45:
            schedule.append("easy")
        elif frac < 0.85:
            schedule.append("medium")
        else:
            schedule.append("hard")
    return schedule


def format_evidence(evidence_descriptions: Dict[str, str]) -> str:
    return "\n".join(f"- {exhibit_id}: {description}" for exhibit_id, description in evidence_descriptions.items())


def reset_text(obs: Any) -> str:
    return (
        f"CASE BRIEF:\n{obs.case_brief}\n\n"
        f"You have {obs.questions_remaining} questions. "
        "Available exhibits with descriptions:\n"
        f"{format_evidence(obs.evidence_descriptions)}\n\n"
        "Use ask_question first. After the witness commits, present the exact matching exhibit ID."
    )


def tool_feedback(obs: Any) -> str:
    components = obs.reward_components or {}
    evidence = ", ".join(obs.available_evidence)
    return (
        f"WITNESS: {obs.witness_response}\n"
        f"STATE: triggered={int(components.get('contradictions_triggered', 0))}/"
        f"{int(components.get('contradictions_total', 0))}, "
        f"surfaced={int(components.get('contradictions_surfaced', 0))}/"
        f"{int(components.get('contradictions_total', 0))}, "
        f"questions_remaining={obs.questions_remaining}, done={obs.done}\n"
        f"VALID_EXHIBITS: {evidence}"
    )


def tool_call(name: str, arguments: Dict[str, Any]) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, sort_keys=True) + "\n</tool_call>"


def create_oracle_sft_dataset(tokenizer: Any, num_samples: int) -> Dataset:
    rows: List[Dict[str, List[int]]] = []
    stages = sample_stage_schedule(num_samples)
    for seed, stage in enumerate(stages, start=314159):
        env = CounselEnvironment()
        obs = env.reset(seed=seed, curriculum_stage=stage, episode_id=f"sft_{seed}")
        messages = [{"role": "user", "content": f"{BASE_PROMPT}\n\n{reset_text(obs)}"}]

        for contradiction in env.witness.contradictions:
            question = f"{contradiction.trigger_keywords[0]}?"
            assistant_message = {"role": "assistant", "content": tool_call("ask_question", {"question": question})}
            rows.append(tokenize_assistant_action(tokenizer, messages, assistant_message))
            messages.append(assistant_message)
            obs = env.step(CounselAction(tool="ask_question", text=question))
            messages.append({"role": "user", "content": f"<tool_response>\n{tool_feedback(obs)}\n</tool_response>"})

            exhibit_id = contradiction.disprover_evidence_id
            assistant_message = {
                "role": "assistant",
                "content": tool_call("present_evidence", {"exhibit_id": exhibit_id}),
            }
            rows.append(tokenize_assistant_action(tokenizer, messages, assistant_message))
            messages.append(assistant_message)
            obs = env.step(CounselAction(tool="present_evidence", exhibit_id=exhibit_id))
            messages.append({"role": "user", "content": f"<tool_response>\n{tool_feedback(obs)}\n</tool_response>"})

        assistant_message = {"role": "assistant", "content": tool_call("rest_case", {})}
        rows.append(tokenize_assistant_action(tokenizer, messages, assistant_message))
        messages.append(assistant_message)
        obs = env.step(CounselAction(tool="rest_case"))
        messages.append({"role": "user", "content": f"<tool_response>\n{tool_feedback(obs)}\n</tool_response>"})

    return Dataset.from_list(rows)


def tokenize_assistant_action(
    tokenizer: Any,
    prompt_messages: List[Dict[str, str]],
    assistant_message: Dict[str, str],
) -> Dict[str, List[int]]:
    """Create one SFT row with loss only on the assistant tool-call span."""
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [assistant_message],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    assistant_start = full_text.rfind(assistant_message["content"])
    if assistant_start < 0:
        raise RuntimeError("assistant tool-call span was not found in rendered chat template")
    prompt_text = full_text[:assistant_start]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        max_length=MAX_SFT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        max_length=MAX_SFT_LENGTH,
        truncation=True,
    )["input_ids"]
    input_ids = encoded["input_ids"]
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    labels = [label if token != tokenizer.pad_token_id else -100 for label, token in zip(labels, input_ids)]
    encoded["labels"] = labels
    return encoded


def run_sft() -> None:
    print(
        {
            "phase": "sft_warm_start",
            "model": MODEL,
            "sft_dataset_size": SFT_DATASET_SIZE,
            "sft_epochs": SFT_EPOCHS,
            "sft_learning_rate": SFT_LEARNING_RATE,
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    dataset = create_oracle_sft_dataset(tokenizer, SFT_DATASET_SIZE)
    args = TrainingArguments(
        output_dir=str(SFT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LEARNING_RATE,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(SFT_DIR))
    tokenizer.save_pretrained(str(SFT_DIR))


def create_grpo_dataset(num_samples: int) -> Dataset:
    prompts = [[{"role": "user", "content": BASE_PROMPT}] for _ in range(num_samples)]
    return Dataset.from_dict({"prompt": prompts, "curriculum_stage": sample_stage_schedule(num_samples)})


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

    def reset(self, **kwargs: Any) -> str:
        self.stage = kwargs.get("curriculum_stage") or "medium"
        result = self.client.reset(curriculum_stage=self.stage)
        obs = result.observation
        self.reward = 0.0
        self.done = False
        self.questions = []
        self.actions = []
        self.last_components = obs.reward_components
        return reset_text(obs)

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
        return tool_feedback(obs)

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
        return tool_feedback(obs)

    def rest_case(self) -> str:
        """End the cross-examination."""
        self.actions.append("rest_case")
        result = self.client.step(CounselAction(tool="rest_case"))
        obs = result.observation
        self.reward = obs.reward
        self.done = True
        self.last_components = obs.reward_components
        return tool_feedback(obs)


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

        evidence = env.actions.count("present_evidence")
        questions = env.actions.count("ask_question")
        rested = bool(env.actions and env.actions[-1] == "rest_case")
        surface_rate = surfaced / total
        timing_rate = timing / total

        reward = 1.35 * surface_rate + 0.35 * timing_rate
        reward += 0.06 if evidence > 0 and surfaced > 0 else 0.0
        reward += 0.03 if triggered > 0 and evidence > 0 else 0.0
        reward += 0.04 if rested and surfaced >= total else 0.0
        if triggered > surfaced and evidence == 0:
            reward -= 0.30 * EVIDENCE_PRESSURE
        if evidence > 0 and surfaced == 0:
            reward -= 0.12 * EVIDENCE_PRESSURE
        if questions > 3 and surfaced == 0:
            reward -= 0.03 * (questions - 3)
        reward -= 0.08 * blind
        reward -= 0.08 * inadmissible
        reward -= 0.05 * duplicate
        reward -= 0.04 * irrelevant
        reward -= 0.01 * max(0, len(env.actions) - 6)
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
        except Exception as exc:
            last_error = exc
            time.sleep(5)
    raise RuntimeError(f"Counsel-Env did not become ready: {last_error}")


def write_summary(path: Path, train_result: Any) -> None:
    summary = {
        "model": MODEL,
        "sft_dir": str(SFT_DIR),
        "env_url": ENV_URL,
        "space_repo": SPACE_REPO,
        "artifact_repo": ARTIFACT_REPO,
        "sft_dataset_size": SFT_DATASET_SIZE,
        "sft_epochs": SFT_EPOCHS,
        "sft_learning_rate": SFT_LEARNING_RATE,
        "grpo_max_steps": MAX_STEPS,
        "grpo_dataset_size": GRPO_DATASET_SIZE,
        "num_generations": NUM_GENERATIONS,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "grpo_learning_rate": GRPO_LEARNING_RATE,
        "evidence_pressure": EVIDENCE_PRESSURE,
        "metrics": getattr(train_result, "metrics", {}) or {},
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
        commit_message=f"Upload SFT+GRPO Counsel-Env checkpoint ({MAX_STEPS} GRPO steps)",
    )
    print(f"Uploaded artifacts to https://huggingface.co/{ARTIFACT_REPO}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wait_for_env()
    run_sft()

    training_args = GRPOConfig(
        use_vllm=USE_VLLM,
        chat_template_kwargs={"enable_thinking": False},
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        temperature=0.9,
        top_p=0.95,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        learning_rate=GRPO_LEARNING_RATE,
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
        model=str(SFT_DIR),
        reward_funcs=reward_func,
        train_dataset=create_grpo_dataset(GRPO_DATASET_SIZE),
        args=training_args,
        environment_factory=CounselToolEnv,
    )
    train_result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    write_summary(OUTPUT_DIR / "training_summary.json", train_result)
    upload_artifacts(OUTPUT_DIR)


if __name__ == "__main__":
    main()
