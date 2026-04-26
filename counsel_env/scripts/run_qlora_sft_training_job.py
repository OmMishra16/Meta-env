# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.12.0",
#   "bitsandbytes>=0.48.0",
#   "datasets>=4.0.0",
#   "huggingface_hub>=1.0.0",
#   "openenv-core>=0.2.1",
#   "peft>=0.18.0",
#   "torch>=2.8.0",
#   "transformers>=5.2.0",
# ]
# ///
"""Fast QLoRA oracle-SFT run for larger Counsel-Env models.

This job is intended for quick 4B/8B ablations. It trains only LoRA adapter
weights on generated oracle next-action rows, uploads the adapter, and leaves
GRPO as an optional follow-up only if SFT evaluation beats the current best.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


SPACE_REPO = os.getenv("COUNSEL_SPACE_REPO", "heavycoderhh/counsel-env")
MODEL = os.getenv("COUNSEL_MODEL", "Qwen/Qwen3-8B")
OUTPUT_DIR = Path(os.getenv("COUNSEL_OUTPUT_DIR", "/tmp/counsel-qwen3-8b-qlora-sft"))
ARTIFACT_REPO = os.getenv("COUNSEL_ARTIFACT_REPO", "heavycoderhh/counsel-env-qwen3-8b-qlora-sft-run4")
SFT_DATASET_SIZE = int(os.getenv("COUNSEL_SFT_DATASET_SIZE", "480"))
SFT_EPOCHS = float(os.getenv("COUNSEL_SFT_EPOCHS", "1"))
SFT_MAX_STEPS = int(os.getenv("COUNSEL_SFT_MAX_STEPS", "320"))
SFT_LEARNING_RATE = float(os.getenv("COUNSEL_SFT_LEARNING_RATE", "2e-4"))
MAX_SFT_LENGTH = int(os.getenv("COUNSEL_MAX_SFT_LENGTH", "2048"))
LORA_R = int(os.getenv("COUNSEL_LORA_R", "16"))
LORA_ALPHA = int(os.getenv("COUNSEL_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("COUNSEL_LORA_DROPOUT", "0.05"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("COUNSEL_GRAD_ACCUM", "4"))
INCLUDE_REST_ROWS = os.getenv("COUNSEL_INCLUDE_REST_ROWS", "0") == "1"


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
    "Return exactly one tool call and no prose. Never invent exhibit IDs. "
    "Do not rest the case until after at least one contradiction has been surfaced."
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


def tokenize_assistant_action(
    tokenizer: Any,
    prompt_messages: List[Dict[str, str]],
    assistant_message: Dict[str, str],
) -> Dict[str, List[int]]:
    """Create one row with loss on the rendered assistant generation span."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": False},
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [assistant_message],
        tools=TOOLS,
        tokenize=False,
        add_generation_prompt=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    encoded = tokenizer(full_text, add_special_tokens=False, max_length=MAX_SFT_LENGTH, truncation=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, max_length=MAX_SFT_LENGTH, truncation=True)[
        "input_ids"
    ]
    input_ids = encoded["input_ids"]
    labels = list(input_ids)
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    encoded["labels"] = labels
    return encoded


def create_oracle_sft_dataset(tokenizer: Any, num_samples: int) -> Dataset:
    rows: List[Dict[str, List[int]]] = []
    stages = sample_stage_schedule(num_samples)
    for seed, stage in enumerate(stages, start=514159):
        env = CounselEnvironment()
        obs = env.reset(seed=seed, curriculum_stage=stage, episode_id=f"qwen8b_sft_{seed}")
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

        if INCLUDE_REST_ROWS:
            assistant_message = {"role": "assistant", "content": tool_call("rest_case", {})}
            rows.append(tokenize_assistant_action(tokenizer, messages, assistant_message))

    return Dataset.from_list(rows)


def write_summary(path: Path, train_result: Any, row_count: int) -> None:
    summary = {
        "recipe": "qwen3_8b_qlora_oracle_sft",
        "base_model": MODEL,
        "artifact_repo": ARTIFACT_REPO,
        "space_repo": SPACE_REPO,
        "sft_case_count": SFT_DATASET_SIZE,
        "sft_row_count": row_count,
        "sft_epochs": SFT_EPOCHS,
        "sft_max_steps": SFT_MAX_STEPS,
        "sft_learning_rate": SFT_LEARNING_RATE,
        "max_sft_length": MAX_SFT_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "include_rest_rows": INCLUDE_REST_ROWS,
        "metrics": getattr(train_result, "metrics", {}) or {},
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def upload_artifacts() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping artifact upload.")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=ARTIFACT_REPO, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=ARTIFACT_REPO,
        repo_type="model",
        folder_path=str(OUTPUT_DIR),
        commit_message=f"Upload Counsel-Env QLoRA SFT adapter ({SFT_MAX_STEPS} max steps)",
    )
    print(f"Uploaded QLoRA SFT adapter to https://huggingface.co/{ARTIFACT_REPO}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = create_oracle_sft_dataset(tokenizer, SFT_DATASET_SIZE)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=SFT_EPOCHS,
        max_steps=SFT_MAX_STEPS if SFT_MAX_STEPS > 0 else -1,
        learning_rate=SFT_LEARNING_RATE,
        optim="paged_adamw_8bit",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    train_result = trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    write_summary(OUTPUT_DIR / "training_summary.json", train_result, len(dataset))
    upload_artifacts()


if __name__ == "__main__":
    main()
