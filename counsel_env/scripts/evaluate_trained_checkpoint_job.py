# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "accelerate>=1.12.0",
#   "huggingface_hub>=1.0.0",
#   "numpy>=2.0.0",
#   "openenv-core>=0.2.1",
#   "pandas>=2.0.0",
#   "peft>=0.18.0",
#   "torch>=2.8.0",
#   "transformers>=5.2.0",
# ]
# ///
"""Evaluate a trained Counsel-Env checkpoint with tool calls."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SPACE_REPO = os.getenv("COUNSEL_SPACE_REPO", "heavycoderhh/counsel-env")
MODEL_REPO = os.getenv("COUNSEL_EVAL_MODEL", "heavycoderhh/counsel-env-qwen3-0.6b-grpo")
BASE_MODEL = os.getenv("COUNSEL_BASE_MODEL", "Qwen/Qwen3-0.6B")
EVAL_LABEL = os.getenv("COUNSEL_EVAL_LABEL", "trained_grpo")
OUTPUT_DIR = Path(os.getenv("COUNSEL_EVAL_OUTPUT_DIR", "/tmp/counsel-eval-output"))
EPISODES = int(os.getenv("COUNSEL_EVAL_EPISODES", "30"))
BASE_EPISODES = int(os.getenv("COUNSEL_BASE_EVAL_EPISODES", "10"))
MAX_TOOL_STEPS = int(os.getenv("COUNSEL_EVAL_MAX_TOOL_STEPS", "8"))
MAX_NEW_TOKENS = int(os.getenv("COUNSEL_EVAL_MAX_NEW_TOKENS", "256"))
START_SEED = int(os.getenv("COUNSEL_EVAL_START_SEED", "20260425"))
UPLOAD_REPO = os.getenv("COUNSEL_EVAL_UPLOAD_REPO", MODEL_REPO)
UPLOAD_PATH = os.getenv("COUNSEL_EVAL_UPLOAD_PATH", "eval")


def prepare_imports() -> None:
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
        raise RuntimeError(f"Could not load Counsel-Env from {source_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["counsel_env"] = module
    spec.loader.exec_module(module)


prepare_imports()

from counsel_env.models import CounselAction  # noqa: E402
from counsel_env.server.counsel_env_environment import CounselEnvironment  # noqa: E402
from counsel_env.evaluation import (  # noqa: E402
    evaluate_agent,
    keyword_spam_agent,
    make_eval_seeds,
    oracle_scripted_agent,
    present_all_agent,
    random_agent,
    summarize,
)


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
    "Return exactly one tool call and no prose. Avoid repeated, irrelevant, leading, or compound questions. "
    "Never invent exhibit IDs: choose only from the available exhibit list. "
    "After the witness commits to a claim, immediately present the matching exhibit. "
    "After all currently surfaced contradictions are exhausted, rest the case."
)


def reset_text(obs: Any) -> str:
    evidence = "\n".join(
        f"- {exhibit_id}: {description}" for exhibit_id, description in obs.evidence_descriptions.items()
    )
    return (
        f"CASE BRIEF:\n{obs.case_brief}\n\n"
        f"You have {obs.questions_remaining} questions. "
        f"Available exhibits with descriptions:\n{evidence}\n\n"
        "Use exactly one of ask_question, present_evidence, or rest_case. "
        "The exhibit_id argument must exactly match one listed exhibit ID."
    )


def parse_tool_call(text: str) -> tuple[Optional[str], Dict[str, Any]]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    raw = match.group(1) if match else None
    if raw is None:
        json_match = re.search(r"\{\s*\"name\"\s*:\s*\"[^\"]+\".*?\}", text, flags=re.DOTALL)
        raw = json_match.group(0) if json_match else None
    if raw is None:
        return None, {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, {}
    return payload.get("name"), payload.get("arguments") or {}


def model_reply(model: Any, tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=False)


def format_tool_feedback(env: CounselEnvironment, obs: Any) -> str:
    components = obs.reward_components or {}
    evidence = ", ".join(obs.available_evidence)
    return (
        f"WITNESS: {obs.witness_response}\n"
        f"STATE: triggered={int(components.get('contradictions_triggered', 0))}/"
        f"{int(components.get('contradictions_total', 0))}, "
        f"surfaced={int(components.get('contradictions_surfaced', 0))}/"
        f"{int(components.get('contradictions_total', 0))}, "
        f"questions_remaining={obs.questions_remaining}, done={obs.done}\n"
        f"VALID_EXHIBITS: {evidence}\n"
        "NEXT_HINT: If a witness just committed to a claim, present the matching exhibit. "
        "If every contradiction you can pursue is surfaced, rest_case."
    )


def execute_tool(env: CounselEnvironment, name: Optional[str], args: Dict[str, Any]) -> tuple[str, bool, str]:
    try:
        if name == "ask_question":
            obs = env.step(CounselAction(tool="ask_question", text=str(args.get("question", ""))))
        elif name == "present_evidence":
            obs = env.step(CounselAction(tool="present_evidence", exhibit_id=str(args.get("exhibit_id", ""))))
        elif name == "rest_case":
            obs = env.step(CounselAction(tool="rest_case"))
        else:
            return f"Invalid or missing tool call: {name}", True, "invalid_tool"
    except Exception as exc:
        return str({"error": str(exc)}), True, "tool_error"
    return format_tool_feedback(env, obs), obs.done, "ok"


def evaluate_model(repo_id: str, episodes: int, label: str) -> tuple[List[dict], List[str]]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        peft_config = PeftConfig.from_pretrained(repo_id)
    except Exception:
        peft_config = None
    load_repo = peft_config.base_model_name_or_path if peft_config is not None else repo_id
    tokenizer_repo = repo_id if peft_config is not None else load_repo
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        load_repo,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if peft_config is not None:
        model = PeftModel.from_pretrained(model, repo_id)
    model.eval()

    rows: List[dict] = []
    transcripts: List[str] = []
    seeds = make_eval_seeds(episodes, START_SEED)
    for index, seed in enumerate(seeds):
        random.seed(seed)
        env = CounselEnvironment()
        obs = env.reset(seed=seed, curriculum_stage="mixed", episode_id=f"{label}_{seed}")
        messages = [{"role": "user", "content": f"{BASE_PROMPT}\n\n{reset_text(obs)}"}]
        invalid_calls = 0

        for _step in range(MAX_TOOL_STEPS):
            reply = model_reply(model, tokenizer, messages)
            tool_name, args = parse_tool_call(reply)
            response, done, status = execute_tool(env, tool_name, args)
            if status != "ok":
                invalid_calls += 1
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"<tool_response>\n{response}\n</tool_response>"})
            if done:
                break

        if not env.done:
            env.step(CounselAction(tool="rest_case"))

        components = env._calculate_reward_components()
        row = {
            "agent": label,
            "model_repo": repo_id,
            "seed": seed,
            "case_id": env.case["case_id"],
            "difficulty": env.case["difficulty"],
            "reward": components["total_reward"],
            "primary_reward": components["primary_reward"],
            "auxiliary_reward": components["auxiliary_reward_raw"],
            "contradictions_total": int(components["contradictions_total"]),
            "contradictions_triggered": int(components["contradictions_triggered"]),
            "contradictions_surfaced": int(components["contradictions_surfaced"]),
            "questions_used": env.questions_used,
            "evidence_presented": env.evidence_presented_count,
            "evidence_timing_successes": int(components["evidence_timing_successes"]),
            "blind_evidence_count": int(components["blind_evidence_count"]),
            "useless_questions_ratio": components["useless_questions_ratio"],
            "avg_question_length": components["avg_question_length"],
            "invalid_tool_calls": invalid_calls,
            "transcript": env.export_transcript_markdown(),
        }
        print(json.dumps({k: v for k, v in row.items() if k != "transcript"}, sort_keys=True))
        rows.append(row)
        if index < 3:
            transcripts.append(f"# Agent: {label}\n\n" + env.export_transcript_markdown())

    return rows, transcripts


def _mean_ci(values: List[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    avg = mean(values)
    if len(values) < 2:
        return avg, 0.0
    return avg, 1.96 * stdev(values) / (len(values) ** 0.5)


def summarize_group(rows: List[dict], group: Dict[str, str]) -> dict:
    rewards = [row["reward"] for row in rows]
    primary = [row["primary_reward"] for row in rows]
    trigger_rates = [
        row["contradictions_triggered"] / max(1, row["contradictions_total"])
        for row in rows
    ]
    surface_rates = [
        row["contradictions_surfaced"] / max(1, row["contradictions_total"])
        for row in rows
    ]
    reward_mean, reward_ci95 = _mean_ci(rewards)
    primary_mean, primary_ci95 = _mean_ci(primary)
    surface_mean, surface_ci95 = _mean_ci(surface_rates)
    return {
        **group,
        "episodes": len(rows),
        "avg_reward": reward_mean,
        "avg_reward_ci95": reward_ci95,
        "avg_primary_reward": primary_mean,
        "avg_primary_reward_ci95": primary_ci95,
        "avg_trigger_rate": mean(trigger_rates) if trigger_rates else 0.0,
        "avg_surface_rate": surface_mean,
        "avg_surface_rate_ci95": surface_ci95,
        "avg_evidence_timing": mean(row["evidence_timing_successes"] for row in rows) if rows else 0.0,
        "avg_useless_ratio": mean(row["useless_questions_ratio"] for row in rows) if rows else 0.0,
        "invalid_tool_calls": sum(int(row.get("invalid_tool_calls", 0)) for row in rows),
    }


def summarize_expanded(rows: List[dict]) -> List[dict]:
    summaries: List[dict] = []
    for agent in sorted({row["agent"] for row in rows}):
        agent_rows = [row for row in rows if row["agent"] == agent]
        summaries.append(summarize_group(agent_rows, {"agent": agent, "slice": "all"}))
        for difficulty in sorted({row["difficulty"] for row in agent_rows}):
            difficulty_rows = [row for row in agent_rows if row["difficulty"] == difficulty]
            summaries.append(summarize_group(difficulty_rows, {"agent": agent, "slice": difficulty}))
    return summaries


def write_outputs(rows: List[dict], baseline_rows: List[dict], transcripts: List[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = baseline_rows + rows
    compact_rows = [{k: v for k, v in row.items() if k != "transcript"} for row in all_rows]
    summary = summarize(compact_rows)
    expanded_summary = summarize_expanded(compact_rows)
    (OUTPUT_DIR / "trained_eval_rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in compact_rows) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "trained_eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "trained_eval_expanded_summary.json").write_text(
        json.dumps(expanded_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "trained_eval_transcripts.md").write_text(
        "\n\n---\n\n".join(transcripts),
        encoding="utf-8",
    )
    pd.DataFrame(compact_rows).to_csv(OUTPUT_DIR / "trained_eval_rows.csv", index=False)
    pd.DataFrame(expanded_summary).to_csv(OUTPUT_DIR / "trained_eval_expanded_summary.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(expanded_summary, indent=2, sort_keys=True))


def upload_outputs() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping upload")
        return
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=UPLOAD_REPO,
        repo_type="model",
        folder_path=str(OUTPUT_DIR),
        path_in_repo=UPLOAD_PATH,
        commit_message="Add held-out trained checkpoint evaluation",
    )
    print(f"Uploaded evaluation outputs to https://huggingface.co/{UPLOAD_REPO}/tree/main/{UPLOAD_PATH}")


def main() -> None:
    seeds = make_eval_seeds(EPISODES, START_SEED)
    baseline_rows: List[dict] = []
    for name, policy in [
        ("random", random_agent),
        ("keyword_spam", keyword_spam_agent),
        ("present_all", present_all_agent),
        ("scripted_oracle", oracle_scripted_agent),
    ]:
        rows, _ = evaluate_agent(name, policy, seeds, transcript_limit=0)
        baseline_rows.extend(rows)

    trained_rows, transcripts = evaluate_model(MODEL_REPO, EPISODES, EVAL_LABEL)
    if BASE_EPISODES > 0:
        base_rows, base_transcripts = evaluate_model(BASE_MODEL, BASE_EPISODES, "base_qwen3_0_6b")
        trained_rows.extend(base_rows)
        transcripts.extend(base_transcripts)

    write_outputs(trained_rows, baseline_rows, transcripts)
    upload_outputs()


if __name__ == "__main__":
    main()
