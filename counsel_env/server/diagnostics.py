import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, List

from counsel_env.models import CounselAction
from counsel_env.server.counsel_env_environment import CounselEnvironment


def _question_pattern(question: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    tokens = ["<num>" if token.isdigit() else token for token in text.split()]
    return " ".join(tokens[:12])


def _scripted_policy(env: CounselEnvironment, rng: random.Random) -> List[str]:
    questions: List[str] = []
    for contradiction in env.witness.contradictions[:2]:
        question = f"{contradiction.trigger_keywords[0]}?"
        questions.append(question)
        env.step(CounselAction(tool="ask_question", text=question))
        env.step(CounselAction(tool="present_evidence", exhibit_id=contradiction.disprover_evidence_id))
        if env.done:
            break
    return questions


def _random_policy(env: CounselEnvironment, rng: random.Random) -> List[str]:
    questions: List[str] = []
    generic_questions = [
        "What happened that day?",
        "Do you remember the weather?",
        "Who else was nearby?",
        "Can you describe your routine?",
        "What did you eat for dinner?",
        "Did anything unusual happen?",
    ]
    for _ in range(8):
        if rng.random() < 0.75:
            question = rng.choice(generic_questions)
            questions.append(question)
            env.step(CounselAction(tool="ask_question", text=question))
        else:
            exhibit = rng.choice(list(env.case["evidence"].keys()))
            env.step(CounselAction(tool="present_evidence", exhibit_id=exhibit))
        if env.done:
            break
    return questions


def _mixed_policy(env: CounselEnvironment, rng: random.Random, episode: int) -> List[str]:
    if episode in {0, 4, 8}:
        return _scripted_policy(env, rng)
    return _random_policy(env, rng)


POLICIES: Dict[str, Callable[[CounselEnvironment, random.Random, int], List[str]]] = {
    "mixed": _mixed_policy,
    "scripted": lambda env, rng, episode: _scripted_policy(env, rng),
    "random": lambda env, rng, episode: _random_policy(env, rng),
}


def run_rollout_diagnostics(
    output_path: str | Path = "rollout_debug.jsonl",
    num_episodes: int = 10,
    policy: str = "mixed",
    seed: int = 13,
) -> List[dict]:
    rng = random.Random(seed)
    path = Path(output_path)
    rows: List[dict] = []
    stages = ["easy", "medium", "hard"]
    policy_fn = POLICIES[policy]

    with path.open("w", encoding="utf-8") as handle:
        for episode in range(num_episodes):
            env = CounselEnvironment()
            stage = stages[episode % len(stages)]
            obs = env.reset(curriculum_stage=stage)
            questions = policy_fn(env, rng, episode)
            obs = env.step(CounselAction(tool="rest_case")) if not env.done else obs
            components = obs.reward_components
            patterns = {_question_pattern(question) for question in questions}
            action_types = [
                line.split(":", 1)[0]
                for line in env.transcript
                if ":" in line
            ]
            row = {
                "episode": episode,
                "stage": stage,
                "difficulty": env.case.get("difficulty"),
                "case_id": env.case.get("case_id"),
                "contradictions_total": int(components["contradictions_total"]),
                "contradictions_triggered": int(components["contradictions_triggered"]),
                "contradictions_surfaced": int(components["contradictions_surfaced"]),
                "questions_asked": env.questions_used,
                "evidence_presented": env.evidence_presented_count,
                "useless_questions_ratio": components["useless_questions_ratio"],
                "avg_question_length": components["avg_question_length"],
                "unique_question_patterns": len(patterns),
                "unique_question_patterns_pct": (len(patterns) / max(1, len(questions))),
                "action_diversity": len(set(action_types)),
                "primary_reward": components["primary_reward"],
                "auxiliary_reward": components["auxiliary_reward_raw"],
                "total_reward": obs.reward,
                "success": obs.reward > 0.0,
            }
            rows.append(row)
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    avg_reward = sum(row["total_reward"] for row in rows) / max(1, len(rows))
    if avg_reward == 0:
        print('WARNING: ENV TOO HARD')
    if avg_reward > 0.7:
        print('WARNING: ENV TOO EASY')
    print(f"Wrote {len(rows)} rollout diagnostics to {path} (avg_reward={avg_reward:.3f})")
    return rows


if __name__ == "__main__":
    run_rollout_diagnostics()
