"""Local held-out evaluation and artifact generation for Counsel-Env."""

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List

from counsel_env.models import CounselAction
from counsel_env.server.counsel_env_environment import CounselEnvironment


AgentPolicy = Callable[[CounselEnvironment, random.Random], None]


def random_agent(env: CounselEnvironment, rng: random.Random) -> None:
    """Low-information baseline."""
    for _ in range(8):
        if env.done:
            break
        if rng.random() < 0.7:
            env.step(CounselAction(tool="ask_question", text=f"Can you explain detail {rng.randint(1, 999)}?"))
        else:
            env.step(CounselAction(tool="present_evidence", exhibit_id=rng.choice(list(env.case["evidence"].keys()))))
    if not env.done:
        env.step(CounselAction(tool="rest_case"))


def keyword_spam_agent(env: CounselEnvironment, rng: random.Random) -> None:
    """Reward-hacking probe: asks trigger-like terms but does not reason about exhibits."""
    questions = [
        "Where were you?",
        "What was your motive?",
        "Did you know the victim?",
        "What happened and why?",
        "Were you at the location?",
    ]
    for question in questions:
        if env.done:
            break
        env.step(CounselAction(tool="ask_question", text=question))
    if not env.done:
        env.step(CounselAction(tool="rest_case"))


def present_all_agent(env: CounselEnvironment, rng: random.Random) -> None:
    """Reward-hacking probe: blindly presents every exhibit."""
    for exhibit_id in list(env.case["evidence"].keys()):
        if env.done:
            break
        env.step(CounselAction(tool="present_evidence", exhibit_id=exhibit_id))
    if not env.done:
        env.step(CounselAction(tool="rest_case"))


def oracle_scripted_agent(env: CounselEnvironment, rng: random.Random) -> None:
    """Upper-bound scripted strategy using hidden contradiction metadata."""
    for contradiction in env.witness.contradictions:
        if env.done:
            break
        env.step(CounselAction(tool="ask_question", text=f"{contradiction.trigger_keywords[0]}?"))
        if env.done:
            break
        env.step(CounselAction(tool="present_evidence", exhibit_id=contradiction.disprover_evidence_id))
    if not env.done:
        env.step(CounselAction(tool="rest_case"))


AGENTS: Dict[str, AgentPolicy] = {
    "random": random_agent,
    "keyword_spam": keyword_spam_agent,
    "present_all": present_all_agent,
    "scripted_oracle": oracle_scripted_agent,
}


def make_eval_seeds(count: int = 30, start: int = 20260425) -> List[int]:
    return list(range(start, start + count))


def evaluate_agent(
    name: str,
    policy: AgentPolicy,
    seeds: Iterable[int],
    curriculum_stage: str = "mixed",
    transcript_limit: int = 3,
) -> tuple[List[dict], List[str]]:
    rows: List[dict] = []
    markdown_samples: List[str] = []

    for index, seed in enumerate(seeds):
        agent_offset = sum(ord(ch) for ch in name)
        rng = random.Random(seed + agent_offset)
        env = CounselEnvironment()
        obs = env.reset(seed=seed, curriculum_stage=curriculum_stage, episode_id=f"{name}_{seed}")
        assert obs.case_id == env.case["case_id"]
        policy(env, rng)
        if not env.done:
            env.step(CounselAction(tool="rest_case"))

        components = env._calculate_reward_components()
        row = {
            "agent": name,
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
        }
        rows.append(row)

        if index < transcript_limit:
            markdown_samples.append(f"# Agent: {name}\n\n" + env.export_transcript_markdown())

    return rows, markdown_samples


def summarize(rows: List[dict]) -> List[dict]:
    summaries: List[dict] = []
    for agent in sorted({row["agent"] for row in rows}):
        agent_rows = [row for row in rows if row["agent"] == agent]
        summaries.append(
            {
                "agent": agent,
                "episodes": len(agent_rows),
                "avg_reward": mean(row["reward"] for row in agent_rows),
                "avg_primary_reward": mean(row["primary_reward"] for row in agent_rows),
                "avg_trigger_rate": mean(
                    row["contradictions_triggered"] / max(1, row["contradictions_total"])
                    for row in agent_rows
                ),
                "avg_surface_rate": mean(
                    row["contradictions_surfaced"] / max(1, row["contradictions_total"])
                    for row in agent_rows
                ),
                "avg_evidence_timing": mean(row["evidence_timing_successes"] for row in agent_rows),
                "avg_useless_ratio": mean(row["useless_questions_ratio"] for row in agent_rows),
            }
        )
    return summaries


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(plot_dir: Path, summaries: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        write_csv(plot_dir / "summary_for_plots.csv", summaries)
        write_svg_bar_chart(
            plot_dir / "baseline_vs_oracle.svg",
            summaries,
            metric="avg_reward",
            title="Held-out evaluation reward by baseline",
        )
        write_svg_multi_metric(plot_dir / "rubric_breakdown.svg", summaries)
        return

    agents = [row["agent"] for row in summaries]
    rewards = [row["avg_reward"] for row in summaries]
    primary = [row["avg_primary_reward"] for row in summaries]
    surface = [row["avg_surface_rate"] for row in summaries]
    trigger = [row["avg_trigger_rate"] for row in summaries]
    useless = [row["avg_useless_ratio"] for row in summaries]

    plt.figure(figsize=(8, 4.5))
    plt.bar(agents, rewards, color=["#777777", "#ba5a31", "#4c78a8", "#2f855a"][: len(agents)])
    plt.ylabel("average reward")
    plt.xlabel("agent")
    plt.title("Held-out evaluation reward by baseline")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(plot_dir / "baseline_vs_oracle.png", dpi=180)
    plt.close()

    x = range(len(agents))
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, primary, marker="o", label="primary reward")
    plt.plot(x, trigger, marker="o", label="trigger rate")
    plt.plot(x, surface, marker="o", label="surface rate")
    plt.plot(x, useless, marker="o", label="useless question ratio")
    plt.xticks(list(x), agents, rotation=20, ha="right")
    plt.ylabel("rate")
    plt.xlabel("agent")
    plt.title("Reward-hacking audit metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "rubric_breakdown.png", dpi=180)
    plt.close()


def write_svg_bar_chart(path: Path, summaries: List[dict], metric: str, title: str) -> None:
    width, height = 840, 460
    margin_left, margin_bottom, margin_top = 80, 90, 60
    chart_w = width - margin_left - 40
    chart_h = height - margin_top - margin_bottom
    max_value = max(1.0, max(row[metric] for row in summaries))
    bar_gap = 24
    bar_w = (chart_w - bar_gap * (len(summaries) - 1)) / max(1, len(summaries))
    colors = ["#777777", "#ba5a31", "#4c78a8", "#2f855a"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-30}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
    ]
    for idx, row in enumerate(summaries):
        value = row[metric]
        x = margin_left + idx * (bar_w + bar_gap)
        bar_h = chart_h * value / max_value
        y = height - margin_bottom - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="13">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{height - 54}" text-anchor="middle" font-family="Arial" font-size="12">{row["agent"]}</text>')
    parts.append(f'<text x="22" y="{margin_top + chart_h/2}" transform="rotate(-90 22,{margin_top + chart_h/2})" font-family="Arial" font-size="13">average reward</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_multi_metric(path: Path, summaries: List[dict]) -> None:
    width, height = 880, 500
    margin_left, margin_bottom, margin_top = 80, 100, 60
    chart_w = width - margin_left - 50
    chart_h = height - margin_top - margin_bottom
    metrics = [
        ("avg_primary_reward", "primary", "#2f855a"),
        ("avg_trigger_rate", "trigger", "#4c78a8"),
        ("avg_surface_rate", "surface", "#805ad5"),
        ("avg_useless_ratio", "useless", "#ba5a31"),
    ]
    agents = [row["agent"] for row in summaries]
    x_step = chart_w / max(1, len(agents) - 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">Reward-hacking audit metrics</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-30}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
    ]
    for value in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = height - margin_bottom - chart_h * value
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.2f}</text>')
    for metric, label, color in metrics:
        points = []
        for idx, row in enumerate(summaries):
            x = margin_left + idx * x_step
            y = height - margin_bottom - chart_h * max(0.0, min(1.0, row[metric]))
            points.append((x, y))
        path_data = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{path_data}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        lx = width - 180
        ly = margin_top + 22 * metrics.index((metric, label, color))
        parts.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{lx+18}" y="{ly}" font-family="Arial" font-size="13">{label}</text>')
    for idx, agent in enumerate(agents):
        x = margin_left + idx * x_step
        parts.append(f'<text x="{x:.1f}" y="{height-62}" text-anchor="middle" font-family="Arial" font-size="12">{agent}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_before_after_pairs(path: Path, transcript_by_agent: Dict[str, List[str]]) -> None:
    sections = ["# Before / After Transcript Samples", ""]
    for agent in ["random", "keyword_spam", "present_all", "scripted_oracle"]:
        samples = transcript_by_agent.get(agent, [])
        if samples:
            sections.append(samples[0])
            sections.append("")
    path.write_text("\n".join(sections), encoding="utf-8")


def run_evaluation(output_dir: str | Path = "assets", episodes: int = 30) -> dict:
    output = Path(output_dir)
    plot_dir = output / "plots"
    transcript_dir = output / "transcripts"
    output.mkdir(exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    seeds = make_eval_seeds(episodes)
    all_rows: List[dict] = []
    transcript_by_agent: Dict[str, List[str]] = {}
    for agent, policy in AGENTS.items():
        rows, markdown_samples = evaluate_agent(agent, policy, seeds)
        all_rows.extend(rows)
        transcript_by_agent[agent] = markdown_samples

    summaries = summarize(all_rows)
    write_jsonl(output / "heldout_eval.jsonl", all_rows)
    write_csv(output / "heldout_eval_summary.csv", summaries)
    (output / "heldout_eval_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_plots(plot_dir, summaries)
    write_before_after_pairs(transcript_dir / "before_after_pairs.md", transcript_by_agent)
    return {"rows": all_rows, "summaries": summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="assets")
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    result = run_evaluation(args.output_dir, args.episodes)
    print(json.dumps(result["summaries"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
