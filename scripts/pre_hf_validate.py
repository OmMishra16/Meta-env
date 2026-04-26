#!/usr/bin/env python3
"""One-command local preflight before spending Hugging Face credits."""

import ast
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from counsel_env import CounselAction, CounselEnv


def run(cmd: list[str], description: str, cwd: Path = ROOT, timeout: int = 180) -> None:
    print(f"\n[RUN] {description}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.stdout.strip():
        print(result.stdout.strip()[-1600:])
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip()[-2000:])
        raise SystemExit(f"[FAIL] {description}")
    print(f"[OK] {description}")


def check_notebook() -> None:
    print("\n[RUN] notebook parse and credit guards")
    path = ROOT / "counsel_env" / "notebooks" / "train_counsel.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code":
            ast.parse("".join(cell.get("source", [])), filename=f"cell_{index}")
    assert "RUN_TRAINING', '0'" in source or 'RUN_TRAINING", "0"' in source
    assert "push_to_hub=False" in source
    print("[OK] notebook parses and remote training is disabled by default")


def check_manifest_and_docs() -> None:
    print("\n[RUN] manifest and docs")
    manifest = (ROOT / "counsel_env" / "openenv.yaml").read_text(encoding="utf-8")
    readme = (ROOT / "counsel_env" / "README.md").read_text(encoding="utf-8")
    assert "name: counsel-env" in manifest
    assert "multi-agent" in manifest
    assert "Cross-Examination Arena" in readme
    assert "We built a courtroom where an LLM learns to catch lies" in readme
    assert "Reward-Hacking Audit" in readme
    assert (ROOT / "assets" / "demo" / "demo_case.md").exists()
    assert (ROOT / "assets" / "demo" / "video_script.md").exists()
    assert (ROOT / "assets" / "demo" / "blog_draft.md").exists()
    assert "No Hugging Face" not in readme
    print("[OK] manifest and README are counsel-env specific")


def check_import_modes() -> None:
    run(
        [sys.executable, "-c", "import counsel_env.server.app as app; print(type(app.app).__name__)"],
        "package-style app import",
    )
    run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'counsel_env'); import server.app; print(type(server.app.app).__name__)",
        ],
        "Docker/Space-style app import",
    )


def wait_for_port(port: int, timeout_s: float = 15.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError(f"server did not listen on port {port}")


def check_server_smoke() -> None:
    print("\n[RUN] local OpenEnv server smoke")
    port = 8017
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "counsel_env.server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        wait_for_port(port)
        with CounselEnv(base_url=f"http://127.0.0.1:{port}").sync() as client:
            reset = client.reset(curriculum_stage="easy").observation
            assert reset.case_brief
            assert reset.questions_remaining == 15
            step = client.step(CounselAction(tool="ask_question", text="Where were you?")).observation
            assert step.questions_remaining == 14
            state = client.state()
            assert state.questions_used == 1
            assert state.contradictions_total >= 1
        print("[OK] local OpenEnv server reset/step/state")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def check_rollout_debug() -> None:
    path = ROOT / "assets" / "diagnostics" / "rollout_debug.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    non_zero = sum(row["total_reward"] > 0 for row in rows)
    avg_reward = sum(row["total_reward"] for row in rows) / len(rows)
    assert non_zero >= 1
    assert avg_reward < 0.7
    print(f"[OK] {path.relative_to(ROOT)} has {non_zero}/{len(rows)} non-zero episodes, avg_reward={avg_reward:.3f}")


def check_evaluation_artifacts() -> None:
    summary_path = ROOT / "assets" / "heldout_eval_summary.json"
    trained_summary_path = ROOT / "assets" / "trained_eval" / "trained_eval_summary.json"
    trained_rows_path = ROOT / "assets" / "trained_eval" / "trained_eval_rows.csv"
    trained_transcripts_path = ROOT / "assets" / "trained_eval" / "trained_eval_transcripts.md"
    run3_summary_path = ROOT / "assets" / "trained_eval_run3" / "trained_eval_summary.json"
    run3_rows_path = ROOT / "assets" / "trained_eval_run3" / "trained_eval_rows.csv"
    run3_transcripts_path = ROOT / "assets" / "trained_eval_run3" / "trained_eval_transcripts.md"
    transcripts_path = ROOT / "assets" / "transcripts" / "before_after_pairs.md"
    plot_paths = [
        ROOT / "assets" / "plots" / "baseline_vs_oracle.svg",
        ROOT / "assets" / "plots" / "rubric_breakdown.svg",
    ]
    assert summary_path.exists(), "missing heldout_eval_summary.json"
    assert trained_summary_path.exists(), "missing trained_eval_summary.json"
    assert trained_rows_path.exists(), "missing trained_eval_rows.csv"
    assert trained_transcripts_path.exists(), "missing trained_eval_transcripts.md"
    assert run3_summary_path.exists(), "missing run3 trained_eval_summary.json"
    assert run3_rows_path.exists(), "missing run3 trained_eval_rows.csv"
    assert run3_transcripts_path.exists(), "missing run3 trained_eval_transcripts.md"
    assert transcripts_path.exists(), "missing before_after_pairs.md"
    for path in plot_paths:
        assert path.exists(), f"missing plot: {path}"
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    by_agent = {row["agent"]: row for row in summaries}
    assert by_agent["scripted_oracle"]["avg_reward"] > by_agent["keyword_spam"]["avg_reward"]
    assert by_agent["scripted_oracle"]["avg_reward"] > by_agent["present_all"]["avg_reward"]
    assert by_agent["present_all"]["avg_reward"] == 0.0
    trained_summaries = json.loads(trained_summary_path.read_text(encoding="utf-8"))
    trained_by_agent = {row["agent"]: row for row in trained_summaries}
    assert trained_by_agent["trained_sft_grpo_run2"]["avg_reward"] > by_agent["keyword_spam"]["avg_reward"]
    assert trained_by_agent["trained_sft_grpo_run2"]["avg_primary_reward"] > 0.0
    run3_summaries = json.loads(run3_summary_path.read_text(encoding="utf-8"))
    run3_by_agent = {row["agent"]: row for row in run3_summaries}
    assert run3_by_agent["trained_sft_grpo_run3"]["avg_reward"] > trained_by_agent["trained_sft_grpo_run2"]["avg_reward"]
    assert run3_by_agent["trained_sft_grpo_run3"]["avg_primary_reward"] > trained_by_agent["trained_sft_grpo_run2"]["avg_primary_reward"]
    assert "Triggered:" in transcripts_path.read_text(encoding="utf-8")
    assert "Agent: trained_sft_grpo_run2" in trained_transcripts_path.read_text(encoding="utf-8")
    assert "Agent: trained_sft_grpo_run3" in run3_transcripts_path.read_text(encoding="utf-8")
    print("[OK] held-out evaluation artifacts and reward-hacking audit")


def main() -> int:
    check_manifest_and_docs()
    check_notebook()
    check_import_modes()
    run([sys.executable, "scripts/validate_components.py"], "component validation")
    run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"], "full pytest suite")
    run([sys.executable, "-m", "counsel_env.server.diagnostics"], "rollout diagnostics")
    check_rollout_debug()
    run([sys.executable, "-m", "counsel_env.evaluation", "--episodes", "30", "--output-dir", "assets"], "held-out baseline evaluation")
    check_evaluation_artifacts()
    check_server_smoke()
    print("\n[OK] PRE-HF PREFLIGHT PASSED")
    print("No Hugging Face credits used by this local validation run. Published Space, checkpoint, and eval artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
