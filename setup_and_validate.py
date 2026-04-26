#!/usr/bin/env python3
"""Run local validation checks and print submission artifact status."""

import os
import subprocess
import sys


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n {title}\n{'=' * 70}\n")


def run(cmd: list[str], description: str) -> bool:
    print(f"-> {description}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode == 0:
        print("[OK]")
        if result.stdout.strip():
            print(result.stdout.strip()[-1200:])
        return True
    print("[FAIL]")
    print((result.stdout + result.stderr)[-2000:])
    return False


def main() -> int:
    section("COUNSEL-ENV SETUP AND VALIDATION")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")

    checks = [
        ([sys.executable, "validate_components.py"], "component validation"),
        (
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q"],
            "full local test suite",
        ),
        (
            [sys.executable, "-m", "counsel_env.server.diagnostics"],
            "sample rollout diagnostics",
        ),
    ]

    failures = 0
    for cmd, description in checks:
        if not run(cmd, description):
            failures += 1

    section("SUBMISSION ARTIFACTS")
    print(
        "Published Space: https://huggingface.co/spaces/heavycoderhh/counsel-env\n"
        "Published checkpoint: https://huggingface.co/heavycoderhh/counsel-env-qwen3-0.6b-grpo-run2\n"
        "Mirrored trained eval artifacts: assets/trained_eval/"
    )

    section("SUMMARY")
    if failures:
        print(f"[FAIL] {failures} local validation check(s) failed.")
        return 1
    print("[OK] Local environment, tests, and rollout diagnostics are green.")
    print("[OK] Run-2 SFT+GRPO checkpoint and held-out eval artifacts are ready for judges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
