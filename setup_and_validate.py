#!/usr/bin/env python3
"""Run local validation checks and print credit-safe next steps."""

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

    section("HF CREDIT SAFETY")
    print(
        "No Hugging Face jobs or pushes were run. Before remote training, budget roughly "
        "$0.50 for an A10G dry run and $6-$10 for a 90 minute A100 GRPO run."
    )

    section("SUMMARY")
    if failures:
        print(f"[FAIL] {failures} local validation check(s) failed.")
        return 1
    print("[OK] Local environment, tests, and rollout diagnostics are green.")
    print("Next: run approved remote GRPO training, then commit real reward curves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
