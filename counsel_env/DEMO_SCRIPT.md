# 90-Second Demo Script

## 0-15s: Hook

Modern LLMs are often polite interviewers. Cross-examination needs a harder skill: make a witness commit, remember that commitment, and use evidence at the right moment.

## 15-35s: Environment

Counsel-Env generates courtroom-style cases with hidden contradictions. The model sees the case brief and evidence list, but not the ground truth. The witness is deterministic, so reward is cheap, fast, and verifiable.

## 35-55s: Failure Modes

Show the benchmark table. Random questioning scores zero. Presenting all evidence scores zero. Keyword spam triggers claims sometimes, but primary reward stays zero because nothing is proved.

## 55-75s: Live Demo

Open `/demo`. Reset an easy case, ask the hinted trigger question, then present the hinted exhibit. The witness stammers, the contradiction is surfaced, and reward appears when counsel rests.

## 75-90s: Why It Matters

This is a compact theory-of-mind task: the agent must reason over another actor's commitments under a budget. The published Qwen3-8B QLoRA SFT checkpoint is close to the scripted oracle ceiling and beats the reward-hacking baselines by learning the trigger-then-evidence loop on held-out cases.

## Shot List

1. Space landing page.
2. `/demo` reset button and case brief.
3. A failed baseline row from the benchmark table.
4. Trigger question.
5. Evidence presentation.
6. Reward/state panel showing surfaced contradiction.
