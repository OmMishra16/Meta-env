# Two-Minute Demo Script

## 0-20s: Hook

Modern LLMs are polite interviewers. Cross-examination needs something sharper: make a witness commit, then use evidence to expose the contradiction.

We built Counsel-Env, a courtroom environment where an LLM learns to catch lies under a question budget.

## 20-50s: The Environment

Each case has a hidden contradiction. The agent sees the case brief and evidence list, but not the ground truth.

The witness is deterministic. If the agent asks the right question, the witness commits to a sealed claim. If the agent then presents the matching exhibit, the contradiction is surfaced and the agent gets primary reward.

## 50-80s: Failure Modes

Here is the random baseline. It asks vague questions and presents evidence too early. Reward: zero.

Here is keyword spam. It triggers the witness, but does not prove anything. Primary reward is still zero.

This is the reward-hacking audit: the environment does not reward empty courtroom theater.

## 80-105s: Success

Now watch the target behavior.

The agent asks about time. The witness commits: the assault happened at 11:00 PM.

The agent presents timestamped surveillance footage. The claim collapses. Contradiction surfaced.

## 105-120s: Why It Matters

This is theory-of-mind under a budget: tracking what another agent has committed to, choosing the right evidence, and recovering from partial information.

After fast Qwen3-8B QLoRA SFT, the run4b checkpoint surfaces contradictions on held-out cases at a 0.928 primary/surface rate, close to the scripted oracle ceiling. It learns the core sequence: trigger first, evidence second.
