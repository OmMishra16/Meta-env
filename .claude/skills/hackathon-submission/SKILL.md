# Hackathon Submission — Requirements & Gotchas

**This skill loads when working on any submission deliverable** (README, video, blog, final checks).

## The judging criteria (memorize the weights)

| Criterion | Weight | What it means |
|---|---|---|
| Environment Innovation | **40%** | Is the env novel, creative, challenging? Does it test agent behavior in a way that hasn't been done? |
| Storytelling & Presentation | **30%** | Can you clearly explain problem, env, agent behavior? Engaging demo for non-technical audience? |
| Showing Improvement in Rewards | **20%** | Reward curves, before/after behavior, comparison to baseline — proof the agent learned. |
| Reward & Training Pipeline | **10%** | Coherent reward logic, pipeline that produces real improvement. |

**70% of points are Innovation + Storytelling.** A messy but ambitious env with a great pitch beats a polished but boring one. The judges' guide explicitly says this.

## Mandatory deliverables (non-negotiable; missing any = serious disadvantage)

1. **OpenEnv-compatible environment hosted on HF Spaces** — the submission URL itself
2. **Working training script** (Unsloth or HF TRL), ideally as a Colab notebook judges can re-run
3. **Evidence of actual training** — at minimum, loss + reward plots from a real run (committed as PNGs to the repo, not just in Colab/Wandb)
4. **A short writeup** — mini-blog on HuggingFace OR <2 min YouTube video OR slide deck. Linked from the README.
5. **README on the HF Space** that motivates the problem, explains how the env works, shows results
6. **All materials linked from the README** — Colab, video, blog, slides, GitHub mirror

## README structure (do not phone this in — it earns 30%)

Order matters. Judges scan top-to-bottom.

1. **Hook image** — a courtroom illustration, a screenshot of a winning rollout with contradictions highlighted, or an architecture diagram. NOT a logo.
2. **One-paragraph elevator pitch** — what this is, why it matters. Concrete, no buzzwords.
3. **Problem framing** — what capability gap does this address? Why is this domain underexplored? Cite the absence in the OpenEnv hub.
4. **Environment design**
   - Architecture diagram (Mermaid block: case_generator → witness → rubric tree → reward)
   - Action / observation tables
   - "How a contradiction works" with a worked example using the alibi case (concrete, with actual sample text)
5. **Reward rubric** — the rubric tree with weights, justification for keeping primary reward simple. Cite TRL guidance.
6. **Training results**
   - Reward curve PNG with baseline as dashed horizontal line
   - Per-rubric breakdown PNG
   - Concrete numbers: "Baseline 0.4 / 4 → trained 2.3 / 4 contradictions surfaced on average"
7. **Before / after transcripts** — at least 2 (ideally 3) side-by-side. THIS IS WHERE JUDGES FALL IN LOVE. Crisp; trim boring parts.
8. **Try it yourself** — copy-paste install + sample script. Test that this actually works on a fresh machine.
9. **Limitations and future work** — be honest. "Witness is rule-based; future: train via self-play. Templates limited; future: LLM-generated cases. Single domain; future: depositions, medical history, journalist interviews."
10. **Links** — Colab notebook, video, blog, slides, GitHub mirror. ALL of them.

### Tone

The brief says: *"engineering quality matters less than ambition."* Lean into ambition. Use voice.

Bad: "This environment trains LLMs on dialogue tasks."
Good: "We built a courtroom where an AI lawyer learns to catch liars."

Bad: "We hope this approach generalizes."
Good: "The next environments could train depositions, customer-service de-escalation, journalist interviews — we open-sourced the first one."

## Demo video script (if going video route)

Length: ≤2 min. Four 30-second beats.

1. **Hook (0–30s)** — "Modern LLMs are bad at finding lies. Watch this." Show baseline rollout where witness escapes clean. End: "Now watch this."
2. **Trained rollout (30–60s)** — model surfaces contradiction with on-screen captions: "*Trigger fired*", "*Evidence presented*", "*Contradiction surfaced (+1.0)*"
3. **Reward plot (60–90s)** — animated reward curve, annotations at key inflection points
4. **Why it matters (90–120s)** — "Theory of mind under a budget. Next: depositions, customer service, journalism. We open-sourced the first one."

Recording tips:
- Screen recording with OBS or Loom
- Voiceover separately, layer in editing
- Actual transcripts on screen (not summarized)
- Music: low, royalty-free, don't dominate

## Mini-blog (if going written route)

~600-800 words on HuggingFace blog format. Title format: "Teaching an LLM to Cross-Examine: Building Counsel-Env on OpenEnv"

Structure:
- Hook paragraph (problem, why hard)
- "What we built" (env mechanic, with one worked example)
- "How it learns" (reward design, training)
- Results (the killer plot, before/after)
- "What this opens up" (where the architecture generalizes)
- "Try it yourself" (install + link to Space)

## Submission checklist (final pass before submit)

- [ ] HF Space `<username>/counsel-env` shows **Running** (not paused/sleeping/error)
- [ ] `SUPPORTS_CONCURRENT_SESSIONS=True` in CounselEnvironment
- [ ] `pip install git+https://huggingface.co/spaces/<username>/counsel-env` works on a fresh machine
- [ ] `openenv.yaml` is valid and tagged
- [ ] No reserved tool names (`reset`, `step`, `state`, `close`)
- [ ] README.md lives on the HF Space (not just GitHub)
- [ ] README links to: Colab notebook URL, video URL, blog URL, GitHub mirror
- [ ] Reward curve PNG committed to /assets/plots/
- [ ] Per-rubric metrics PNG committed
- [ ] At least 2 before/after transcript pairs in README
- [ ] Colab notebook re-runs end-to-end on a fresh A100 (verify by re-running)
- [ ] Demo video published (YouTube unlisted is fine) and linked
- [ ] Blog post published and linked
- [ ] Smoke test passes on a fresh machine
- [ ] Submit URL is the **HF Space URL**

## Critical rules from the hackathon brief

- **Submission URL = HF Space URL.** Not GitHub. Not Colab. The Space.
- **Changes after submission deadline = ignored.** Don't push commits after submitting.
- **No big video files in the env submission repo** — link to YouTube/blog, don't bloat the Space.
- **README must have everything linked.** If judges have to hunt, you lose.
- **One submission per team.** Pick the best, go all in.

## Mentor talking points (when Meta engineers walk by)

- "Theme 1, theory-of-mind angle. Cross-examination is underexplored in the hub — verified by checking the catalog."
- "Reward rubric uses `openenv.core.rubrics`: WeightedSum over 5 components. Primary signal binary per TRL guidance."
- "Witness is intentionally deterministic — verifiable rewards, no GPU contention with trainer, swappable for LLM-witness in v2."
- "5 case templates, ~30k unique slot-fill combinations. Procgen scales to unlimited training data."

Avoid: hedging language ("we hope," "we're not sure"), comparing to Wordle ("we built a Wordle variant"), or anything self-deprecating. You did this work. Own it.

## Anti-patterns that lose points

- Reproducing an existing env (Wordle/Sudoku/2048/Tic-Tac-Toe). Even with novel reward, judges have seen 50+ of these.
- README that's an API doc, not a story. (Bad: "The action space contains the following types..." Good: "Cross-examining a witness is a budgeted negotiation between what you know and what they admit.")
- Plots only in Wandb / Colab cells (judges may not have access; commit PNGs)
- LLM-as-judge for primary reward (noisy, gameable, fails the "hard to game" criterion)
- Generic descriptions ("an environment for training agents on dialogue") — be specific about what capability you're teaching
- No link to a runnable notebook. Judges want to re-run.
