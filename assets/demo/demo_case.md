# Demo Case Walkthrough

## The Hook

Counsel-Env is a courtroom training ground where an LLM learns that polite questions are not enough. It has to pin the witness down, make them commit, and then use evidence at the right moment.

## Same Case, Four Behaviors

Seeded case: `timeline_255d67`

Brief:

> Esme Dubois is accused of assault at the public library on Walnut Street on May 28th. The victim claims the assault occurred at 11:45 PM. The defendant claims they were elsewhere at that time.

The hidden contradiction:

| Step | Mechanic | Example |
| --- | --- | --- |
| Trigger | Ask about time / when it happened | `time?` |
| Sealed claim | Witness commits to false time | `The assault happened at 11:00 PM, not 11:45 PM.` |
| Disprover | Present timestamped footage | `surveillance_timestamp` |
| Surface | Witness cannot reconcile the claim | `[Witness stammers] ...` |

## Baseline Failure

The random baseline asks vague questions and sometimes presents evidence too early:

```text
Q: Can you explain detail 571?
A: I don't recall.

Evidence: surveillance_timestamp
A: [Witness] I have no comment on that exhibit.
```

Reward: `0.0`

Why it fails: evidence was presented before the witness committed to a false claim.

## Reward-Hacking Failure

The keyword-spam baseline asks trigger-like questions, but never proves the contradiction:

```text
Q: What happened and why?
A: The assault happened at 11:00 PM, not 11:45 PM.

Q: Were you at the location?
A: I don't recall.
```

Reward: `0.03`

Why it fails: triggering a sealed claim gives only tiny auxiliary reward. Primary reward stays zero until the disprover is presented.

## Successful Cross-Examination

The scripted oracle shows the target behavior we want a trained model to learn:

```text
Q: time?
A: The assault happened at 11:00 PM, not 11:45 PM.

Evidence: surveillance_timestamp
A: [Witness stammers] I... I'm not sure what to say.
```

Reward: `0.88`

Why it succeeds: trigger first, evidence second, contradiction surfaced.

## What To Show Judges

1. Show the random baseline transcript.
2. Show keyword-spam getting almost no reward.
3. Show the trigger-then-evidence transcript.
4. Show `assets/plots/rubric_breakdown.svg`.
5. After training, replace the oracle example with the trained model on the same seed.
