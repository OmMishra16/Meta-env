# Counsel-Env: Teaching AI to Cross-Examine

We imagined a simple but frustrating courtroom situation.

A person is fighting a case. They know something in the witness's story does not add up. There is evidence somewhere that can prove it. But the lawyer asks broad questions, misses the exact contradiction, or shows the evidence too early before the witness has fully committed to a false claim.

The chance is lost.

That idea became the starting point for Counsel-Env.

We wanted to build an environment where an AI does not just talk well, but learns to question carefully, listen for contradictions, and use evidence at the right moment.

## The Idea

Counsel-Env is a courtroom-style training environment.

The AI acts like a cross-examining lawyer. It gets a case brief, a witness, a list of evidence, and a limited number of questions.

Its goal is to:

1. Ask the right question.
2. Make the witness commit to a claim.
3. Present the evidence that proves the claim wrong.
4. Surface the contradiction.

If the AI asks vague questions, it does not score well.

If it throws evidence randomly, it fails.

If it only uses trigger words without proving anything, it also fails.

The model has to follow the actual reasoning sequence.

## Why It Matters

Most AI models are trained to be helpful and fluent. But real-world reasoning often needs more than fluent answers.

Sometimes the important skill is knowing what to ask next.

In law, investigation, compliance, journalism, and fact-checking, people often need to compare claims against evidence. The challenge is not just having information. The challenge is using it at the right time.

That is what Counsel-Env tries to teach.

## How We Built It

We built Counsel-Env using OpenEnv.

Each case is generated with a hidden contradiction. The witness has a story. The evidence contains something that can disprove part of that story.

The AI does not directly see the hidden answer. It has to uncover it through questioning.

The reward system is designed so the model cannot cheat. It only gets strong reward when it actually exposes a contradiction. Asking random questions, dumping evidence, or sounding confident is not enough.

## Training and Results

We trained a Qwen3-8B model using a fast QLoRA SFT setup.

The model learned the main pattern we wanted:

> first make the witness commit, then present the matching evidence.

We tested it against simple baselines like random actions, keyword spam, and blind evidence dumping.

On the expanded 150-seed evaluation, the trained model achieved:

- average reward: `0.864`
- contradiction surface rate: `0.943`
- invalid tool calls: `0`

That means it learned to surface contradictions reliably on new cases, not just memorized one example.

## What Makes It Interesting

The core lesson is timing.

Evidence is not always useful the moment you have it. In cross-examination, evidence becomes powerful after the witness has made a claim that the evidence can disprove.

That makes the task more strategic than normal question answering.

The AI has to track what the witness said, choose the right next move, and avoid wasting its question budget.

## Features

Counsel-Env includes:

- courtroom-style generated cases
- deterministic witness behavior
- hidden contradictions
- evidence exhibits
- limited question budget
- reward system that discourages shortcuts
- trained model checkpoint
- baseline comparisons
- evaluation charts
- live Hugging Face Space demo

## Real-World Use

This is a research environment, not a replacement for lawyers.

But the skill it trains can be useful in many places:

- legal education
- deposition practice
- investigative interviews
- compliance review
- fact-checking
- evidence-based reasoning tasks

The broader goal is to train AI systems that do not just answer confidently, but can test claims against evidence.

## Final Thought

We built Counsel-Env around one simple idea:

An AI should not only know facts. It should know how to challenge a claim, ask the right question, and bring the right evidence at the right time.

That is what makes the difference between simply talking and actually reasoning.
