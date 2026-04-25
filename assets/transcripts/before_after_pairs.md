# Before / After Transcript Samples

# Agent: random

## timeline_255d67 (easy)

Esme Dubois is accused of assault at the public library on Walnut Street on May 28th. The victim claims the assault occurred at 11:45 PM. The defendant claims they were elsewhere at that time.

### Transcript
- Step 1 `ask_question`: Can you explain detail 571?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 2 `ask_question`: Can you explain detail 251?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 3 `ask_question`: Can you explain detail 483?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 4 `present_evidence`: witness_alibi
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 5 `ask_question`: Can you explain detail 686?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 6 `present_evidence`: surveillance_timestamp
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 7 `ask_question`: Can you explain detail 425?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 8 `ask_question`: Can you explain detail 869?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 9 `rest_case`: rest_case
  - Witness: [Counsel rests.]
  - Triggered: none; Surfaced: none

### Reward
```json
{"primary_reward": 0.0, "auxiliary_reward_raw": -0.4, "total_reward": 0.0, "contradictions_total": 1.0, "contradictions_triggered": 0.0, "contradictions_surfaced": 0.0, "keyword_questions": 0.0, "evidence_timing_successes": 0.0, "blind_evidence_count": 2.0, "duplicate_questions": 0.0, "irrelevant_questions": 6.0, "inadmissible_actions": 0.0, "useless_questions_ratio": 1.0, "avg_question_length": 5.0}
```

# Agent: keyword_spam

## timeline_255d67 (easy)

Esme Dubois is accused of assault at the public library on Walnut Street on May 28th. The victim claims the assault occurred at 11:45 PM. The defendant claims they were elsewhere at that time.

### Transcript
- Step 1 `ask_question`: Where were you?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 2 `ask_question`: What was your motive?
  - Witness: I had no reason to hurt anyone.
  - Triggered: none; Surfaced: none
- Step 3 `ask_question`: Did you know the victim?
  - Witness: I don't recall.
  - Triggered: none; Surfaced: none
- Step 4 `ask_question`: What happened and why?
  - Witness: The assault happened at 11:00 PM, not 11:45 PM.
  - Triggered: c1_time_of_assault; Surfaced: none
- Step 5 `ask_question`: Were you at the location?
  - Witness: I don't recall.
  - Triggered: c1_time_of_assault; Surfaced: none
- Step 6 `rest_case`: rest_case
  - Witness: [Counsel rests.]
  - Triggered: c1_time_of_assault; Surfaced: none

### Reward
```json
{"primary_reward": 0.0, "auxiliary_reward_raw": 0.15000000000000002, "total_reward": 0.030000000000000006, "contradictions_total": 1.0, "contradictions_triggered": 1.0, "contradictions_surfaced": 0.0, "keyword_questions": 1.0, "evidence_timing_successes": 0.0, "blind_evidence_count": 0.0, "duplicate_questions": 0.0, "irrelevant_questions": 3.0, "inadmissible_actions": 0.0, "useless_questions_ratio": 0.6, "avg_question_length": 4.2}
```

# Agent: present_all

## timeline_255d67 (easy)

Esme Dubois is accused of assault at the public library on Walnut Street on May 28th. The victim claims the assault occurred at 11:45 PM. The defendant claims they were elsewhere at that time.

### Transcript
- Step 1 `present_evidence`: victim_statement
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 2 `present_evidence`: surveillance_timestamp
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 3 `present_evidence`: phone_records
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 4 `present_evidence`: witness_alibi
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 5 `present_evidence`: motive_email
  - Witness: [Witness] I have no comment on that exhibit.
  - Triggered: none; Surfaced: none
- Step 6 `rest_case`: rest_case
  - Witness: [Counsel rests.]
  - Triggered: none; Surfaced: none

### Reward
```json
{"primary_reward": 0.0, "auxiliary_reward_raw": -0.25, "total_reward": 0.0, "contradictions_total": 1.0, "contradictions_triggered": 0.0, "contradictions_surfaced": 0.0, "keyword_questions": 0.0, "evidence_timing_successes": 0.0, "blind_evidence_count": 5.0, "duplicate_questions": 0.0, "irrelevant_questions": 0.0, "inadmissible_actions": 0.0, "useless_questions_ratio": 0.0, "avg_question_length": 0.0}
```

# Agent: scripted_oracle

## timeline_255d67 (easy)

Esme Dubois is accused of assault at the public library on Walnut Street on May 28th. The victim claims the assault occurred at 11:45 PM. The defendant claims they were elsewhere at that time.

### Transcript
- Step 1 `ask_question`: time?
  - Witness: The assault happened at 11:00 PM, not 11:45 PM.
  - Triggered: c1_time_of_assault; Surfaced: none
- Step 2 `present_evidence`: surveillance_timestamp
  - Witness: [Witness stammers] I... I'm not sure what to say.
  - Triggered: c1_time_of_assault; Surfaced: c1_time_of_assault
- Step 3 `rest_case`: rest_case
  - Witness: [Counsel rests.]
  - Triggered: c1_time_of_assault; Surfaced: c1_time_of_assault

### Reward
```json
{"primary_reward": 1.0, "auxiliary_reward_raw": 0.4, "total_reward": 0.8800000000000001, "contradictions_total": 1.0, "contradictions_triggered": 1.0, "contradictions_surfaced": 1.0, "keyword_questions": 1.0, "evidence_timing_successes": 1.0, "blind_evidence_count": 0.0, "duplicate_questions": 0.0, "irrelevant_questions": 0.0, "inadmissible_actions": 0.0, "useless_questions_ratio": 0.0, "avg_question_length": 1.0}
```
