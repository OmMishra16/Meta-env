"""Smoke tests for the Cross-Examination Arena environment."""

import sys
import os

# Ensure the repo root is on sys.path so package imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from counsel_env.models import CounselAction
from counsel_env.server.case_generator import generate_alibi_case
from counsel_env.server.counsel_environment import CounselEnvironment


# ---------------------------------------------------------------------------
# Test 1 — reset() produces a valid initial observation
# ---------------------------------------------------------------------------
def test_reset_produces_valid_observation():
    env = CounselEnvironment()
    obs = env.reset()

    assert obs.case_brief != "", "case_brief should be non-empty"
    assert obs.questions_remaining == 15, "should start with full question budget"
    assert isinstance(obs.available_evidence, list), "available_evidence must be a list"
    assert len(obs.available_evidence) > 0, "should have at least one exhibit"
    assert not obs.done, "episode should not start as done"
    assert obs.reward == 0.0 or obs.reward is None, "reward should be 0 at episode start"


# ---------------------------------------------------------------------------
# Test 2 — alibi trigger fires on the correct keyword
# ---------------------------------------------------------------------------
def test_alibi_trigger_fires():
    env = CounselEnvironment(case_fn=generate_alibi_case)
    env.reset()

    obs = env.step(CounselAction(tool="ask_question", text="where were you that night"))

    # Witness should have committed a specific sealed claim, not the default fallback
    assert obs.witness_response != "I don't recall.", (
        "Expected sealed alibi claim, got fallback response"
    )
    assert obs.witness_response != "", "Should have a non-empty response"

    # Internally, c1_alibi_location must be triggered
    triggered = [c for c in env._witness.contradictions if c.triggered]
    assert len(triggered) >= 1, "At least one contradiction should be triggered"
    triggered_cids = [c.cid for c in triggered]
    assert "c1_alibi_location" in triggered_cids, (
        f"c1_alibi_location expected in triggered set, got {triggered_cids}"
    )

    # Not yet surfaced — evidence hasn't been presented
    surfaced = [c for c in env._witness.contradictions if c.surfaced]
    assert len(surfaced) == 0, "No contradiction should be surfaced before presenting evidence"


# ---------------------------------------------------------------------------
# Test 3 — presenting the disprover after triggering surfaces the contradiction
# ---------------------------------------------------------------------------
def test_evidence_surfaces_contradiction():
    env = CounselEnvironment(case_fn=generate_alibi_case)
    env.reset()

    # Step 1: trigger c1
    env.step(CounselAction(tool="ask_question", text="where were you that night"))

    # Step 2: present the disprover for c1
    obs = env.step(CounselAction(tool="present_evidence", exhibit_id="phone_tower_log"))

    assert "[Witness stammers]" in obs.witness_response, (
        f"Expected contradiction reaction, got: {obs.witness_response!r}"
    )

    state = env.state
    assert state.contradictions_surfaced == 1, (
        f"Expected 1 contradiction surfaced, got {state.contradictions_surfaced}"
    )

    # Internally confirm c1 is marked surfaced
    c1 = next(c for c in env._witness.contradictions if c.cid == "c1_alibi_location")
    assert c1.surfaced, "c1_alibi_location should be marked surfaced"


# ---------------------------------------------------------------------------
# Test 4 — completed episode returns reward > 0.5 when all contradictions surfaced
# ---------------------------------------------------------------------------
def test_completed_episode_returns_reward():
    env = CounselEnvironment(case_fn=generate_alibi_case)
    env.reset()

    # Surface contradiction 1: alibi location
    env.step(CounselAction(tool="ask_question", text="where were you that night"))
    env.step(CounselAction(tool="present_evidence", exhibit_id="phone_tower_log"))

    # Surface contradiction 2: warehouse familiarity
    env.step(CounselAction(tool="ask_question", text="have you ever been to the warehouse"))
    env.step(CounselAction(tool="present_evidence", exhibit_id="ride_receipt"))

    # Surface contradiction 3: motive denial
    env.step(CounselAction(tool="ask_question", text="did you have any grievance with the victim"))
    env.step(CounselAction(tool="present_evidence", exhibit_id="victim_letter"))

    # Rest the case
    obs = env.step(CounselAction(tool="rest_case"))

    assert obs.done, "Episode should be done after rest_case"
    assert obs.reward is not None, "Reward should be set"
    assert obs.reward > 0.5, (
        f"Expected reward > 0.5 after surfacing all contradictions, got {obs.reward}"
    )

    state = env.state
    assert state.contradictions_surfaced == 3
    assert state.contradictions_total == 3


# ---------------------------------------------------------------------------
# Test 5 — inadmissible questions accumulate in state
# ---------------------------------------------------------------------------
def test_inadmissibility_penalty():
    env = CounselEnvironment(case_fn=generate_alibi_case)
    env.reset()

    # Three leading questions
    env.step(CounselAction(tool="ask_question", text="isn't it true that you were there"))
    env.step(CounselAction(tool="ask_question", text="isn't it true that you had motive"))
    env.step(CounselAction(tool="ask_question", text="isn't it true that you lied to police"))

    env.step(CounselAction(tool="rest_case"))

    state = env.state
    assert state.inadmissible_count == 3, (
        f"Expected inadmissible_count == 3, got {state.inadmissible_count}"
    )
    # Inadmissible questions should NOT consume the question budget
    assert state.questions_used == 0, (
        f"Inadmissible questions should not consume budget, got questions_used={state.questions_used}"
    )
