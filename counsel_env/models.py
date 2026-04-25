"""Data models for the Cross-Examination Arena environment."""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CounselAction(Action):
    """Action for the Cross-Examination Arena environment.

    Args:
        tool: Tool to invoke — ask_question, present_evidence, make_objection, rest_case.
        text: Question or statement text (for ask_question and make_objection).
        exhibit_id: Evidence exhibit ID to present (for present_evidence).
        reason: Reason for objection (for make_objection).

    Returns:
        Submitted to CounselEnvironment.step() for processing.
    """

    tool: str = Field(..., description="Tool: ask_question | present_evidence | make_objection | rest_case")
    text: Optional[str] = Field(default=None, description="Question or statement text")
    exhibit_id: Optional[str] = Field(default=None, description="Exhibit ID to present")
    reason: Optional[str] = Field(default=None, description="Reason for objection")


class CounselObservation(Observation):
    """Observation from the Cross-Examination Arena environment.

    Args:
        witness_response: Witness's verbal response to the last action.
        available_evidence: List of exhibit IDs the lawyer may present.
        questions_remaining: Budget of questions still available.
        transcript_tail: Most recent exchanges in the transcript.
        case_brief: Case summary provided at episode start.
        done: Whether the episode has ended.
        reward: Reward signal (non-zero only at episode end).

    Returns:
        Returned by reset() and step() after each interaction.
    """

    witness_response: str = Field(default="", description="Witness's response to the last action")
    available_evidence: List[str] = Field(default_factory=list, description="Available exhibit IDs")
    questions_remaining: int = Field(default=0, description="Questions remaining in budget")
    transcript_tail: str = Field(default="", description="Recent transcript exchanges")
    case_brief: str = Field(default="", description="Case summary")


class CounselState(State):
    """State metadata for the Cross-Examination Arena environment.

    Args:
        case_id: Unique case identifier (encodes case template + slot values).
        contradictions_total: Total number of seeded contradictions in the case.
        contradictions_surfaced: Contradictions surfaced so far by presenting evidence.
        questions_used: Number of non-inadmissible questions asked.
        inadmissible_count: Number of inadmissible questions asked (leading/compound).

    Returns:
        Returned by env.state property for diagnostic introspection.
    """

    case_id: str = Field(default="", description="Case identifier")
    contradictions_total: int = Field(default=0, description="Total contradictions seeded")
    contradictions_surfaced: int = Field(default=0, description="Contradictions surfaced")
    questions_used: int = Field(default=0, description="Non-inadmissible questions asked")
    inadmissible_count: int = Field(default=0, description="Inadmissible questions asked")
