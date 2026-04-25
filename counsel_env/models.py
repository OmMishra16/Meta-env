from typing import Dict, List, Optional

from pydantic import Field

from openenv.core import Action, Observation, State


class CounselAction(Action):
    tool: str = Field(description="Action tool: ask_question, present_evidence, make_objection, or rest_case")
    text: Optional[str] = Field(default=None, description="Question text for ask_question")
    exhibit_id: Optional[str] = Field(default=None, description="Evidence id for present_evidence")
    reason: Optional[str] = Field(default=None, description="Objection reason for make_objection")


class CounselObservation(Observation):
    witness_response: str = ""
    available_evidence: List[str] = Field(default_factory=list)
    evidence_descriptions: Dict[str, str] = Field(default_factory=dict)
    questions_remaining: int = 0
    transcript_tail: str = ""
    case_brief: str = ""
    case_id: str = ""
    difficulty: str = ""
    reward_components: Dict[str, float] = Field(default_factory=dict)


class CounselState(State):
    case_id: str = ""
    difficulty: str = ""
    contradictions_total: int = 0
    contradictions_triggered: int = 0
    contradictions_surfaced: int = 0
    questions_used: int = 0
    action_count: int = 0
    duplicate_question_count: int = 0
    irrelevant_question_count: int = 0
    inadmissible_count: int = 0
    evidence_timing_successes: int = 0
