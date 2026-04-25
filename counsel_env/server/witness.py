"""Deterministic rule-based witness responder. No LLM calls."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Contradiction:
    """A seeded contradiction the lawyer can surface by asking the right questions then presenting evidence.

    The flow: question with trigger_keyword → witness commits to sealed_claim (triggered=True)
              → present disprover_evidence_id → contradiction is surfaced (surfaced=True).
    """

    cid: str
    trigger_keywords: List[str]
    sealed_claim: str
    disprover_evidence_id: str
    triggered: bool = False
    surfaced: bool = False


class Witness:
    """A deterministic rule-based witness that responds based on keyword matching.

    No LLMs involved — responses are looked up from the story dict or
    triggered via contradiction keywords. This ensures verifiable, reproducible rewards.
    """

    def __init__(self, story: Dict[str, str], contradictions: List[Contradiction]) -> None:
        self.story = story
        self.contradictions = contradictions
        self.committed_claims: List[str] = []

    def respond_to_question(self, q: str) -> str:
        """Match question against contradiction triggers, then story topics, then default.

        Args:
            q: The question text from the lawyer.

        Returns:
            The witness's verbal response string.
        """
        q_lower = (q or "").lower()
        for c in self.contradictions:
            if any(kw in q_lower for kw in c.trigger_keywords):
                if not c.triggered:
                    c.triggered = True
                    self.committed_claims.append(c.sealed_claim)
                return c.sealed_claim
        for topic, resp in self.story.items():
            if topic in q_lower:
                return resp
        return "I don't recall."

    def react_to_evidence(self, exhibit_id: str) -> str:
        """React to a presented exhibit — surfaces a contradiction if already triggered.

        A contradiction is only surfaced if the witness has already committed to the
        contradicting claim (triggered=True). Presenting evidence without triggering
        first yields no reaction.

        Args:
            exhibit_id: The exhibit ID being presented.

        Returns:
            The witness's reaction string.
        """
        for c in self.contradictions:
            if c.disprover_evidence_id == exhibit_id and c.triggered and not c.surfaced:
                c.surfaced = True
                return "[Witness stammers] I... I'm not sure what to say."
        return "[Witness] I have no comment on that exhibit."
