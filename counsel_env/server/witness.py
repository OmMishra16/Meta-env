from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Contradiction:
    cid: str
    trigger_keywords: List[str]
    sealed_claim: str
    disprover_evidence_id: str
    triggered: bool = False
    surfaced: bool = False

class Witness:
    def __init__(self, story: Dict[str, str], contradictions: List[Contradiction]):
        self.story = story
        self.contradictions = contradictions
        self.committed_claims: List[str] = []

    def respond_to_question(self, q: str) -> str:
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
        for c in self.contradictions:
            if c.disprover_evidence_id == exhibit_id and c.triggered and not c.surfaced:
                c.surfaced = True
                return "[Witness stammers] I... I'm not sure what to say."
        return "[Witness] I have no comment on that exhibit."