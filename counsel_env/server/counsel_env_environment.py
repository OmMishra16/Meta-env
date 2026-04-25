import random
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CounselAction, CounselObservation, CounselState
except ImportError:  # pragma: no cover - supports direct module execution
    from models import CounselAction, CounselObservation, CounselState

try:
    from .case_generator import generate_case
    from .rubrics import Total
    from .witness import Contradiction, Witness
except ImportError:  # pragma: no cover - supports direct module execution
    from case_generator import generate_case
    from rubrics import Total
    from witness import Contradiction, Witness


QUESTION_BUDGET = 15
MAX_ACTIONS = QUESTION_BUDGET * 3
PRIMARY_WEIGHT = 0.8
AUXILIARY_WEIGHT = 0.2


class CounselEnvironment(Environment):
    """Cross-examination arena with deterministic witness mechanics."""

    SUPPORTS_CONCURRENT_SESSIONS = True
    QUESTION_BUDGET = QUESTION_BUDGET
    MAX_ACTIONS = MAX_ACTIONS

    def __init__(self):
        super().__init__(rubric=Total())
        self._initialize_empty_state()

    def _initialize_empty_state(self) -> None:
        self.case: Dict[str, Any] = {}
        self.witness: Optional[Witness] = None
        self.episode_id: Optional[str] = None
        self.seed: Optional[int] = None
        self.questions_used = 0
        self.action_count = 0
        self.transcript: List[str] = []
        self.transcript_events: List[Dict[str, Any]] = []
        self.asked_question_keys: set[str] = set()
        self.question_lengths: List[int] = []
        self.duplicate_question_count = 0
        self.irrelevant_question_count = 0
        self.inadmissible_count = 0
        self.invalid_exhibit_count = 0
        self.keyword_question_count = 0
        self.contradictions_triggered_count = 0
        self.trigger_action_by_cid: Dict[str, int] = {}
        self.evidence_presented_count = 0
        self.evidence_timing_successes = 0
        self.blind_evidence_count = 0
        self.done = False
        self.latest_response = ""
        self.reward_components: Dict[str, float] = {}

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        curriculum_stage: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> CounselObservation:
        """Reset the environment with a curriculum-aware generated case."""
        if seed is not None:
            random.seed(seed)

        self._initialize_empty_state()
        self.seed = seed
        self.episode_id = episode_id
        stage = curriculum_stage or kwargs.get("stage") or "medium"
        self.case = generate_case(difficulty=difficulty, curriculum_stage=stage)
        contradictions = [
            Contradiction(
                cid=c.cid if hasattr(c, "cid") else c["cid"],
                trigger_keywords=(
                    c.trigger_keywords if hasattr(c, "trigger_keywords") else c["trigger_keywords"]
                ),
                sealed_claim=c.sealed_claim if hasattr(c, "sealed_claim") else c["sealed_claim"],
                disprover_evidence_id=(
                    c.disprover_evidence_id
                    if hasattr(c, "disprover_evidence_id")
                    else c["disprover_evidence_id"]
                ),
            )
            for c in self.case["contradictions"]
        ]
        self.witness = Witness(self.case["witness_story"], contradictions)
        return self._obs("")

    def step(
        self,
        action: CounselAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CounselObservation:
        """Execute an action and return the next observation."""
        if self.done:
            return self._obs("[Episode already over.]")

        self.action_count += 1
        tool = (action.tool or "").strip()

        if self.action_count > self.MAX_ACTIONS:
            self.done = True
            return self._obs("[Safety stop: maximum action count reached.]")

        if tool == "ask_question":
            response = self._handle_question(action.text or "")
        elif tool == "present_evidence":
            response = self._handle_evidence(action.exhibit_id or "")
        elif tool == "make_objection":
            response = self._handle_objection(action.reason or "")
        elif tool == "rest_case":
            response = "[Counsel rests.]"
            self.done = True
        else:
            self.inadmissible_count += 1
            response = f"Invalid action: {tool or '<empty>'}."

        self.latest_response = response
        self.transcript.append(self._format_transcript_line(action, response))

        if self.questions_used >= self.QUESTION_BUDGET:
            self.done = True

        return self._obs(response)

    @property
    def state(self) -> CounselState:
        """Return compact state for OpenEnv clients and diagnostics."""
        contradictions = self.witness.contradictions if self.witness is not None else []
        return CounselState(
            case_id=self.case.get("case_id", ""),
            difficulty=self.case.get("difficulty", ""),
            episode_id=self.episode_id,
            step_count=self.action_count,
            contradictions_total=len(contradictions),
            contradictions_triggered=sum(1 for c in contradictions if c.triggered),
            contradictions_surfaced=sum(1 for c in contradictions if c.surfaced),
            questions_used=self.questions_used,
            action_count=self.action_count,
            duplicate_question_count=self.duplicate_question_count,
            irrelevant_question_count=self.irrelevant_question_count,
            inadmissible_count=self.inadmissible_count,
            evidence_timing_successes=self.evidence_timing_successes,
        )

    def _handle_question(self, question: str) -> str:
        if self.questions_used >= self.QUESTION_BUDGET:
            self.done = True
            return "Question budget exhausted."

        clean_question = question.strip()
        if not clean_question:
            self.inadmissible_count += 1
            return "Please ask a proper question."

        if len(clean_question) > 1000:
            clean_question = clean_question[:1000]

        self.questions_used += 1
        self.question_lengths.append(len(clean_question.split()))

        if _is_inadmissible(clean_question):
            self.inadmissible_count += 1
            return "[Objection sustained: inadmissible question.]"

        question_key = _semantic_key(clean_question)
        if question_key in self.asked_question_keys:
            self.duplicate_question_count += 1
            return "You've already asked that question."
        self.asked_question_keys.add(question_key)

        before_triggered = self._triggered_count()
        contains_trigger_keyword = self._contains_trigger_keyword(clean_question)
        if contains_trigger_keyword:
            self.keyword_question_count += 1

        assert self.witness is not None
        triggered_before = {c.cid for c in self.witness.contradictions if c.triggered}
        response = self.witness.respond_to_question(clean_question)
        triggered_after = {c.cid for c in self.witness.contradictions if c.triggered}
        for cid in sorted(triggered_after - triggered_before):
            self.trigger_action_by_cid[cid] = self.action_count
        after_triggered = self._triggered_count()
        self.contradictions_triggered_count += max(0, after_triggered - before_triggered)

        if not contains_trigger_keyword and response == "I don't recall.":
            self.irrelevant_question_count += 1

        return response

    def _handle_evidence(self, exhibit_id: str) -> str:
        if not exhibit_id or exhibit_id not in self.case.get("evidence", {}):
            self.inadmissible_count += 1
            self.invalid_exhibit_count += 1
            return "Invalid exhibit ID."

        self.evidence_presented_count += 1
        assert self.witness is not None
        surfaced_before = {c.cid for c in self.witness.contradictions if c.surfaced}
        response = self.witness.react_to_evidence(exhibit_id)
        surfaced_after = {c.cid for c in self.witness.contradictions if c.surfaced}
        newly_surfaced = surfaced_after - surfaced_before
        if newly_surfaced:
            for cid in newly_surfaced:
                trigger_step = self.trigger_action_by_cid.get(cid)
                if trigger_step is not None and self.action_count - trigger_step <= 2:
                    self.evidence_timing_successes += 1
        elif not any(c.disprover_evidence_id == exhibit_id and c.triggered for c in self.witness.contradictions):
            self.blind_evidence_count += 1
        return response

    def _handle_objection(self, reason: str) -> str:
        self.inadmissible_count += 1
        if not reason.strip():
            return "Objection requires a reason."
        return "Objection overruled: no objection window is currently open."

    def _obs(self, latest_response: str) -> CounselObservation:
        components = self._calculate_reward_components()
        reward = components["total_reward"] if self.done else 0.0
        return CounselObservation(
            witness_response=latest_response,
            available_evidence=list(self.case.get("evidence", {}).keys()),
            evidence_descriptions=deepcopy(self.case.get("evidence", {})),
            questions_remaining=max(0, self.QUESTION_BUDGET - self.questions_used),
            transcript_tail=self._get_transcript_tail(),
            case_brief=self.case.get("case_brief", ""),
            case_id=self.case.get("case_id", ""),
            difficulty=self.case.get("difficulty", ""),
            done=self.done,
            reward=reward,
            reward_components=components,
        )

    def _calculate_reward(self) -> float:
        components = self._calculate_reward_components()
        reward = (
            PRIMARY_WEIGHT * components["primary_reward"]
            + AUXILIARY_WEIGHT * components["auxiliary_reward_raw"]
        )
        return max(0.0, min(1.0, reward))

    def _calculate_reward_components(self) -> Dict[str, float]:
        contradictions = self.witness.contradictions if self.witness is not None else []
        total = max(1, len(contradictions))
        surfaced = sum(1 for c in contradictions if c.surfaced)
        triggered = sum(1 for c in contradictions if c.triggered)
        question_total = max(1, self.questions_used)

        primary_reward = surfaced / total
        auxiliary = 0.0
        auxiliary += 0.2 * triggered
        auxiliary += 0.1 * self.keyword_question_count
        auxiliary += 0.1 * self.evidence_timing_successes
        auxiliary -= 0.05 * (self.duplicate_question_count + self.irrelevant_question_count)
        auxiliary -= 0.05 * self.blind_evidence_count
        auxiliary -= 0.1 * self.inadmissible_count

        self.reward_components = {
            "primary_reward": primary_reward,
            "auxiliary_reward_raw": auxiliary,
            "total_reward": max(0.0, min(1.0, PRIMARY_WEIGHT * primary_reward + AUXILIARY_WEIGHT * auxiliary)),
            "contradictions_total": float(len(contradictions)),
            "contradictions_triggered": float(triggered),
            "contradictions_surfaced": float(surfaced),
            "keyword_questions": float(self.keyword_question_count),
            "evidence_timing_successes": float(self.evidence_timing_successes),
            "blind_evidence_count": float(self.blind_evidence_count),
            "duplicate_questions": float(self.duplicate_question_count),
            "irrelevant_questions": float(self.irrelevant_question_count),
            "inadmissible_actions": float(self.inadmissible_count),
            "useless_questions_ratio": (
                (self.duplicate_question_count + self.irrelevant_question_count) / question_total
            ),
            "avg_question_length": (
                sum(self.question_lengths) / len(self.question_lengths)
                if self.question_lengths
                else 0.0
            ),
        }
        return self.reward_components

    def _triggered_count(self) -> int:
        if self.witness is None:
            return 0
        return sum(1 for c in self.witness.contradictions if c.triggered)

    def _contains_trigger_keyword(self, question: str) -> bool:
        if self.witness is None:
            return False
        q_lower = question.lower()
        for contradiction in self.witness.contradictions:
            if any(keyword.lower() in q_lower for keyword in contradiction.trigger_keywords):
                return True
        return False

    def _format_transcript_line(self, action: CounselAction, response: str) -> str:
        if action.tool == "ask_question":
            action_text = action.text or ""
            prefix = "Q"
        elif action.tool == "present_evidence":
            action_text = action.exhibit_id or ""
            prefix = "Evidence"
        elif action.tool == "make_objection":
            action_text = action.reason or ""
            prefix = "Objection"
        else:
            action_text = action.tool or ""
            prefix = "Action"
        event = {
            "step": self.action_count,
            "tool": action.tool or "",
            "input": action_text,
            "response": response,
            "questions_remaining": max(0, self.QUESTION_BUDGET - self.questions_used),
            "triggered": [c.cid for c in self.witness.contradictions if c.triggered] if self.witness else [],
            "surfaced": [c.cid for c in self.witness.contradictions if c.surfaced] if self.witness else [],
        }
        self.transcript_events.append(event)
        return f"{prefix}: {action_text}\nA: {response}"

    def _get_transcript_tail(self) -> str:
        return "\n".join(self.transcript[-5:])

    def export_transcript_json(self) -> Dict[str, Any]:
        """Return a replayable transcript payload for evaluation artifacts."""
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "case_id": self.case.get("case_id", ""),
            "difficulty": self.case.get("difficulty", ""),
            "case_brief": self.case.get("case_brief", ""),
            "evidence": deepcopy(self.case.get("evidence", {})),
            "reward_components": self._calculate_reward_components(),
            "events": deepcopy(self.transcript_events),
        }

    def export_transcript_markdown(self) -> str:
        """Return a compact human-readable transcript with contradiction labels."""
        payload = self.export_transcript_json()
        lines = [
            f"## {payload['case_id']} ({payload['difficulty']})",
            "",
            payload["case_brief"],
            "",
            "### Transcript",
        ]
        for event in payload["events"]:
            lines.append(
                f"- Step {event['step']} `{event['tool']}`: {event['input']}\n"
                f"  - Witness: {event['response']}\n"
                f"  - Triggered: {', '.join(event['triggered']) or 'none'}; "
                f"Surfaced: {', '.join(event['surfaced']) or 'none'}"
            )
        lines.extend(
            [
                "",
                "### Reward",
                "```json",
                str(payload["reward_components"]).replace("'", '"'),
                "```",
            ]
        )
        return "\n".join(lines)


def _semantic_key(question: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = [word for word in text.split() if word not in {"the", "a", "an", "you", "your", "did"}]
    return " ".join(words)


def _is_inadmissible(question: str) -> bool:
    q_lower = question.lower()
    leading_patterns = ["isn't it true that", "didn't you", "wouldn't you agree"]
    if any(pattern in q_lower for pattern in leading_patterns):
        return True
    if question.count("?") > 1:
        return True
    if q_lower.count(" and ") > 2:
        return True
    return False
