"""CounselEnvironment — Cross-Examination Arena OpenEnv environment."""

import random
from copy import deepcopy
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CounselAction, CounselObservation, CounselState
    from .case_generator import generate_case
    from .rubrics import CounselRubric
    from .witness import Contradiction, Witness
except ImportError:
    from models import CounselAction, CounselObservation, CounselState  # type: ignore
    from server.case_generator import generate_case  # type: ignore
    from server.rubrics import CounselRubric  # type: ignore
    from server.witness import Contradiction, Witness  # type: ignore


QUESTION_BUDGET = 15


class CounselEnvironment(Environment):
    """Cross-Examination Arena — train LLMs to surface witness contradictions.

    Each episode is a procgen courtroom case with seeded contradictions. The
    lawyer (agent) asks questions and presents evidence within a QUESTION_BUDGET.
    Reward is granted only at episode end: surfaced/total minus inadmissibility penalty.

    SUPPORTS_CONCURRENT_SESSIONS is True — each WebSocket session gets its own
    isolated instance, enabling parallel GRPO rollout without shared state.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, case_fn: Optional[Callable[[], dict]] = None) -> None:
        """Initialize the environment.

        Args:
            case_fn: Optional callable that returns a case dict. Defaults to
                generate_case() (random template). Inject for testing.
        """
        super().__init__(rubric=CounselRubric())
        self._case_fn: Callable[[], dict] = case_fn or generate_case

        # Episode state — all initialized properly in reset()
        self._case_id: str = ""
        self._case_brief: str = ""
        self._witness: Optional[Witness] = None
        self._available_evidence: Dict[str, str] = {}
        self._questions_remaining: int = QUESTION_BUDGET
        self._questions_used: int = 0
        self._inadmissible_count: int = 0
        self._done: bool = False
        self._final_reward: float = 0.0
        self._step_count: int = 0
        self._transcript: List[str] = []
        self._trigger_turns: Dict[str, int] = {}
        self._evidence_turns: Dict[str, int] = {}
        self._episode_id: str = ""

    # ------------------------------------------------------------------
    # OpenEnv required interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> CounselObservation:
        """Start a new episode with a freshly generated case.

        Args:
            seed: Optional random seed for reproducible case generation.
            episode_id: Optional explicit episode identifier.

        Returns:
            Initial CounselObservation with case_brief, available evidence,
            and questions_remaining == QUESTION_BUDGET.
        """
        if seed is not None:
            random.seed(seed)

        self._reset_rubric()

        case = self._case_fn()
        self._case_id = case["case_id"]
        self._case_brief = case["case_brief"]
        self._available_evidence = dict(case["evidence"])

        # Fresh contradiction instances so state doesn't carry over between episodes
        contradictions = [
            Contradiction(
                cid=c.cid,
                trigger_keywords=list(c.trigger_keywords),
                sealed_claim=c.sealed_claim,
                disprover_evidence_id=c.disprover_evidence_id,
            )
            for c in case["contradictions"]
        ]
        self._witness = Witness(story=dict(case["witness_story"]), contradictions=contradictions)

        self._questions_remaining = QUESTION_BUDGET
        self._questions_used = 0
        self._inadmissible_count = 0
        self._done = False
        self._final_reward = 0.0
        self._step_count = 0
        self._transcript = []
        self._trigger_turns = {}
        self._evidence_turns = {}
        self._episode_id = episode_id or str(uuid4())

        return CounselObservation(
            witness_response="Court is in session. The witness has been sworn in.",
            available_evidence=list(self._available_evidence.keys()),
            questions_remaining=self._questions_remaining,
            transcript_tail="",
            case_brief=self._case_brief,
            done=False,
            reward=0.0,
            metadata={
                "contradictions_surfaced": 0,
                "contradictions_total": len(contradictions),
                "inadmissible_count": 0,
                "evidence_timing_score": 0.0,
            },
        )

    def step(
        self,
        action: CounselAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> CounselObservation:
        """Execute one action in the cross-examination.

        Routes on action.tool:
            ask_question    — consume budget, detect inadmissibility, get witness response
            present_evidence — show exhibit, potentially surface a contradiction
            make_objection  — acknowledged (v1: no mechanical effect)
            rest_case       — end episode, compute final reward

        Args:
            action: CounselAction specifying tool and parameters.
            timeout_s: Ignored (deterministic env).

        Returns:
            CounselObservation with witness response, updated budget, and
            non-zero reward only when done=True.
        """
        if self._done:
            return self._build_obs(
                witness_response="[COURT] The case has already concluded.",
                done=True,
                reward=self._final_reward,
            )

        tool = action.tool

        if tool == "ask_question":
            return self._handle_ask_question(action)
        elif tool == "present_evidence":
            return self._handle_present_evidence(action)
        elif tool == "make_objection":
            return self._handle_make_objection(action)
        elif tool == "rest_case":
            return self._handle_rest_case()
        else:
            return self._build_obs(witness_response=f"[COURT] Unknown tool: {tool!r}.")

    @property
    def state(self) -> CounselState:
        """Return current episode metadata.

        Returns:
            CounselState with case_id, contradiction counts, question budget usage,
            and inadmissible question count.
        """
        total = len(self._witness.contradictions) if self._witness else 0
        surfaced = (
            sum(1 for c in self._witness.contradictions if c.surfaced)
            if self._witness
            else 0
        )
        return CounselState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            case_id=self._case_id,
            contradictions_total=total,
            contradictions_surfaced=surfaced,
            questions_used=self._questions_used,
            inadmissible_count=self._inadmissible_count,
        )

    # ------------------------------------------------------------------
    # Action handlers (private)
    # ------------------------------------------------------------------

    def _handle_ask_question(self, action: CounselAction) -> CounselObservation:
        self._step_count += 1
        text = action.text or ""

        if self._is_inadmissible(text):
            self._inadmissible_count += 1
            response = "[COURT] Objection sustained. That question is inadmissible."
            self._append_transcript(f"Q (inadmissible): {text}\n[COURT]: {response}")
            return self._build_obs(witness_response=response)

        if self._questions_remaining <= 0:
            self._done = True
            reward = self._compute_reward()
            self._final_reward = reward
            return self._build_obs(
                witness_response="[COURT] Question budget exhausted. The defense rests.",
                done=True,
                reward=reward,
            )

        # Record which contradictions are triggered before this question
        already_triggered = {c.cid for c in self._witness.contradictions if c.triggered}

        self._questions_used += 1
        self._questions_remaining -= 1
        response = self._witness.respond_to_question(text)

        # Record turn for any newly triggered contradictions
        for c in self._witness.contradictions:
            if c.triggered and c.cid not in already_triggered:
                self._trigger_turns[c.cid] = self._step_count

        self._append_transcript(f"Q: {text}\nW: {response}")
        return self._build_obs(witness_response=response)

    def _handle_present_evidence(self, action: CounselAction) -> CounselObservation:
        self._step_count += 1
        exhibit_id = action.exhibit_id or ""

        if exhibit_id not in self._available_evidence:
            return self._build_obs(
                witness_response=f"[COURT] Exhibit '{exhibit_id}' not found in evidence list."
            )

        self._evidence_turns[exhibit_id] = self._step_count
        response = self._witness.react_to_evidence(exhibit_id)
        self._append_transcript(f"[Exhibit: {exhibit_id}]\nW: {response}")
        return self._build_obs(witness_response=response)

    def _handle_make_objection(self, action: CounselAction) -> CounselObservation:
        self._step_count += 1
        reason = action.reason or "reason not stated"
        response = f"[COURT] Objection noted by the defense: {reason}."
        self._append_transcript(f"[Objection: {reason}]")
        return self._build_obs(witness_response=response)

    def _handle_rest_case(self) -> CounselObservation:
        self._step_count += 1
        self._done = True
        reward = self._compute_reward()
        self._final_reward = reward
        return self._build_obs(
            witness_response="[COURT] The defense rests. Court is adjourned.",
            done=True,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    def _is_inadmissible(self, text: str) -> bool:
        """Detect leading questions and compound questions.

        Args:
            text: Question text to check.

        Returns:
            True if the question is inadmissible.
        """
        if not text:
            return False
        t = text.lower()
        leading_phrases = ["isn't it true", "didn't you", "wouldn't you agree"]
        if any(phrase in t for phrase in leading_phrases):
            return True
        if t.count("?") > 1:
            return True
        return False

    def _compute_reward(self) -> float:
        """Compute final episode reward: surfaced/total minus inadmissibility penalty.

        Returns:
            Float in [0, 1]. Clamped. Only called when done=True.
        """
        if not self._witness:
            return 0.0
        total = len(self._witness.contradictions)
        if total == 0:
            return 0.0
        surfaced = sum(1 for c in self._witness.contradictions if c.surfaced)
        raw = surfaced / total - 0.1 * self._inadmissible_count
        return max(0.0, min(1.0, raw))

    def _compute_evidence_timing_score(self) -> float:
        """Score how quickly evidence followed triggers (diagnostic).

        Returns:
            Float in [0, 1]. Average over triggered contradictions.
        """
        if not self._witness:
            return 0.0
        triggered = [c for c in self._witness.contradictions if c.triggered]
        if not triggered:
            return 0.0
        scores = []
        for c in triggered:
            trigger_turn = self._trigger_turns.get(c.cid, -1)
            present_turn = self._evidence_turns.get(c.disprover_evidence_id, -1)
            if c.surfaced and trigger_turn >= 0 and present_turn >= 0:
                gap = present_turn - trigger_turn
                score = 1.0 if gap <= 3 else max(0.0, 1.0 - (gap - 3) * 0.2)
            elif c.surfaced:
                score = 0.5
            else:
                score = 0.0
            scores.append(score)
        return sum(scores) / len(scores)

    def _append_transcript(self, exchange: str) -> None:
        self._transcript.append(exchange)

    def _get_transcript_tail(self, n: int = 5) -> str:
        return "\n---\n".join(self._transcript[-n:])

    def _build_obs(
        self,
        witness_response: str = "",
        done: Optional[bool] = None,
        reward: Optional[float] = None,
    ) -> CounselObservation:
        """Construct a CounselObservation from current environment state."""
        if done is None:
            done = self._done
        if reward is None:
            reward = self._final_reward if self._done else 0.0

        n_surfaced = (
            sum(1 for c in self._witness.contradictions if c.surfaced) if self._witness else 0
        )
        n_total = len(self._witness.contradictions) if self._witness else 0

        return CounselObservation(
            witness_response=witness_response,
            available_evidence=list(self._available_evidence.keys()),
            questions_remaining=self._questions_remaining,
            transcript_tail=self._get_transcript_tail(),
            case_brief=self._case_brief,
            done=done,
            reward=reward,
            metadata={
                "contradictions_surfaced": n_surfaced,
                "contradictions_total": n_total,
                "inadmissible_count": self._inadmissible_count,
                "evidence_timing_score": self._compute_evidence_timing_score(),
            },
        )
