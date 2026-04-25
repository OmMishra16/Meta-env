# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Counsel Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CounselAction, CounselObservation, CounselState


class CounselEnv(
    EnvClient[CounselAction, CounselObservation, CounselState]
):
    """
    Client for the Counsel Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CounselEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(CounselAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CounselEnv.from_docker_image("counsel_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CounselAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CounselAction) -> Dict:
        """
        Convert CounselAction to JSON payload for step message.

        Args:
            action: CounselAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tool": action.tool,
            "text": action.text,
            "exhibit_id": action.exhibit_id,
            "reason": action.reason,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CounselObservation]:
        """
        Parse server response into StepResult[CounselObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CounselObservation
        """
        obs_data = payload.get("observation", {})
        observation = CounselObservation(
            witness_response=obs_data.get("witness_response", ""),
            available_evidence=obs_data.get("available_evidence", []),
            evidence_descriptions=obs_data.get("evidence_descriptions", {}),
            questions_remaining=obs_data.get("questions_remaining", 0),
            transcript_tail=obs_data.get("transcript_tail", ""),
            case_brief=obs_data.get("case_brief", ""),
            case_id=obs_data.get("case_id", ""),
            difficulty=obs_data.get("difficulty", ""),
            done=obs_data.get("done", payload.get("done", False)),
            reward=obs_data.get("reward", payload.get("reward", 0.0)),
            reward_components=obs_data.get("reward_components", {}),
        )

        return StepResult(
            observation=observation,
            reward=observation.reward,
            done=observation.done,
        )

    def _parse_state(self, payload: Dict) -> CounselState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        data = payload.get("data", payload)
        return CounselState(
            case_id=data.get("case_id", ""),
            difficulty=data.get("difficulty", ""),
            contradictions_total=data.get("contradictions_total", 0),
            contradictions_triggered=data.get("contradictions_triggered", 0),
            contradictions_surfaced=data.get("contradictions_surfaced", 0),
            questions_used=data.get("questions_used", 0),
            action_count=data.get("action_count", data.get("step_count", 0)),
            duplicate_question_count=data.get("duplicate_question_count", 0),
            irrelevant_question_count=data.get("irrelevant_question_count", 0),
            inadmissible_count=data.get("inadmissible_count", 0),
            evidence_timing_successes=data.get("evidence_timing_successes", 0),
            step_count=data.get("step_count", data.get("action_count", 0)),
        )
