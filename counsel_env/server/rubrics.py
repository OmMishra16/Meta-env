"""Composable rubric tree for diagnostic per-component reward logging.

The canonical episode reward is computed by CounselEnvironment._compute_reward().
These rubrics exist for per-component diagnostic logging during training.

Tree structure (weights normalized to sum 1.0):
    CounselRubric (WeightedSum)
    ├── ContradictionsSurfacedRubric  weight=0.625  (primary signal)
    ├── EvidenceTimingRubric          weight=0.188  (was evidence timely?)
    └── AdmissibilityRubric           weight=0.188  (penalize leading/compound Qs)
"""

from openenv.core.rubrics import Rubric
from openenv.core.rubrics.containers import WeightedSum


class ContradictionsSurfacedRubric(Rubric):
    """Fraction of seeded contradictions surfaced this episode."""

    def forward(self, action, observation) -> float:
        """Return contradictions_surfaced / contradictions_total in [0, 1].

        Args:
            action: CounselAction (unused here).
            observation: CounselObservation with metadata keys contradictions_surfaced
                and contradictions_total.

        Returns:
            Float in [0, 1] representing surfacing ratio.
        """
        meta = getattr(observation, "metadata", {}) or {}
        total = meta.get("contradictions_total", 0)
        if total == 0:
            return 0.0
        surfaced = meta.get("contradictions_surfaced", 0)
        return min(1.0, surfaced / total)


class EvidenceTimingRubric(Rubric):
    """Reward for presenting disprover evidence within 3 turns of triggering a contradiction."""

    def forward(self, action, observation) -> float:
        """Return the pre-computed evidence timing score from observation metadata.

        Args:
            action: CounselAction (unused here).
            observation: CounselObservation with metadata key evidence_timing_score.

        Returns:
            Float in [0, 1]; 1.0 = evidence followed trigger within 3 turns.
        """
        meta = getattr(observation, "metadata", {}) or {}
        return float(meta.get("evidence_timing_score", 0.0))


class AdmissibilityRubric(Rubric):
    """Penalize inadmissible questions (leading or compound)."""

    def forward(self, action, observation) -> float:
        """Return 1 - min(inadmissible_count / 5, 1.0) — decreases with each bad question.

        Args:
            action: CounselAction (unused here).
            observation: CounselObservation with metadata key inadmissible_count.

        Returns:
            Float in [0, 1]; 1.0 = no inadmissible questions asked.
        """
        meta = getattr(observation, "metadata", {}) or {}
        bad = meta.get("inadmissible_count", 0)
        return max(0.0, 1.0 - bad / 5.0)


class CounselRubric(Rubric):
    """Top-level diagnostic rubric for the Cross-Examination Arena.

    Weights are normalized to sum to 1.0 from the intended ratios
    (ContradictionsSurfaced:EvidenceTiming:Admissibility = 1.0:0.3:0.3).
    """

    def __init__(self) -> None:
        super().__init__()
        self.surfaced = ContradictionsSurfacedRubric()
        self.timing = EvidenceTimingRubric()
        self.admissibility = AdmissibilityRubric()
        # Normalized so weights sum to 1.0: 1.0/1.6, 0.3/1.6, 0.3/1.6
        self.combined = WeightedSum(
            [self.surfaced, self.timing, self.admissibility],
            weights=[0.625, 0.1875, 0.1875],
        )

    def forward(self, action, observation) -> float:
        """Return weighted diagnostic score in [0, 1].

        Args:
            action: CounselAction.
            observation: CounselObservation.

        Returns:
            Float in [0, 1].
        """
        return self.combined(action, observation)
