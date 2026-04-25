from typing import Any

from openenv.core.rubrics import Rubric
from openenv.core.rubrics.containers import WeightedSum


def _component(observation: Any, name: str, default: float = 0.0) -> float:
    components = getattr(observation, "reward_components", None)
    if isinstance(components, dict):
        return float(components.get(name, default))
    if isinstance(observation, dict):
        return float(observation.get(name, default))
    return float(getattr(observation, name, default))


class ContradictionsSurfaced(Rubric):
    """Primary signal: normalized binary-per-contradiction success."""

    def forward(self, action: Any, observation: Any) -> float:
        return _component(observation, "primary_reward", 0.0)


class AuxiliaryProgress(Rubric):
    """Dense shaping signal used only as a minority component."""

    def forward(self, action: Any, observation: Any) -> float:
        return max(-1.0, min(1.0, _component(observation, "auxiliary_reward_raw", 0.0)))


class Total(Rubric):
    """OpenEnv rubric tree: primary reward dominates auxiliary shaping."""

    def __init__(self):
        super().__init__()
        self.primary = ContradictionsSurfaced()
        self.auxiliary = AuxiliaryProgress()
        self.combined = WeightedSum(
            [self.primary, self.auxiliary],
            weights=[0.8, 0.2],
        )

    def forward(self, action: Any, observation: Any) -> float:
        return self.combined(action, observation)
