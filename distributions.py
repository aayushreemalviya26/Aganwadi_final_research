from typing import Tuple
import numpy as np

from config import SimulationConfig
from entities import HealthState


class RandomSampler:
    """Single random-number source to prevent hidden, non-reproducible randomness."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def bernoulli(self, probability: float) -> bool:
        return bool(self.rng.random() < probability)

    def integer_inclusive(self, bounds: Tuple[int, int]) -> int:
        low, high = bounds
        return int(self.rng.integers(low, high + 1))

    def uniform(self, low: float, high: float) -> float:
        return float(self.rng.uniform(low, high))

    def initial_health_state(self, config: SimulationConfig) -> HealthState:
        return HealthState.SAM if self.bernoulli(config.sam_share) else HealthState.MAM

    def initial_complication(
        self,
        state: HealthState,
        config: SimulationConfig,
    ) -> bool:
        probability = (
            config.sam_complication_probability
            if state == HealthState.SAM
            else config.mam_complication_probability
        )
        return self.bernoulli(probability)
