from typing import Tuple
import numpy as np

from config import SimulationConfig
from entities import HealthState


class RandomSampler:
    """
    Central random-number generator for all stochastic events
    in the simulation.

    Using one seeded generator makes each simulation replication
    reproducible.
    """

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def bernoulli(self, probability: float) -> bool:
        """Return True with the given probability."""
        return bool(self.rng.random() < probability)

    def integer_inclusive(self, bounds: Tuple[int, int]) -> int:
        """
        Sample an integer uniformly from an inclusive range.

        Example:
        (1, 7) can return 1, 2, ..., 7.
        """
        low, high = bounds
        return int(self.rng.integers(low, high + 1))

    def uniform(self, low: float, high: float) -> float:
        """Sample a continuous value uniformly between low and high."""
        return float(self.rng.uniform(low, high))

    def initial_health_state(
        self,
        config: SimulationConfig
    ) -> HealthState:
        """
        Assign MAM or SAM when the child enters the
        post-detection model.
        """
        if self.bernoulli(config.sam_share):
            return HealthState.SAM

        return HealthState.MAM

    def initial_complication(
        self,
        state: HealthState,
        config: SimulationConfig,
    ) -> bool:
        """
        Determine whether the child has a medical complication
        at entry, conditional on nutritional state.
        """

        if state == HealthState.SAM:
            probability = config.sam_complication_probability
        else:
            probability = config.mam_complication_probability

        return self.bernoulli(probability)