from syn_grid.config.models import PerceptionConf
from syn_grid.core.orbs.orb_meta import OrbCategory, DirectType, SynergyType
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.grid_world import GridWorld

import numpy as np
from abc import ABC, abstractmethod
from gymnasium import spaces
from typing import Final, Any


class BasePerception(ABC):
    # ================= #
    #        Init       #
    # ================= #

    _OLD_MISSING_VALUE = -1.0  # TODO: remove this when fixed
    _MISSING_ORB_VALUE: Final[float] = 0.0
    _ORB_ACTIVE_FLAG: Final[float] = 1.0

    def __init__(self, conf: PerceptionConf, orbs: int) -> None:
        # Global values
        self._orbs_in_env = orbs
        self._max_active_orbs = conf.max_active_orbs
        self._max_steps = conf.max_steps
        self._max_grid_y = conf.grid_rows - 1
        self._max_grid_x = conf.grid_cols - 1

        # Droid data
        self._max_score = conf.max_score
        self._max_tier_chain = conf.max_tier

        # Orb data
        self._max_category = len(OrbCategory) - 1
        self._max_type = max(len(DirectType) - 1, len(SynergyType) - 1)
        self._max_tier = conf.max_tier
        self._max_orb_lifespan = BaseOrb._life_span

    # ================= #
    #      Helpers      #
    # ================= #

    # === Global data getters === #
    def _get_max_global_values(self) -> np.ndarray:
        return np.array([self._max_steps], dtype=np.float32)

    # === Droid data getters === #
    def _get_max_droid_positions(self) -> np.ndarray:
        return np.array([self._max_grid_y, self._max_grid_x], dtype=np.float32)

    def _get_max_droid_data(self) -> np.ndarray:
        # TODO: Try without max score, return it later if needed
        return np.array([self._max_tier_chain], dtype=np.float32)

    # === Orb data getters === #
    def _get_max_orb_positions(self) -> np.ndarray:
        return np.array([self._max_grid_y, self._max_grid_x], dtype=np.float32)

    def _get_max_orb_identity(self) -> np.ndarray:
        return np.array(
            [self._max_category, self._max_type, self._max_tier], dtype=np.float32
        )

    def _get_max_orb_data(self) -> np.ndarray:
        return np.array([self._max_orb_lifespan], dtype=np.float32)

    def _sort_orbs_by_droid_proximity(
        self, orbs: list[BaseOrb], droid_y: int, droid_x: int
    ) -> list[BaseOrb]:
        def manhattan_distance(orb: BaseOrb) -> float:
            if not orb.is_active:
                return float("inf")
            return abs(orb.position[0] - droid_y) + abs(orb.position[1] - droid_x)

        return sorted(orbs, key=manhattan_distance)[: self._max_active_orbs]

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def setup_obs_space(self) -> spaces.Space: ...

    @abstractmethod
    def get_observation(self, state: GridWorld, steps_left: int) -> Any:
        """
        Get current observation from the environment.

        Returns:
            An observation for the agent, format depends on concrete implementation

            **CompositePerception**:
                - Returns Dict[str, np.ndarray]
                - Each np.ndarray can have any shape (1D, 2D, 3D, HWC, etc.)

            **VectorPerception**:
                - Returns np.ndarray of shape (N,)

            **SpatialPerception**:
                - Returns np.ndarray of shape (C, H, W)

        The return type must match the observation_space defined in setup_obs_space().
        """
        ...
