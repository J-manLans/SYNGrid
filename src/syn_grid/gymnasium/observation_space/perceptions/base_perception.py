from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
from gymnasium import spaces

from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.observation import ObservationRules


class BasePerception(ABC):
    # ================= #
    #        Init       #
    # ================= #

    _MISSING_ORB_VALUE: Final[float] = 0.0
    _ACTIVE_FLAG: Final[float] = 1.0

    def __init__(
        self,
        observation: ObservationRules,
        grid_dimensions: tuple[int, int],
        max_identity: int,
    ) -> None:
        # Grid bounds
        self._max_row: Final[int] = grid_dimensions[0] - 1
        self._max_col: Final[int] = grid_dimensions[1] - 1

        # Orb slots
        self._slot_count: Final[int] = observation.observation_slot_count
        self._sort_limit: Final[int] = observation.sort_limit

        # Orb data
        self._max_identity: Final[int] = max_identity

    # ================= #
    #      Helpers      #
    # ================= #

    # ======= setup_obs_space() helpers ======= #

    # --- Droid data getters --- #
    def _get_max_droid_position(self) -> np.ndarray:
        return np.array([self._max_row, self._max_col], dtype=np.float32)

    # --- Orb data getters --- #
    def _get_max_orb_values(self) -> np.ndarray:
        return np.array(
            [self._ACTIVE_FLAG, self._max_row, self._max_col, self._max_identity],
            dtype=np.float32,
        )

    # ======= get_observation() helpers ======= #

    def _get_droid_position(self, droid_y: int, droid_x: int) -> np.ndarray:
        return np.array([droid_y, droid_x], dtype=np.float32)

    def _get_orb_values(self, orb: BaseOrb) -> np.ndarray:
        orb_y, orb_x = orb.position

        return np.array(
            [self._ACTIVE_FLAG, orb_y, orb_x, orb.META.IDENTITY], dtype=np.float32
        )

    def _sort_orbs_by_manhattan_dist_to_droid(
        self, orbs: list[BaseOrb], droid_y: int, droid_x: int
    ) -> list[BaseOrb]:
        """Sort orbs by distance to droid, inactive orbs go to the bottom"""

        return sorted(
            orbs,
            key=lambda orb: (
                abs(orb.position[0] - droid_y) + abs(orb.position[1] - droid_x)
                if orb.is_active
                else float("inf")
            ),
        )[: self._sort_limit]

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def setup_obs_space(self) -> spaces.Space: ...

    @abstractmethod
    def get_observation(self, state: GridWorld) -> Any:
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
