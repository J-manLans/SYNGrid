from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
from gymnasium import spaces

from syn_grid.config.models import PerceptionConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb


class BasePerception(ABC):
    # ================= #
    #        Init       #
    # ================= #

    _MISSING_ORB_VALUE: Final[float] = 0.0
    _ACTIVE_FLAG: Final[float] = 1.0

    def __init__(self, conf: PerceptionConf, orbs: int, max_identity: int) -> None:
        self._perception_conf = conf

        # Global values
        self._orbs_in_env = orbs

        # # Orb data
        self._max_identity = max_identity
        self._max_orb_lifespan = BaseOrb._life_span

    # ================= #
    #      Helpers      #
    # ================= #

    # ======= setup_obs_space() helpers ======= #

    # --- Global data getters --- #
    def _get_max_global_values(self) -> np.ndarray:
        return np.array(
            [
                self._perception_conf.max_steps,
                self._perception_conf.max_score,
                self._perception_conf.max_tier,
            ],
            dtype=np.float32,
        )

    # --- Droid data getters --- #
    def _get_max_droid_positions(self) -> np.ndarray:
        return np.array(
            [self._perception_conf.grid_rows, self._perception_conf.grid_cols],
            dtype=np.float32,
        )

    # --- Orb data getters --- #
    def _get_max_orb_base(self) -> np.ndarray:
        return np.array(
            [
                self._perception_conf.grid_rows,
                self._perception_conf.grid_cols,
                self._max_identity,
            ],
            dtype=np.float32,
        )

    def _get_max_orb_extended(self) -> np.ndarray:
        return np.array([self._max_orb_lifespan], dtype=np.float32)

    def _get_max_orb_type_flags(self) -> np.ndarray:
        return np.ones(
            sum(self._perception_conf.enabled_orbs.model_dump().values()),
            dtype=np.float32,
        )

    def _get_observable_orb_count(self) -> int:
        """
        Returns the number of orbs to include in the observation. In single chain mode all orbs up to max tier are always present, so max_tier is used. Otherwise, max_active_orbs is used.
        """

        if self._perception_conf.curriculum_training:
            return (
                self._perception_conf.tiers
                if self._perception_conf.single_chain_mode
                else self._perception_conf.max_active_orbs
            )

        return (
            self._perception_conf.max_tier
            if self._perception_conf.single_chain_mode
            else self._perception_conf.max_active_orbs
        )

    # ======= get_observation() helpers ======= #

    def _get_global_values(self, steps_left: int, state: GridWorld) -> np.ndarray:
        return np.array(
            [
                steps_left,
                min(state.droid.score, self._perception_conf.max_score),
                state.droid.digestion_engine.chained_tiers,
            ],
            dtype=np.float32,
        )

    def _get_droid_values(self, droid_y: int, droid_x: int) -> np.ndarray:
        return np.array([droid_y, droid_x], dtype=np.float32)

    def _get_orb_values(self, orb: BaseOrb, include_timer: bool = False) -> np.ndarray:
        orb_y, orb_x = orb.position

        values = [self._ACTIVE_FLAG, orb_y, orb_x, orb.META.IDENTITY]

        if include_timer:
            values.append(orb.TIMER.remaining)

        return np.array(values, dtype=np.float32)

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
        )[
            : (
                self._perception_conf.max_tier
                if self._perception_conf.single_chain_mode
                else self._perception_conf.max_active_orbs
            )
        ]

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
