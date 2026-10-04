from abc import ABC, abstractmethod
from typing import Any, Final

import numpy as np
from gymnasium import spaces

from syn_grid.config.models.scenario.scenario_common import PerceptionConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.rules.observation import ObservationRules


class BasePerception(ABC):
    """Encodes the world for the agent.

    The *shape* of an observation is not a free choice: how many orb slots it
    holds and how far the tier channel reaches follow from the world, so the
    scenario states them and this class is told. Perceptions used to re-derive
    both from a set of scenario booleans carried in the perception config, which
    meant the observation space a scenario produced was only knowable by
    reading four files at once.
    """

    # ================= #
    #        Init       #
    # ================= #

    _MISSING_ORB_VALUE: Final[float] = 0.0
    _ACTIVE_FLAG: Final[float] = 1.0

    def __init__(
        self,
        conf: PerceptionConf,
        observation_rules: ObservationRules,
        orbs: int,
        max_identity: int,
    ) -> None:
        self._perception_conf = conf
        self._observation_rules = observation_rules

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
                self._observation_rules.max_tier,
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
        Returns the number of orb slots the observation is built to hold, as the
        scenario defines it. In single chain mode all orbs up to max tier are
        always present, so max_tier is used. Otherwise, max_active_orbs is used.
        """

        return self._observation_rules.observation_slot_count

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
        )[: self._observation_rules.sort_limit]

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
