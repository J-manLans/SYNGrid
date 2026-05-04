from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb

from gymnasium import spaces
import numpy as np


class VectorFullyPOMDP(BasePerception):

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        self._orb_slot_map: dict[int, int] = {}
        self._obs_data.fill(self._MISSING_ORB_VALUE)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        droid_high = self._get_max_droid_positions()

        orb_high = np.concatenate(
            [
                np.array([self._ACTIVE_FLAG], dtype=np.float32),
                self._get_max_orb_positions(),
                self._get_max_orb_identity(),
            ]
        )
        self._orb_features = orb_high.shape[0]
        orb_high = np.tile(orb_high, self._max_active_orbs)
        self._orb_start_index = droid_high.shape[0]

        high = np.concatenate([droid_high, orb_high])

        # Initialize the array used for giving the observation
        self._obs_data = np.zeros_like(high, dtype=np.float32)

        # Return observation space definition
        return spaces.Box(
            low=0.0,
            high=high,
            shape=high.shape,
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        # Droid data
        droid_y, droid_x = state.droid.position
        self._obs_data[0 : self._orb_start_index] = self._get_droid_values(
            droid_y, droid_x
        )

        sorted_orbs = self._sort_orbs_by_manhattan_dist_to_droid(
            state.ALL_ORBS, droid_y, droid_x
        )

        # Orb data
        obs_index = self._orb_start_index
        for orb in sorted_orbs:
            if orb.is_active:
                self._obs_data[obs_index : obs_index + self._orb_features] = (
                    self._get_orb_values(orb)
                )
            else:
                self._obs_data[obs_index : obs_index + self._orb_features] = (
                    self._MISSING_ORB_VALUE
                )

            obs_index += self._orb_features

        return self._obs_data
