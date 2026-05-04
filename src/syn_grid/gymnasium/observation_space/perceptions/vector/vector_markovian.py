from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.config.models import PerceptionConf
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class VectorMarkovian(BasePerception):

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        # Reset the observation arrays
        self._obs_data.fill(self._MISSING_ORB_VALUE)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        global_high = self._get_max_global_values()

        droid_high = self._get_max_droid_positions()
        self._droid_start_index = global_high.shape[0]

        orb_parts = [
            np.array([self._ORB_ACTIVE_FLAG], dtype=np.float32),
            self._get_max_orb_positions(),
            self._get_max_orb_identity(),
        ]
        if self._include_timer:
            orb_parts.append(self._get_max_orb_data())
        orb_high = np.concatenate(orb_parts)
        self._orb_features = orb_high.shape[0]
        orb_high = np.tile(orb_high, self._max_active_orbs)
        self._orb_start_index = self._droid_start_index + droid_high.shape[0]

        high = np.concatenate([global_high, droid_high, orb_high])

        # Initialize the array used for giving the observation
        self._obs_data = np.zeros_like(high, dtype=np.float32)

        # Return observation space definition
        return spaces.Box(
            low=0,
            high=high,
            shape=high.shape,
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        # Global data
        self._obs_data[0 : self._droid_start_index] = self._get_global_values(
            steps_left, state
        )

        # Droid data
        droid_y, droid_x = state.droid.position
        self._obs_data[self._droid_start_index : self._orb_start_index] = (
            self._get_droid_values(droid_y, droid_x)
        )

        sorted_orbs = self._sort_orbs_by_manhattan_dist_to_droid(
            state.ALL_ORBS, droid_y, droid_x
        )

        # Orb data
        obs_index = self._orb_start_index
        for orb in sorted_orbs:
            if orb.is_active:
                self._obs_data[obs_index : obs_index + self._orb_features] = (
                    self._get_orb_values(orb, self._include_timer)
                )
            else:
                self._obs_data[obs_index : obs_index + self._orb_features] = (
                    self._MISSING_ORB_VALUE
                )

            obs_index += self._orb_features

        return self._obs_data
