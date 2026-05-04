from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class CompositeMarkovian(BasePerception):
    # TODO: remake this with nested dicts and create a custom extractor, right now it is merely a
    # 1D Box vector

    # ================= #
    #        Init       #
    # ================= #

    _GLOBAL_KEY = "global_data"
    _DROID_KEY = "droid_data"
    _ORB_KEY = "orb_data"

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        # Reset the observation arrays
        self._global_data.fill(0.0)
        self._droid_data.fill(0.0)
        self._orb_data.fill(self._MISSING_ORB_VALUE)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        global_high = self._get_max_global_values()
        droid_high = self._get_max_droid_positions()

        orb_parts = [
            np.array([self._ORB_ACTIVE_FLAG], dtype=np.float32),
            self._get_max_orb_positions(),
            self._get_max_orb_identity(),
        ]
        if self._include_timer:
            orb_parts.append(self._get_max_orb_data())
        orb_high = np.tile(
            np.concatenate(orb_parts),
            (self._max_active_orbs, 1),
        )

        # Initialize the arrays used for giving the observation
        self._global_data = np.zeros_like(global_high, dtype=np.float32)
        self._droid_data = np.zeros_like(droid_high, dtype=np.float32)
        self._orb_data = np.zeros_like(orb_high, dtype=np.float32)

        # Return observation space definition
        return spaces.Dict(
            {
                self._GLOBAL_KEY: spaces.Box(
                    low=0,
                    high=global_high,
                    shape=global_high.shape,
                    dtype=np.float32,
                ),
                self._DROID_KEY: spaces.Box(
                    low=0,
                    high=droid_high,
                    shape=droid_high.shape,
                    dtype=np.float32,
                ),
                self._ORB_KEY: spaces.Box(
                    low=self._MISSING_ORB_VALUE,
                    high=orb_high,
                    shape=orb_high.shape,
                    dtype=np.float32,
                ),
            }
        )

    def get_observation(
        self, state: GridWorld, steps_left: int
    ) -> dict[str, np.ndarray]:
        # Global data
        self._global_data[0:] = self._get_global_values(steps_left, state)

        # Droid data
        droid_y, droid_x = state.droid.position
        self._droid_data[0:] = self._get_droid_values(droid_y, droid_x)

        # Sort orbs by distance to droid, inactive orbs go to the bottom
        sorted_orbs = self._sort_orbs_by_manhattan_dist_to_droid(
            state.ALL_ORBS, droid_y, droid_x
        )

        # Orb data
        for i, orb in enumerate(sorted_orbs):
            if orb.is_active:
                self._orb_data[i] = self._get_orb_values(orb, self._include_timer)
            else:
                self._orb_data[i] = self._MISSING_ORB_VALUE

        return {
            self._GLOBAL_KEY: self._global_data,
            self._DROID_KEY: self._droid_data,
            self._ORB_KEY: self._orb_data,
        }
