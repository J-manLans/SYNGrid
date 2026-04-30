from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models import PerceptionConf

import numpy as np
from gymnasium import spaces


class MediumSpatialPerception(BasePerception):

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        self._grid.fill(0)

    def setup_obs_space(self) -> spaces.Space:
        # Initialize spatial specific values and construct the high array
        max_agent_present = 1
        high = np.concatenate([
            np.array([max_agent_present]),
            self._get_max_orb_identity(),
            self._get_max_orb_data()
        ])

        # Create H,W,C and let C be the length of the list
        rows, cols = self._get_max_droid_positions()
        rows = int(rows) + 1
        cols = int(cols) + 1
        channels = high.shape[0]

        # Initialize grid
        self._grid = np.zeros((rows, cols, channels), dtype=np.float32)

        # Build low array - one value per channel
        low = np.zeros(channels, dtype=np.float32)
        low[3] = -1  # Set the tier index to be -1 for non tier orbs

        return spaces.Box(
            low=low,
            high=high,
            shape=(rows, cols, channels),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        # Reset grid
        self._grid.fill(0)

        # Droid data
        droid_y, droid_x = state.droid.position

        self._grid[droid_y, droid_x, 0] = 1

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                y, x = orb.position

                self._grid[y, x, 1] = orb.META.CATEGORY.value
                self._grid[y, x, 2] = orb.META.TYPE.value
                self._grid[y, x, 3] = orb.META.TIER if orb.META.TIER > 0 else -1
                self._grid[y, x, 4] = orb.TIMER.remaining

        return self._grid
