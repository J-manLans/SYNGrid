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

    def reset(self) -> None: ...

    def setup_obs_space(self) -> spaces.Space:
        self._max_vals = []

        # Initialize spatial specific values
        max_agent_present = 1

        # Add the spatial values to the list
        self._max_vals.insert(0, max_agent_present)

        # Create H,W,C and let C be the length of the list
        droid_positions = self._get_max_droid_positions()
        self._rows = int(droid_positions[0])
        self._cols = int(droid_positions[1])
        self._channels = len(self._max_vals)

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._rows, self._cols, self._channels),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        # TODO: Think about if this is worth it:
        #
        # But I want my obs's to be very much plug and play. So this was a good suggestion:
        # You could handle global info by dedicating separate channels entirely (full 5x5
        # filled with the same normalized value) instead of grouping global values like score,
        # steps left and chained tiers with the droid, sort of letting the agent understand they
        # aren't tied to a specific cell. This way I don't need to add more spaces for them and put
        # them in a dict, since I don't think most agents handle those observations out of the box.

        grid = np.zeros((self._rows, self._cols, self._channels), dtype=np.float32)

        # Droid data
        droid_y, droid_x = state.droid.position

        grid[droid_y, droid_x, 0] = 1
        grid[droid_y, droid_x, 1] = steps_left
        grid[droid_y, droid_x, 2] = state.droid.score
        grid[droid_y, droid_x, 3] = state.droid.digestion_engine.chained_tiers

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                y, x = orb.position

                grid[y, x, 4] = orb.META.CATEGORY.value
                grid[y, x, 5] = orb.META.TYPE.value
                grid[y, x, 6] = orb.META.TIER
                grid[y, x, 7] = orb.TIMER.remaining

        return grid
