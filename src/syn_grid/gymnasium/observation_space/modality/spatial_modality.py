from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models import ModalityConf
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.gymnasium.observation_space.modality.base_modality import BaseModality

from gymnasium import spaces
import numpy as np


class SpatialModality(BaseModality):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, modality_conf: ModalityConf):
        self._modality_conf = modality_conf

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space:
        self._max_vals = difficulty.get_max_values()

        # Initialize spatial specific values
        max_agent_present = 1

        # Add the spatial values to the list
        self._max_vals.insert(0, max_agent_present)

        # Create H,W,C and let C be the length of the list
        self._ROWS = self._modality_conf.grid_rows
        self._COLS = self._modality_conf.grid_cols
        self._CHANNELS = len(self._max_vals)

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._ROWS, self._COLS, self._CHANNELS),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        # TODO: Think about if this is worth it:
        #
        # But I want my obs's to be very much plug and play. So this was a good suggestion:
        # You could handle global info by either dedicating separate channels entirely (full 5x5
        # filled with the same normalized value)

        grid = np.zeros((self._ROWS, self._COLS, self._CHANNELS), dtype=np.float32)

        # Droid data
        droid_y, droid_x = state.droid.position

        grid[droid_y, droid_x, 0] = 1
        grid[droid_y, droid_x, 1] = steps_left / self._max_vals[1]
        grid[droid_y, droid_x, 2] = state.droid.score / self._max_vals[2]
        grid[droid_y, droid_x, 3] = (
            state.droid.digestion_engine.chained_tiers / self._max_vals[3]
        )

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                y, x = orb.position

                grid[y, x, 4] = orb.meta.category.value / self._max_vals[4]
                grid[y, x, 5] = orb.meta.type.value / self._max_vals[5]
                grid[y, x, 6] = orb.meta.tier / self._max_vals[6]
                grid[y, x, 7] = orb.timer.remaining / self._max_vals[7]

        return grid
