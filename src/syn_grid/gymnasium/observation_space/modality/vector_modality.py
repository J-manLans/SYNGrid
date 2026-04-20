from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.config.models import ModalityConf
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.gymnasium.observation_space.modality.base_modality import BaseModality

from gymnasium import spaces
import numpy as np


class VectorModality(BaseModality):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, modality_conf: ModalityConf):
        self._orb_timers = {}
        self._orb_slot_map: dict[int, int] = {}
        self._available_slots = [5, 11, 17]
        self._modality_conf = modality_conf

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space:
        self._max_vals = difficulty.get_max_values()

        # Initialize vector specific values
        max_droid_y = self._modality_conf.grid_rows - 1
        max_droid_x = self._modality_conf.grid_cols - 1
        max_orb_y = self._modality_conf.grid_rows - 1
        max_orb_x = self._modality_conf.grid_cols - 1

        # Add the orb values
        self._max_vals[3:3] = [max_orb_y, max_orb_x]
        self._max_vals[3:3] = self._max_vals[3:] * (
            self._modality_conf.max_active_orbs - 1
        )

        # Add the droid values
        self._max_vals[1:1] = [max_droid_y, max_droid_x]

        # Let the shape be the length of the list
        self._SHAPE = len(self._max_vals)

        low = np.zeros(self._SHAPE, dtype=np.float32)
        low[5:] = -1.0

        return spaces.Box(
            low=low,
            high=1.0,
            shape=(self._SHAPE,),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        grid = np.full(self._SHAPE, -1.0, dtype=np.float32)

        # Episode data
        grid[0] = steps_left / self._max_vals[0]

        # Droid data
        droid_y, droid_x = state.droid.position

        grid[1] = droid_y / self._max_vals[1]
        grid[2] = droid_x / self._max_vals[2]
        grid[3] = state.droid.score / self._max_vals[3]
        grid[4] = state.droid.digestion_engine.chained_tiers / self._max_vals[4]

        self._prune_orb_slot_map(state)

        # Orb data
        for orb_index, orb in enumerate(state.ALL_ORBS):
            if orb.is_active:

                # Assign a permanent grid slot if this orb is new
                if orb_index not in self._orb_slot_map:
                    for grid_idx in [5, 11, 17]:
                        if grid_idx not in self._orb_slot_map.values():
                            self._orb_slot_map[orb_index] = grid_idx
                            break

                # Write orb data to its assigned slot
                grid_idx = self._orb_slot_map.get(orb_index)
                if grid_idx is not None:
                    self._add_orb_data(orb, grid, grid_idx)

        return grid

    # ================= #
    #      Helpers      #
    # ================= #

    def _prune_orb_slot_map(self, state: GridWorld):
        if not self._orb_slot_map:
            return

        active_indices = {i for i, orb in enumerate(state.ALL_ORBS) if orb.is_active}

        for orb_index in list(self._orb_slot_map.keys()):
            if orb_index not in active_indices:
                del self._orb_slot_map[orb_index]

    def _add_orb_data(self, orb: BaseOrb, grid: np.ndarray, index: int) -> None:
        orb_y, orb_x = orb.position

        grid[index] = orb_y / self._max_vals[index]
        index += 1
        grid[index] = orb_x / self._max_vals[index]
        index += 1
        grid[index] = orb.meta.category.value / self._max_vals[index]
        index += 1
        grid[index] = orb.meta.type.value / self._max_vals[index]
        index += 1
        grid[index] = orb.meta.tier / self._max_vals[index]
        index += 1
        grid[index] = orb.timer.remaining / self._max_vals[index]
        index += 1
