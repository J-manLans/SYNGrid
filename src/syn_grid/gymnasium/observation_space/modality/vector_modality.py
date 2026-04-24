from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.config.models import ModalityConf
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.gymnasium.observation_space.modality.base_modality import BaseModality

from gymnasium import spaces
import numpy as np
from typing import Final


class VectorModality(BaseModality):
    # ================= #
    #       Init        #
    # ================= #

    _AVAILABLE_SLOTS: Final[list[int]] = [2, 7, 12]

    def __init__(self, modality_conf: ModalityConf):
        self._MODALITY_CONF = modality_conf

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        self._orb_slot_map: dict[int, int] = {}

    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Space:
        self._max_vals = difficulty.get_max_values()

        # Initialize vector specific values
        max_droid_y = self._MODALITY_CONF.grid_rows - 1
        max_droid_x = self._MODALITY_CONF.grid_cols - 1
        max_orb_y = self._MODALITY_CONF.grid_rows - 1
        max_orb_x = self._MODALITY_CONF.grid_cols - 1

        # Add the orb values
        self._max_vals[0:0] = [max_orb_y, max_orb_x]
        self._max_vals[0:0] = self._max_vals[0:] * (
            self._MODALITY_CONF.max_active_orbs - 1
        )

        # Add the droid values
        self._max_vals[0:0] = [max_droid_y, max_droid_x]

        # Let the shape be the length of the list
        self._SHAPE = len(self._max_vals)

        low = np.zeros(self._SHAPE, dtype=np.float32)
        low[2:] = -1.0

        return spaces.Box(
            low=low,
            high=1.0,
            shape=(self._SHAPE,),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        obs = np.full(self._SHAPE, -1.0, dtype=np.float32)

        # Episode data
        # obs[0] = steps_left / self._max_vals[0]

        # Droid data
        droid_y, droid_x = state.DROID.position

        obs[0] = droid_y / self._max_vals[0]
        obs[1] = droid_x / self._max_vals[1]
        # obs[2] = state.DROID.score / self._max_vals[2]
        # obs[4] = state.DROID.DIGESTION_ENGINE.chained_tiers / self._max_vals[4]

        self._prune_orb_slot_map(state)

        # Available orb data
        for orb_index_in_all_orbs_list, orb in enumerate(state.ALL_ORBS):
            if orb.is_active:

                # Assign a permanent grid slot if this orb is new
                if orb_index_in_all_orbs_list not in self._orb_slot_map:
                    for obs_start_index in self._AVAILABLE_SLOTS:
                        if obs_start_index not in self._orb_slot_map.values():
                            self._orb_slot_map[orb_index_in_all_orbs_list] = obs_start_index
                            break

                # Write orb data to its assigned slot
                obs_start_index = self._orb_slot_map.get(orb_index_in_all_orbs_list)
                if obs_start_index is not None:
                    self._add_orb_data(orb, obs, obs_start_index)

        return obs

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

    def _add_orb_data(self, orb: BaseOrb, grid: np.ndarray, grid_index: int) -> None:
        orb_y, orb_x = orb.position

        grid[grid_index] = orb_y / self._max_vals[grid_index]
        grid_index += 1
        grid[grid_index] = orb_x / self._max_vals[grid_index]
        grid_index += 1
        grid[grid_index] = orb.META.CATEGORY.value / self._max_vals[grid_index]
        grid_index += 1
        grid[grid_index] = orb.META.TYPE.value / self._max_vals[grid_index]
        grid_index += 1
        grid[grid_index] = orb.META.TIER / self._max_vals[grid_index]
        grid_index += 1
        # grid[grid_index] = orb.TIMER.remaining / self._max_vals[grid_index]
        # grid_index += 1
