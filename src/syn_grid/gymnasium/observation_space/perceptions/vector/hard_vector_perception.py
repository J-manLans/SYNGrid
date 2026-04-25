from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.config.models import PerceptionConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb

from gymnasium import spaces
import numpy as np


class HardVectorPerception(BasePerception):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, conf: PerceptionConf, orbs: int) -> None:
        super().__init__(conf, orbs)

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self) -> spaces.Space:
        self._max_vals = []
        droid_data = []
        orb_data = []

        droid_data.extend(self._get_max_droid_positions())
        self._max_vals.extend(droid_data)
        orb_data.extend(self._get_max_orb_positions())
        orb_data.extend(self._get_max_orb_identity())
        self._max_vals.extend(orb_data * self._MAX_ACTIVE_ORBS)

        num_orb_slots = len(orb_data)
        num_droid_slots = len(droid_data)
        self._initialize_available_slots_list(num_droid_slots, num_orb_slots)

        self._SHAPE = len(self._max_vals)

        low = np.full(self._SHAPE, -1.0, dtype=np.float32)
        low[0:num_droid_slots] = 0.0

        return spaces.Box(
            low=low,
            high=1.0,
            shape=(self._SHAPE,),
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._orb_slot_map: dict[int, int] = {}

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        obs = np.full(self._SHAPE, -1.0, dtype=np.float32)

        # Droid data
        droid_y, droid_x = state.DROID.position
        obs[0] = droid_y / self._max_vals[0]
        obs[1] = droid_x / self._max_vals[1]

        self._prune_orb_slot_map(state)

        # Available orb data
        for orb_index_in_all_orbs_list, orb in enumerate(state.ALL_ORBS):
            if orb.is_active:

                # Assign a permanent grid slot if this orb is new
                if orb_index_in_all_orbs_list not in self._orb_slot_map:
                    for obs_start_index in self._AVAILABLE_SLOTS:
                        if obs_start_index not in self._orb_slot_map.values():
                            self._orb_slot_map[orb_index_in_all_orbs_list] = (
                                obs_start_index
                            )
                            break

                # Write orb data to its assigned slot
                obs_start_index = self._orb_slot_map.get(orb_index_in_all_orbs_list)
                if obs_start_index is not None:
                    self._add_orb_data(orb, obs, obs_start_index)

        return obs

    # ================= #
    #      Helpers      #
    # ================= #

    def _initialize_available_slots_list(
        self, orb_start_index: int, num_orb_slots: int
    ) -> None:
        self._AVAILABLE_SLOTS = [orb_start_index]

        for i in range(1, self._MAX_ACTIVE_ORBS):
            self._AVAILABLE_SLOTS.append(self._AVAILABLE_SLOTS[i - 1] + num_orb_slots)

    def _prune_orb_slot_map(self, state: GridWorld):
        if not self._orb_slot_map:
            return

        active_indices = {i for i, orb in enumerate(state.ALL_ORBS) if orb.is_active}

        for orb_index in list(self._orb_slot_map.keys()):
            if orb_index not in active_indices:
                del self._orb_slot_map[orb_index]

    def _add_orb_data(self, orb: BaseOrb, obs: np.ndarray, obs_index: int) -> None:
        orb_y, orb_x = orb.position

        obs[obs_index] = orb_y / self._max_vals[obs_index]
        obs_index += 1
        obs[obs_index] = orb_x / self._max_vals[obs_index]
        obs_index += 1
        obs[obs_index] = orb.META.CATEGORY.value / self._max_vals[obs_index]
        obs_index += 1
        obs[obs_index] = orb.META.TYPE.value / self._max_vals[obs_index]
        obs_index += 1
        obs[obs_index] = orb.META.TIER / self._max_vals[obs_index]
        obs_index += 1
