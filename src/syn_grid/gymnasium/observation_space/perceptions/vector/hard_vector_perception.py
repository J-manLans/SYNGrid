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
    #        API        #
    # ================= #

    def reset(self) -> None:
        self._orb_slot_map: dict[int, int] = {}
        self._obs_data[:] = self._MISSING_ORB_VALUE

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        droid_high = np.concatenate([self._get_max_droid_positions()])
        orb_high = np.concatenate(
            [self._get_max_orb_positions(), self._get_max_orb_identity()]
        )
        self._orb_features = orb_high.shape[0]
        orb_high = np.tile(orb_high, self._max_active_orbs)
        high = np.concatenate([droid_high, orb_high])

        # Configure slot mapping for runtime observation filling
        num_droid_slots = droid_high.shape[0]
        self._initialize_available_slots_list(num_droid_slots, self._orb_features)

        # Initialize the array used for giving the observation and finalize observation space
        # definition
        self._obs_data = np.full(
            high.shape[0], self._MISSING_ORB_VALUE, dtype=np.float32
        )
        low = self._obs_data
        low[0:num_droid_slots] = 0.0

        return spaces.Box(
            low=low,
            high=high,
            shape=(high.shape[0],),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:

        # Droid positions [y, x]
        self._obs_data[0], self._obs_data[1] = state.droid.position

        self._cleanup_inactive_orbs(state)

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
                    self._add_orb_data(orb, obs_start_index)

        return self._obs_data

    # ================= #
    #      Helpers      #
    # ================= #

    def _initialize_available_slots_list(
        self, orb_start_index: int, num_orb_slots: int
    ) -> None:
        self._AVAILABLE_SLOTS = [orb_start_index]

        for i in range(1, self._max_active_orbs):
            self._AVAILABLE_SLOTS.append(self._AVAILABLE_SLOTS[i - 1] + num_orb_slots)

    def _cleanup_inactive_orbs(self, state: GridWorld):
        """Remove mapping and reset observation slots for orbs that are no longer active."""

        if not self._orb_slot_map:
            return

        active_indices = {i for i, orb in enumerate(state.ALL_ORBS) if orb.is_active}

        for orb_index in list(self._orb_slot_map.keys()):
            if orb_index not in active_indices:
                obs_start_index = self._orb_slot_map[orb_index]
                # Reset the slot data in observation before removing mapping from the dict
                self._obs_data[
                    obs_start_index : obs_start_index + self._orb_features
                ] = self._MISSING_ORB_VALUE
                del self._orb_slot_map[orb_index]

    def _add_orb_data(self, orb: BaseOrb, obs_index: int) -> None:
        orb_y, orb_x = orb.position
        self._obs_data[obs_index : obs_index + self._orb_features] = [
            orb_y,
            orb_x,
            orb.META.CATEGORY.value,
            orb.META.TYPE.value,
            orb.META.TIER,
        ]
