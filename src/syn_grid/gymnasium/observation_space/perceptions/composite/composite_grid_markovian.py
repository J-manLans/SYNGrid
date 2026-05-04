from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectType, SynergyType
from syn_grid.config.models import PerceptionConf


import numpy as np
from gymnasium import spaces


class CompositeGridMarkovian(BasePerception):
    # NOTE: this needs a custom extractor to work, right now it is merely a 1D Box vector because
    # SB3 flattens it. But this is post-thesis work.

    # ================= #
    #        Init       #
    # ================= #

    _GLOBAL_KEY = "global_data"
    _GRID_KEY = "grid_data"

    def __init__(self, conf: PerceptionConf, orbs: int):
        super().__init__(conf, orbs)
        self._orb_type_channels = self._build_orb_type_channel_map()

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None:
        self._global_data.fill(0.0)
        self._grid_data.fill(0.0)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        global_high = self._get_max_global_values()
        grid_channels_high = np.concatenate(
            [
                np.array([self._ACTIVE_FLAG], dtype=np.float32),
                self._get_max_orb_type_flags(),
                np.array([self._max_tier], dtype=np.float32),
                self._get_max_orb_data(),
            ]
        )

        # Create H,W,C and let C be the length of the list
        rows, cols = self._get_max_droid_positions()
        rows = int(rows) + 1
        cols = int(cols) + 1
        channels = grid_channels_high.shape[0]

        # Initialize the array and grid used for giving the observation
        self._global_data = np.zeros_like(global_high, dtype=np.float32)
        self._grid_data = np.zeros((rows, cols, channels), dtype=np.float32)
        high_3d = np.broadcast_to(grid_channels_high, self._grid_data.shape).copy()

        # Return observation space definition
        return spaces.Dict(
            {
                self._GLOBAL_KEY: spaces.Box(
                    low=0.0, high=global_high, shape=global_high.shape, dtype=np.float32
                ),
                self._GRID_KEY: spaces.Box(
                    low=0.0, high=high_3d, shape=self._grid_data.shape, dtype=np.float32
                ),
            }
        )

    def get_observation(
        self, state: GridWorld, steps_left: int
    ) -> dict[str, np.ndarray]:
        """
        NOTE: This is what I need to work towards when having more orbs:

        type_channel = self._get_orb_type_channel_index(orb)
        self._grid_data[y, x, type_channel] = self._ACTIVE_FLAG
        self._grid_data[y, x, channels - 2] = orb.META.TIER
        self._grid_data[y, x, channels - 1] = orb.TIMER.remaining
        """

        # Global data
        self._global_data[:] = self._get_global_values(steps_left, state)

        # Reset grid
        self._grid_data.fill(0.0)

        # Droid data
        droid_y, droid_x = state.droid.position
        self._grid_data[droid_y, droid_x, 0] = self._ACTIVE_FLAG

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                y, x = orb.position
                channel = self._orb_type_channels[orb.META.TYPE]

                self._grid_data[y, x, channel] = self._ACTIVE_FLAG
                self._grid_data[y, x, -2] = orb.META.TIER
                self._grid_data[y, x, -1] = orb.TIMER.remaining

        return {
            self._GLOBAL_KEY: self._global_data,
            self._GRID_KEY: self._grid_data,
        }

    # ================= #
    #      Helpers      #
    # ================= #

    def _build_orb_type_channel_map(self) -> dict[DirectType | SynergyType, int]:
        channel = 1  # channel 0 is droid
        mapping = {}

        if self._enabled_orbs.neg_enabled:
            mapping[DirectType.NEGATIVE] = channel
            channel += 1
        if self._enabled_orbs.tier_enabled:
            mapping[SynergyType.TIER] = channel
            channel += 1

        # Check if any implemented orbs are forgotten
        expected = sum(self._enabled_orbs.model_dump().values())
        if len(mapping) != expected:
            raise ValueError(
                f"Orb type channel map is missing entries — "
                f"expected {expected} but got {len(mapping)}. "
                f"Did you forget to add a new orb type to _build_orb_type_channel_map?"
            )

        return mapping
