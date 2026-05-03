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
        self._obs_data.fill(self._OLD_MISSING_VALUE)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        world_high = np.concatenate(
            [
                self._get_max_global_values(),
                self._get_max_droid_positions(),
                self._get_max_droid_data(),
            ]
        )
        orb_high = np.concatenate(
            [
                self._get_max_orb_positions(),
                self._get_max_orb_identity(),
                self._get_max_orb_data(),
            ]
        )
        self._orb_features = orb_high.shape[0]
        orb_high = np.tile(orb_high, self._orbs_in_env)
        high = np.concatenate([world_high, orb_high])

        # Initialize the array used for giving the observation and finalize observation
        # space definition
        self._obs_data = np.full(
            high.shape[0], self._OLD_MISSING_VALUE, dtype=np.float32
        )
        low = self._obs_data
        low[0 : world_high.shape[0]] = 0.0

        return spaces.Box(
            low=low,
            high=high,
            shape=(len(high),),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        obs_index = 0

        # Global data
        self._obs_data[obs_index] = steps_left
        obs_index += 1

        # Droid data
        droid_y, droid_x = state.droid.position
        self._obs_data[obs_index : obs_index + 4] = [
            droid_y,
            droid_x,
            state.droid.score,
            state.droid.digestion_engine.chained_tiers,
        ]
        obs_index += 4

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                orb_y, orb_x = orb.position

                self._obs_data[obs_index : obs_index + self._orb_features] = [
                    orb_y,
                    orb_x,
                    orb.META.CATEGORY.value,
                    orb.META.TYPE.value,
                    orb.META.TIER,
                    orb.TIMER.remaining,
                ]
            else:
                self._obs_data[obs_index : obs_index + self._orb_features] = -1

            obs_index += self._orb_features

        return self._obs_data
