from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class MediumCompositePerception(BasePerception):
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

        droid_high = np.concatenate(
            [self._get_max_droid_positions(), self._get_max_droid_data()]
        )

        orb_high = np.tile(
            np.concatenate(
                [
                    np.array([self._ORB_ACTIVE_FLAG], dtype=np.float32),
                    self._get_max_orb_positions(),
                    self._get_max_orb_identity(),
                    # self._get_max_orb_data(), # TODO: Re-add this after thesis experiments, I wont use timer for them, so removing it simplifies observation
                ]
            ),
            (self._orbs_in_env, 1),
        )
        orb_features = orb_high.shape[1]

        # Initialize the arrays used for giving the observation
        self._global_data = np.zeros_like(global_high, dtype=np.float32)
        self._droid_data = np.zeros_like(droid_high, dtype=np.float32)
        self._orb_data = np.zeros((self._orbs_in_env, orb_features), dtype=np.float32)

        return spaces.Dict(
            {
                self._GLOBAL_KEY: spaces.Box(
                    low=0,
                    high=global_high,
                    shape=(len(global_high),),
                    dtype=np.float32,
                ),
                self._DROID_KEY: spaces.Box(
                    low=0,
                    high=droid_high,
                    shape=(len(droid_high),),
                    dtype=np.float32,
                ),
                self._ORB_KEY: spaces.Box(
                    low=self._MISSING_ORB_VALUE,
                    high=orb_high,
                    shape=(self._orbs_in_env, orb_features),
                    dtype=np.float32,
                ),
            }
        )

    def get_observation(
        self, state: GridWorld, steps_left: int
    ) -> dict[str, np.ndarray]:
        # Global data
        self._global_data[0] = steps_left

        # Droid data
        self._droid_data[0], self._droid_data[1] = state.droid.position
        self._droid_data[2] = state.droid.score
        self._droid_data[3] = state.droid.digestion_engine.chained_tiers

        # Orb data
        for i, orb in enumerate(state.ALL_ORBS):
            if orb.is_active:
                orb_y, orb_x = orb.position

                self._orb_data[i] = [
                    self._ORB_ACTIVE_FLAG,
                    orb_y,
                    orb_x,
                    orb.META.CATEGORY.value,
                    orb.META.TYPE.value,
                    orb.META.TIER,
                    # orb.TIMER.remaining, # TODO: Re-add this after thesis experiments, I wont use timer for them, so removing it simplifies observation
                ]
            else:
                self._orb_data[i] = self._MISSING_ORB_VALUE

        return {
            self._GLOBAL_KEY: self._global_data,
            self._DROID_KEY: self._droid_data,
            self._ORB_KEY: self._orb_data,
        }
