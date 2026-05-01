from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class HardCompositePerception(BasePerception):
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
        self._droid_data.fill(0.0)
        self._orb_data.fill(self._MISSING_ORB_VALUE)

    def setup_obs_space(self) -> spaces.Space:
        # Define observation layout
        droid_high = np.concatenate([self._get_max_droid_positions()])

        orb_high = np.tile(
            np.concatenate(
                [
                    np.array([self._ORB_ACTIVE_FLAG], dtype=np.float32),
                    self._get_max_orb_positions(),
                    self._get_max_orb_identity(),
                ]
            ),
            (self._orbs_in_env, 1),
        )
        orb_features = orb_high.shape[1]

        # Initialize the arrays used for giving the observation
        self._droid_data = np.zeros_like(droid_high, dtype=np.float32)
        self._orb_data = np.zeros((self._orbs_in_env, orb_features), dtype=np.float32)

        return spaces.Dict(
            {
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
        # Droid data [y, x]
        self._droid_data[0], self._droid_data[1] = state.droid.position

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
                ]
            else:
                self._orb_data[i] = self._MISSING_ORB_VALUE

        return {
            self._DROID_KEY: self._droid_data,
            self._ORB_KEY: self._orb_data,
        }
