from syn_grid.gymnasium.observation_space.perceptions.base_perception import (
    BasePerception,
)
from syn_grid.config.models import PerceptionConf
from syn_grid.core.grid_world import GridWorld

import numpy as np
from gymnasium import spaces


class MediumVectorPerception(BasePerception):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, conf: PerceptionConf, orbs: int) -> None:
        super().__init__(conf, orbs)

    # ================= #
    #        API        #
    # ================= #

    def reset(self) -> None: ...

    def setup_obs_space(self) -> spaces.Space:
        max_vals = []
        orb_data = []

        # define observation layout
        max_vals.extend(self._get_max_global_values())
        max_vals.extend(self._get_max_droid_positions())
        max_vals.extend(self._get_max_droid_data())
        orb_data.extend(self._get_max_orb_positions())
        orb_data.extend(self._get_max_orb_identity())
        orb_data.extend(self._get_max_orb_data())
        self._num_orb_slots = len(orb_data)
        max_vals.extend(orb_data * self._orbs_in_env)

        # finalize observation space definition
        self._SHAPE = len(max_vals)
        low = np.full(self._SHAPE, -1.0, dtype=np.float32)
        low[0:4] = 0.0
        high = np.asarray(max_vals, dtype=np.float32)

        return spaces.Box(
            low=low,
            high=high,
            shape=(self._SHAPE,),
            dtype=np.float32,
        )

    def get_observation(self, state: GridWorld, steps_left: int) -> np.ndarray:
        obs = np.full(self._SHAPE, -1.0, dtype=np.float32)
        obs_index = 0

        # Global data
        obs[obs_index] = steps_left
        obs_index += 1

        # Droid data
        droid_y, droid_x = state.droid.position

        obs[obs_index] = droid_y
        obs_index += 1
        obs[obs_index] = droid_x
        obs_index += 1
        obs[obs_index] = state.droid.score
        obs_index += 1
        obs[obs_index] = state.droid.digestion_engine.chained_tiers
        obs_index += 1

        # Orb data
        for orb in state.ALL_ORBS:
            if orb.is_active:
                orb_y, orb_x = orb.position

                obs[obs_index] = orb_y
                obs_index += 1
                obs[obs_index] = orb_x
                obs_index += 1
                obs[obs_index] = orb.META.CATEGORY.value
                obs_index += 1
                obs[obs_index] = orb.META.TYPE.value
                obs_index += 1
                obs[obs_index] = orb.META.TIER
                obs_index += 1
                obs[obs_index] = orb.TIMER.remaining
                obs_index += 1
            else:
                obs_index += self._num_orb_slots

        return obs
