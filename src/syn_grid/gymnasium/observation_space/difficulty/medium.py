from syn_grid.core.orbs.orb_meta import OrbCategory, DirectType, SynergyType
from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models import ObsConfig, MediumDifficultyConf

from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, obs_conf: ObsConfig):
        self._medium_conf = obs_conf.medium_difficulty

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, hard_obs_high: NDArray) -> spaces.Space:
        spatial_obs = self._setup_spatial_obs(hard_obs_high)

        return self._setup_spatial_obs(hard_obs_high)

    def get_observation(self, state: GridWorld, steps_left: int)-> np.ndarray:
        return self._get_spatial_obs(state, steps_left)

    # ================= #
    #      Helpers      #
    # ================= #

    # === Setup obs === #

    def _setup_spatial_obs(self, hard_obs_high: NDArray) -> spaces.Box:
        max_agent_present = 1
        max_steps = self._medium_conf.max_steps
        max_score = self._medium_conf.max_score
        max_tier_chain = self._medium_conf.max_tier
        max_category = len(OrbCategory) - 1
        max_type = max(len(DirectType) - 1, len(SynergyType) - 1)
        max_tier = self._medium_conf.max_tier
        max_orb_lifespan = (self._medium_conf.grid_rows - 1) + (self._medium_conf.grid_cols - 1)

        self._max_vals = [
            # agent values
            max_agent_present,
            max_steps,
            max_score,
            max_tier_chain,
            # orb values
            max_category,
            max_type,
            max_tier,
            max_orb_lifespan
        ]

        self._ROWS = self._medium_conf.grid_rows
        self._COLS = self._medium_conf.grid_cols
        self._CHANNELS = len(self._max_vals)

        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._medium_conf.grid_rows, self._medium_conf.grid_cols, self._CHANNELS),
            dtype=np.float32,
        )

    # def _setup_episode_meta(self) -> spaces.Box:
    #     return spaces.Box(
    #         low=0,
    #         high=self._medium_conf.max_steps,
    #         shape=(1,),
    #         dtype=np.float32,
    #     )

    # def _setup_droid_meta(self) -> spaces.Box:
    #     max_score = self._medium_conf.max_score
    #     max_tier_chain = self._medium_conf.max_tier

    #     high = np.asarray([max_score, max_tier_chain], dtype=np.float32)

    #     return spaces.Box(
    #         low=0,
    #         high=high,
    #         shape=(2,),
    #         dtype=np.float32,
    #     )

    # === Get obs === #

    def _get_spatial_obs(self, state: GridWorld, steps_left: int) -> np.ndarray:
        grid = np.zeros((self._ROWS, self._COLS, self._CHANNELS), dtype=np.float32)

        # Droid data
        droid_y, droid_x = state.droid.position
        grid[droid_y, droid_x, 0] = 1
        grid[droid_y, droid_x, 1] = steps_left / self._max_vals[1]
        grid[droid_y, droid_x, 2] = state.droid.score / self._max_vals[2]
        grid[droid_y, droid_x, 3] = state.droid.digestion_engine.chained_tiers / self._max_vals[3]

        # Orb data
        for orb in state.ALL_ORBS:
            if not orb.is_active:
                continue

            y, x = orb.position

            grid[y, x, 4] = orb.meta.category.value / self._max_vals[4]
            grid[y, x, 5] = orb.meta.type.value / self._max_vals[5]
            grid[y, x, 6] = orb.meta.tier / self._max_vals[6]
            grid[y, x, 7] = orb.timer.remaining / self._max_vals[7]

        return grid

    # def _get_episode_meta(self, steps_left: int) -> np.ndarray:
    #     return np.asarray([steps_left], dtype=np.float32)

    # def _get_droid_meta(self, state: GridWorld) -> np.ndarray:
    #     score = state.droid.score
    #     current_tier_chain = state.droid.digestion_engine.chained_tiers

    #     return np.asarray([score, current_tier_chain], dtype=np.float32)