from syn_grid.gymnasium.observation_space.difficulty.base_difficulty import (
    BaseDifficulty,
)
from syn_grid.core.grid_world import GridWorld
from syn_grid.config.models import MediumDifficultyConf


from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, world: GridWorld, medium_conf: MediumDifficultyConf):
        self._world = world
        self._medium_conf = medium_conf

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, hard_obs_high: NDArray) -> spaces.Space:
        spatial_obs = self._setup_spatial_obs(hard_obs_high)
        episode_meta = self._setup_episode_meta()
        droid_meta = self._setup_droid_meta()

        return spaces.Dict({
            "grid": spatial_obs,
            "episode_meta": episode_meta,
            "droid_meta": droid_meta
        })

    def apply(self, state)-> dict[str, NDArray]:
        # TODO: Continue here, but test run first.
        ...

    # ================= #
    #      Helpers      #
    # ================= #

    def _setup_spatial_obs(self, hard_obs_high: NDArray) -> spaces.Box:
        max_orb_lifespan = (self._medium_conf.grid_rows - 1) + (self._medium_conf.grid_cols - 1)
        high = np.asarray([*hard_obs_high, max_orb_lifespan],dtype=np.float32)

        ROWS = self._medium_conf.grid_rows
        COLS = self._medium_conf.grid_cols
        CHANNELS = len(high)

        high = np.tile(high, (ROWS, COLS, 1))

        return spaces.Box(
            low=0,
            high=high,
            shape=(self._medium_conf.grid_rows, self._medium_conf.grid_cols, CHANNELS),
            dtype=np.float32,
        )

    def _setup_episode_meta(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=self._medium_conf.max_steps,
            shape=(1,),
            dtype=np.float32,
        )

    def _setup_droid_meta(self) -> spaces.Box:
        max_score = self._medium_conf.max_score
        max_tier_chain = self._medium_conf.max_tier

        high = np.asarray([max_score, max_tier_chain], dtype=np.float32)

        return spaces.Box(
            low=0,
            high=high,
            shape=(2,),
            dtype=np.float32,
        )
