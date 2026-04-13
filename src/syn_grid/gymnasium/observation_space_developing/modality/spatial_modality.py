from syn_grid.config.models import ObservationConf, OrbFactoryConf
from syn_grid.gymnasium.observation_space_developing.difficulty.base_difficulty import BaseDifficulty

from gymnasium import spaces
import numpy as np
from typing import Final


class SpatialModality:
    # ================= #
    #       Init        #
    # ================= #
    hard_observation_space: Final[spaces.Box]

    def __init__(
        self, orb_conf: OrbFactoryConf, obs_conf: ObservationConf
    ):
        orb_count = self._get_orb_count(orb_conf, obs_conf)
        self.hard_observation_space = self._build_space(obs_conf, orb_count)

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, difficulty: BaseDifficulty) -> spaces.Dict:
        return difficulty.setup_obs_space(self.hard_observation_space)

    def encode(self, difficulty: BaseDifficulty) -> spaces.Space:
        ...

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _get_orb_count(
        self, orb_conf: OrbFactoryConf, obs_conf: ObservationConf
    ) -> int:
        num_orb_types = len(orb_conf.types.__class__.model_fields)

        if orb_conf.types.tier.enabled:
            num_orb_types += obs_conf.max_curriculum_tier

        return num_orb_types

    def _build_space(self, obs_conf: ObservationConf, orb_count: int) -> spaces.Box:
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_conf.grid_rows, obs_conf.grid_cols, orb_count),
            dtype=np.float32,
        )


