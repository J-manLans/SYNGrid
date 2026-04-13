from syn_grid.gymnasium.observation_space_developing.difficulty.base_difficulty import BaseDifficulty
from syn_grid.core.grid_world import GridWorld

from typing import Any
from gymnasium import spaces
import numpy as np


class MediumDifficulty(BaseDifficulty):
    # ================= #
    #        Init       #
    # ================= #

    def __init__(self, world: GridWorld):
        self._world = world

    # ================= #
    #        API        #
    # ================= #

    def setup_obs_space(self, spatial_obs: spaces.Box) -> spaces.Dict:
        grid_meta = self._setup_grid_meta()
        agent_meta = self._setup_agent_meta()
        orb_meta = self._setup_orb_meta()

        return spaces.Dict ({
            "grid": spatial_obs,
            "grid meta": grid_meta,
            "agent meta": agent_meta,
            "orb meta": orb_meta
        })

    def apply(self)-> dict[str, Any]:
        ...

    # ================= #
    #      Helpers      #
    # ================= #

    def _setup_grid_meta(self) -> spaces.Box:
        ...

    def _setup_agent_meta(self) -> spaces.Box:
        ...

    def _setup_orb_meta(self) -> spaces.Box:
        ...
