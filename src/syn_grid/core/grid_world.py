from syn_grid.config.models import (
    GridWorldConf,
    OrbConf,
    OrbFactoryConf,
    DroidConf,
    NegativeConf,
    TierConf,
)
from syn_grid.gymnasium.action_space import DroidAction
from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.orbs.orb_meta import OrbMeta
from syn_grid.core.orbs.orb_factory import OrbFactory
from syn_grid.core.orbs.base_orb import BaseOrb

import numpy as np
from numpy.random import Generator, default_rng
from typing import Final


class GridWorld:
    # ================= #
    #       Init        #
    # ================= #

    ALL_ORBS: Final[list[BaseOrb]]
    _inactive_orbs: list[BaseOrb] = []
    _active_orbs: list[BaseOrb] = []

    def __init__(
        self,
        conf: GridWorldConf,
        orb_manager_conf: OrbFactoryConf,
        droid_conf: DroidConf,
        negative_orb_conf: NegativeConf,
        tier_orb_conf: TierConf,
    ):
        """
        Initializes the grid world. Defines the game world's size and initializes the droid and orbs.
        """

        if conf.grid_rows <= 0 or conf.grid_cols <= 0:
            raise ValueError("grid_cols and grid_rows should be larger than 0")

        if conf.max_active_orbs <= 0:
            raise ValueError("max_active_orbs should be larger than 0")

        self._max_active_orbs = conf.max_active_orbs
        self._grid_rows = conf.grid_rows
        self._grid_cols = conf.grid_cols

        self.droid = SynergyDroid(droid_conf)

        self.ALL_ORBS = OrbFactory(
            orb_manager_conf, negative_orb_conf, tier_orb_conf
        ).create_orbs()

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the droid to its starting position and re-spawns the orb at a random location
        """

        self.droid.reset()  # Initialize Droids starting position

        self._active_orbs.clear()
        self._inactive_orbs.clear()
        self._inactive_orbs = list(self.ALL_ORBS)
        for orb in self.ALL_ORBS:
            orb.reset()

        if rng == None:
            rng = default_rng()

        self.rng = rng

        # Initialize the orb's position
        self._spawn_random_orb()

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_agent_action(self, agent_action: DroidAction) -> float:
        reward = self.droid.perform_action(agent_action)

        for orb in self.ALL_ORBS:
            if orb.is_active:
                if self._update_timer_and_return_is_completed(orb):
                    orb.deplete_orb()
                    self._remove_orb(orb)
                elif self.droid.position == orb.position:
                    reward = self.droid.consume_orb(orb)
                    self._remove_orb(orb)
            else:
                if self._update_timer_and_return_is_completed(orb):
                    self._spawn_random_orb()

        return reward

    # === Getters === #

    def get_orb_positions(self, only_active: bool) -> list[list[np.int64]]:
        if only_active:
            return [o.position for o in self._active_orbs]

        return [o.position for o in self.ALL_ORBS]

    def get_orb_is_active_status(self, only_active: bool) -> list[bool]:
        if only_active:
            return [o.is_active for o in self._active_orbs]

        return [o.is_active for o in self.ALL_ORBS]

    def get_orb_meta(self, only_active: bool) -> list[OrbMeta]:
        if only_active:
            return [o.meta for o in self._active_orbs]

        return [o.meta for o in self.ALL_ORBS]

    def get_orb_categories(self) -> list[int]:
        return [o.meta.category.value for o in self.ALL_ORBS]

    def get_orb_types(self) -> list[int]:
        return [o.meta.type.value for o in self.ALL_ORBS]

    def get_orb_life(self) -> list[int]:
        return [o.timer.remaining for o in self.ALL_ORBS]

    def get_orb_tiers(self) -> list[int]:
        return [o.meta.tier for o in self.ALL_ORBS]

    # ================= #
    #      Helpers      #
    # ================= #

    # === API === #

    def _update_timer_and_return_is_completed(self, orb: BaseOrb) -> bool:
        orb.timer.tick()
        return orb.timer.is_completed()

    def _remove_orb(self, orb: BaseOrb):
        idx = self._active_orbs.index(orb)
        depleted = self._active_orbs.pop(idx)
        self._inactive_orbs.append(depleted)

    # === Global === #

    def _spawn_random_orb(self):
        available_orbs = [r for r in self._inactive_orbs if r.timer.is_completed()]

        if len(available_orbs) > 0 and len(self._active_orbs) < self._max_active_orbs:
            orb_idx = self.rng.integers(0, len(available_orbs))
            orb = self._inactive_orbs.pop(orb_idx)

            while True:
                position = [
                    self.rng.integers(0, self._grid_rows),
                    self.rng.integers(0, self._grid_cols),
                ]

                if self._empty_spawn_cell(position):
                    orb.spawn(position)
                    self._active_orbs.append(orb)
                    break

    def _empty_spawn_cell(self, position: list[np.int64]) -> bool:
        # Check against droid
        if position == self.droid.position:
            return False

        # If there are no active orbs we can spawn right away
        if len(self._active_orbs) == 0:
            return True

        # Else check against all active orbs
        for r in self._active_orbs:
            if position == r.position:
                return False

        return True
