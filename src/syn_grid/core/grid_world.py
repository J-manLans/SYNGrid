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

        self._CONF: Final[GridWorldConf] = conf
        self._DE_SPAWN_TIERS = orb_manager_conf.de_spawn_tiers

        self.DROID: Final[SynergyDroid] = SynergyDroid(droid_conf)

        self._ACTIVE_ORBS: Final[list[BaseOrb]] = []
        self._inactive_orbs: list[BaseOrb] = []
        self.ALL_ORBS: Final[list[BaseOrb]] = OrbFactory(
            orb_manager_conf, negative_orb_conf, tier_orb_conf
        ).create_orbs()

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the droid to its starting position and re-spawns the orb at a random location
        """

        # Initialize Droid
        self.DROID.reset()

        # Reset the orb arrays
        self._ACTIVE_ORBS.clear()
        self._inactive_orbs.clear()
        self._inactive_orbs = self.ALL_ORBS.copy()
        for orb in self.ALL_ORBS:
            orb.reset()

        if rng == None:
            rng = default_rng()

        self.rng = rng

        # Initialize the first orb
        self._spawn_random_ready_orb()

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_agent_action(self, agent_action: DroidAction) -> float:
        reward = self.DROID.perform_action(agent_action)

        for orb in self.ALL_ORBS:
            # decrease timer both for active orbs and orbs on cooldown
            orb.TIMER.tick()
            if orb.is_active:
                # only de-spawn non-tier orbs
                if orb.TIMER.is_completed() and (orb.META.TIER == 0 or self._DE_SPAWN_TIERS):
                    orb.de_spawn()
                    self._remove_orb(orb)
                elif self.DROID.position == orb.position:
                    reward += self.DROID.consume_orb(orb)
                    self._remove_orb(orb)

        if len(self._ACTIVE_ORBS) < self._CONF.max_active_orbs:
            self._spawn_random_ready_orb()

        return reward

    # === Getters === #

    def get_orb_positions(self, only_active: bool) -> list[list[int]]:
        if only_active:
            return [o.position for o in self._ACTIVE_ORBS]

        return [o.position for o in self.ALL_ORBS]

    def get_orb_is_active_status(self, only_active: bool) -> list[bool]:
        if only_active:
            return [o.is_active for o in self._ACTIVE_ORBS]

        return [o.is_active for o in self.ALL_ORBS]

    def get_orb_meta(self, only_active: bool) -> list[OrbMeta]:
        if only_active:
            return [o.META for o in self._ACTIVE_ORBS]

        return [o.META for o in self.ALL_ORBS]

    def get_orb_categories(self) -> list[int]:
        return [o.META.CATEGORY.value for o in self.ALL_ORBS]

    def get_orb_types(self) -> list[int]:
        return [o.META.TYPE.value for o in self.ALL_ORBS]

    def get_orb_life(self) -> list[int]:
        return [o.TIMER.remaining for o in self.ALL_ORBS]

    def get_orb_tiers(self) -> list[int]:
        return [o.META.TIER for o in self.ALL_ORBS]

    # ================= #
    #      Helpers      #
    # ================= #

    # === API === #

    def _remove_orb(self, orb: BaseOrb):
        idx = self._ACTIVE_ORBS.index(orb)
        depleted = self._ACTIVE_ORBS.pop(idx)
        self._inactive_orbs.append(depleted)

    # === Global === #

    def _spawn_random_ready_orb(self):
        ready_orbs = [o for o in self._inactive_orbs if o.TIMER.is_completed()]
        if not ready_orbs:
            return

        orb = self.rng.choice(ready_orbs)  # type: ignore[arg-type]
        self._inactive_orbs.remove(orb)

        while True:
            position = [
                int(self.rng.integers(0, self._CONF.grid_rows)),
                int(self.rng.integers(0, self._CONF.grid_cols)),
            ]

            if self._empty_spawn_cell(position):
                orb.spawn(position)
                self._ACTIVE_ORBS.append(orb)
                break

    def _empty_spawn_cell(self, position: list[int]) -> bool:
        # Check against droid
        if position == self.DROID.position:
            return False

        # If there are no active orbs we can spawn right away
        if len(self._ACTIVE_ORBS) == 0:
            return True

        # Else check against all active orbs
        for r in self._ACTIVE_ORBS:
            if position == r.position:
                return False

        return True
