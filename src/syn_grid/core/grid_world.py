from syn_grid.config.models import (
    GridWorldConf,
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

        # World
        self._conf: Final[GridWorldConf] = conf

        # Droid
        self.droid: Final[SynergyDroid] = SynergyDroid(droid_conf)

        # Orbs
        self._active_orbs: Final[list[BaseOrb]] = []
        self._inactive_orbs: list[BaseOrb] = []
        self.ALL_ORBS: Final[list[BaseOrb]] = OrbFactory(
            orb_manager_conf, negative_orb_conf, tier_orb_conf
        ).create_orbs()

    def reset(self, rng: Generator | None = None) -> None:
        """
        Reset the droid to its starting position and re-spawns the orb at a random location
        """

        # Reset Droid
        self.droid.reset()

        # Reset the orb arrays
        self._active_orbs.clear()
        self._inactive_orbs.clear()
        self._inactive_orbs = self.ALL_ORBS.copy()
        for orb in self.ALL_ORBS:
            orb.reset()

        if rng == None:
            rng = default_rng()

        self._rng = rng

        # Spawn the first orb
        self._spawn_random_orb_if_ready()

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_droid_action(self, agent_action: DroidAction) -> float:
        reward = 0
        step_penalty = self.droid.perform_action(agent_action)

        for orb in self.ALL_ORBS:
            if orb.is_active:
                # only decrease timer for tier orbs if de-spawning is activated in the configs
                if orb.META.TIER == 0 or self._conf.de_spawn_tiers:
                    orb.TIMER.tick()
                if orb.TIMER.is_completed():
                    orb.de_spawn()
                    self._toggle_orb_to_inactive(orb)
                elif self.droid.position == orb.position:
                    # consume orb
                    reward = self.droid.consume_orb(orb)
                    self._toggle_orb_to_inactive(orb)
            else:
                # decrease the cooldown for inactive orbs
                orb.TIMER.tick()

        if len(self._active_orbs) < self._conf.max_active_orbs:
            self._spawn_random_orb_if_ready()

        return step_penalty + reward

    # === Getters === #

    def get_orb_positions(self, only_active: bool) -> list[list[int]]:
        if only_active:
            return [o.position for o in self._active_orbs]

        return [o.position for o in self.ALL_ORBS]

    def get_orb_is_active_status(self, only_active: bool) -> list[bool]:
        if only_active:
            return [o.is_active for o in self._active_orbs]

        return [o.is_active for o in self.ALL_ORBS]

    def get_orb_meta(self, only_active: bool) -> list[OrbMeta]:
        if only_active:
            return [o.META for o in self._active_orbs]

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

    def _toggle_orb_to_inactive(self, orb: BaseOrb):
        idx = self._active_orbs.index(orb)
        depleted = self._active_orbs.pop(idx)
        self._inactive_orbs.append(depleted)

    # === Global === #

    def _spawn_random_orb_if_ready(self):
        ready_orbs = [o for o in self._inactive_orbs if o.TIMER.is_completed()]
        if not ready_orbs:
            return

        orb = self._rng.choice(ready_orbs)  # type: ignore[arg-type]

        while True:
            position = [
                int(self._rng.integers(0, self._conf.grid_rows)),
                int(self._rng.integers(0, self._conf.grid_cols)),
            ]

            if self._empty_spawn_cell(position):
                self._inactive_orbs.remove(orb)
                orb.spawn(position)
                self._active_orbs.append(orb)
                break

    def _empty_spawn_cell(self, position: list[int]) -> bool:
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
