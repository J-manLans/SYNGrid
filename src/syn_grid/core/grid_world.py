from __future__ import annotations

from typing import TYPE_CHECKING, Final

from numpy.random import Generator, default_rng

from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbMeta
from syn_grid.gymnasium.action_space import DroidAction

if TYPE_CHECKING:
    from syn_grid.scenario.blocks.population import OrbPopulation
    from syn_grid.scenario.blocks.spawning import SpawningRules


class GridWorld:
    """
    The simulation.

    Generic mechanics only: move, tick timers, spawn, consume, score. Every
    decision about *how* those behave belongs to the scenario, which arrives
    already composed. There is no conditional in this class whose answer depends
    on which scenario is running -- that is the whole point of the boundary, and
    it used to be three of them.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        grid_dimensions: tuple[int, int],
        droid: SynergyDroid,
        orb_population: OrbPopulation,
        spawning: SpawningRules,
    ):
        """
        Initializes the grid world from the pieces a scenario composed for it.

        :param grid_dimensions: the world's size as (rows, cols).
        :param droid: the droid that moves through this world.
        :param population: builds this world's orb pool.
        :param spawning: the rules defining how this world's orbs behave.
        """

        # World
        self._grid_rows, self._grid_cols = grid_dimensions
        self._spawning: Final[SpawningRules] = spawning

        # Droid
        self.droid: Final[SynergyDroid] = droid

        # Orbs
        self._active_orbs: Final[list[BaseOrb]] = []
        self._inactive_orbs: list[BaseOrb] = []
        # Class-wide and shared by every world in the process; it is the grid's
        # Manhattan diameter, so it is set here rather than by a scenario rule.
        BaseOrb.set_life_span(self._grid_rows, self._grid_cols)
        self.ALL_ORBS: Final[list[BaseOrb]] = orb_population.create()

        self._remap_sparse_identities_to_dense()

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

        if rng is None:
            rng = default_rng()

        self._rng = rng

        self._spawning.on_reset(self)

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_droid_action(self, agent_action: DroidAction) -> float:
        reward = 0.0
        step_penalty = self.droid.perform_action(agent_action)

        for orb in self.ALL_ORBS:
            if orb.is_active:
                # Tier orbs only age if the scenario lets them; a tier chain
                # holds its sequence still, a continuous world may not.
                if orb.META.TIER == 0 or self._spawning.tier_orb_expires:
                    orb.TIMER.tick()
                if orb.TIMER.is_completed():
                    orb.de_spawn()
                    self._toggle_orb_to_inactive(orb)
                elif self.droid.position == orb.position:
                    # consume orb
                    reward = self.droid.consume_orb(orb)
                    self._toggle_orb_to_inactive(orb)
                    self._spawning.on_orb_consumed(self)
            else:
                # decrease the cooldown for inactive orbs
                orb.TIMER.tick()

        self._spawning.after_step(self)

        return step_penalty + reward

    # === Getters === #

    @property
    def grid_dimensions(self) -> tuple[int, int]:
        """The world's size as (rows, cols)."""
        return self._grid_rows, self._grid_cols

    @property
    def active_orbs(self) -> list[BaseOrb]:
        return self._active_orbs

    @property
    def max_active_orbs(self) -> int:
        return self._spawning.max_active_orbs

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

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    def _remap_sparse_identities_to_dense(self) -> None:
        """Remap radix identities to dense sequential indices to simplify learning"""

        sorted_orbs = sorted(self.ALL_ORBS, key=lambda o: o.META.IDENTITY)

        identity_map = {}
        next_dense = 1

        for orb in sorted_orbs:
            radix_id = orb.META.IDENTITY

            if radix_id not in identity_map:
                identity_map[radix_id] = next_dense
                next_dense += 1

            orb.META.IDENTITY = identity_map[radix_id]

        self.max_identity = next_dense - 1

    # === Orb field ===
    # Invoked by the scenario's spawning rules, and public because the rules
    # are not part of this class.

    def spawn_orb_if_ready(self) -> None:
        """Place one ready orb on a random empty cell, if any is ready."""

        ready_orbs = [o for o in self._inactive_orbs if o.TIMER.is_completed()]
        if not ready_orbs:
            return

        orb = self._rng.choice(ready_orbs)  # type: ignore[arg-type]

        while True:
            position = [
                int(self._rng.integers(0, self._grid_rows)),
                int(self._rng.integers(0, self._grid_cols)),
            ]

            if self._empty_spawn_cell(position):
                self._inactive_orbs.remove(orb)
                orb.spawn(position)
                self._active_orbs.append(orb)
                break

    def deactivate_all_orbs(self, delay: int) -> None:
        """Put the whole field on cooldown, where it sits."""

        for orb in self._active_orbs:
            orb.reset()
            orb.TIMER.set(delay)

    def reactivate_all_orbs(self) -> None:
        """Bring back every orb whose cooldown has run out, where it was."""

        for orb in self._active_orbs:
            if orb.TIMER.is_completed():
                orb.spawn(orb.position)

    # === API === #

    def _toggle_orb_to_inactive(self, orb: BaseOrb) -> None:
        idx = self._active_orbs.index(orb)
        depleted = self._active_orbs.pop(idx)
        self._inactive_orbs.append(depleted)

    # === Global === #

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
