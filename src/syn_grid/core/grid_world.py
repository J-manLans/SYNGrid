from typing import Final

from numpy.random import Generator

from syn_grid.core.droid.synergy_droid import SynergyDroid
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.droid.droid_action import DroidAction


class GridWorld:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        grid_dimensions: tuple[int, int],
        max_active_orbs: int,
        max_steps: int,
        orbs: list[BaseOrb],
        droid: SynergyDroid,
    ):
        """
        Holds the grid's size, the droid and the orbs it is handed, how many orbs the field may hold at once, and the
        step clock of an episode.
        """

        # World
        self._max_active_orbs: Final[int] = max_active_orbs
        self._rows: Final[int] = grid_dimensions[0]
        self._cols: Final[int] = grid_dimensions[1]

        # Step clock
        self._max_steps: Final[int] = max_steps
        self._steps_left: int = max_steps

        # Droid
        self.droid: Final[SynergyDroid] = droid

        # Orbs
        self.ALL_ORBS: Final[list[BaseOrb]] = orbs

        self._remap_sparse_identities_to_dense()

    def reset(self, rng: Generator) -> None:
        """
        Reset the droid to its starting position, fill the orb field and restart the step clock. The orbs are placed
        with `rng`, so the same generator gives the same layout.
        """

        # Reset the step clock
        self._steps_left = self._max_steps

        # Reset Droid
        self.droid.reset()

        # Reset the orbs
        for orb in self.ALL_ORBS:
            orb.reset()

        self._rng = rng

        # Fill the field
        for _ in range(self._max_active_orbs):
            self._spawn_random_orb_if_ready()

    # ================= #
    #        API        #
    # ================= #

    # === Logic === #

    def perform_droid_action(self, agent_action: DroidAction) -> float:
        self._steps_left -= 1

        reward = 0.0
        step_penalty = self.droid.take_step(agent_action)

        for orb in self.ALL_ORBS:
            orb.TIMER.tick()

            # The contact check comes first, so the droid wins an orb on its last tick of life
            if orb.is_active and self.droid.position == orb.position:
                reward = self.droid.consume_orb(orb)
            elif orb.has_expired():
                orb.de_spawn()

        # Bring back one ready orb if the field has room
        if len(self.orbs_on_field()) < self._max_active_orbs:
            self._spawn_random_orb_if_ready()

        return step_penalty + reward

    # === Getters === #

    @property
    def steps_left(self) -> int:
        """How many steps the episode has left. Each action takes one, and a reset restarts it at `max_steps`."""

        return self._steps_left

    def orbs_on_field(self) -> list[BaseOrb]:
        """The orbs that are on the field now, which are the ones that can be seen and eaten."""

        return [o for o in self.ALL_ORBS if o.is_active]

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

    # === API === #

    def _deactivate_all_orbs(self) -> None:
        for orb in self.orbs_on_field():
            orb.reset()
            orb.TIMER.set(self._conf.delay)

    def _reactivate_all_orbs(self) -> None:
        for orb in self.orbs_on_field():
            if orb.TIMER.is_completed():
                orb.spawn(orb.position)

    # === Global === #

    def _spawn_random_orb_if_ready(self) -> None:
        ready_orbs = [o for o in self.ALL_ORBS if o.is_ready()]
        if not ready_orbs:
            return

        orb = self._rng.choice(ready_orbs)  # type: ignore[arg-type]

        while True:
            position = [
                int(self._rng.integers(0, self._rows)),
                int(self._rng.integers(0, self._cols)),
            ]

            if self._empty_spawn_cell(position):
                orb.spawn(position)
                break

    def _empty_spawn_cell(self, position: list[int]) -> bool:
        # Check against droid
        if position == self.droid.position:
            return False

        # Check against the orbs on the field
        for orb in self.orbs_on_field():
            if position == orb.position:
                return False

        return True
