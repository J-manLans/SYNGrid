from dataclasses import dataclass
from typing import Final

from numpy.random import Generator

from syn_grid.core.droid.synergy_droid import DroidAction, SynergyDroid
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import OrbMeta


@dataclass(frozen=True)
class StepResult:
    """What happened in one tick of the world.

    Attributes:
        reward: the step's cost plus what the eaten orb paid, already taken in by the droid.
        consumed: the meta of the orb the droid ate this step, or None. The droid eats at most one
            orb a step, since no two orbs share a cell.
    """

    reward: float
    consumed: OrbMeta | None


class GridWorld:
    """
    The grid an episode is played on. It runs the orb field and the step clock around the droid it is handed.

    The world has one rule for every scenario. Orbs leave the field when the droid eats them or their lifespan runs
    out, and one orb that is ready returns each step if the field has room. A scenario shapes the field through the
    orbs it hands over, and states only how many may be on the field at once.
    """

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

    def perform_droid_action(self, agent_action: DroidAction) -> StepResult:
        """
        Run one tick of the world, set off by one droid action.

        The droid takes its step, every orb's timer ticks, the droid eats the orb it stands on, an orb whose lifespan
        ran out leaves the field, and one ready orb returns if the field has room. The droid is checked first, so it
        wins an orb on the last tick of its life. Returns the step's cost plus what the eaten orb paid, and the eaten
        orb's meta.
        """

        self._steps_left -= 1

        reward = 0.0
        consumed: OrbMeta | None = None
        step_penalty = self.droid.take_step(agent_action)

        for orb in self.ALL_ORBS:
            orb.TIMER.tick()

            # The contact check comes first, so the droid wins an orb on its last tick of life
            if orb.is_active and self.droid.position == orb.position:
                reward = self.droid.consume_orb(orb)
                consumed = orb.META
            elif orb.has_expired():
                orb.de_spawn()

        # Bring back one ready orb if the field has room
        if len(self.orbs_on_field()) < self._max_active_orbs:
            self._spawn_random_orb_if_ready()

        return StepResult(step_penalty + reward, consumed)

    # === Getters === #

    @property
    def grid_dimensions(self) -> tuple[int, int]:
        """The grid's size as (rows, cols)."""
        return self._rows, self._cols

    @property
    def steps_left(self) -> int:
        """How many steps the episode has left. Each action takes one, and a reset restarts it at `max_steps`."""

        return self._steps_left

    def orbs_on_field(self) -> list[BaseOrb]:
        """The orbs that are on the field now, which are the ones that can be seen and eaten."""

        return [o for o in self.ALL_ORBS if o.is_active]

    def is_cell_free(self, position: list[int]) -> bool:
        """Whether a cell is free of the droid and of every orb on the field, so an orb can be
        placed in it."""

        # Check against droid
        if position == self.droid.position:
            return False

        # Check against the orbs on the field
        for orb in self.orbs_on_field():
            if position == orb.position:
                return False

        return True

    # ================= #
    #      Helpers      #
    # ================= #

    # === Init === #

    # TODO(phase 7): the remap and `max_identity` move into a table the scenario builds, in 7a (see
    # "An orb's identity is looked up in a table the scenario builds" in docs/dev/decisions/observation.md).
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

    # === Global === #

    def _spawn_random_orb_if_ready(self) -> None:
        """Put one random ready orb in a random free cell. Does nothing when no orb is ready."""

        ready_orbs = [o for o in self.ALL_ORBS if o.is_ready()]
        if not ready_orbs:
            return

        orb = self._rng.choice(ready_orbs)  # type: ignore[arg-type]

        while True:
            position = [
                int(self._rng.integers(0, self._rows)),
                int(self._rng.integers(0, self._cols)),
            ]

            if self.is_cell_free(position):
                orb.spawn(position)
                break
