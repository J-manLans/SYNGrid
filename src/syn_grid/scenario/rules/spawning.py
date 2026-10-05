"""
How a scenario drives the orb field over the course of an episode.

Three independent decisions live here, and they are genuinely independent:
whether the pool starts full, whether consuming an orb starts a cooldown, and
what happens to the field once a step is done. A tier chain with delay
reactivates the orbs it just put on cooldown; a continuous world instead tops
the field back up. So these compose rather than multiply.

The world still owns the mechanics -- spawn, cooldown, reactivate. A rule only
decides which of them to invoke and when.

The three strategies are stateless and are frozen dataclasses so that
``SpawningRules`` compares by value. Without that, two rules built for the same
world compared unequal purely because they held different strategy instances,
which makes a scenario impossible to assert about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from syn_grid.core.grid_world import GridWorld


class AfterAction(Protocol):
    """What the orb field does once a step's worth of movement is resolved."""

    def apply(self, world: GridWorld) -> None: ...


@dataclass(frozen=True)
class LeaveFieldAlone:
    """Nothing. A tier chain without delay simply waits for the next action."""

    def apply(self, world: GridWorld) -> None:
        return None


@dataclass(frozen=True)
class ReactivateAllOrbs:
    """Bring back every orb whose cooldown has run out.

    This is the delay scenario's resolution step: consuming an orb put the
    whole field on cooldown, and once the cooldown lapses the same orbs
    return to the positions they were consumed from.
    """

    def apply(self, world: GridWorld) -> None:
        world.reactivate_all_orbs()


@dataclass(frozen=True)
class RefillOrbPool:
    """Top the field back up to its configured size, by one orb.

    One orb per step, not a loop. A continuous world fills in over time
    rather than instantly, and the rate is part of the difficulty.
    """

    def apply(self, world: GridWorld) -> None:
        if len(world.active_orbs) < world.max_active_orbs:
            world.spawn_orb_if_ready()


@dataclass(frozen=True)
class SpawningRules:
    """The scenario's orb-field rules.

    Attributes:
        fill_pool_on_reset: spawn the whole field at reset (a tier chain's
            sequence is all present at once) versus a single orb.
        max_active_orbs: how large the field is allowed to get. A tier chain's
            is its length, so this is the scenario's answer and not a value the
            world config gets to contradict.
        tier_orb_expires: whether tier orbs run down and de-spawn on their
            timer like direct orbs do. Continuous worlds can choose; a tier
            chain cannot, because a chain that loses a link is not a chain.
        delay_on_consume: whether consuming an orb starts a cooldown across the
            whole field.
        after_action: what the field does at the end of a step.
    """

    fill_pool_on_reset: bool
    max_active_orbs: int
    tier_orb_expires: bool
    delay_on_consume: int | None
    after_action: AfterAction

    def on_reset(self, world: GridWorld) -> None:
        if self.fill_pool_on_reset:
            for _ in range(world.max_active_orbs):
                world.spawn_orb_if_ready()
        else:
            world.spawn_orb_if_ready()

    def on_orb_consumed(self, world: GridWorld) -> None:
        if self.delay_on_consume is not None:
            world.deactivate_all_orbs()

    def after_step(self, world: GridWorld) -> None:
        self.after_action.apply(world)
