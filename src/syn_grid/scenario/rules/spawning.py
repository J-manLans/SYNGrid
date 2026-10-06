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


# ################## #
#     Interface      #
# ################## #


class AfterAction(Protocol):
    """What the orb field does once a step's worth of movement is resolved."""

    def apply(self, world: GridWorld) -> None: ...


# ################## #
#     Strategies     #
# ################## #


@dataclass(frozen=True)
class LeaveOrbFieldAlone:
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
class RefillOrbField:
    """Top the field back up to its configured size, by one orb.

    One orb per step, not a loop. A continuous world fills in over time
    rather than instantly, and the rate is part of the difficulty.
    """

    def apply(self, world: GridWorld) -> None:
        if len(world.active_orbs) < world.max_active_orbs:
            world.spawn_orb_if_ready()


# ################## #
#       Rules        #
# ################## #


@dataclass(frozen=True)
class SpawningRules:
    """The scenario's orb-field rules.

    Attributes:
        spawn_max_on_episode_start: spawn the whole orb field according to `max_active_orbs` at
            episode start.
        max_active_orbs: how large the active orb field is allowed to get. In a clean tier chain
            scenario for example, it will be as large as the complete tier chain decided by the
            tier configs `max_tier` parameter, for a continuous scenario its up to the
            max_active_orbs` config parameter. So each scenario decides for itself what'll it be.
        tier_orb_expires: whether tier orbs run down and de-spawn on their timer like direct orbs
            do. Continuous scenarios can choose; a tier chain scenario cannot, because a chain that
            loses a link is not a chain.
        _delay_on_consume: whether consuming an orb starts a cooldown across the whole field.
        _after_action: what happens to the orb field at the end of a step.
    """

    spawn_max_on_episode_start: bool
    max_active_orbs: int
    tier_orb_expires: bool
    _delay_on_consume: int | None
    _after_action: AfterAction

    def on_reset(self, world: GridWorld) -> None:
        if self.spawn_max_on_episode_start:
            for _ in range(world.max_active_orbs):
                world.spawn_orb_if_ready()
        else:
            world.spawn_orb_if_ready()

    def on_orb_consumed(self, world: GridWorld) -> None:
        if self._delay_on_consume is not None:
            world.deactivate_all_orbs(self._delay_on_consume)

    def after_step(self, world: GridWorld) -> None:
        self._after_action.apply(world)
