"""
How a fixed field gets its negative orb.

A field that is not sampled holds exactly one negative orb. This is the
implementation of ``OrbPopulation`` for it, shared by every scenario family
with a fixed field.
"""

from __future__ import annotations

from dataclasses import dataclass

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb

# ################### #
# Population Strategy #
# ################### #


@dataclass(frozen=True)
class NegPopulation:
    """One negative orb.

    The orb leaves the field when it is eaten or when its lifespan runs out,
    and returns elsewhere after its cool-down. Frozen so it compares by value.

    Attributes:
        reward: what eating the orb pays, 0 or negative.
        cool_down: the number of steps the orb is off the field before it can
            appear again.
    """

    reward: float
    cool_down: int

    # TODO(phase 3): hand the orb its lifespan here once `BaseOrb` takes one per orb. It follows
    # from the grid size, so the builder has to pass it in.

    def create(self) -> list[BaseOrb]:
        return [NegativeOrb(self.reward, self.cool_down)]
