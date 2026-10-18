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
    and returns elsewhere after its cool-down. Frozen so two scenarios built
    from the same config hold equal populations.

    Attributes:
        reward: what eating the orb pays, 0 or negative.
        cool_down: the number of steps the orb is off the field before it can
            appear again.
        life_span: the number of steps the orb stays on the field before it
            leaves.
    """

    reward: float
    cool_down: int
    life_span: int

    @property
    def count(self) -> int:
        """Always 1: a fixed field holds exactly one negative orb, and how many is not config."""

        return 1

    def create(self) -> list[BaseOrb]:
        """Return the negative orb, with this population's reward, cool-down and lifespan.

        It comes back off the field, and the world places it.
        """

        return [NegativeOrb(self.reward, self.cool_down, self.life_span)]
