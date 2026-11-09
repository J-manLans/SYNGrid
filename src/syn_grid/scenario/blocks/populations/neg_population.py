"""
How a fixed field gets its negative orbs.

This is the implementation of ``OrbPopulation`` for negative orbs, shared by
every scenario family with a fixed field. How many a field holds is the
family's choice, which its builder passes in.
"""

from __future__ import annotations

from dataclasses import dataclass

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.fixed_reward_orb import FixedRewardOrb
from syn_grid.core.orbs.orb_meta import DirectKind

# ################### #
# Population Strategy #
# ################### #


@dataclass(frozen=True)
class NegPopulation:
    """A number of negative orbs, all alike.

    Each orb leaves the field when it is eaten or when its lifespan runs out,
    and returns elsewhere after its cool-down. Frozen so two scenarios built
    from the same config hold equal populations.

    Attributes:
        orbs: how many negative orbs, at least 1. A bundle with no orbs would
            give the world a digester with nothing to digest.
        reward: what eating an orb pays, 0 or negative.
        cool_down: the number of steps an orb is off the field before it can
            appear again.
        life_span: the number of steps an orb stays on the field before it
            leaves.
    """

    orbs: int
    reward: float
    cool_down: int
    life_span: int

    def __post_init__(self) -> None:
        if self.orbs < 1:
            raise ValueError("A negative population needs at least 1 orb")

    @property
    def count(self) -> int:
        """The number of orbs, `orbs`."""

        return self.orbs

    def create(self) -> list[BaseOrb]:
        """Return `orbs` negative orbs, each with this population's reward, cool-down and lifespan.

        They come back off the field, and the world places them.
        """

        return [
            FixedRewardOrb(
                DirectKind.NEGATIVE, self.reward, self.cool_down, self.life_span
            )
            for _ in range(self.orbs)
        ]
