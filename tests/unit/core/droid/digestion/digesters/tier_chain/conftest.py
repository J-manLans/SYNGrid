"""The tier chain the digester tests eat from: three tiers, worth 10, 20 and 30.

A digester fixture takes `max_tier` from here, so the digester and the chain cannot disagree on
how long the chain is.
"""

from collections.abc import Callable

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def max_tier() -> int:
    return 3


@pytest.fixture
def orbs(max_tier: int) -> list[BaseOrb]:
    """One tier orb of each tier, worth 10, 20 and 30."""

    return TierChainPopulation(
        max_tier=max_tier,
        base_reward=10.0,
        growth_factor=1.0,
        cool_down=None,
        life_span=None,
    ).create()


@pytest.fixture
def eat(orbs: list[BaseOrb]) -> Callable[..., list[float]]:
    """`eat(digester, *tiers)` digests a tier orb of each tier, in the order given, and returns
    what each paid."""

    def _eat(digester: TierDigester, *tiers: int) -> list[float]:
        return [
            digester.digest(next(o for o in orbs if o.META.TIER == tier))
            for tier in tiers
        ]

    return _eat
