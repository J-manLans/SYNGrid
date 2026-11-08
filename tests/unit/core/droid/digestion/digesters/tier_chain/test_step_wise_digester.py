"""What a break pays under step-wise scoring."""

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 3


# Tier orbs are worth 10, 20 and 30
BASE_REWARD = 10.0


GROWTH_FACTOR = 1.0


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#      Helpers      #
# ================= #


def _eat(digester: StepWiseDigester, orbs: list[BaseOrb], *tiers: int) -> list[float]:
    """Digest a tier orb of each tier, in the order given; returns what each paid."""

    return [
        digester.digest(next(o for o in orbs if o.META.TIER == tier)) for tier in tiers
    ]


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def orbs() -> list[BaseOrb]:
    return TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=BASE_REWARD,
        growth_factor=GROWTH_FACTOR,
        cool_down=None,
        life_span=None,
    ).create()


@pytest.fixture
def digester() -> StepWiseDigester:
    return StepWiseDigester(max_tier=MAX_TIER, chain_break_penalty=CHAIN_BREAK_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_tier_1_that_breaks_a_chain_pays_the_penalty_alone(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        # Tiers 1 and 2 were paid as they were eaten; the tier 1 that breaks is not
        paid = _eat(digester, orbs, 1, 2, 1)

        assert paid == [10.0, 20.0, CHAIN_BREAK_PENALTY]
