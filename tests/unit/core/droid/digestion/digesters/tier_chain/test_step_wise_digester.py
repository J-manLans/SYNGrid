"""What step-wise scoring pays: every orb eaten in order pays its own reward when it is eaten."""

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


class TestProgress:
    """The next tier, not the last."""

    def test_an_orb_that_grows_the_chain_pays_its_reward(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        paid = _eat(digester, orbs, 1, 2)

        assert paid == [10.0, 20.0]


class TestCompletion:
    """The next tier, and the last."""

    def test_a_completed_chain_pays_its_full_worth_one_orb_at_a_time(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        paid = _eat(digester, orbs, 1, 2, 3)

        assert paid == [10.0, 20.0, 30.0]

    def test_after_a_completed_chain_a_tier_1_starts_a_new_one(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        # A completion pays like a progress here, so only the next orb shows the chain was over
        paid = _eat(digester, orbs, 1, 2, 3, 1)

        assert paid[-1] == 10.0


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_tier_1_that_breaks_a_chain_pays_the_penalty_alone(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        # Tiers 1 and 2 were paid as they were eaten; the tier 1 that breaks is not
        paid = _eat(digester, orbs, 1, 2, 1)

        assert paid == [10.0, 20.0, CHAIN_BREAK_PENALTY]


class TestUndigested:
    """What a running chain is owed when the episode ends."""

    def test_a_running_chain_is_owed_nothing_since_its_orbs_were_paid(
        self, digester: StepWiseDigester, orbs: list[BaseOrb]
    ):
        _eat(digester, orbs, 1, 2)

        assert digester.undigested() == 0.0
