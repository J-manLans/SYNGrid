"""What max-tier scoring pays: a chain pays its whole worth when it is completed, and nothing before."""

from collections.abc import Callable

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.max_tier_digester import (
    MaxTierDigester,
)

# ================= #
#     Constants     #
# ================= #


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def digester(max_tier: int) -> MaxTierDigester:
    return MaxTierDigester(max_tier=max_tier, chain_break_penalty=CHAIN_BREAK_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestProgress:
    """The next tier, not the last."""

    def test_an_orb_that_grows_the_chain_pays_nothing(
        self, digester: MaxTierDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2)

        assert paid == [0.0, 0.0]


class TestCompletion:
    """The next tier, and the last."""

    def test_the_orb_that_completes_the_chain_pays_its_full_worth(
        self, digester: MaxTierDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2, 3)

        assert paid[-1] == 10.0 + 20.0 + 30.0


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_break_pays_the_penalty_and_what_the_chain_gathered_is_lost(
        self, digester: MaxTierDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2, 1)

        assert paid[-1] == CHAIN_BREAK_PENALTY
        assert digester.undigested() == 0.0


class TestUndigested:
    """What a running chain is owed when the episode ends."""

    def test_a_running_chain_is_owed_nothing(
        self, digester: MaxTierDigester, eat: Callable[..., list[float]]
    ):
        eat(digester, 1, 2)

        assert digester.undigested() == 0.0
