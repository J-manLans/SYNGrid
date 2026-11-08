"""What step-wise scoring pays: every orb eaten in order pays its own reward when it is eaten."""

from collections.abc import Callable

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)

# ================= #
#     Constants     #
# ================= #


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def digester(max_tier: int) -> StepWiseDigester:
    return StepWiseDigester(max_tier=max_tier, chain_break_penalty=CHAIN_BREAK_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestProgress:
    """The next tier, not the last."""

    def test_an_orb_that_grows_the_chain_pays_its_reward(
        self, digester: StepWiseDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2)

        assert paid == [10.0, 20.0]


class TestCompletion:
    """The next tier, and the last."""

    def test_a_completed_chain_pays_its_full_worth_one_orb_at_a_time(
        self, digester: StepWiseDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2, 3)

        assert paid == [10.0, 20.0, 30.0]

    def test_after_a_completed_chain_a_tier_1_starts_a_new_one(
        self, digester: StepWiseDigester, eat: Callable[..., list[float]]
    ):
        # A completion pays like a progress here, so only the next orb shows the chain was over
        paid = eat(digester, 1, 2, 3, 1)

        assert paid[-1] == 10.0


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_tier_1_that_breaks_a_chain_pays_the_penalty_alone(
        self, digester: StepWiseDigester, eat: Callable[..., list[float]]
    ):
        # Tiers 1 and 2 were paid as they were eaten; the tier 1 that breaks is not
        paid = eat(digester, 1, 2, 1)

        assert paid == [10.0, 20.0, CHAIN_BREAK_PENALTY]


class TestUndigested:
    """What a running chain is owed when the episode ends."""

    def test_a_running_chain_is_owed_nothing_since_its_orbs_were_paid(
        self, digester: StepWiseDigester, eat: Callable[..., list[float]]
    ):
        eat(digester, 1, 2)

        assert digester.undigested() == 0.0
