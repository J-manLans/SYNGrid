"""What threshold scoring pays: a chain holds its worth while it grows, and pays it when it ends."""

from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from syn_grid.core.droid.digestion.digesters.tier_chain.step_wise_digester import (
    StepWiseDigester,
)
from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)

# ================= #
#     Constants     #
# ================= #


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def digester(max_tier: int) -> ThresholdDigester:
    return ThresholdDigester(max_tier=max_tier, chain_break_penalty=CHAIN_BREAK_PENALTY)


# ================= #
#       Tests       #
# ================= #


class TestProgress:
    """The next tier, not the last."""

    def test_an_orb_that_grows_the_chain_pays_nothing_and_is_held(
        self, digester: ThresholdDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2)

        assert paid == [0.0, 0.0]
        assert digester.undigested() == 10.0 + 20.0


class TestCompletion:
    """The next tier, and the last."""

    def test_the_orb_that_completes_the_chain_pays_its_full_worth(
        self, digester: ThresholdDigester, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2, 3)

        assert paid[-1] == 10.0 + 20.0 + 30.0
        assert digester.undigested() == 0.0


class TestBreak:
    """A tier orb eaten out of order."""

    def test_a_tier_1_that_breaks_a_chain_pays_the_penalty_and_what_the_chain_held(
        self, digester: ThresholdDigester, eat: Callable[..., list[float]]
    ):
        # The tier 1 that breaks is not held, so nothing is owed afterwards
        paid = eat(digester, 1, 2, 1)

        assert paid[-1] == CHAIN_BREAK_PENALTY + 10.0 + 20.0
        assert digester.undigested() == 0.0


class TestSameTotalAsStepWise:
    """Threshold pays what step-wise pays, only later."""

    # The fixtures are built once for all the examples, which is safe: digesting never changes
    # an orb, and each example builds its own digesters
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(data=st.data())
    def test_any_sequence_pays_the_same_total_once_what_is_held_is_paid(
        self, max_tier: int, eat: Callable[..., list[float]], data: st.DataObject
    ):
        tiers = data.draw(
            st.lists(st.integers(min_value=1, max_value=max_tier), max_size=12)
        )
        threshold = ThresholdDigester(max_tier, CHAIN_BREAK_PENALTY)
        step_wise = StepWiseDigester(max_tier, CHAIN_BREAK_PENALTY)

        threshold_total = sum(eat(threshold, *tiers)) + threshold.undigested()
        step_wise_total = sum(eat(step_wise, *tiers))

        assert threshold_total == pytest.approx(step_wise_total)
