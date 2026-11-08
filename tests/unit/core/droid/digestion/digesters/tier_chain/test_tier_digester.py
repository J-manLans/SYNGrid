"""How the tier digester moves the chain, counts what happens to it, and which payout it asks for."""

from collections.abc import Callable
from typing import override

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.direct.negative_orb import NegativeOrb

# ================= #
#     Constants     #
# ================= #


# A different payout from each hook, so what digest returns shows which hook it called. None
# is 0, since a hook that paid 0 would look the same as a payout that was never asked for
PROGRESS_PAY = 1


COMPLETION_PAY = 2


BREAK_PAY = 3


# ================= #
#    Stand-ins      #
# ================= #


class _Recorder(TierDigester):
    """A tier digester that pays a different amount from each hook, and records the chain each
    hook saw: the hook's name, the chain's tier and the chain's worth."""

    def __init__(self, max_tier: int):
        super().__init__(max_tier)
        self.seen: list[tuple[str, int, float]] = []

    @override
    def undigested(self) -> float:
        return 0.0

    @override
    def _score_progress(self, orb: BaseOrb) -> float:
        self.seen.append(("progress", self.chain_tier, self.chain_worth))
        return PROGRESS_PAY

    @override
    def _score_break(self, orb: BaseOrb) -> float:
        self.seen.append(("break", self.chain_tier, self.chain_worth))
        return BREAK_PAY

    @override
    def _score_completion(self, orb: BaseOrb) -> float:
        self.seen.append(("completion", self.chain_tier, self.chain_worth))
        return COMPLETION_PAY


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def digester(max_tier: int) -> _Recorder:
    return _Recorder(max_tier)


# ================= #
#       Tests       #
# ================= #


class TestProgress:
    """The next tier, not the last."""

    def test_the_next_tier_grows_the_chain_adds_its_reward_and_pays_progress(
        self, digester: _Recorder, eat: Callable[..., list[float]]
    ):
        # Tier 1 with no chain running is an ordinary first step, not a break
        paid = eat(digester, 1, 2)

        assert paid == [PROGRESS_PAY, PROGRESS_PAY]
        assert digester.chain_tier == 2
        assert digester.chain_worth == 10.0 + 20.0
        assert digester.chain_progresses == 2
        assert digester.chains_broken == 0


class TestCompletion:
    """The next tier, and the last."""

    def test_the_last_tier_in_order_completes_clears_the_chain_and_pays_completion(
        self, digester: _Recorder, eat: Callable[..., list[float]]
    ):
        paid = eat(digester, 1, 2, 3)

        assert paid[-1] == COMPLETION_PAY
        assert digester.chains_completed == 1
        assert digester.chain_tier == 0
        assert digester.chain_worth == 0.0
        # A completion is not a progress
        assert digester.chain_progresses == 2


class TestBreak:
    """Any other tier."""

    @pytest.mark.parametrize(
        "tiers",
        [
            pytest.param((1, 2, 2), id="the same tier twice"),
            pytest.param((2,), id="a tier above 1 with no chain"),
            pytest.param((1, 3), id="a skipped tier"),
            pytest.param((1, 2, 1), id="a tier 1 mid-chain"),
        ],
    )
    def test_a_tier_out_of_order_breaks_clears_the_chain_and_pays_the_break(
        self,
        digester: _Recorder,
        eat: Callable[..., list[float]],
        tiers: tuple[int, ...],
    ):
        paid = eat(digester, *tiers)

        assert paid[-1] == BREAK_PAY
        assert digester.chains_broken == 1
        assert digester.chain_tier == 0
        assert digester.chain_worth == 0.0


class TestCounters:
    """What the episode's counters add up to."""

    def test_each_tier_orb_counts_once_so_the_counters_add_up_to_the_orbs_eaten(
        self, digester: _Recorder, eat: Callable[..., list[float]]
    ):
        # Two completed chains, then two breaks
        tiers = (1, 2, 3, 1, 2, 3, 2, 2)

        eat(digester, *tiers)

        assert digester.chain_progresses == 4
        assert digester.chains_completed == 2
        assert digester.chains_broken == 2


class TestHooks:
    """What each payout hook is shown."""

    def test_each_hook_sees_the_chain_as_it_was_before_the_orb(
        self, digester: _Recorder, eat: Callable[..., list[float]]
    ):
        # A chain broken by a tier 1, then a new chain run to completion
        eat(digester, 1, 2, 1, 1, 2, 3)

        assert digester.seen == [
            ("progress", 0, 0.0),
            ("progress", 1, 10.0),
            ("break", 2, 30.0),
            ("progress", 0, 0.0),
            ("progress", 1, 10.0),
            ("completion", 2, 30.0),
        ]


class TestRejection:
    """An orb the digester cannot digest."""

    def test_an_orb_with_no_tier_is_rejected(self, digester: _Recorder):
        with pytest.raises(ValueError, match="tier"):
            digester.digest(NegativeOrb(penalty=-1.0, cool_down=0, life_span=1))


class TestReset:
    """A new episode."""

    def test_reset_clears_the_chain_and_the_counters(
        self, digester: _Recorder, eat: Callable[..., list[float]]
    ):
        # A completion, then a break, then a chain left running at tier 1
        eat(digester, 1, 2, 3, 2, 1)

        digester.reset()

        assert digester.chain_tier == 0
        assert digester.chain_worth == 0.0
        assert digester.chain_progresses == 0
        assert digester.chains_broken == 0
        assert digester.chains_completed == 0
