"""How the droid moves, what each move costs, how it takes in rewards, and how it settles and
resets."""

import pytest

from syn_grid.core.droid.digestion.digesters.tier_chain.threshold_digester import (
    ThresholdDigester,
)
from syn_grid.core.droid.synergy_droid import DroidAction, SynergyDroid
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Constants     #
# ================= #


# Not square, so mixing up rows and columns shows in the walls and the centre
GRID = (3, 5)


CENTRE = [1, 2]


MAX_TIER = 3


BASE_REWARD = 1.0


GROWTH_FACTOR = 1.0


# Tier orbs are worth 1, 2 and 3, so a completed chain pays 6 under threshold scoring
CHAIN_WORTH = 6.0


MAX_ENERGY = 10.0


# Distinct values, so a total shows which pieces are in it
STEP_PENALTY = -0.01


BOUNDARY_PENALTY = -0.3


CHAIN_BREAK_PENALTY = -0.5


# ================= #
#      Helpers      #
# ================= #


def _eat_tiers(droid: SynergyDroid, orbs: list[BaseOrb], *tiers: int) -> float:
    """The droid eats a tier orb of each tier, in the order given; returns what the last one paid."""

    paid = 0.0
    for tier in tiers:
        paid = droid.consume_orb(next(o for o in orbs if o.META.TIER == tier))

    return paid


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def orbs() -> list[BaseOrb]:
    """A three-tier chain worth 1, 2 and 3."""

    return TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=BASE_REWARD,
        growth_factor=GROWTH_FACTOR,
        cool_down=None,
        life_span=None,
    ).create()


@pytest.fixture
def droid() -> SynergyDroid:
    """A droid on a 3x5 grid with threshold scoring, at the start of an episode.

    Threshold scoring holds a running chain's worth, so a droid can be owed something when the
    episode ends.
    """

    droid = SynergyDroid(
        [ThresholdDigester(max_tier=MAX_TIER, chain_break_penalty=CHAIN_BREAK_PENALTY)],
        GRID,
        max_energy=MAX_ENERGY,
        step_penalty=STEP_PENALTY,
        boundary_penalty=BOUNDARY_PENALTY,
    )
    droid.reset()

    return droid


# ================= #
#       Tests       #
# ================= #


class TestTakeStep:
    """Where a move takes the droid, and what it costs."""

    @pytest.mark.parametrize(
        ("action", "cell"),
        [
            (DroidAction.LEFT, [1, 1]),
            (DroidAction.DOWN, [2, 2]),
            (DroidAction.RIGHT, [1, 3]),
            (DroidAction.UP, [0, 2]),
        ],
    )
    def test_a_free_step_moves_one_cell_and_costs_the_step_penalty(
        self, droid: SynergyDroid, action: DroidAction, cell: list[int]
    ):
        cost = droid.take_step(action)

        assert droid.position == cell
        assert cost == STEP_PENALTY
        assert droid.energy == pytest.approx(MAX_ENERGY + STEP_PENALTY)
        assert droid.score == pytest.approx(STEP_PENALTY)

    @pytest.mark.parametrize(
        ("action", "edge"),
        [
            (DroidAction.LEFT, [1, 0]),
            (DroidAction.DOWN, [2, 2]),
            (DroidAction.RIGHT, [1, 4]),
            (DroidAction.UP, [0, 2]),
        ],
    )
    def test_a_step_into_a_wall_stays_on_the_edge_and_costs_both_penalties(
        self, droid: SynergyDroid, action: DroidAction, edge: list[int]
    ):
        droid.position = list(edge)

        cost = droid.take_step(action)

        assert droid.position == edge
        assert cost == pytest.approx(STEP_PENALTY + BOUNDARY_PENALTY)
        assert droid.score == pytest.approx(STEP_PENALTY + BOUNDARY_PENALTY)


class TestTakingInRewards:
    """Every reward goes to the score; energy keeps between 0 and the max energy."""

    def test_eating_takes_in_what_the_orb_pays_and_takes_it_off_the_field(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        droid.energy = 1.0
        last = next(o for o in orbs if o.META.TIER == 3)

        paid = _eat_tiers(droid, orbs, 1, 2, 3)

        assert paid == CHAIN_WORTH
        assert droid.energy == 1.0 + CHAIN_WORTH
        assert droid.score == CHAIN_WORTH
        assert not last.is_active

    def test_at_full_energy_a_reward_raises_the_score_and_leaves_energy_at_the_max(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        _eat_tiers(droid, orbs, 1, 2, 3)

        assert droid.energy == MAX_ENERGY
        assert droid.score == CHAIN_WORTH

    def test_a_loss_never_takes_energy_below_zero_while_the_score_takes_all_of_it(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        droid.energy = 0.2

        droid.take_step(DroidAction.LEFT)
        _eat_tiers(droid, orbs, 2)

        assert droid.energy == 0.0
        assert droid.score == pytest.approx(STEP_PENALTY + CHAIN_BREAK_PENALTY)

    def test_an_orb_eaten_after_a_step_that_drained_the_droid_pays_in_full(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        # Whatever the orb pays goes to energy, whichever digester paid it
        _eat_tiers(droid, orbs, 1, 2)
        droid.energy = 0.005

        droid.take_step(DroidAction.LEFT)
        assert droid.energy == 0.0

        paid = _eat_tiers(droid, orbs, 3)
        assert paid > 0.0
        assert droid.energy == paid


class TestOnTermination:
    """Settling the droid when the episode is terminated."""

    def test_settling_adds_what_is_owed_and_the_terminal_reward_to_the_score_only(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        # A step puts something in the score before settling. Tiers 1 and 2 grow the chain, pay
        # nothing and leave it owed 3
        droid.take_step(DroidAction.LEFT)
        _eat_tiers(droid, orbs, 1, 2)
        droid.energy = 4.0

        settled = droid.on_termination(-0.7)

        assert settled == pytest.approx(3.0 - 0.7)
        assert droid.score == pytest.approx(STEP_PENALTY + 3.0 - 0.7)
        assert droid.energy == 4.0


class TestReset:
    """A new episode."""

    def test_reset_puts_the_droid_back_in_the_centre_at_full_energy_with_its_digesters_cleared(
        self, droid: SynergyDroid, orbs: list[BaseOrb]
    ):
        # Tier 2 first breaks the chain, and tier 1 then starts one
        droid.take_step(DroidAction.UP)
        _eat_tiers(droid, orbs, 2, 1)

        droid.reset()

        digester = droid.get_digester(ThresholdDigester)
        assert droid.position == CENTRE
        assert droid.energy == MAX_ENERGY
        assert droid.score == 0.0
        assert digester.chain_tier == 0
        assert digester.chains_broken == 0
