"""What a tier chain population makes: one tier orb per tier, worth its tier, with its timing."""

from syn_grid.core.orbs.orb_meta import SynergyKind
from syn_grid.scenario.blocks.populations.tier_chain_population import (
    TierChainPopulation,
)

# ================= #
#     Constants     #
# ================= #


MAX_TIER = 3


# ================= #
#      Helpers      #
# ================= #


def _population(
    *,
    base_reward: float = 1.0,
    growth_factor: float = 1.0,
    cool_down: int | None = None,
    life_span: int | None = None,
) -> TierChainPopulation:
    return TierChainPopulation(
        max_tier=MAX_TIER,
        base_reward=base_reward,
        growth_factor=growth_factor,
        cool_down=cool_down,
        life_span=life_span,
    )


# ================= #
#       Tests       #
# ================= #


class TestCount:
    """How many orbs it makes, known before it makes them."""

    def test_it_is_one_tier_orb_for_each_tier_up_to_the_max(self):
        population = _population()

        orbs = population.create()

        assert population.count == len(orbs) == MAX_TIER
        assert all(orb.META.KIND == SynergyKind.TIER for orb in orbs)
        assert sorted(orb.META.TIER or 0 for orb in orbs) == [1, 2, 3]


class TestCreate:
    """The orbs each call makes."""

    def test_each_tier_is_worth_the_base_reward_times_the_tier_to_the_growth_factor(
        self,
    ):
        orbs = _population(base_reward=10.0, growth_factor=2.0).create()

        # 10 * 1 ** 2, 10 * 2 ** 2 and 10 * 3 ** 2
        assert {orb.META.TIER: orb.REWARD for orb in orbs} == {
            1: 10.0,
            2: 40.0,
            3: 90.0,
        }

    def test_it_hands_each_orb_its_lifespan_and_cool_down(self):
        # An orb does not show its lifespan or cool-down, so they are read from what it does, as the
        # world would: it expires on its second step, is removed, and is ready after one more. The
        # two values differ, so a swap shows
        orbs = _population(cool_down=1, life_span=2).create()

        for orb in orbs:
            orb.spawn([0, 0])
            orb.TIMER.tick()
            assert not orb.has_expired()
            orb.TIMER.tick()
            assert orb.has_expired()

            orb.de_spawn()
            assert not orb.is_ready()
            orb.TIMER.tick()
            assert orb.is_ready()

    def test_each_call_makes_new_orbs(self):
        population = _population()

        first, second = population.create(), population.create()

        # id() is the object, not the orb's identity: no orb object is in both lists
        assert not {id(orb) for orb in first} & {id(orb) for orb in second}
