"""What a negative population makes: as many negative orbs as it is told, with its reward and timing."""

import pytest

from syn_grid.core.orbs.orb_meta import DirectKind
from syn_grid.scenario.blocks.populations.neg_population import NegPopulation

# ================= #
#     Constants     #
# ================= #


REWARD = -2.0


# ================= #
#      Helpers      #
# ================= #


def _population(
    *, orbs: int = 1, cool_down: int = 1, life_span: int = 1
) -> NegPopulation:
    return NegPopulation(
        orbs=orbs, reward=REWARD, cool_down=cool_down, life_span=life_span
    )


# ================= #
#       Tests       #
# ================= #


class TestCount:
    """How many orbs it makes, known before it makes them."""

    def test_it_makes_as_many_negative_orbs_as_it_is_told_each_with_its_reward(self):
        population = _population(orbs=3)

        orbs = population.create()

        assert population.count == len(orbs) == 3
        assert all(orb.META.KIND == DirectKind.NEGATIVE for orb in orbs)
        assert all(orb.REWARD == REWARD for orb in orbs)
        # id() is the object, not the orb's identity: three separate orbs, not one listed three times
        assert len({id(orb) for orb in orbs}) == 3

    def test_fewer_than_1_is_refused_and_1_is_allowed(self):
        assert _population(orbs=1).count == 1

        with pytest.raises(ValueError):
            _population(orbs=0)


class TestCreate:
    """The orbs each call makes."""

    def test_it_hands_each_orb_its_lifespan_and_cool_down(self):
        # An orb does not show its lifespan or cool-down, so they are read from what it does, as the
        # world would: it expires on its second step, is removed, and is ready after one more. The
        # two values differ, so a swap shows
        orbs = _population(orbs=2, cool_down=1, life_span=2).create()

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

        assert population.create()[0] is not population.create()[0]
