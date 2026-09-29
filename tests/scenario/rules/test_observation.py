"""Observation rules, and the two counts that are deliberately not the same.

`ObservationRules` carries two numbers. They usually agree and in the spatial
scenario they do not: the observation is sized for `tiers` slots while the
distance sort that fills them still caps at max_tier, so the trailing slots exist
and stay zero forever. That asymmetry is preserved from the old
`_get_observable_orb_count` / `_sort_orbs_by_manhattan_dist_to_droid` pair, which
each computed their own answer from the flags.

These tests pin the numbers a scenario hands out, and then pin the consequence --
that an observation really is wider than the number of orbs that can ever occupy
it -- because the consequence is the thing a reader would otherwise assume is a
bug and "fix".
"""

import pytest
import yaml
from gymnasium import spaces

from syn_grid.config.models import FullConf
from syn_grid.core.grid_world import GridWorld
from syn_grid.gymnasium.observation_space.perceptions.vector import (
    VectorFogOfWar,
    VectorMarkovian,
)
from syn_grid.scenario.registry import build_scenario

CONFIG_DIR = "src/syn_grid/config/scenarios"

# Scenario name -> the config file that selects it.
CONFIG_FOR = {
    "goal_tier_chain_spatial": "tier_chain_spatial",
    "goal_tier_chain_scaling_sparse": "tier_chain_scaling_sparse",
    "continuous": "continuous_step_wise",
}


def _conf(name: str) -> FullConf:
    with open(f"{CONFIG_DIR}/{CONFIG_FOR[name]}.yaml") as f:
        return FullConf(**yaml.safe_load(f))


def _rules(name: str):
    """The observation rules a named scenario ships with."""

    conf = _conf(name)
    return build_scenario(conf.scenario, conf.world, conf.obs).observation


def _perception(cls, conf: FullConf, rules, orbs: int, max_identity: int = 3):
    p = cls(conf.obs.perception, rules, orbs, max_identity)
    p.setup_obs_space()
    p.reset()
    return p


def _world(conf: FullConf) -> GridWorld:
    scenario = build_scenario(conf.scenario, conf.world, conf.obs)
    world = GridWorld(
        scenario,
        conf.world.grid_world_conf,
        conf.world.orb_factory_conf,
        conf.world.droid_conf,
        conf.world.negative_orb_conf,
        conf.world.tier_orb_conf,
    )
    world.reset()
    return world


# ============ #
#    Counts     #
# ============ #


class TestCounts:
    def test_a_goal_scenario_under_curriculum_widens_the_observation(self):
        rules = _rules("goal_tier_chain_spatial")

        assert rules.observation_slot_count == 5

    def test_the_sort_limit_does_not_follow_it(self):
        """The asymmetry, stated plainly. Under curriculum the observation is sized
        wider than the chain, and the sort that fills it has never followed."""

        rules = _rules("goal_tier_chain_spatial")

        assert rules.sort_limit == 3

    def test_without_curriculum_the_two_agree(self):
        rules = _rules("goal_tier_chain_scaling_sparse")

        assert rules.observation_slot_count == rules.sort_limit == 5

    def test_a_continuous_scenario_uses_the_field_size_for_both(self):
        rules = _rules("continuous")

        assert rules.observation_slot_count == rules.sort_limit == 3

    def test_the_tier_channel_bound_comes_from_the_world(self):
        rules = _rules("goal_tier_chain_scaling_sparse")

        assert rules.max_tier == 5


# ============ #
#  Consequence  #
# ============ #


class TestConsequence:
    def test_the_observation_is_wider_than_the_world_can_ever_fill(self):
        """The spatial scenario's observation reserves five orb slots for a
        three-orb chain. Two slots are dead weight, permanently. Pinned so it is
        not later read as a bug."""

        rules = _rules("goal_tier_chain_spatial")
        world = _world(_conf("goal_tier_chain_spatial"))

        assert rules.observation_slot_count == 5
        assert len(world.ALL_ORBS) == 3
        assert rules.observation_slot_count > len(world.ALL_ORBS)

    def test_the_dead_slots_are_actually_zero(self):
        conf = _conf("goal_tier_chain_spatial")
        rules = _rules("goal_tier_chain_spatial")
        world = _world(conf)

        perception = _perception(
            VectorMarkovian, conf, rules, len(world.ALL_ORBS), world.max_identity
        )
        obs = perception.get_observation(world, 10)

        stride = len(perception._get_orb_values(world.active_orbs[0]))
        start = 2  # after the droid's row and col
        slots = [
            obs[start + i * stride : start + (i + 1) * stride]
            for i in range(rules.observation_slot_count)
        ]

        # Occupancy is the active flag, not the row: an orb sitting on row 0 is
        # indistinguishable from an empty slot by position alone.
        occupied = [s for s in slots if s[0] == 1.0]

        assert len(occupied) == len(world.active_orbs)
        assert sorted(float(s[1]) for s in occupied) == sorted(
            float(o.position[0]) for o in world.active_orbs
        )
        assert len(slots) - len(occupied) == rules.observation_slot_count - len(
            world.active_orbs
        )

    def test_padding_the_observation_does_not_change_the_reward(self):
        """Padding is free. Confirmed because it is the reason the asymmetry is
        tolerable at all: the policy sees zeros, not noise."""

        narrow_conf = _conf("continuous")
        wide_conf = _conf("goal_tier_chain_spatial")

        narrow = _perception(VectorMarkovian, narrow_conf, _rules("continuous"), 3, 3)
        wide = _perception(
            VectorMarkovian, wide_conf, _rules("goal_tier_chain_spatial"), 3, 3
        )

        assert narrow.setup_obs_space().shape != wide.setup_obs_space().shape

    def test_fog_of_war_only_reports_adjacent_orbs(self):
        """The spatial scenario's partial observability. Orbs further than one cell
        away in either axis are not written into the observation at all."""

        conf = _conf("goal_tier_chain_spatial")
        rules = _rules("goal_tier_chain_spatial")
        world = _world(conf)

        perception = _perception(
            VectorFogOfWar, conf, rules, len(world.ALL_ORBS), world.max_identity
        )
        world.droid.position = [0, 0]
        obs = perception.get_observation(world, 10)

        dy, dx = world.droid.position
        visible = [
            o
            for o in world.ALL_ORBS
            if o.is_active
            and max(abs(o.position[0] - dy), abs(o.position[1] - dx)) <= 1
        ]
        stride = len(perception._get_orb_values(world.active_orbs[0]))
        reported = sum(
            1 for i in range(rules.observation_slot_count) if obs[2 + i * stride] == 1.0
        )

        assert reported == len(visible)


# ============ #
#    Spaces     #
# ============ #


class TestSpaces:
    @pytest.mark.parametrize("perception", [VectorMarkovian, VectorFogOfWar])
    def test_the_declared_space_matches_the_scenarios_slot_count(self, perception):
        conf = _conf("goal_tier_chain_spatial")
        rules = _rules("goal_tier_chain_spatial")
        world = _world(conf)

        p = _perception(
            perception, conf, rules, len(world.ALL_ORBS), world.max_identity
        )
        space = p.setup_obs_space()

        stride = len(p._get_orb_values(world.active_orbs[0]))
        assert isinstance(space, spaces.Box)
        assert space.shape == (2 + stride * rules.observation_slot_count,)

    def test_the_observation_satisfies_its_own_declared_space(self):
        """A scenario that hands the perception a slot count the Box does not match
        would train on an observation the space rejects."""

        conf = _conf("goal_tier_chain_spatial")
        rules = _rules("goal_tier_chain_spatial")
        world = _world(conf)

        p = _perception(
            VectorMarkovian, conf, rules, len(world.ALL_ORBS), world.max_identity
        )
        space = p.setup_obs_space()
        obs = p.get_observation(world, 10)

        assert space.contains(obs)

    def test_a_perception_no_longer_needs_scenario_flags_to_size_itself(self):
        """The config block that used to carry these counts is gone. The same config
        with different rules must give a different width, which is the whole claim
        that shape follows from the scenario."""

        conf = _conf("goal_tier_chain_spatial")
        for gone in (
            "single_chain_mode",
            "curriculum_training",
            "max_active_orbs",
            "max_tier",
        ):
            assert not hasattr(conf.obs.perception, gone)

        narrow = _perception(
            VectorMarkovian, conf, _rules("continuous"), 3, world_identity := 3
        )
        wide = _perception(
            VectorMarkovian, conf, _rules("goal_tier_chain_spatial"), 3, world_identity
        )

        assert narrow.setup_obs_space().shape != wide.setup_obs_space().shape
