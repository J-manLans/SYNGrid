"""Scenario selection: a name picks the rules, and the rules police their config.

The old schema had no name. A scenario was a combination of booleans, which
meant the only way to ask "what kind of world is this?" was to re-derive it, and
the only way to catch an impossible combination was a conditional inside a
config model. These tests cover the two things that replaced that: the name
resolves, and an impossible parameter set is refused by whoever owns it.
"""

import pytest
import yaml

from syn_grid.config.models import FullConf, ScoringMode
from syn_grid.scenario.registry import SCENARIOS, build_scenario
from syn_grid.scenario.rules.population import (
    TierChainPopulation,
    WeightedPopulation,
)
from syn_grid.scenario.rules.spawning import (
    LeaveFieldAlone,
    ReactivateAllOrbs,
    RefillOrbPool,
)
from syn_grid.scenario.rules.termination import (
    ContinuousTermination,
    GoalTermination,
)
from syn_grid.scenario.scenario import ScenarioType

CONFIG_DIR = "src/syn_grid/config/scenarios"

# Registered scenario name -> the config file that selects it. Not the same
# thing: the perception variants and the branch-coverage configs all select a
# scenario that already has a canonical file of its own.
CANONICAL_CONFIG = {
    "goal_tier_chain_spatial": "tier_chain_spatial",
    "goal_tier_chain_delay": "tier_chain_delay",
    "goal_tier_chain_scaling_dense": "tier_chain_scaling_dense",
    "goal_tier_chain_scaling_sparse": "tier_chain_scaling_sparse",
    "continuous": "continuous_step_wise",
    "continuous_delay": "continuous_delay",
}


def _load(name: str) -> FullConf:
    with open(f"{CONFIG_DIR}/{name}.yaml") as f:
        return FullConf(**yaml.safe_load(f))


# ============ #
#   Selection  #
# ============ #


class TestSelection:
    def test_the_name_in_the_config_is_the_scenario_that_gets_built(self):
        conf = _load("tier_chain_spatial")

        scenario = build_scenario(conf.scenario, conf.world, conf.obs)

        assert scenario.name == conf.scenario == "goal_tier_chain_spatial"
        assert scenario.type is ScenarioType.GOAL

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_registered_scenario_builds_from_its_own_config(self, name: str):
        """Each registered name has a config that selects it, and that config is
        accepted. A registered scenario nobody can configure is a dead entry."""

        assert name in CANONICAL_CONFIG, f"no canonical config registered for {name}"

        conf = _load(CANONICAL_CONFIG[name])

        assert conf.scenario == name
        assert build_scenario(name, conf.world, conf.obs).name == name

    def test_an_unknown_scenario_name_says_what_is_available(self):
        conf = _load("tier_chain_spatial")

        with pytest.raises(KeyError) as exc:
            build_scenario("tier_chain_martian", conf.world, conf.obs)

        message = str(exc.value)
        assert "tier_chain_martian" in message
        assert "goal_tier_chain_spatial" in message

    def test_a_bad_scenario_name_is_rejected_when_the_config_loads(self):
        """Not only when an environment is built. A config is worth validating
        before anything expensive happens to it."""

        with open(f"{CONFIG_DIR}/tier_chain_spatial.yaml") as f:
            raw = yaml.safe_load(f)
        raw["scenario"] = "tier_chain_martian"

        with pytest.raises(ValueError, match="tier_chain_martian"):
            FullConf(**raw)

    def test_the_two_roots_are_distinguishable(self):
        goal = _scenario(_load("tier_chain_spatial"))
        continuous = _scenario(_load("continuous_step_wise"))

        assert goal.is_goal and goal.type is ScenarioType.GOAL
        assert not continuous.is_goal and continuous.type is ScenarioType.CONTINUOUS


# ============ #
#  Composition  #
# ============ #


class TestGoalComposition:
    def test_the_pool_is_the_chain(self):
        scenario = _scenario(_load("tier_chain_spatial"))

        assert isinstance(scenario.population, TierChainPopulation)

    def test_the_field_starts_full_and_nothing_expires(self):
        scenario = _scenario(_load("tier_chain_spatial"))

        assert scenario.spawning.fill_pool_on_reset is True
        assert scenario.spawning.tier_orb_expires is False
        assert isinstance(scenario.spawning.after_action, LeaveFieldAlone)

    def test_a_delay_scenario_puts_the_field_on_cooldown_and_reactivates_it(self):
        plain = _scenario(_load("tier_chain_spatial"))
        delay = _scenario(_load("tier_chain_delay"))

        assert plain.spawning.delay_on_consume is False
        assert delay.spawning.delay_on_consume is True
        assert isinstance(delay.spawning.after_action, ReactivateAllOrbs)

    def test_the_field_size_is_the_chain_length_not_the_configured_count(self):
        """This is the value the old schema used to overwrite inside a validator,
        so the number in the YAML was routinely a lie. The scenario states it, and
        the config it came from says something else on purpose."""

        conf = _load("tier_chain_scaling_sparse")

        assert conf.world.grid_conf.max_active_orbs == 3
        assert conf.world.grid_conf.max_tier == 5

        scenario = build_scenario(conf.scenario, conf.world, conf.obs)

        assert scenario.spawning.max_active_orbs == 5

    def test_curriculum_widens_the_observation_but_not_the_sort(self):
        """Two counts, deliberately unequal. Under curriculum the observation is
        sized for `tiers` slots while the distance sort still caps at max_tier, so
        the trailing slots are sized and never written. Preserved, not tidied."""

        conf = _load("tier_chain_spatial")
        rules = build_scenario(conf.scenario, conf.world, conf.obs).observation

        assert conf.obs.perception_conf.tiers == 5
        assert conf.world.grid_conf.max_tier == 3
        assert rules.observation_slot_count == 5
        assert rules.sort_limit == 3

    def test_without_curriculum_the_two_counts_coincide(self):
        conf = _load("tier_chain_scaling_sparse")
        rules = build_scenario(conf.scenario, conf.world, conf.obs).observation

        assert rules.observation_slot_count == rules.sort_limit == 5


class TestContinuousComposition:
    def test_the_pool_is_weighted(self):
        scenario = _scenario(_load("continuous_step_wise"))

        assert isinstance(scenario.population, WeightedPopulation)

    def test_the_field_starts_with_one_orb_and_refills_one_at_a_time(self):
        scenario = _scenario(_load("continuous_step_wise"))

        assert scenario.spawning.fill_pool_on_reset is False
        assert isinstance(scenario.spawning.after_action, RefillOrbPool)

    def test_the_field_size_is_the_configured_count(self):
        conf = _load("continuous_step_wise")
        scenario = build_scenario(conf.scenario, conf.world, conf.obs)

        assert (
            scenario.spawning.max_active_orbs
            == conf.world.grid_conf.max_active_orbs
        )

    def test_tier_orbs_may_expiry_depending_on_the_config(self):
        conf = _load("continuous_de_spawn")
        assert build_scenario(
            conf.scenario, conf.world, conf.obs
        ).spawning.tier_orb_expires

        conf = _load("continuous_step_wise")
        assert not build_scenario(
            conf.scenario, conf.world, conf.obs
        ).spawning.tier_orb_expires

    def test_the_observation_is_as_wide_as_the_field(self):
        conf = _load("continuous_step_wise")
        rules = build_scenario(conf.scenario, conf.world, conf.obs).observation

        assert rules.observation_slot_count == rules.sort_limit == 3

    def test_curriculum_changes_nothing_here(self):
        """Recorded because it is easy to assume otherwise. The setting only ever
        reached the observation when a chain was driving the world, so a
        continuous scenario with curriculum on is the same world with it off."""

        on = _scenario(_load("continuous_curriculum"))
        off = _scenario(_load("continuous_step_wise"))

        assert on.observation == off.observation
        assert on.spawning == off.spawning


class TestTerminationComposition:
    def test_a_goal_scenario_gets_goal_termination(self):
        scenario = _scenario(_load("tier_chain_spatial"))

        assert isinstance(scenario.termination, GoalTermination)
        assert scenario.termination.curriculum is True
        assert scenario.termination.delay is False

    def test_a_continuous_scenario_gets_continuous_termination(self):
        scenario = _scenario(_load("continuous_step_wise"))

        assert isinstance(scenario.termination, ContinuousTermination)

    def test_the_timeout_penalty_reaches_the_rules_from_the_droid_config(self):
        conf = _load("tier_chain_spatial")
        scenario = build_scenario(conf.scenario, conf.world, conf.obs)

        assert (
            scenario.termination.timeout_penalty
            == conf.world.droid_conf.timeout_penalty
        )

    def test_the_scoring_mode_reaches_the_rules_from_the_tier_config(self):
        conf = _load("continuous_threshold")
        scenario = build_scenario(conf.scenario, conf.world, conf.obs)

        assert scenario.termination.scoring is ScoringMode.THRESHOLD


# ============ #
#   Legality   #
# ============ #


class TestLegality:
    """A scenario refuses parameter sets it cannot honour, instead of the config
    model refusing combinations of flags that named nothing."""

    def test_a_goal_scenario_refuses_de_spawning_tiers(self):
        conf = _load("tier_chain_spatial")
        conf = _with_grid(conf, de_spawn_tiers=True)

        with pytest.raises(ValueError, match="cannot de-spawn tiers"):
            build_scenario(conf.scenario, conf.world, conf.obs)

    def test_a_chain_longer_than_the_grid_is_refused(self):
        conf = _load("tier_chain_spatial")
        conf = _with_grid(conf, max_tier=25)

        with pytest.raises(ValueError, match="no space for orbs"):
            build_scenario(conf.scenario, conf.world, conf.obs)

    def test_a_continuous_scenario_needs_a_positive_field_size(self):
        conf = _load("continuous_step_wise")
        conf = _with_grid(conf, max_active_orbs=0)

        with pytest.raises(ValueError, match="max_active_orbs"):
            build_scenario(conf.scenario, conf.world, conf.obs)

    def test_dense_scaling_refuses_the_wrong_scoring_mode(self):
        """The scenario is defined by its scoring mode, so a config that selects
        a different one is asking for a different scenario."""

        conf = _load("tier_chain_scaling_dense")
        conf = _with_scoring(conf, ScoringMode.MAX_TIER)

        with pytest.raises(ValueError, match="threshold"):
            build_scenario(conf.scenario, conf.world, conf.obs)

    def test_sparse_scaling_refuses_the_wrong_scoring_mode(self):
        conf = _load("tier_chain_scaling_sparse")
        conf = _with_scoring(conf, ScoringMode.THRESHOLD)

        with pytest.raises(ValueError, match="max_tier"):
            build_scenario(conf.scenario, conf.world, conf.obs)

    def test_a_legality_failure_is_reported_at_config_load(self):
        with open(f"{CONFIG_DIR}/tier_chain_scaling_dense.yaml") as f:
            raw = yaml.safe_load(f)
        raw["world"]["tier_orb_conf"]["scoring"] = "max_tier"

        with pytest.raises(ValueError, match="threshold"):
            FullConf(**raw)


# ============ #
#    Helpers    #
# ============ #


def _scenario(conf: FullConf):
    """Resolve the scenario the way the environment does."""

    return build_scenario(conf.scenario, conf.world, conf.obs)


def _with_grid(conf: FullConf, **updates):
    grid = conf.world.grid_conf.model_copy(update=updates)
    return conf.model_copy(
        update={"world": conf.world.model_copy(update={"grid_world_conf": grid})}
    )


def _with_scoring(conf: FullConf, scoring: ScoringMode):
    tier = conf.world.tier_orb_conf.model_copy(update={"scoring": scoring})
    return conf.model_copy(
        update={"world": conf.world.model_copy(update={"tier_orb_conf": tier})}
    )
