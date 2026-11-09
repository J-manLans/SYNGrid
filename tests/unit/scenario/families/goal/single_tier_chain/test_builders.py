"""How the single tier chain family composes a scenario: its tag, tier chain, metrics, negative orb and watcher."""

from typing import Any

import pytest
import yaml

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.core.droid.digestion.digesters.direct_digester import NegDigester
from syn_grid.core.droid.digestion.digesters.tier_chain.tier_digester import (
    TierDigester,
)
from syn_grid.core.orbs.orb_meta import DirectKind, SynergyKind
from syn_grid.scenario.blocks.orb_bundle import OrbBundle
from syn_grid.scenario.blocks.populations.neg_population import NegPopulation
from syn_grid.scenario.blocks.recipes.world_recipe import WorldRecipe
from syn_grid.scenario.blocks.watcher import NoWatcher
from syn_grid.scenario.families.goal.single_tier_chain.watcher import (
    DelayWatcherRecipe,
)
from syn_grid.scenario.registry import SCENARIOS
from syn_grid.scenario.scenario import Scenario

# ================= #
#      Helpers      #
# ================= #


def _base_raw(name: ScenarioName) -> dict[str, Any]:
    """A locked scenario's base config as plain data, so a run's YAML cannot move the test."""

    lock = SCENARIOS[name].lock
    assert lock is not None
    return yaml.safe_load(lock.base_config_path.read_text())


def _conf(name: ScenarioName, *, negative_orb: bool = False) -> Any:
    """A locked scenario's base config, with its negative orb switched on if asked."""

    raw = _base_raw(name)
    raw["world_conf"]["neg_orb_conf"]["enabled"] = negative_orb

    return SCENARIOS[name].conf_class(**raw)


def _scenario(name: ScenarioName, *, negative_orb: bool = False) -> Scenario:
    """A locked scenario built from `_conf`."""

    return SCENARIOS[name].build(name.value, _conf(name, negative_orb=negative_orb))


# ================= #
#       Tests       #
# ================= #


class TestTag:
    """Each scenario's tag: where the run sits on its axis, then the steps, then `_Neg`."""

    @pytest.mark.parametrize(
        ("name", "start"),
        [
            (ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL, "{rows}x{cols}_steps"),
            (
                ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE,
                "tier{tier}_{rows}x{cols}_steps",
            ),
            (ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY, "delay{delay}_steps"),
        ],
    )
    def test_names_the_axis_then_the_steps(self, name: ScenarioName, start: str):
        conf = _conf(name)
        world = conf.world_conf
        expected = start.format(
            rows=world.grid_conf.grid_rows,
            cols=world.grid_conf.grid_cols,
            tier=world.tier_orb_conf.max_tier,
            delay=getattr(getattr(conf, "delay_conf", None), "steps", None),
        )

        tag = SCENARIOS[name].build(name.value, conf).scenario_tag

        assert tag == f"{expected}{world.max_steps}"

    def test_a_negative_orb_adds_neg(self):
        scenario = _scenario(
            ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL, negative_orb=True
        )

        assert scenario.scenario_tag.endswith("_Neg")


class TestTierChain:
    """The chain a scenario's droid builds from the tier orbs its world holds."""

    def test_eating_the_tiers_in_order_completes_it_at_the_configs_max_tier_and_not_before(
        self,
    ):
        # Scaling sparse, whose axis is the max tier, so a digester that assumed another top tier fails
        name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE
        conf = _conf(name)
        world = SCENARIOS[name].build(name.value, conf).build_world()
        digester = world.droid.get_digester(TierDigester)
        tier_orbs = sorted(
            (orb for orb in world.ALL_ORBS if orb.META.KIND == SynergyKind.TIER),
            key=lambda orb: orb.META.TIER or 0,
        )

        *below_the_top, top = tier_orbs
        for orb in below_the_top:
            digester.digest(orb)

        assert digester.chains_completed == 0

        digester.digest(top)

        assert top.META.TIER == conf.world_conf.tier_orb_conf.max_tier
        assert digester.chains_completed == 1


class TestMetrics:
    """What a scenario reports at the end of an episode, each number from what counts it."""

    def test_each_metric_reads_its_own_count(self):
        # Spatial's chain tops out at tier 3. Each count ends on a different number, so two
        # metrics that read each other's count fail too.
        scenario = _scenario(
            ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL, negative_orb=True
        )
        world = scenario.build_world()
        tiers = world.droid.get_digester(TierDigester)
        negatives = world.droid.get_digester(NegDigester)
        tier_orb = {
            orb.META.TIER: orb
            for orb in world.ALL_ORBS
            if orb.META.KIND == SynergyKind.TIER
        }
        negative_orb = next(
            orb for orb in world.ALL_ORBS if orb.META.KIND == DirectKind.NEGATIVE
        )

        # Tiers 1 and 2 progress and 3 completes; then each tier 2 breaks, as no chain runs
        for tier in (1, 2, 3, 2, 2, 2):
            tiers.digest(tier_orb[tier])
        for _ in range(4):
            negatives.digest(negative_orb)

        assert {metric.name: metric.read(world) for metric in scenario.metrics} == {
            "chain_progresses": 2,
            "chains_completed": 1,
            "chains_broken": 3,
            "negatives_eaten": 4,
        }


class TestNegativeOrb:
    """The negative orb a scenario gets when its config enables it."""

    def test_it_has_the_configs_values_and_lives_a_corner_to_corner_walk(self):
        name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
        conf = _conf(name, negative_orb=True)
        negative = conf.world_conf.neg_orb_conf
        grid = conf.world_conf.grid_conf

        recipe = SCENARIOS[name].build(name.value, conf).build_world

        assert isinstance(recipe, WorldRecipe)
        assert recipe.orb_bundles[-1] == OrbBundle(
            population=NegPopulation(
                reward=negative.reward,
                cool_down=negative.cool_down,
                life_span=(grid.grid_rows - 1) + (grid.grid_cols - 1),
            ),
            make_digester=NegDigester,
        )


class TestWatcher:
    """Which scenarios act after the world's step."""

    def test_delay_watches_with_its_delay(self):
        name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY
        conf = _conf(name)

        scenario = SCENARIOS[name].build(name.value, conf)

        assert scenario.make_watcher == DelayWatcherRecipe(conf.delay_conf.steps)

    def test_a_scenario_without_one_never_acts(self):
        scenario = _scenario(ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL)

        assert scenario.make_watcher is NoWatcher
