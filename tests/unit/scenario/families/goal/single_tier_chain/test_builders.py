"""How the single tier chain family composes a scenario: its tag, negative orb and watcher."""

import pytest

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.core.droid.digestion.digesters.neg_digester import NegDigester
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


def _scenario(name: ScenarioName, **overrides: dict) -> Scenario:
    manager = ConfigManager()
    raw = manager.load_raw(manager.scenario_config_path(name))
    raw["world_conf"] = {**raw.get("world_conf", {}), **overrides}
    entry = SCENARIOS[name]

    return entry.build(name.value, entry.full_conf(raw))


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
        entry = SCENARIOS[name]
        manager = ConfigManager()
        conf = entry.full_conf(manager.load_raw(manager.scenario_config_path(name)))
        world = conf.world_conf
        expected = start.format(
            rows=world.grid_conf.grid_rows,
            cols=world.grid_conf.grid_cols,
            tier=world.tier_orb_conf.max_tier,
            delay=getattr(getattr(conf, "delay_conf", None), "steps", None),
        )

        tag = entry.build(name.value, conf).scenario_tag

        assert tag == f"{expected}{world.max_steps}"

    def test_a_negative_orb_adds_neg(self):
        scenario = _scenario(
            ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL,
            neg_orb_conf={"enabled": True},
        )

        assert scenario.scenario_tag.endswith("_Neg")


class TestNegativeOrb:
    """The negative orb a scenario gets when its config enables it."""

    def test_it_has_the_configs_values_and_lives_a_corner_to_corner_walk(self):
        name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
        manager = ConfigManager()
        raw = manager.load_raw(manager.scenario_config_path(name))
        raw["world_conf"] = {
            **raw.get("world_conf", {}),
            "neg_orb_conf": {"enabled": True},
        }
        conf = SCENARIOS[name].full_conf(raw)
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
        manager = ConfigManager()
        conf = SCENARIOS[name].full_conf(
            manager.load_raw(manager.scenario_config_path(name))
        )

        scenario = SCENARIOS[name].build(name.value, conf)

        assert scenario.make_watcher == DelayWatcherRecipe(conf.delay_conf.steps)

    def test_a_scenario_without_one_never_acts(self):
        scenario = _scenario(ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL)

        assert scenario.make_watcher is NoWatcher
