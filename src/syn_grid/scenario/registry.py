"""
Scenario selection.

This module is the whole of "which scenario is this". The global config names one, its entry here
says which config class validates its config, which builder composes its pieces, which perceptions
it allows, and whether it is locked. Nothing downstream re-derives identity from the config. The
config classes and the builders live in their family's folder, such as
``families/goal/single_tier_chain/config.py`` and ``families/goal/single_tier_chain/builders.py``.
Registering a new scenario means writing its builder there and adding an entry to ``SCENARIOS``
here.

Names that are configured identically share a config class. A name gets its own when it needs a
field the family lacks, and that subclass lives beside the family it departs from.

A locked scenario's run chooses only what its small model holds. The entry merges those choices
onto the scenario's base config, a YAML in the family's ``base_configs/`` folder, and the result is
the full config the builder takes. Nothing past the entry knows whether a scenario is locked.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import yaml
from pydantic import BaseModel

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.scenario.common_models import Perception, ScenarioConf
from syn_grid.scenario.families.goal.single_tier_chain.builders import (
    build_single_tier_chain_delay,
    build_single_tier_chain_scaling_dense,
    build_single_tier_chain_scaling_sparse,
    build_single_tier_chain_spatial,
)
from syn_grid.scenario.families.goal.single_tier_chain.config import (
    ScalingScenarioConf,
    SpatialScenarioConf,
    TierDelayScenarioConf,
    TierScenarioConf,
)
from syn_grid.scenario.scenario import Scenario
from syn_grid.utils.paths_util import get_syn_grid_path

# ============ #
#   Registry   #
# ============ #


C = TypeVar("C", bound=ScenarioConf)


@dataclass(frozen=True)
class Lock:
    """What makes a scenario locked: the small model of what a run may choose, and the base config
    the choices are merged onto."""

    choices_class: type[BaseModel]
    base_config: Path


@dataclass(frozen=True)
class ScenarioEntry(Generic[C]):
    """What a scenario name stands for.

    `conf_class` is the full model the builder takes. `perceptions` are the ones with the
    scenario's visibility, and every config the entry builds is held to them. `lock` is set for a
    locked scenario; without one, a run's config is the full config.
    """

    conf_class: type[C]
    builder: Callable[[str, C], Scenario]
    perceptions: tuple[Perception, ...]
    lock: Lock | None = None

    def full_conf(self, raw: dict[str, Any]) -> C:
        """
        The full config for a run's raw config.

        For a locked scenario, the raw config is checked against the small model, so a fixed value
        it states is rejected, and merged onto the base config. Otherwise it is the full config.
        """

        if self.lock is None:
            return self.conf_class(**raw)

        self.lock.choices_class(**raw)

        with self.lock.base_config.open("r") as f:
            base = yaml.safe_load(f)

        return self.conf_class(**_merge(base, raw))

    def build(self, scenario_name: str, scenario_conf: ScenarioConf) -> Scenario:
        """Check that the config is this entry's config class and that its perception is allowed,
        then run the builder on it."""

        if not isinstance(scenario_conf, self.conf_class):
            raise TypeError(
                f"Scenario '{scenario_name}' is configured by {self.conf_class.__name__}, "
                f"but was given a {type(scenario_conf).__name__}"
            )

        perception = scenario_conf.obs_conf.observation_handler_conf.perception

        if perception not in self.perceptions:
            allowed = ", ".join(p.value for p in self.perceptions)
            raise ValueError(
                f"Scenario '{scenario_name}' does not allow the perception "
                f"'{perception.value}'. Allowed: {allowed}"
            )

        return self.builder(scenario_name, scenario_conf)


_SINGLE_TIER_CHAIN_BASE_CONFIGS = get_syn_grid_path(
    "scenario", "families", "goal", "single_tier_chain", "base_configs"
)

SCENARIOS: dict[ScenarioName, ScenarioEntry] = {
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL: ScenarioEntry(
        TierScenarioConf,
        build_single_tier_chain_spatial,
        perceptions=(Perception.VECTOR_FOG_OF_WAR,),
        lock=Lock(
            SpatialScenarioConf, _SINGLE_TIER_CHAIN_BASE_CONFIGS / "spatial.yaml"
        ),
    ),
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_SPARSE: ScenarioEntry(
        TierScenarioConf,
        build_single_tier_chain_scaling_sparse,
        perceptions=(Perception.VECTOR_MARKOVIAN,),
        lock=Lock(
            ScalingScenarioConf,
            _SINGLE_TIER_CHAIN_BASE_CONFIGS / "scaling_sparse.yaml",
        ),
    ),
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_SCALING_DENSE: ScenarioEntry(
        TierScenarioConf,
        build_single_tier_chain_scaling_dense,
        perceptions=(Perception.VECTOR_MARKOVIAN,),
        lock=Lock(
            ScalingScenarioConf,
            _SINGLE_TIER_CHAIN_BASE_CONFIGS / "scaling_dense.yaml",
        ),
    ),
    ScenarioName.GOAL_SINGLE_TIER_CHAIN_DELAY: ScenarioEntry(
        TierDelayScenarioConf,
        build_single_tier_chain_delay,
        # TODO(delay): delay's visibility is not decided, so it allows every perception.
        perceptions=tuple(Perception),
    ),
}


def build_scenario(scenario: ScenarioName, scenario_conf: ScenarioConf) -> Scenario:
    """Look the name up in `SCENARIOS` and build the scenario through its entry."""

    try:
        entry = SCENARIOS[scenario]
    except KeyError:
        raise KeyError(
            f"Unknown scenario '{scenario}'. Available: {sorted(SCENARIOS)}"
        ) from None

    return entry.build(scenario.value, scenario_conf)


# ----------- #
#   Helpers   #
# ----------- #


def _merge(base: dict[str, Any], choices: dict[str, Any]) -> dict[str, Any]:
    """The base config with the choices written over it, block by block."""

    merged = dict(base)

    for key, value in choices.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value

    return merged
