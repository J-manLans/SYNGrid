"""
What a scenario name stands for: its entry.

An entry is the whole of one scenario, which a family's catalogue states in one place: the full
config model, the perceptions with its visibility, its lock if it is locked, and what the family
needs to compose it. `ScenarioEntry` holds what every family shares and does the generic work, the
merge of a locked run's tunables and the checks before building. Each family subclasses it once,
with the fields its scenarios differ by, and composes the `Scenario` in `_compose`.

A locked scenario's run sets only its tunables. The entry merges those tunables onto the
scenario's base config, a YAML in the family's ``base_configs/`` folder, and the result is the
full config the family composes from. Nothing past the entry knows whether a scenario is locked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from syn_grid.config.models.scenario.common_models import Perception, ScenarioConf
from syn_grid.scenario.scenario import Scenario

# ============ #
#     Lock     #
# ============ #


@dataclass(frozen=True)
class Lock:
    """What makes a scenario locked: the model of its tunables, and the base config they are
    merged onto."""

    scenario_tunables: type[BaseModel]
    base_config_path: Path


# ============ #
#    Entry     #
# ============ #


@dataclass(frozen=True, kw_only=True)
class ScenarioEntry[C: ScenarioConf](ABC):
    """One scenario: its config, its visibility, its lock, and how its family composes it.

    `conf_class` is the full model the scenario is composed from. `perceptions` are the ones with
    the scenario's visibility, and every config the entry builds is held to them. `lock` is set
    for a locked scenario; without one, a run's config is the full config.
    """

    conf_class: type[C]
    perceptions: tuple[Perception, ...]
    lock: Lock | None = None

    def full_conf(self, raw: dict[str, Any]) -> C:
        """
        The full config for a run's raw config.

        For a locked scenario, the raw config is checked against its tunables, so a fixed value
        it states is rejected, and merged onto the base config. Otherwise it is the full config.
        """

        if self.lock is None:
            return self.conf_class(**raw)

        # Only a check: it rejects what a run may not set, and the object is not kept.
        self.lock.scenario_tunables(**raw)

        # The merged config is the complete one, so this is what gets validated as the full model.
        return self.conf_class(**_merge(self._base_raw(), raw))

    def base_conf(self) -> C:
        """
        The full config a locked scenario was run with: its base config, validated as the full
        model. It is what a run's tunables are merged onto, and what a test of the scenario's rules
        builds from, since a run's YAML is edited to train.

        Raises:
            ValueError: the scenario is not locked, so it has no base config.
        """

        return self.conf_class(**self._base_raw())

    def _base_raw(self) -> dict[str, Any]:
        if self.lock is None:
            raise ValueError(
                f"{self.conf_class.__name__}'s scenario is not locked: it has no base config"
            )

        with self.lock.base_config_path.open("r") as f:
            return yaml.safe_load(f)

    def build(self, scenario_name: str, scenario_conf: ScenarioConf) -> Scenario:
        """Check that the config is this entry's config class and that its perception is allowed,
        then compose the scenario."""

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

        return self._compose(scenario_name, scenario_conf)

    @abstractmethod
    def _compose(self, scenario_name: str, scenario_conf: C) -> Scenario:
        """The scenario, composed from a config that has passed the checks."""


# ----------- #
#   Helpers   #
# ----------- #


def _merge(base: dict[str, Any], tunables: dict[str, Any]) -> dict[str, Any]:
    """The base config with the tunables written over it, block by block.
    Whatever the tunables do not state keeps the base's value.
    """

    # Start from the base. The loop touches only the tunables, so everything else stays as it is.
    merged = base.copy()

    for key, value in tunables.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # A block on both sides: dig down into it.
            merged[key] = _merge(merged[key], value)
        else:
            # The bottom: the run's value replaces the base's.
            merged[key] = value

    return merged
