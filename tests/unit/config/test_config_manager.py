"""How config files are found, loaded and validated.

The tests read the real run YAMLs only for what does not depend on their contents, so editing a run
to train cannot turn them red: that a file exists and that it parses.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.utils.paths_util import get_syn_grid_path

# ================= #
#     Constants     #
# ================= #


GLOBAL = "global_config.yaml"


# ================= #
#     Stand-ins     #
# ================= #


class _TakesAnything(BaseModel, extra="allow"):
    """A model any mapping passes, so a load does not depend on what a run YAML holds."""


class _NeedsAFieldNoYamlHas(BaseModel):
    """A model every run YAML fails, for the same reason."""

    a_field_no_yaml_has: int


# ================= #
#      Helpers      #
# ================= #


def _real_yaml(relative_path: str) -> Path:
    return get_syn_grid_path("config", "yaml", relative_path)


# ================= #
#       Tests       #
# ================= #


class TestScenarioConfigPath:
    """Where each scenario's run YAML is."""

    @pytest.mark.parametrize("name", list(ScenarioName))
    def test_every_scenario_has_a_run_yaml_there(self, name: ScenarioName):
        manager = ConfigManager()

        raw = manager.load_raw(manager.scenario_config_path(name))

        assert isinstance(raw, dict)


class TestLoad:
    """Reading a YAML file, checked against a model or as it is."""

    def test_a_config_is_validated_by_its_model(self):
        manager = ConfigManager()

        loaded = manager.load_config(GLOBAL, _TakesAnything)

        assert isinstance(loaded, _TakesAnything)
        with pytest.raises(ValidationError):
            manager.load_config(GLOBAL, _NeedsAFieldNoYamlHas)

    def test_raw_is_the_yaml_as_it_is_written(self):
        assert ConfigManager().load_raw(GLOBAL) == yaml.safe_load(
            _real_yaml(GLOBAL).read_text()
        )

    def test_a_missing_file_is_refused_with_its_path(self):
        with pytest.raises(FileNotFoundError, match="no_such_config.yaml"):
            ConfigManager().load_raw("no_such_config.yaml")
