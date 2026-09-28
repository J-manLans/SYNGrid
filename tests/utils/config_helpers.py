from typing import Any, TypeVar

from pydantic import BaseModel

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models import FullConf
from syn_grid.scenario.registry import build_scenario
from syn_grid.scenario.scenario import Scenario

T = TypeVar("T", bound=BaseModel)


def get_test_config(path: str = "test_configs.yaml") -> FullConf:
    """Load and return a FullConf from a test config file."""

    return ConfigManager(path).load_config(FullConf)


def get_scenario(conf: FullConf) -> Scenario:
    """Resolve the scenario a config selects.

    Tests that only need a world to poke at should ask for the scenario the
    same way the environment does, rather than reading scenario flags -- there
    are none left to read.
    """

    return build_scenario(conf.scenario, conf.world, conf.obs)


def update_conf(conf: T, updates: dict[str, Any]) -> T:
    """
    Return a new immutable BaseModel of type T with updates applied.
    Nested updates should be dicts matching the nested structure.
    """

    new_conf = conf
    for key, value in updates.items():
        sub_conf = getattr(new_conf, key)
        if isinstance(sub_conf, BaseModel) and isinstance(value, dict):
            # recursively update nested BaseModel
            sub_conf = update_conf(sub_conf, value)
        else:
            sub_conf = value
        new_conf = new_conf.model_copy(update={key: sub_conf})
    return new_conf
