from src.syn_grid.config.config_manager import ConfigManager
from src.syn_grid.config.models import FullConf

from typing import Any
from typing import TypeVar, Any
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def get_test_config(path: str = "test_configs.yaml") -> FullConf:
    """Load and return a FullConf from a test config file."""

    return ConfigManager("test_configs.yaml").load_config(FullConf)


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
