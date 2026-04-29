from syn_grid.utils.paths_util import get_package_path, get_project_path
from syn_grid.utils.date_utils import get_date
from syn_grid.config.models import ExperimentConfig

import yaml
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
import sys

T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, config_file: str):
        self.yaml_path = Path(get_package_path("config", config_file))
        self.save_conf_path = Path(get_project_path("output", "saved_configs"))
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.yaml_path}")

    # ================= #
    #       API        #
    # ================= #

    def load_config(self, model_class: Type[T]) -> T:
        """
        Load a YAML file into a Pydantic model instance.

        This method reads the YAML file specified in `self.yaml_path` and
        parses it into an instance of the given Pydantic model class.

        Args:
            model_class: A subclass of `BaseModel` that the YAML data will be parsed into.
        Returns:
            An instance of `model_class` populated with data from the YAML file.
        """

        with self.yaml_path.open("r") as f:
            raw = yaml.safe_load(f)

        return model_class(**raw)

    def save_snapshot(self, save_conf_id: str) -> None:
        """
        Save a timestamped snapshot of the config file to the saved_configs folder.

        Args:
            save_conf_id: Identifier used as prefix in the snapshot filename.
        """

        self.save_conf_path.mkdir(parents=True, exist_ok=True)

        timestamp = get_date()
        snapshot_file = self.save_conf_path / f"{save_conf_id}_{timestamp}.yaml"

        snapshot_file.write_bytes(self.yaml_path.read_bytes())
