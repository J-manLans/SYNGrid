from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from syn_grid.utils.paths_util import get_project_path, get_syn_grid_path

T = TypeVar("T", bound=BaseModel)

# TODO: this one probably also needs some restructuring, especially since the goal is to move
# towards a GUI, but I will do that job whenever I cross that river.


class ConfigManager:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self):
        self.save_conf_path = get_project_path("output", "saved_configs")

    # ================= #
    #        API        #
    # ================= #

    def load_config(self, config_name: str, model_class: type[T]) -> T:
        """
        Load a YAML file into a Pydantic model instance.

        Args:
            config_name: Name of the YAML configuration file.
            model_class: Pydantic model class used to validate the configuration.

        Returns:
            An instance of `model_class` populated with the YAML data.
        """

        yaml_path = self._create_yaml_path(config_name)

        with yaml_path.open("r") as f:
            raw = yaml.safe_load(f)

        return model_class(**raw)

    def save_snapshot(self, config_name: str, save_conf_id: str) -> None:
        """
        Save a timestamped snapshot of the config file to the saved_configs folder.

        Args:
            save_conf_id: Identifier used as prefix in the snapshot filename.
        """

        self.save_conf_path.mkdir(parents=True, exist_ok=True)

        snapshot_file = self.save_conf_path / f"{save_conf_id}.yaml"
        yaml_path = self._create_yaml_path(config_name)

        snapshot_file.write_bytes(yaml_path.read_bytes())

    # ================= #
    #      Helpers      #
    # ================= #

    def _create_yaml_path(self, config_name: str) -> Path:
        yaml_path = get_syn_grid_path("config", 'yaml', config_name)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        return yaml_path
