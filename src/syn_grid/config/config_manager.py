from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from syn_grid.config.models.global_models import ScenarioName
from syn_grid.utils.paths_util import get_project_path, get_syn_grid_path

M = TypeVar("M", bound=BaseModel)

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

    def scenario_config_path(self, scenario: ScenarioName) -> str:
        """
        Return the path of a scenario's YAML file, relative to `config/yaml/`.

        Args:
            scenario: The scenario whose config file is wanted.
        """

        return f"scenarios/{scenario.value}.yaml"

    def load_config(self, relative_path: str, model_class: type[M]) -> M:
        """
        Load a YAML file into a Pydantic model instance.

        Args:
            relative_path: Path of the YAML file, relative to `config/yaml/`.
            model_class: Pydantic model class used to validate the configuration.

        Returns:
            An instance of `model_class` populated with the YAML data.
        """

        yaml_path = self._create_yaml_path(relative_path)

        with yaml_path.open("r") as f:
            raw = yaml.safe_load(f)

        return model_class(**raw)

    def save_snapshot(self, relative_path: str, save_conf_id: str) -> None:
        """
        Save a timestamped snapshot of the config file to the saved_configs folder.

        Args:
            relative_path: Path of the YAML file, relative to `config/yaml/`.
            save_conf_id: Identifier used as prefix in the snapshot filename.
        """

        self.save_conf_path.mkdir(parents=True, exist_ok=True)

        snapshot_file = self.save_conf_path / f"{save_conf_id}.yaml"
        yaml_path = self._create_yaml_path(relative_path)

        snapshot_file.write_bytes(yaml_path.read_bytes())

    # ================= #
    #      Helpers      #
    # ================= #

    def _create_yaml_path(self, relative_path: str) -> Path:
        yaml_path = get_syn_grid_path("config", "yaml", relative_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        return yaml_path
