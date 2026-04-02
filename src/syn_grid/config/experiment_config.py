from syn_grid.utils.paths_util import get_package_path, get_project_path

import yaml
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel
import datetime

T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self):
        self.yaml_path = Path(get_package_path("config", "configs.yaml"))
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

    def save_snapshot(self, config: BaseModel, save_conf_id: str) -> None:
        """
        Save a timestamped snapshot of a config model to the saved_configs folder.

        Args:
            config: Pydantic model instance to serialize and save.
            save_conf_id: Identifier used as prefix in the snapshot filename.
        """

        self.save_conf_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_file = self.save_conf_path / f"{save_conf_id}_{timestamp}.yaml"

        with snapshot_file.open("w") as f:
            yaml.safe_dump(config.model_dump(), f)
