from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig
from syn_grid.utils.paths_util import get_project_path
from syn_grid.gymnasium.env_factory import make, check_my_env

import sys, datetime
from pathlib import Path
from abc import ABC, abstractmethod
from gymnasium import Env


class BaseAgentRunner(ABC):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        conf: AgentConfig,
        obs_conf: ObsConfig,
        run_conf: WorldConfig,
    ):
        self.conf = conf.global_agent_conf
        self.train_conf = conf.train_agent_conf
        self.eval_conf = conf.eval_agent_conf
        self.obs_conf = obs_conf
        self.run_conf = run_conf

        # Get current date and time to us as an identifier for unique file naming
        self.date = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

        # Create directories for saving models and logs
        self.model_dir = Path(get_project_path("output", "models"))
        self.log_dir = Path(get_project_path("output", "results", "logs"))

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    # ================= #
    #  Abstract methods #
    # ================= #

    @abstractmethod
    def train(self): ...

    @abstractmethod
    def eval(self): ...

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_raw_env(self, render_mode: str | None,) -> Env:
        env = make(render_mode, self.run_conf, self.obs_conf)

        if self.conf.check_env:
            check_my_env(env)
            sys.exit("Environment is fine")

        return env

    def _get_model_path(self) -> Path:
        """
        Returns a path that matches the latest model of the timesteps specified in the config file
        """

        if self.conf.agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        file_name = f"{self.conf.agent_steps}_{self.conf.id}_{self.conf.alg}*"

        # list all files
        matches = list(self.model_dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(f"No model found for path: {file_name}")

        # return the one with the highest value of the timestamp
        return max(matches, key=lambda p: p.stat().st_mtime)

    def _get_log_identifier(self) -> str:
        return f"{self.conf.id}_{self.conf.alg}_{self.date}"
