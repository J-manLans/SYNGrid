from syn_grid.config.models import GlobalAgentConf
from syn_grid.config.models import WorldConfig, ObsConfig
from syn_grid.gymnasium.env_factory import register_env
from syn_grid.utils.paths_util import get_project_path

import sys
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from gymnasium import Env


class AgentRunner:
    # ================= #
    #       Init        #
    # ================= #
    hyper_parameters = {"policy": "MlpPolicy", "device": "cpu", "ent_coef": 0.02}
    algorithms = {"PPO": PPO, "DQN": DQN, "A2C": A2C}

    def __init__(
        self,
        conf: GlobalAgentConf,
        run_conf: WorldConfig,
        obs_conf: ObsConfig,
    ):
        register_env()

        algorithm_names = list(self.algorithms.keys())
        self.algorithm = algorithm_names[conf.algorithm_index]
        self.AlgorithmClass: type[BaseAlgorithm] = self.algorithms[self.algorithm]

        self.agent_steps = conf.agent_steps
        self.identifier = conf.identifier

        self.run_conf = run_conf
        self.obs_conf = obs_conf

    # ================= #
    #        API        #
    # ================= #

    def get_model(self, env: Env | None) -> BaseAlgorithm:
        """Create a path to match the latest model of the specified timesteps and load it"""

        if self.agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        base_dir = Path(get_project_path("output", "models"))
        file_name = f"{self.identifier}_{self.algorithm}_{self.agent_steps}*"

        matches = list(base_dir.glob(file_name))
        if not matches:
            raise FileNotFoundError(f"No model found for path: {file_name}")

        return self.AlgorithmClass.load(matches[-1], env=env, device=self.hyper_parameters["device"])
