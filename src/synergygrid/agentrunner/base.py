import sys
from pathlib import Path
from synergygrid.config.configs import algorithms, environment
from synergygrid.gymnasium.env_factory import register_env
from stable_baselines3.common.base_class import BaseAlgorithm


class AgentRunner:
    def __init__(self, algorithm_index: int):
        register_env()
        self.environment = list(environment.keys())[0]

        algorithm_names = list(algorithms.keys())
        self.algorithm = algorithm_names[algorithm_index]
        self.AlgorithmClass: type[BaseAlgorithm] = algorithms[self.algorithm]

        self.model = None

    def get_model(self, agent_steps: str, env) -> BaseAlgorithm:
        """Create a path to match the latest model of the specified timesteps and load it"""

        if agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        base_dir = Path("results/models") / self.environment
        file_name = f"{self.algorithm}_{self.environment}_{agent_steps}*"
        return self.AlgorithmClass.load(list(base_dir.glob(file_name))[-1], env=env)
