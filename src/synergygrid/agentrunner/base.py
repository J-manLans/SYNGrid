import sys
from pathlib import Path
from synergygrid import algorithms, environment, register_env


class AgentRunner:
    def __init__(self, algorithm: int):
        register_env()
        self.environment = list(environment.keys())[0]

        alg = list(algorithms.keys())
        self.algorithm = alg[algorithm]
        self.AlgorithmClass = algorithms.get(self.algorithm, {})

        self.model = None

    def get_model(self, agent_steps: str, env):
        """Create a path to match the latest model of the specified timesteps and load it"""

        if agent_steps == "":
            sys.exit("You forgot to specify the models steps")

        base_dir = Path("results/models") / self.environment
        file_name = f"{self.algorithm}_{self.environment}_{agent_steps}*"
        return self.AlgorithmClass.load(list(base_dir.glob(file_name))[-1], env=env)
