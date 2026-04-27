from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

from sb3_contrib import QRDQN


class FrameStackDQN(BaseSB3Runner):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig
    ):
        hyper_parameters = {"policy": "MlpPolicy", "device": "cpu", "ent_coef": 0.02}
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, QRDQN)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None: ...

    def eval(self) -> None: ...
