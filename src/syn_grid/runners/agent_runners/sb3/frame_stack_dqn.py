from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig


class FrameStackDQN(BaseSB3Runner):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig
    ): ...

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None: ...

    def eval(self) -> None: ...
