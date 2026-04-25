from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

from sb3_contrib import RecurrentPPO


class LstmPPO(BaseSB3Runner):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        hyper_parameters = {
            "policy": "MlpLstmPolicy",
            "device": "cpu",
            "ent_coef": 0.008,
            "n_steps": 128,
            "batch_size": 64,
            "n_epochs": 5,
            "policy_kwargs": {
                "lstm_hidden_size": 64,
                "n_lstm_layers": 1,
                "shared_lstm": False,
            },
        }
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, RecurrentPPO)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._get_raw_env(self.train_conf.render_mode)

        if self.train_conf.enable_output:
            env = self._wrap_in_monitor(env)

    def eval(self) -> None: ...
