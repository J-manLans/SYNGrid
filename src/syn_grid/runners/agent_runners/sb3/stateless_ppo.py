from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from syn_grid.config.models import AgentConfig, ObsConfig, WorldConfig
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner


class StatelessPPO(BaseSB3Runner[PPO]):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        policy = self._get_policy_from_perception(
            obs_conf.observation_handler.perception
        )
        hyper_parameters = {
            "policy": policy,
            "device": "cpu",
            "ent_coef": 0.025,
            "n_steps": 128,
            "batch_size": 128,
            "n_epochs": 4,
        }
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, PPO)
        print("Initializing stateless PPO...")

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._wrap_env(self._train_conf.render_mode, self._TRAIN)
        model = self._get_model(env, self._TRAIN)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._wrap_env(self._eval_conf.render_mode, self._EVAL)
        model = self._load_model(env)

        self._eval_model(env, model)

    # ================= #
    #      Helpers      #
    # ================= #

    def _wrap_env(self, render_mode: str | None, sub_dir: str) -> VecNormalize:
        env = self._make_wrapped_dummy_vec_env(render_mode, sub_dir)
        return self._get_normalized_env(env)
