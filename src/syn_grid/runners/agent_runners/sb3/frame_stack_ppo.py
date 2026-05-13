from syn_grid.runners.agent_runners.sb3.stateless_ppo import StatelessPPO
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig
from stable_baselines3.common.vec_env import VecFrameStack


class FrameStackPPO(StatelessPPO):
    # ================= #
    #       Init        #
    # ================= #

    _N_STACK = 10

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        super().__init__(conf, obs_conf, run_conf)
        print("Initializing frame stacking PPO...")

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._wrap_env(self._train_conf.render_mode, self._TRAIN)
        env = self._get_frame_stacked_env(env)
        model = self._get_model(env, self._TRAIN)

        self._train_model(model, env)

    def eval(self) -> None:
        # prep model and env
        env = self._wrap_env(self._eval_conf.render_mode, self._EVAL)
        env = self._get_frame_stacked_env(env)
        model = self._load_model(env)

        self._eval_model(env, model)

    # ================= #
    #      Helpers      #
    # ================= #

    def _get_frame_stacked_env(self, env):
        return VecFrameStack(env, self._N_STACK)
