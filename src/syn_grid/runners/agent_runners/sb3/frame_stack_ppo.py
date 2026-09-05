from stable_baselines3.common.vec_env import VecFrameStack

from syn_grid.config.models import AgentConfig, ObsConfig, WorldConfig
from syn_grid.runners.agent_runners.sb3.stateless_ppo import StatelessPPO


class FrameStackPPO(StatelessPPO):
    # ================= #
    #       Init        #
    # ================= #

    _N_STACK = 4

    def __init__(self, world_conf: WorldConfig, obs_conf: ObsConfig, conf: AgentConfig):
        super().__init__(world_conf, obs_conf, conf)
        print("Adding frame stacking on top...")

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
