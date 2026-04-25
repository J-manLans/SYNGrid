from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

from stable_baselines3 import PPO


class StatelessPPO(BaseSB3Runner):
    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: AgentConfig, obs_conf: ObsConfig, run_conf: WorldConfig):
        hyper_parameters = {"policy": "MlpPolicy", "device": "cpu", "ent_coef": 0.02}
        super().__init__(conf, obs_conf, run_conf, hyper_parameters, PPO)

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._get_raw_env(self.train_conf.render_mode)

        if self.train_conf.enable_output:
            env = self._wrap_in_monitor(env)

        if self.train_conf.continue_training:
            model = self._load_model(env)
        else:
            model = self._create_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        env = self._get_raw_env("human")
        model = self._load_model(env)
        obs, _ = env.reset()

        done = False
        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                done = truncated or terminated
        except Exception as e:
            print(f"System crashed: {e}")
            raise  # exit function gracefully

        env.close()
