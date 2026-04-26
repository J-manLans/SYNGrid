from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.config.models import AgentConfig, WorldConfig, ObsConfig

import numpy as np
from sb3_contrib import RecurrentPPO


class LstmPPO(BaseSB3Runner[RecurrentPPO]):
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
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "policy_kwargs": {
                "lstm_hidden_size": 256,
                "n_lstm_layers": 1,
                "shared_lstm": False,
            },
        }
        super().__init__(
            conf,
            obs_conf,
            run_conf,
            hyper_parameters,
            RecurrentPPO,
            hyper_parameters["policy_kwargs"]["lstm_hidden_size"],
        )

    # ================= #
    #        API        #
    # ================= #

    def train(self) -> None:
        env = self._make_wrapped_dummy_vec_env(self.train_conf.render_mode)
        model = self._get_model(env)

        self._train_model(model, env)

    def eval(self) -> None:
        env = self._make_wrapped_dummy_vec_env("human")
        model = self._get_model(env)
        model.set_env(env)

        lstm_states = None
        num_envs = len(env.envs)
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        obs = env.reset()
        done = False
        try:
            while not done:
                action, lstm_states = model.predict(
                    obs,  # type: ignore[arg-type]
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )

                obs, rewards, dones, info = env.step(action)

                # Exit environment if terminated or truncated.
                done = dones.any()
                episode_starts = dones
        except Exception as e:
            print(f"System crashed: {e}")
            raise  # exit function gracefully

        env.close()
