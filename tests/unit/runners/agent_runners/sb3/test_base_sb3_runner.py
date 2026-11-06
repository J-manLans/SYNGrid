import numpy as np

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.runner_models import Algorithm, RunnerConf
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.runners.agent_runners.runner_bundle import RunnerBundle
from syn_grid.runners.agent_runners.runner_registry import build_runner
from syn_grid.runners.agent_runners.sb3.base_sb3_runner import BaseSB3Runner
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Fixtures      #
# ================= #


def _runner(alg: Algorithm) -> BaseSB3Runner:
    register_env()
    name = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL
    manager = ConfigManager()
    entry = SCENARIOS[name]
    scenario = entry.build(
        name.value,
        entry.full_conf(manager.load_raw(manager.scenario_config_path(name))),
    )
    base = manager.load_config("runner_config.yaml", RunnerConf)
    runner_conf = base.model_copy(
        update={
            "common_conf": base.common_conf.model_copy(
                update={"alg": alg, "training": True}
            ),
            "train_conf": base.train_conf.model_copy(
                update={"n_envs": 1, "render_mode": None, "csv_output": False}
            ),
            "eval_conf": base.eval_conf.model_copy(
                update={"num_eval_episodes": 1, "render_mode": None}
            ),
        }
    )
    runner = build_runner(RunnerBundle(scenario=scenario, runner_conf=runner_conf))
    assert isinstance(runner, BaseSB3Runner)

    return runner


class _RecordingModel:
    """Stands in for an SB3 model: moves left, and records the state it is handed."""

    def __init__(self):
        self.handed: list[tuple[object, np.ndarray]] = []

    def predict(self, obs, state, episode_start, deterministic):
        self.handed.append((state, np.array(episode_start)))
        return np.zeros((len(obs),), dtype=np.int64), len(self.handed)


# ================= #
#      Tests        #
# ================= #


class TestBuildEnv:
    """The env a spec asks for."""

    def test_a_frame_stack_stacks_the_observation(self):
        plain = _runner(Algorithm.PPO)._build_env(None, "train")
        stacked = _runner(Algorithm.FSPPO)._build_env(None, "train")
        plain_shape = plain.observation_space.shape
        assert plain_shape is not None

        assert stacked.observation_space.shape == (4 * plain_shape[0],)


class TestEvalModel:
    """The eval loop's hand-off to the model."""

    def test_each_step_gets_the_state_and_episode_starts_the_last_one_left(self):
        runner = _runner(Algorithm.PPO)
        model = _RecordingModel()
        env = runner._build_env(None, "eval")
        env.seed(0)

        runner._eval_model(env, model)  # type: ignore[arg-type]

        states = [state for state, _ in model.handed]
        starts = [start for _, start in model.handed]
        assert len(model.handed) >= 2
        assert states == [None, *range(1, len(states))]
        assert starts[0].all() and not any(s.any() for s in starts[1:])
