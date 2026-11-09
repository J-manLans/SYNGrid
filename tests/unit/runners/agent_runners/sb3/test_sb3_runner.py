"""How the SB3 runner builds its env, creates, saves and loads its model, trains and evaluates."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.runner_models import Algorithm, RunnerConf
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.gymnasium.utils.episode_logging.csv_episode_logger import (
    CSVEpisodeLogger,
)
from syn_grid.runners.agent_runners.runner_registry import RUNNER, build_runner
from syn_grid.runners.agent_runners.sb3.run_controls_callback import (
    RunControlsCallback,
)
from syn_grid.runners.agent_runners.sb3.sb3_runner import SB3Runner, SB3Spec
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL


# ================= #
#     Stand-ins     #
# ================= #


class _RecordingModel:
    """Stands in for an SB3 model: moves left, and records what it is handed."""

    def __init__(self):
        self.handed: list[tuple[object, np.ndarray]] = []
        self.deterministic: list[bool] = []

    def predict(self, obs, state, episode_start, deterministic):
        self.handed.append((state, np.array(episode_start)))
        self.deterministic.append(deterministic)
        return np.zeros((len(obs),), dtype=np.int64), len(self.handed)


class _LearningModel:
    """Stands in for an SB3 model in training: records each `learn` call, and can be told to
    stop from the window or be interrupted."""

    def __init__(self, *, quit_from_window: bool = False, interrupt: bool = False):
        self.learned: list[dict[str, Any]] = []
        self._quit = quit_from_window
        self._interrupt = interrupt

    def learn(self, **kwargs: Any) -> None:
        self.learned.append(kwargs)
        if self._interrupt:
            raise KeyboardInterrupt
        if self._quit:
            kwargs["callback"].stopped = True


class _ClosingEnv:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _TitleRecorder:
    """Stands in for the human renderer the eval loop titles."""

    def __init__(self):
        self.titles: list[str] = []

    def set_title(self, title: str) -> None:
        self.titles.append(title)


class _OneStepEpisodes:
    """Stands in for a human eval env with one env, whose every episode ends after a step."""

    num_envs = 1

    def __init__(self):
        self.renderer = _TitleRecorder()
        self.closed = False

    def reset(self) -> np.ndarray:
        return np.zeros((1, 1))

    def step(self, action: np.ndarray):
        return np.zeros((1, 1)), np.zeros(1), np.ones(1, dtype=bool), [{}]

    def get_attr(self, name: str, indices: int) -> list[Any]:
        assert indices == 0
        return [getattr(self, name)]

    def close(self) -> None:
        self.closed = True


# ================= #
#      Helpers      #
# ================= #


def _runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alg: Algorithm = Algorithm.PPO,
    *,
    common: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
) -> SB3Runner:
    """A runner of spatial, working in `tmp_path`, so what it writes lands there. The runner
    config's values that a test reads are set here, not taken from the user's YAML."""

    register_env()
    manager = ConfigManager()
    entry = SCENARIOS[SPATIAL]
    scenario = entry.build(
        SPATIAL.value,
        SCENARIOS[SPATIAL].base_conf(),
    )
    base = manager.load_config("runner_config.yaml", RunnerConf)
    runner_conf = base.model_copy(
        update={
            "common_conf": base.common_conf.model_copy(
                update={
                    "alg": alg,
                    "seed": 3,
                    "tag": None,
                    "training": True,
                    **(common or {}),
                }
            ),
            "train_conf": base.train_conf.model_copy(
                update={
                    "n_envs": 1,
                    "render_mode": None,
                    "csv_output": False,
                    "record_video": False,
                    "continue_training": False,
                    "tensorboard_output": False,
                    "model_output": True,
                    **(train or {}),
                }
            ),
            "eval_conf": base.eval_conf.model_copy(
                update={
                    "num_eval_episodes": 1,
                    "render_mode": None,
                    "csv_output": False,
                    "record_video": False,
                    **(eval or {}),
                }
            ),
        }
    )
    monkeypatch.chdir(tmp_path)
    runner = build_runner(scenario, runner_conf)
    assert isinstance(runner, SB3Runner)

    return runner


def _wrappers(env: Any) -> list[type]:
    """The classes in a gym env's wrapper chain, outermost first."""

    chain = []
    while env is not None:
        chain.append(type(env))
        env = getattr(env, "env", None)
    return chain


def _record_training_side_effects(
    runner: SB3Runner, monkeypatch: pytest.MonkeyPatch, model: object, env: object
) -> dict[str, int]:
    """Count the saves of `model` and `env` and the stop sounds, instead of writing a model or
    playing a sound."""

    counts = {"saves": 0, "sounds": 0}

    def save(saved_model, saved_env, unique_model_id):
        assert saved_model is model and saved_env is env
        assert unique_model_id == runner.get_unique_model_id()
        counts["saves"] += 1

    def sound():
        counts["sounds"] += 1

    monkeypatch.setattr(runner._artifact_manager, "save_model", save)
    monkeypatch.setattr(runner, "_play_training_stopped_sound", sound)

    return counts


# ================= #
#       Tests       #
# ================= #


class TestBuildEnv:
    """The env a run trains or evaluates on."""

    def test_a_frame_stack_stacks_the_observation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        plain = _runner(tmp_path, monkeypatch)._build_env(None, "train")
        stacked = _runner(tmp_path, monkeypatch, Algorithm.FSPPO)._build_env(
            None, "train"
        )
        plain_shape = plain.observation_space.shape
        assert plain_shape is not None

        assert stacked.observation_space.shape == (4 * plain_shape[0],)

    def test_training_runs_its_envs_and_eval_one_in_the_render_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        training = _runner(tmp_path, monkeypatch, train={"n_envs": 3})
        evaluating = _runner(
            tmp_path, monkeypatch, common={"training": False}, train={"n_envs": 3}
        )
        # Eval loads its normalization stats, so give it some to find
        evaluating._artifact_manager.load_normalize_wrapper = (  # type: ignore[method-assign]
            lambda env, training: VecNormalize(env)
        )

        trained_on = training._build_env("rgb_array", "train")
        evaluated_on = evaluating._build_env("rgb_array", "eval")

        assert trained_on.num_envs == 3
        assert evaluated_on.num_envs == 1
        assert trained_on.get_attr("render_mode") == ["rgb_array"] * 3

    def test_each_training_env_writes_its_own_csv_into_the_train_folder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, train={"n_envs": 3, "csv_output": True})

        runner._build_env(None, "train").close()

        run_folder = runner._log_dir / "train" / runner.get_unique_model_id()
        assert sorted(p.name for p in run_folder.iterdir()) == [
            "_env0.csv",
            "_env1.csv",
            "_env2.csv",
        ]


class TestMakeEnv:
    """The wrappers each env of a run gets."""

    def test_training_records_statistics_and_writes_csv_only_when_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        quiet = _runner(tmp_path, monkeypatch, eval={"csv_output": True})
        logged = _runner(tmp_path, monkeypatch, train={"csv_output": True})

        env = quiet._make_env(None, "train", env_idx=0)
        logged_env = logged._make_env(None, "train", env_idx=1)
        logged_env.close()

        assert RecordEpisodeStatistics in _wrappers(env)
        assert CSVEpisodeLogger not in _wrappers(env)
        assert isinstance(logged_env, CSVEpisodeLogger)
        csv = logged._log_dir / "train" / logged.get_unique_model_id() / "_env1.csv"
        assert csv.is_file()

    def test_eval_records_statistics_only_to_write_its_own_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        quiet = _runner(
            tmp_path,
            monkeypatch,
            common={"training": False},
            train={"csv_output": True},
        )
        logged = _runner(
            tmp_path, monkeypatch, common={"training": False}, eval={"csv_output": True}
        )

        env = quiet._make_env(None, "eval", env_idx=0)
        logged_env = logged._make_env(None, "eval", env_idx=0)
        logged_env.close()

        assert RecordEpisodeStatistics not in _wrappers(env)
        assert isinstance(logged_env, CSVEpisodeLogger)
        csv = logged._log_dir / "eval" / logged.get_unique_model_id() / "_env0.csv"
        assert csv.is_file()

    def test_only_the_first_env_records_video(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, train={"record_video": True})

        first = runner._make_env("rgb_array", "train", env_idx=0)
        second = runner._make_env("rgb_array", "train", env_idx=1)

        assert isinstance(first, RecordVideo)
        assert RecordVideo not in _wrappers(second)
        first.close()


class TestModel:
    """The model a fresh training run creates, and what is saved and loaded with it."""

    def test_a_fresh_run_gets_its_seed_its_device_and_a_tensorboard_folder_when_asked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, train={"tensorboard_output": True})
        quiet = _runner(tmp_path, monkeypatch)

        model = runner._resolve_training_model(runner._build_env(None, "train"))
        no_board = quiet._resolve_training_model(quiet._build_env(None, "train"))

        assert isinstance(model, PPO)
        assert model.seed == 3
        assert model.device == torch.device("cpu")
        assert model.tensorboard_log == str(runner._log_dir / "train")
        assert no_board.tensorboard_log is None

    def test_a_specs_hyper_parameter_wins_over_the_shared_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        shared = _runner(tmp_path, monkeypatch)
        monkeypatch.setitem(
            RUNNER,
            Algorithm.PPO,
            SB3Spec(PPO, device="cpu", hyper_parameters={"ent_coef": 0.5}),
        )
        own = _runner(tmp_path, monkeypatch)

        shared_model = shared._resolve_training_model(shared._build_env(None, "train"))
        own_model = own._resolve_training_model(own._build_env(None, "train"))

        assert isinstance(shared_model, PPO) and isinstance(own_model, PPO)
        assert shared_model.ent_coef != 0.5
        assert own_model.ent_coef == 0.5

    def test_a_saved_run_is_loaded_to_continue_and_to_evaluate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        fresh = _runner(tmp_path, monkeypatch)
        env = fresh._build_env(None, "train")
        model = fresh._resolve_training_model(env)
        fresh._maybe_save_model(model, env)
        resumed = _runner(
            tmp_path,
            monkeypatch,
            common={"agent_steps": "0"},
            train={"continue_training": True},
        )
        resumed_env = resumed._build_env(None, "train")
        resumed_model = resumed._resolve_training_model(resumed_env)
        evaluated_env = _runner(
            tmp_path, monkeypatch, common={"agent_steps": "0", "training": False}
        )._build_env(None, "eval")

        for saved, loaded in zip(
            model.policy.parameters(), resumed_model.policy.parameters()
        ):
            assert torch.equal(saved, loaded)
        assert isinstance(resumed_env, VecNormalize) and resumed_env.training
        assert isinstance(evaluated_env, VecNormalize) and not evaluated_env.training

    def test_nothing_is_saved_when_model_output_is_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, train={"model_output": False})
        env = runner._build_env(None, "train")

        runner._maybe_save_model(runner._resolve_training_model(env), env)

        assert list(runner._model_dir.iterdir()) == []


class TestTrainModel:
    """The training loop around `model.learn`."""

    def test_each_iteration_learns_on_from_the_last_and_saves(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path, monkeypatch, train={"iterations": 2, "timesteps": 64}
        )
        model, env = _LearningModel(), _ClosingEnv()
        counts = _record_training_side_effects(runner, monkeypatch, model, env)

        runner._train_model(model, env)  # type: ignore[arg-type]

        assert (
            model.learned
            == [
                {
                    "total_timesteps": 64,
                    "callback": None,
                    "tb_log_name": runner.get_unique_model_id(),
                    "reset_num_timesteps": False,
                }
            ]
            * 2
        )
        assert counts == {"saves": 2, "sounds": 1}
        assert env.closed

    def test_an_interrupt_saves_where_it_stopped_and_closes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, train={"iterations": 3})
        model, env = _LearningModel(interrupt=True), _ClosingEnv()
        counts = _record_training_side_effects(runner, monkeypatch, model, env)

        runner._train_model(model, env)  # type: ignore[arg-type]

        assert len(model.learned) == 1
        assert counts == {"saves": 1, "sounds": 1}
        assert env.closed

    def test_quitting_from_the_window_ends_the_remaining_iterations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path, monkeypatch, train={"iterations": 3, "render_mode": "human"}
        )
        model, env = _LearningModel(quit_from_window=True), _ClosingEnv()
        counts = _record_training_side_effects(runner, monkeypatch, model, env)

        runner._train_model(model, env)  # type: ignore[arg-type]

        assert len(model.learned) == 1
        assert isinstance(model.learned[0]["callback"], RunControlsCallback)
        assert counts["saves"] == 1


class TestEvalModel:
    """The eval loop's hand-off to the model."""

    def test_each_step_gets_the_state_and_episode_starts_the_last_one_left(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, common={"training": False})
        model = _RecordingModel()
        env = runner._make_dummy_vec_env(None, "eval")
        env.seed(0)

        runner._eval_model(env, model)  # type: ignore[arg-type]

        states = [state for state, _ in model.handed]
        starts = [start for _, start in model.handed]
        assert len(model.handed) >= 2
        assert states == [None, *range(1, len(states))]
        assert starts[0].all()
        assert all(s.shape == (1,) and not s.any() for s in starts[1:])
        assert all(model.deterministic)

    def test_a_human_run_titles_each_episode_and_stops_when_the_player_quits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path,
            monkeypatch,
            common={"training": False},
            eval={"render_mode": "human", "num_eval_episodes": 2},
        )
        played, quit_at_once = _OneStepEpisodes(), _OneStepEpisodes()
        handed: list[object] = []

        def keep_playing(renderer):
            handed.append(renderer)
            return False

        monkeypatch.setattr(runner, "_handle_run_controls", keep_playing)
        runner._eval_model(played, _RecordingModel())  # type: ignore[arg-type]
        monkeypatch.setattr(runner, "_handle_run_controls", lambda renderer: True)
        model = _RecordingModel()
        runner._eval_model(quit_at_once, model)  # type: ignore[arg-type]

        assert played.renderer.titles == [
            "SYNGrid · episode 1/2",
            "SYNGrid · episode 2/2",
        ]
        assert handed and all(r is played.renderer for r in handed)
        assert model.handed == []
        assert played.closed and quit_at_once.closed
