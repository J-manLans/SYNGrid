"""What every agent runner gets from the base: its id, its folders, its env and wrappers, the run
controls, and finding a saved file."""

import os
import time
from pathlib import Path
from typing import Any, override

import pygame
import pytest
from gymnasium.wrappers import RecordVideo

from syn_grid.config.config_manager import ConfigManager
from syn_grid.config.models.global_models import ScenarioName
from syn_grid.config.models.runner_models import Algorithm, RunnerConf
from syn_grid.gymnasium.utils.env_factory import register_env
from syn_grid.runners.agent_runners.base_agent_runner import BaseAgentRunner
from syn_grid.scenario.registry import SCENARIOS

# ================= #
#     Constants     #
# ================= #


SPATIAL = ScenarioName.GOAL_SINGLE_TIER_CHAIN_SPATIAL


# ================= #
#     Stand-ins     #
# ================= #


class _OwnRunner(BaseAgentRunner):
    """A runner of a user's own, as the base expects one."""

    @override
    def train(self) -> None: ...

    @override
    def eval(self) -> None: ...


class _PauseRecorder:
    """Stands in for the renderer the run controls are handed: counts the pause screens."""

    def __init__(self):
        self.pauses = 0

    def draw_pause(self) -> None:
        self.pauses += 1


# ================= #
#      Helpers      #
# ================= #


def _runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    common: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    eval: dict[str, Any] | None = None,
) -> _OwnRunner:
    """A runner of spatial, working in `tmp_path`, so its output folders are made there. The
    runner config's values that a test reads are set here, not taken from the user's YAML."""

    register_env()
    manager = ConfigManager()
    entry = SCENARIOS[SPATIAL]
    scenario = entry.build(
        SPATIAL.value,
        entry.full_conf(manager.load_raw(manager.scenario_config_path(SPATIAL))),
    )
    base = manager.load_config("runner_config.yaml", RunnerConf)
    runner_conf = base.model_copy(
        update={
            "common_conf": base.common_conf.model_copy(
                update={
                    "alg": Algorithm.PPO,
                    "seed": 3,
                    "tag": None,
                    "training": True,
                    **(common or {}),
                }
            ),
            "train_conf": base.train_conf.model_copy(
                update={"record_video": False, **(train or {})}
            ),
            "eval_conf": base.eval_conf.model_copy(
                update={"record_video": False, **(eval or {})}
            ),
        }
    )
    monkeypatch.chdir(tmp_path)

    return _OwnRunner(scenario, runner_conf)


def _key(key: int) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))


def _later(event: pygame.event.Event) -> None:
    """Post `event` once the pause has started. The run controls read the queue in one batch and
    drop what follows Space in it, so an event meant for the pause screen has to come after."""

    pygame.time.set_timer(event, 50, loops=1)


# ================= #
#     Fixtures      #
# ================= #


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch):
    """Pygame's event queue on a dummy display, emptied before each test."""

    monkeypatch.setitem(os.environ, "SDL_VIDEODRIVER", "dummy")
    pygame.display.init()
    pygame.event.clear()
    yield
    pygame.display.quit()


# ================= #
#       Tests       #
# ================= #


class TestRunId:
    """The run's id, which names its files and finds its checkpoints by prefix."""

    def test_is_perception_seed_scenario_tag_and_algorithm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch)
        tag = runner.scenario.scenario_tag

        assert runner._id == f"vector_fog_of_war_seed3__TAG_{tag}__PPO"
        assert runner.get_unique_model_id() == f"{runner._id}_{runner._date}"

    def test_a_users_tag_follows_the_scenarios_and_the_algorithm_stays_last(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, common={"tag": "mine"})
        tag = runner.scenario.scenario_tag

        assert runner._id == f"vector_fog_of_war_seed3__TAG_{tag}_mine__PPO"


class TestWrapEpisodeStats:
    """Episode statistics, with the CSV logger on top when a folder is given."""

    def test_each_env_writes_its_own_csv_under_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch)
        run = runner._log_dir / "train" / runner.get_unique_model_id()

        first = runner._wrap_episode_stats(runner._make_raw_env(None), "train")
        second = runner._wrap_episode_stats(runner._make_raw_env(None), "train", 1)
        first.close()
        second.close()

        assert sorted(p.name for p in run.iterdir()) == ["_env0.csv", "_env1.csv"]


class TestMaybeWrapVideo:
    """Video recording, when the config asks for it."""

    def test_training_records_every_interval_shared_among_the_envs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path,
            monkeypatch,
            train={
                "record_video": True,
                "rec_interval": 100,
                "n_envs": 3,
                "rec_length": 7,
            },
        )

        video = runner._maybe_wrap_video(runner._make_raw_env("rgb_array"))

        assert isinstance(video, RecordVideo)
        assert video.step_trigger is not None
        assert [t for t in range(70) if video.step_trigger(t)] == [0, 33, 66]
        assert video.video_length == 7
        assert Path(video.video_folder) == (
            tmp_path / "output" / "results" / "videos" / runner.get_unique_model_id()
        )
        video.close()

    def test_an_interval_below_the_env_count_records_every_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path,
            monkeypatch,
            train={"record_video": True, "rec_interval": 2, "n_envs": 4},
        )

        video = runner._maybe_wrap_video(runner._make_raw_env("rgb_array"))

        assert isinstance(video, RecordVideo) and video.step_trigger is not None
        assert all(video.step_trigger(t) for t in range(5))
        video.close()

    def test_eval_records_the_configured_episode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(
            tmp_path,
            monkeypatch,
            common={"training": False},
            eval={"record_video": True, "rec_episode": 3},
        )

        video = runner._maybe_wrap_video(runner._make_raw_env("rgb_array"))

        assert isinstance(video, RecordVideo) and video.episode_trigger is not None
        assert [e for e in range(6) if video.episode_trigger(e)] == [3]
        video.close()

    def test_without_recording_the_env_is_left_as_it_is(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Video switched on only for the other mode, so neither branch may record
        training = _runner(tmp_path, monkeypatch, eval={"record_video": True})
        evaluating = _runner(
            tmp_path,
            monkeypatch,
            common={"training": False},
            train={"record_video": True},
        )

        for runner in (training, evaluating):
            env = runner._make_raw_env(None)
            assert runner._maybe_wrap_video(env) is env


class TestRunControls:
    """The window's input between two steps of a run with a human renderer."""

    def test_escape_or_closing_the_window_quits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events
    ):
        runner = _runner(tmp_path, monkeypatch)

        _key(pygame.K_ESCAPE)
        assert runner._handle_run_controls(_PauseRecorder())  # type: ignore[arg-type]

        pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert runner._handle_run_controls(_PauseRecorder())  # type: ignore[arg-type]

    def test_other_input_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events
    ):
        runner = _runner(tmp_path, monkeypatch)
        renderer = _PauseRecorder()

        _key(pygame.K_a)

        assert not runner._handle_run_controls(renderer)  # type: ignore[arg-type]
        assert renderer.pauses == 0

    def test_space_pauses_until_space_resumes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events
    ):
        runner = _runner(tmp_path, monkeypatch)
        renderer = _PauseRecorder()

        _key(pygame.K_SPACE)
        _later(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE))

        assert not runner._handle_run_controls(renderer)  # type: ignore[arg-type]
        assert renderer.pauses == 1

    def test_quitting_while_paused_quits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events
    ):
        runner = _runner(tmp_path, monkeypatch)

        _key(pygame.K_SPACE)
        _later(pygame.event.Event(pygame.QUIT))

        assert runner._handle_run_controls(_PauseRecorder())  # type: ignore[arg-type]


class TestFindLatestSavedPath:
    """The saved file a run loads: the configured steps and this run's id."""

    def test_the_newest_match_wins_and_other_runs_are_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        runner = _runner(tmp_path, monkeypatch, common={"agent_steps": "512"})
        older = tmp_path / f"512_{runner._id}_older.zip"
        newer = tmp_path / f"512_{runner._id}_newer.zip"
        other_steps = tmp_path / f"1024_{runner._id}_newest.zip"
        for path in (older, newer, other_steps):
            path.touch()
        now = time.time()
        os.utime(older, (now - 20, now - 20))
        os.utime(newer, (now - 10, now - 10))

        assert runner._find_latest_saved_path(tmp_path) == newer

    def test_no_steps_or_no_match_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        no_steps = _runner(tmp_path, monkeypatch, common={"agent_steps": ""})
        no_match = _runner(tmp_path, monkeypatch, common={"agent_steps": "512"})

        with pytest.raises(ValueError, match="steps"):
            no_steps._find_latest_saved_path(tmp_path)
        with pytest.raises(FileNotFoundError, match=f"512_{no_match._id}"):
            no_match._find_latest_saved_path(tmp_path)
