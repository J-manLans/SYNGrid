import pytest
from pydantic import ValidationError

from syn_grid.config.models import EvalAgentConf, TrainAgentConf


class TestTrainAgentConfValidators:
    """
    Covers TrainAgentConf.validate_config, with a focus on the record_video / render_mode
    decoupling introduced alongside the PygameRenderer
    """

    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def base_kwargs(self) -> dict:
        return {
            "continue_training": False,
            "csv_output": False,
            "tensorboard_output": False,
            "model_output": False,
            "n_envs": 1,
            "timesteps": 1000,
            "iterations": 1,
            "render_mode": None,
            "record_video": False,
            "rec_interval": 100,
            "rec_length": 10,
        }

    # ================= #
    #       Tests       #
    # ================= #

    def test_invalid_render_mode_value_raises(self, base_kwargs):
        base_kwargs["render_mode"] = "not_a_real_mode"

        with pytest.raises(ValidationError, match="render mode"):
            TrainAgentConf(**base_kwargs)

    def test_human_mode_with_multiple_envs_raises(self, base_kwargs):
        base_kwargs["render_mode"] = "human"
        base_kwargs["n_envs"] = 4

        with pytest.raises(ValidationError, match="n_envs=1"):
            TrainAgentConf(**base_kwargs)

    @pytest.mark.parametrize(
        ("render_mode", "record_video", "should_raise"),
        [
            # record_video requires rgb_array specifically
            ("rgb_array", True, False),
            ("human", True, True),
            (None, True, True),
            # rgb_array on its own must NOT require record_video — that's
            # the actual point of decoupling them: manual frame capture and
            # future pixel observations (GridPixel) need rgb_array without
            # triggering RecordVideo.
            ("rgb_array", False, False),
            ("human", False, False),
            (None, False, False),
        ],
    )
    def test_record_video_render_mode_combinations(
        self, base_kwargs, render_mode, record_video, should_raise
    ):
        base_kwargs["render_mode"] = render_mode
        base_kwargs["record_video"] = record_video

        if should_raise:
            with pytest.raises(ValidationError, match="record_video requires"):
                TrainAgentConf(**base_kwargs)
        else:
            TrainAgentConf(**base_kwargs)


class TestEvalAgentConfValidators:
    """Same coverage as TrainAgentConf, minus the n_envs concern (eval always uses one env)."""

    # ================= #
    #     Fixtures      #
    # ================= #

    @pytest.fixture
    def base_kwargs(self) -> dict:
        return {
            "num_eval_episodes": 5,
            "render_mode": None,
            "record_video": False,
            "rec_episode": 1,
            "csv_output": False,
        }

    # ================= #
    #       Tests       #
    # ================= #

    def test_invalid_render_mode_value_raises(self, base_kwargs):
        base_kwargs["render_mode"] = "not_a_real_mode"

        with pytest.raises(ValidationError, match="render mode"):
            EvalAgentConf(**base_kwargs)

    @pytest.mark.parametrize(
        ("render_mode", "record_video", "should_raise"),
        [
            ("rgb_array", True, False),
            ("human", True, True),
            (None, True, True),
            ("rgb_array", False, False),
            ("human", False, False),
            (None, False, False),
        ],
    )
    def test_record_video_render_mode_combinations(
        self, base_kwargs, render_mode, record_video, should_raise
    ):
        base_kwargs["render_mode"] = render_mode
        base_kwargs["record_video"] = record_video

        if should_raise:
            with pytest.raises(ValidationError, match="record_video requires"):
                EvalAgentConf(**base_kwargs)
        else:
            EvalAgentConf(**base_kwargs)
