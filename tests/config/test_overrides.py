from syn_grid.config.overrides import apply_overrides
from tests.utils.config_helpers import get_test_config


class TestApplyOverrides:
    # ================= #
    #       Tests       #
    # ================= #

    def test_updates_global_agent_conf_field(self):
        agent_conf = get_test_config().agent

        apply_overrides(agent_conf, {"human_control": True})

        assert agent_conf.global_agent_conf.human_control is True

    def test_updates_train_agent_conf_field(self):
        agent_conf = get_test_config().agent

        apply_overrides(agent_conf, {"timesteps": 999})

        assert agent_conf.train_agent_conf.timesteps == 999

    def test_updates_eval_agent_conf_field(self):
        agent_conf = get_test_config().agent

        apply_overrides(agent_conf, {"num_eval_episodes": 5})

        assert agent_conf.eval_agent_conf.num_eval_episodes == 5

    def test_ignores_unrecognized_keys(self):
        agent_conf = get_test_config().agent
        original = agent_conf.model_copy(deep=True)

        apply_overrides(agent_conf, {"not_a_real_field": 123})

        assert agent_conf == original

    def test_empty_overrides_changes_nothing(self):
        agent_conf = get_test_config().agent
        original = agent_conf.model_copy(deep=True)

        apply_overrides(agent_conf, {})

        assert agent_conf == original

    def test_multiple_overrides_across_sub_confs_in_one_call(self):
        agent_conf = get_test_config().agent

        apply_overrides(
            agent_conf,
            {"human_control": True, "timesteps": 42, "num_eval_episodes": 7},
        )

        assert agent_conf.global_agent_conf.human_control is True
        assert agent_conf.train_agent_conf.timesteps == 42
        assert agent_conf.eval_agent_conf.num_eval_episodes == 7
