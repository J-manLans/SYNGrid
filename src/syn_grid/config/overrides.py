from syn_grid.config.models import AgentConfig


def apply_overrides(agent_conf: AgentConfig, overrides: dict) -> None:
    """
    Apply a dict of overrides onto an AgentConfig, in place.

    Overrides may come from any source — CLI args (via
    `syn_grid.utils.args_utils.args_to_overrides`), a GUI, a test — as
    long as keys match fields on the agent's global, training, or
    evaluation configuration sub-models. Unrecognized keys are silently
    ignored.

    This function deliberately has no dependency on argparse or on
    which algorithms are registered, so it can be called from any
    override source without pulling in unrelated machinery.

    Args:
        agent_conf: The AgentConfig instance to update in place.
        overrides: Mapping of field name to override value.
    """

    for key, val in overrides.items():
        if hasattr(agent_conf.global_agent_conf, key):
            setattr(agent_conf.global_agent_conf, key, val)
        elif hasattr(agent_conf.train_agent_conf, key):
            setattr(agent_conf.train_agent_conf, key, val)
        elif hasattr(agent_conf.eval_agent_conf, key):
            setattr(agent_conf.eval_agent_conf, key, val)
