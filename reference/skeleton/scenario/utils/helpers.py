from syn_grid.config.models.scenarios.common_models import ScenarioConf


def neg_orb(scenario_conf: ScenarioConf) -> str:
    return "_Neg" if scenario_conf.world_conf.orb_conf.negative else ""