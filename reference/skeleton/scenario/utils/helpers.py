from syn_grid.config.models.common_scenario_models import ScenarioConf


def neg_orb(scenario_conf: ScenarioConf) -> str:
    return "_Neg" if scenario_conf.neg_orb_conf else ""