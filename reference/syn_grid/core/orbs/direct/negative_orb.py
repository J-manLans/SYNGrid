from syn_grid.config.models.scenario.orb_models import NegOrbConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    DirectType,
    OrbMeta,
)


class NegativeOrb(BaseOrb):
    """
    An orb that gives the agent a negative score.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: NegOrbConf):
        super().__init__(
            conf.reward,
            conf.cool_down,
            OrbMeta(type=DirectType.NEGATIVE),
        )
