from syn_grid.legacy.config_legacy.models import NegativeConf
from syn_grid.legacy.core_legacy.orbs.base_orb import BaseOrb
from syn_grid.legacy.core_legacy.orbs.orb_meta import (
    DirectType,
    OrbCategory,
    OrbMeta,
)


class NegativeOrb(BaseOrb):
    """
    An orb that gives the agent a negative score.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, conf: NegativeConf):
        super().__init__(
            conf.reward,
            conf.cool_down,
            OrbMeta(category=OrbCategory.DIRECT, type=DirectType.NEGATIVE),
        )
