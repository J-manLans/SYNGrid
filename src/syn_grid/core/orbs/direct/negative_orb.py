from syn_grid.config.models import NegativeConf
from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    OrbMeta,
    OrbCategory,
    DirectType,
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

    # ================= #
    #        API        #
    # ================= #

    def consume(self) -> "NegativeOrb":
        super()._consume()
        return self
