from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import (
    DirectKind,
    OrbMeta,
)


class NegativeOrb(BaseOrb):
    """
    An orb that gives the agent a negative reward (penalty).
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(self, penalty: float, cool_down: int, life_span: int):
        super().__init__(
            penalty,
            cool_down,
            life_span,
            OrbMeta(DirectKind.NEGATIVE),
        )
