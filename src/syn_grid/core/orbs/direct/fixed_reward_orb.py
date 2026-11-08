"""
The fixed reward orb: a direct orb that pays the same reward every time it is eaten.
"""

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.core.orbs.orb_meta import DirectKind, OrbMeta


class FixedRewardOrb(BaseOrb):
    """A direct orb of the kind it is given, paying the reward it is given.

    The kind is what tells a negative orb from a positive one, to the agent, to
    digestion and to the renderer. The reward's sign is the config's to check.
    """

    # ================= #
    #       Init        #
    # ================= #

    def __init__(
        self,
        kind: DirectKind,
        reward: float,
        cool_down: int | None,
        life_span: int | None,
    ):
        super().__init__(reward, cool_down, life_span, OrbMeta(kind))
