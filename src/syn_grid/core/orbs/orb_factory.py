from syn_grid.config.models.common_models import NegOrbConf
from syn_grid.config.models.tier_chain import TierOrbConf

from syn_grid.core.orbs.base_orb import BaseOrb
from syn_grid.scenario.rules.population import OrbPopulation


class OrbFactory:
    """Sets up orb class state, then hands pool construction to the scenario.

    The ``_life_span`` write here is load-bearing and stays: it is a class
    attribute shared by every GridWorld in the process, several tests depend on
    that, and a perception snapshots the value at construction. It is also not
    scenario-specific -- it is the grid's Manhattan diameter -- so it belongs to
    infrastructure rather than to a scenario's rules.

    ``TierOrb.max_tier`` used to be written here too and is not any more. It was
    only ever read inside ``TierOrb.__init__``, so passing it in removed a
    global that two worlds in one process would overwrite, and that made the
    population rules untestable without standing up the factory that calls them.

    What *is* scenario-specific is how the pool is built, and that arrives as an
    ``OrbPopulation``. This class no longer decides which kind of world it is
    building.
    """

    def __init__(
        self,
        grid_dimension: tuple[int, int],
        negative_orb_conf: NegOrbConf,
        tier_orb_conf: TierOrbConf,
        population: OrbPopulation,
    ):
        self._grid_rows, self._grid_cols = grid_dimension
        self.negative_orb_conf = negative_orb_conf
        self.tier_orb_conf = tier_orb_conf
        self._population = population

    # ================= #
    #        API        #
    # ================= #

    def create_orbs(self) -> list[BaseOrb]:
        """Create all orbs according to the scenario's population rules"""

        # Shared setup
        BaseOrb.set_life_span(
            self._grid_rows, self._grid_cols
        )

        return self._population.create()
