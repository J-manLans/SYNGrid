"""What the orb config blocks accept."""

import pytest
from pydantic import ValidationError

from syn_grid.config.models.scenario.orb_models import TierOrbConf

# ================= #
#       Tests       #
# ================= #


class TestTierOrbConf:
    """The tier orb's block."""

    def test_a_max_tier_of_1_is_rejected(self):
        with pytest.raises(ValidationError, match="max_tier"):
            TierOrbConf(
                max_tier=1, base_reward=1.0, growth_factor=1.0, chain_break_penalty=-0.1
            )
