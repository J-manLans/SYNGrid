import pytest
from synergygrid.core.resources import TierResource, PositiveResource

# ================= #
#      Helpers      #
# ================= #


def _tier_params(max_tier=10):
    return [(TierResource(t, (5, 5)), list(range(t))) for t in range(1, max_tier + 1)]


class TestTier:
    """
    Unit tests for the TierResource class.

    Verifies:
    - Step-wise reward behavior for correctly chained tiers
    - Step-wise reward behavior for incomplete or invalid tier chains
    - Combo reward behavior for correctly chained tiers
    - Combo reward behavior for incomplete or invalid tier chains
    - That the resource updates _chained_tiers correctly during consumption
    """

    @pytest.fixture
    def resource(self):
        """
        Creates a Tier 1 resource with the world boundaries set to 5x5 used to calculate its life span.
        """
        tr = TierResource(1, (5, 5))
        tr.reset()

        return tr

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_tier_is_correct_in_meta(self, resource: TierResource, resource_chain):
        """Test that the tier is correct."""

        assert resource.meta.tier == len(resource_chain)

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_successful_step_wise_consume(self, resource: TierResource, resource_chain):
        """Test that the tier reward is given correctly for a valid step-wise chain."""

        resource.reset()
        resource._step_wise_scoring_type = True
        resource._chained_tiers.extend(resource_chain)

        assert resource.consume() == resource._REWARD

    def test_unsuccessful_step_wise_consume(self, resource: TierResource):
        """Test that step-wise consume returns 0 when the tier chain is invalid for the resource."""

        resource._step_wise_scoring_type = True
        resource._chained_tiers.extend([0, 1, 2])

        assert resource.consume() == 0

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_successful_combo_consume(self, resource: TierResource, resource_chain):
        """Test that combo consume gives the correct reward when the tier chain is complete."""

        resource.reset()
        resource._step_wise_scoring_type = False
        resource._chained_tiers.extend(resource_chain)

        assert resource.consume() == 0

    def test_unsuccessful_combo_consume(self, resource: TierResource):
        """
        Test that combo consume returns the expected value when the tier chain is invalid for the resource.
        """

        resource._step_wise_scoring_type = False
        resource._chained_tiers.extend([0, 1, 2])

        assert resource.consume() == resource._REWARD

    @pytest.mark.parametrize("resource, resource_chain", _tier_params())
    def test_consume_updates_tiers_list(self, resource: TierResource, resource_chain):
        """Test that consume adds the current tier to the chain correctly."""

        resource.reset()
        resource._chained_tiers.extend(resource_chain)  # setup initial chain
        resource.consume()

        assert resource._chained_tiers[-1] == resource.meta.tier
        expected_chain = list(range(resource.meta.tier + 1))  # Check correct order
        assert resource._chained_tiers == expected_chain

    def test_bad_consume_breaks_tiers_list(self, resource: TierResource):
        """Test that bad consume breaks the tier chain."""

        resource.reset()
        resource._chained_tiers.extend([0, 1, 2, 3])  # setup initial chain
        resource.consume()

        assert len(resource._chained_tiers) == 0

    def test_base_consume_breaks_and_restarts_tiers_list(self):
        """Test that bad consume of a tier 0 resource breaks and restarts the tier chain."""

        base_resource = PositiveResource((5, 5))

        base_resource.reset()
        base_resource._chained_tiers.extend([0, 1, 2, 3])  # setup initial chain
        base_resource.consume()

        assert base_resource._chained_tiers[-1] == base_resource.meta.tier
        assert len(base_resource._chained_tiers) == 1
