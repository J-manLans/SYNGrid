import pytest
from synergygrid.core.resources import (
    DirectType,
    ResourceCategory,
    ResourceMeta,
    SynergyType,
)


class TestResourceMeta:
    """
    Unit tests for ResourceMeta.

    These tests validate:
    - Correct enum assignments
    - Tier default behavior
    - Explicit tier behavior
    - Validation of invalid inputs
    """

    def test_direct_positive_initialization(self):
        """
        Verify that a DIRECT POSITIVE resource initializes correctly.

        Expected behavior:
        - category is ResourceCategory.DIRECT
        - type is DirectType.POSITIVE
        - tier defaults to 0 when not provided
        """
        meta = ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER_BASE)

        assert meta.category == ResourceCategory.SYNERGY
        assert meta.type == SynergyType.TIER_BASE
        assert meta.tier == 0

    def test_direct_negative_initialization(self):
        """
        Verify that a DIRECT NEGATIVE resource initializes correctly.

        Ensures:
        - category and type are stored correctly
        - tier defaults to 0
        """
        meta = ResourceMeta(ResourceCategory.DIRECT, DirectType.NEGATIVE)

        assert meta.category == ResourceCategory.DIRECT
        assert meta.type == DirectType.NEGATIVE
        assert meta.tier == 0

    def test_synergy_tier_initialization(self):
        """
        Verify that a SYNERGY TIER resource initializes correctly.

        Ensures:
        - category is ResourceCategory.SYNERGY
        - type is SynergyType.TIER
        - tier defaults to 0
        """
        meta = ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER)

        assert meta.category == ResourceCategory.SYNERGY
        assert meta.type == SynergyType.TIER
        assert meta.tier == 0

    @pytest.mark.parametrize("tier", [1, 2, 5])
    def test_explicit_tier(self, tier):
        """
        Verify that providing a positive tier value
        correctly overrides the default.

        Expected behavior:
        - tier equals the explicitly provided value
        """
        meta = ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER, tier)

        assert meta.tier == tier

    def test_explicit_none_tier_defaults_to_zero(self):
        """
        Ensure that explicitly passing tier=None
        results in tier being set to 0.

        This confirms consistent default handling.
        """
        meta = ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER_BASE, None)

        assert meta.tier == 0

    def test_negative_tier_raises(self):
        """
        Ensure that passing a negative tier value
        raises a ValueError.

        Negative tiers are considered invalid domain input.
        """
        with pytest.raises(ValueError):
            ResourceMeta(ResourceCategory.SYNERGY, SynergyType.TIER, -1)

    def test_mismatch_category_and_type_raises(self):
        """
        Ensure that mismatched category/type combinations
        raise a TypeError.

        Example:
        - DIRECT category must use DirectType
        - SYNERGY category must use SynergyType
        """
        with pytest.raises(TypeError):
            ResourceMeta(ResourceCategory.DIRECT, SynergyType.TIER)
