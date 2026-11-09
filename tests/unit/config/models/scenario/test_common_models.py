"""What the blocks every scenario is built from accept: the penalties, the droid's energy and the step limit."""

import pytest
from pydantic import ValidationError

from syn_grid.config.models.scenario.common_models import (
    DroidConf,
    GridConf,
    PenaltyCheckedConf,
    WorldConf,
)

# ================= #
#     Stand-ins     #
# ================= #


class _WithANewPenalty(PenaltyCheckedConf, frozen=True, extra="forbid", strict=True):
    """A block a later family might add, with a penalty no current block has."""

    new_penalty: float


# ================= #
#      Helpers      #
# ================= #


def _droid(
    *,
    max_energy: float = 10.0,
    step_penalty: float = -0.1,
    boundary_penalty: float = -1.0,
) -> DroidConf:
    return DroidConf(
        max_energy=max_energy,
        step_penalty=step_penalty,
        boundary_penalty=boundary_penalty,
    )


def _world(*, max_steps: int = 100) -> WorldConf:
    return WorldConf(
        max_steps=max_steps,
        grid_conf=GridConf(grid_rows=5, grid_cols=5),
        droid_conf=_droid(),
    )


# ================= #
#       Tests       #
# ================= #


class TestPenalties:
    """A penalty costs or is free, and never pays: every field named `*_penalty` is 0 or negative."""

    def test_a_penalty_of_0_is_accepted_and_one_above_0_is_refused(self):
        _droid(step_penalty=0.0, boundary_penalty=0.0)

        with pytest.raises(ValidationError, match="step_penalty"):
            _droid(step_penalty=0.01)
        with pytest.raises(ValidationError, match="boundary_penalty"):
            _droid(boundary_penalty=0.01)

    def test_a_block_that_extends_it_has_its_penalties_checked_by_their_name_alone(
        self,
    ):
        _WithANewPenalty(new_penalty=0.0)

        with pytest.raises(ValidationError, match="new_penalty"):
            _WithANewPenalty(new_penalty=0.01)


class TestDroidConf:
    """The droid's energy."""

    def test_it_needs_some_energy_to_start_with(self):
        # At 0 the episode would end on its first step, before the droid could do anything
        _droid(max_energy=0.01)

        with pytest.raises(ValidationError, match="max_energy"):
            _droid(max_energy=0.0)


class TestWorldConf:
    """How long an episode can last."""

    def test_it_lasts_at_least_one_step(self):
        _world(max_steps=1)

        with pytest.raises(ValidationError, match="max_steps"):
            _world(max_steps=0)
