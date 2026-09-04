from syn_grid.utils.args_utils import args_to_overrides

from argparse import Namespace


class TestArgsToOverrides:
    # ================= #
    #       Tests       #
    # ================= #

    def test_filters_out_none_values(self):
        args = Namespace(alg_index=None, steps="1000", id=None, train=True)

        overrides = args_to_overrides(args)

        assert overrides == {"steps": "1000", "train": True}

    def test_returns_empty_dict_when_all_none(self):
        args = Namespace(alg_index=None, steps=None, id=None)

        overrides = args_to_overrides(args)

        assert overrides == {}

    def test_preserves_falsy_but_explicit_values(self):
        # False/0/"" are explicitly set values, not the None sentinel,
        # and must survive the filter.
        args = Namespace(train=False, timesteps=0, id="")

        overrides = args_to_overrides(args)

        assert overrides == {"train": False, "timesteps": 0, "id": ""}

