"""Hypothesis's profiles for the whole suite.

`ci`, the default, is small and derandomized, so a run is quick and cannot fail by chance. `dev`
searches as far as wanted, for a run by hand: `HYPOTHESIS_PROFILE=dev pytest`.
"""

import os

from hypothesis import settings

settings.register_profile("ci", max_examples=25, derandomize=True, deadline=None)
settings.register_profile("dev", max_examples=1000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
