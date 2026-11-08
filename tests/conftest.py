"""Hypothesis's profiles for the whole suite.

`ci`, the default, is small and derandomized, so a run is quick and cannot fail by chance. `dev`
searches as far as wanted, for a run by hand: `HYPOTHESIS_PROFILE=dev pytest`.

Both let a test method run on more than one instance of its class. mutmut runs the suite more
than once in one process, which gives each method a new instance each time, and the test classes
hold no state for that to change.
"""

import os

from hypothesis import HealthCheck, settings

_SUPPRESSED = [HealthCheck.differing_executors]

settings.register_profile(
    "ci",
    max_examples=25,
    derandomize=True,
    deadline=None,
    suppress_health_check=_SUPPRESSED,
)
settings.register_profile(
    "dev", max_examples=1000, deadline=None, suppress_health_check=_SUPPRESSED
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
