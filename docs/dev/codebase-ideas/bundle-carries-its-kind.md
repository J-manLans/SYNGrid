# A bundle that carries everything its orb kind brings (2026-10-12)

**Returns when:** a second family uses an orb kind that one already uses, or phase 7 or 8b settles what a kind adds
to the observation or the HUD.

**Idea:** An `OrbBundle` holds a population and a way to make a digester. A kind brings more than that to a scenario:
the metrics that read its digester, a tag suffix, HUD elements and observation features. The bundle could carry some
of these, and a family's entry type would collect them from its bundles as it now sums their counts.

**Where it came from:** Adding the negative orb to the single tier chain family. Its `if` block in
`SingleTierChainEntry._compose` adds the bundle, extends the tag and adds the `negatives_eaten` metric: three lines that
together mean "add this brick".

**Why it might matter:**

- The contributions are of three sorts. The population and the digester must exist together: an orb with no digester
  makes the engine raise. The count and the metrics follow from the kind being present, with nothing to decide. What
  the HUD shows and what the agent observes are the scenario's choice, and a kind should not make it.
- Only the first sort is in the bundle, and the count is taken from the population. That is enough while one family
  uses the negative orb, because its `if` block is one readable place.
- The moment to act is a second family using the same kind. Each would then repeat the lines, and they could drift.
  The first candidate is a `metrics` field on the bundle.
- A metric's reader is the same in every family: take the kind's digester from the engine and read a counter. So the
  readers would move out of a family's `builders.py` and live with the kind. Followed to the end, that is one module per
  kind, which groups by kind where `scenario/blocks/populations/` and `scenario/blocks/recipes/` group by role.
- The HUD's form and the observation's features are open, in phases 8b and 7. Putting either in the bundle decides
  them early.
