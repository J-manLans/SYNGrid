# Phase 13: Close out

**Produces:** fresh docs for people, and a naming pass once legacy is gone.

- Write `docs/architecture/` fresh, for people.
- A naming pass once everything is wired and legacy is gone. The scenario and family names are done, and the
  mechanic's names (`digesters/tier_chain/`, `TierChainPopulation`) stay.
- Drop the diagram tooling: `scripts/diagrams/` and the `diagrams` extra in `pyproject.toml`. The class diagrams
  ship as the owner's draw.io drawing, exported as a PNG with the diagram embedded, so draw.io can open it again.

**Done when:** the test suite still passes.
