# Decisions: Orb identity

The number the agent sees for an orb: how it is built, why tier order shows in it, and where its dense form comes from.
The orb meta it is built from is in `orbs.md`.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## An orb's identity is the radix scalar, and tier order is meant to be seen (2026-11-12)

**Built by:** orb features, for the identity as an orb feature in the slot (see "An orb slot's columns are orb
features the scenario lists" in `observation.md`). The rest is built.

**Decision:** The identity the agent sees is one number per orb, built from the mixed-radix formula on category,
kind and tier, with tier in the top place, so the identity grows with tier. The agent sees it as its rank among the
scenario's orbs, the dense identity (see "An orb's identity is looked up in a table the scenario builds"), which keeps
the order. That the agent can read the chain's order from it is intended, not a leak. It is one orb feature in the
slot.

**Why:** The thesis tried three designs. Identity began as three fields, category, type and tier. After the move to
distance sorting, one-hot gave cleaner and faster convergence, but every tier was an identity of its own, so the
width grew with tier depth and performance fell from tier 3 to tier 10. The radix scalar kept one-hot's single clean
value at a fixed width, and the thesis named its cost: it is a magnitude, not a pure identity, and its limitations
section notes it may hand the stateless agent the order. The delay scenario even relied on the agent reading the
order from the identity.

The vision settles that cost. Tier depth makes a task longer for an agent but not harder for a person, who sees the
sequence at once, and what the agent sees should match what a person watching sees. A readable rank is what a person
gets, so it is not a leak.

The remaining cost is that the magnitude means rank for tier orbs and nothing between kinds. With effect orbs the
catalogue grows to around fifteen kinds, but a scenario will mostly hold about three, and a small network separates
three values easily. The owner's call is that the radix handles that.

Background, from the talk on 2026-11-11, so the reasoning can be found again:

- **MiniGrid** gives the agent a 7x7 view in which each cell holds three integers: object type, colour and state
  (open, closed, locked). It is a grid of facts, not pixels; pixels come from an optional wrapper. Most baselines run
  a small CNN over it and treat the integers as plain numbers, magnitude and all. It is close to the thesis's early
  three fields: each field means one thing, none carries a meant order.
- **MiniHack**, on the NetHack Learning Environment, gives each cell a glyph, an integer up to 5991 for every
  distinct entity, and baselines run it through an embedding layer, a learned vector per id, before a CNN. An
  embedding reads no magnitude, so nothing leaks and the width does not grow. Its own authors note the weak spot:
  NetHack's glyph ids are not well suited to learning, as more than half of them are of a type that never shows up
  in play. An embedding also learns every id on its own, so neighbouring tiers share nothing until training finds it,
  and ids the agent seldom sees, such as high tiers, learn poorly. It needs a custom network.
- **SYNGrid's radix** is a small id space designed for its job, with the one ordered property placed on purpose. It
  pays with a magnitude that means nothing between kinds, where MiniHack pays with a large, mostly empty id space and
  an embedding.

**Rejected:**

- Tier as a number of its own beside one flag per kind in the scenario: honest about which value is ordered, but
  wider, and its advantage at three kinds is small. It is the first thing to try if this is reopened.
- An embedding of the id in every observation: it needs a custom extractor for every run. It is the composite's way
  out instead (see the 7b note in the roadmap).
- MiniGrid's separate integers per field: the thesis moved away from composing identity from several fields.
- One-hot over identities: its width grows with tier depth.
- Hiding the order with shuffled ids: it would make the agent learn what a person sees at once.

**Revisit if:** a scenario holds clearly more than about three kinds, or an effect-orb scenario learns slower than
expected. The cheap check is an ablation on that scenario and its seeds: the radix against tier plus kind flags.

## An orb's identity is looked up in a table the scenario builds (2026-11-12)

**Built by:** phase 7a, the orb identity step.

**Decision:** The dense identity stays: the agent sees the orb's rank among the radix identities of the scenario's
orbs, 1 to N, in radix order. It no longer lives in the orb's meta. The meta holds kind and tier, its radix is a pure
function of them, and it is frozen. The scenario builds a table from radix identity to dense identity once, from the
kinds and tiers it holds, and the identity orb feature reads the orb's dense id from it. The world no longer
remaps.

**Why:** The dense id is a fact about the set of orbs, not about one orb, so a meta that knows only itself cannot
compute it. That is why the world remapped after every orb was built and wrote the result back into each meta, which
kept `IDENTITY` from being `Final`. What the agent sees is the observation's business; the renderer and digestion
never read the dense id. The table numbers the same orbs the same way the world does today, so the observation
should not change.

Where the remap came from is undecided. Two accounts:

- **The thesis's lessons notes** record it as a lesson of its own: the mixed-radix identity flatlined the LSTM under
  VecNormalize warm-up, and the dense sequential remap fixed it. They call it a narrower, debugging-flavoured lesson,
  kept apart from the dimensionality story of one-hot against scalar.
- **The owner recalls** it coming in with the one-hot experiments, where each identity needed a column of its own and
  dense ids number those columns, and staying when one-hot was dropped because an ablation showed it learning a
  little faster. The owner tried one-hot, and it did not scale, since tiers 1, 2 and 3 were each a kind of their own;
  how the remap fits with that is not clear.

`git log -S` does not settle it. The remap first appears in `ab70676` (2026-05-13), the same commit that brought in
the radix identity, which replaced three fields (category, type and tier). No one-hot encoding was ever committed
under `src/`, so the one-hot experiments may not have been committed, and nothing in the code mentions VecNormalize
or an LSTM.

The remap stays, for the even spacing it gives the scalar. Whether the plain radix does as well is for the experiment
named under "Revisit if".

**Rejected:**

- Dropping the remap (decided on 2026-11-11 and reversed here, once the radix scalar was kept): the only evidence
  there is says it helped.
- The meta handed its dense id when it is built: possible, since the scenario knows its kinds before any orb exists,
  but it keeps an observation concern in a core object.
- The world remapping and writing into the meta, as before: the meta cannot be frozen.

**Revisit if:** spatial, trained in 7a with the radix against the remapped radix on the same seeds, shows the remap
earns nothing. The same orb gets different dense ids in different scenarios, so revisit it too if one model has to
move between scenarios.
