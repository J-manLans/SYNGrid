# Decisions: Orbs

What an orb is and carries: its categories and meta, its classes, its timers, and how a population weighs it. The
identity the agent sees, built from the meta, is in `orb-identity.md`.

Written in the present tense, in the format of the template in `../design-decisions.md`. A decision added here also
gets its title in that index.

---

## Two orb categories, and an orb meta that stores only kind and tier (2026-10-12)

**Decision:** There are two orb categories and there will not be a third.

- **Direct:** changes the score and nothing else, on the step it is eaten. The number may be fixed, drawn, or depend
  on the orb's own age.
- **Synergy:** everything else. It leaves state behind that changes a later outcome.

The test: does its digester need anything but the orb itself? If it needs the droid, the world or a memory of earlier
orbs, it is synergy, even if it happens at once (an orb bomb, a teleport).

Inside synergy are two groups, as folders only: **chain** (the payoff depends on which orbs of the kind came before;
the tier orb) and **effect** (it changes something other than itself).

`OrbMeta` says what an orb is: `kind`, a member of `DirectKind` or `SynergyKind`, one per orb that behaves
differently; and `tier`, an int for `SynergyKind.TIER` and `None` for every other kind. The category follows from the
enum. The routing key is `OrbKind = DirectKind | SynergyKind`, defined in `orb_meta.py`. The word is kind
everywhere, since `type` is a builtin.

The values of `OrbCategory`, `DirectKind` and `SynergyKind` start at 1, so 0 stays free for an observation to mean
"no orb", as a 2D observation that writes the kind into a cell would need. The perception writes that 0; it is not a
member of the enums.

**Why:** Direct stays a small closed set, which lines up with the digesters: a direct digester holds nothing a later
orb's payout depends on (its count of orbs eaten is for the metrics), while a synergy digester holds episode state that
changes what a later orb pays. A new orb is an enum member and a digester, with no new field.

**Rejected:**

- An `EffectKind` enum beside `SynergyKind`: it reads as if effect orbs were not synergy.
- One `SynergyKind.EFFECT` with a sub-field: each effect needs its own digester, so the key must tell them apart.
- Storing how an effect acts in the meta: those are capabilities of its digester.
- Direct meaning "happens at once": an orb bomb happens at once and changes more than the score.
- A stored `CATEGORY`, or a separate meta class for tier orbs: the enum already says it, and readers would check the
  kind anyway.
- A `NONE` member in each enum to hold the 0: it is a kind no orb can be, an `OrbMeta` accepted it, and it widened
  every slot of the radix identity by one.

**Revisit if:** an orb turns up that fits neither test.

## A direct orb is one class per way of paying, and one digester per kind (2026-11-08)

**Decision:** `FixedRewardOrb` is the direct orb that pays the same reward every time. It takes its kind and its
reward, so the negative orb and the positive orb are one class: `FixedRewardOrb(DirectKind.NEGATIVE, ...)` and
`FixedRewardOrb(DirectKind.POSITIVE, ...)`. The kind is what the agent, digestion and the renderer tell them apart by.
The reward's sign is checked by the kind's config block, not by the orb.

`DirectDigester` holds the rules for every direct kind: it pays the orb's reward, counts it in `eaten`, and owes
nothing. Each kind has a subclass that only names its kind, `NegDigester` and `PosDigester`, in the same file, and a
new direct kind adds its subclass there: the file is the whole package, so there is no `direct/` folder. The
orb decides how much it pays and the digester collects it, so a drawn or an ageing direct orb would be a class of its
own beside `FixedRewardOrb` and use the same base digester. No shared base orb class is written until such an orb
exists.

**Why:** The negative and the positive orb differ only in their kind and their sign. One class per kind would be two
near copies. The engine routes on the kind and readers look a digester up by its class, so a subclass per kind fits
both without changing the engine.

**Rejected:**

- One kind for both, with the sign in the reward: the agent could not tell a hazard from a treat. Orbs that look alike
  and pay differently are a scenario of their own, and this class serves it with one kind.
- `DirectOrb` as the name: the category also covers drawn and ageing rewards, and the name would claim them.
- One `DirectDigester` instance per kind with no subclasses: `get_digester` could not tell them apart, so the engine
  would have to look digesters up by kind.

## An orb carries its own cool-down and lifespan (2026-10-18)

**Decision:** Whether an orb returns after leaving the field, and whether it expires on it, are values of the orb,
set by the population that makes it. Each is a number of steps or `None`: no cool-down is an orb that does not return
in the episode, no lifespan is one that does not expire. `BaseOrb` answers `has_expired()` and `is_ready()`, and a
timer set to `None` never completes.

The world treats every orb alike and asks no kind. An orb with no cool-down stays inactive for the episode but is not
thrown away: the next reset places it again, and the observation keeps its slot.

Populations take these as values and know no scenario. They are in `scenario/blocks/populations/`, shared by every
family, as the counterpart of `make_digester`.

An orb that despawns gets `_reachable_life_span(rows, cols)`, the longest walk between two cells, so the droid can
reach it from anywhere.

**Why:** An orb kind stays one brick: adding the negative orb adds its bundle and changes nothing in the world. The
rule that an eaten tier orb does not return is in the orb, where no spawning rule can forget it, and the lifespan is a
value per orb, not a class attribute shared by every env in the process.

**Rejected:**

- A cool-down of 0 on tier orbs with spawning deciding per kind: the world would ask each orb its kind.
- Dropping an eaten orb with no cool-down from the world: the reset and the observation need it.
- The populations in a family folder or in `core/orbs/`: one negative orb is not a tier-chain idea, and the world is
  handed its orbs; what it owns is spawning.

## A weight belongs to a sampled field, and a tier chain holds one negative orb (2026-10-12)

**Decision:** A field is filled one of two ways, a separate question from goal or continuous:

- **Fixed:** every orb is stated and present, as in a tier chain.
- **Sampled:** orbs are drawn from a pool, and each kind's `weight` sets the mix.

`weight` belongs to a kind's block only in a sampled family, through a weighted version of the block. `NegOrbConf` is
`enabled`, `reward` and `cool_down`. A tier chain with an enabled negative block holds exactly one negative orb; the
count is not config. The one is the tier chain's choice, made in its builder: `NegPopulation` makes as many orbs as
it is told, at least 1, so a later family can hold several without a second negative population (2026-11-09).

Every population states how many orbs it makes, through `count`. A fixed field sums its bundles' counts for the field
size, the observation slots and the sort limit. In a sampled field the field size comes from the config, the counts
are the pool, and the family's entry type apportions the pool.

**Why:** Each block holds only what its family reads. A weight means something only relative to other weights in one
pool.

**Rejected:**

- `weight` on every negative block: a fixed field had to supply a value nothing read.
- `weight` as a continuous-only idea: a goal scenario can have a sampled field.
- One pool block listing every weight: a kind's block should stay self-contained.
- A configurable number of negative orbs in a tier chain: one is enough.
- A tally kept by hand in the family's entry type: a forgotten `+= 1` leaves the observation a slot short with no error.
- `len(population.create())`: it makes orbs only to count them.
