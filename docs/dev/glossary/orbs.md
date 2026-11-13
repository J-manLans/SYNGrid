# Orbs

- **Orb:** something on the grid the droid can eat.
- **Eat, consume:** the same thing. The code says `consume`, the documents mostly say eat.
- **Kind:** what an orb is: a member of `DirectKind` or `SynergyKind`. Digestion routes on it, and each kind has its
  own block in the config.
- **Category:** direct or synergy. It is not stored; it follows from which enum the kind belongs to.
- **Direct orb:** pays a reward and nothing else, on the step it is eaten. The negative orb is one.
- **Fixed reward orb:** a direct orb that pays the same reward every time, `FixedRewardOrb`. The negative and the
  positive orb are both one, told apart by their kind.
- **Negative orb, positive orb:** a fixed reward orb of `DirectKind.NEGATIVE` or `DirectKind.POSITIVE`. The positive
  orb is written but no scenario holds one yet.
- **Synergy orb:** leaves something behind that changes a later outcome. The tier orb is one.
- **Chain orb, effect orb:** the two groups inside synergy. A chain orb's payoff depends on which orbs of its kind
  came before. An effect orb changes something other than itself. They are folders, not kinds, and no effect orb is
  written.
- **Tier:** a tier orb's place in the chain, from 1 up to `max_tier`. Every other orb has no tier.
- **Identity:** the number the agent sees for an orb. Orbs of the same kind and tier share it.
- **Radix identity:** an orb's kind and tier folded into one number, tier in the top place, computed by its meta.
- **Dense identity:** an orb's rank among the radix identities of a scenario's orbs, 1 to N. It is what the agent
  sees.
- **Lifespan:** how many steps an orb stays on the field before it despawns, or `None` for an orb that stays until
  it is eaten. It belongs to the orb.
- **Despawn:** an orb leaving the field without being eaten.
- **Cool-down:** how many steps an orb is off the field after it is eaten or despawns, or `None` for an orb that
  does not return in the episode. The next episode's reset places every orb again.
- **Expired, ready:** the two questions an orb answers about its timer, `has_expired()` and `is_ready()`. An orb
  has expired when it is on the field and its lifespan has run out, and is ready when it is off the field and its
  cool-down has run out. A timer set to `None` never completes, so such an orb is never expired or ready.
