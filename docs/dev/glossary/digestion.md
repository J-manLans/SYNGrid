# Digestion

- **Digestion:** turning an eaten orb into a reward.
- **Digester:** the rules for one orb kind. It keeps its own episode state and knows no other digester.
- **Engine:** the router that hands an eaten orb to the digester of its kind.
- **Direct digester:** the digester of a direct kind. `DirectDigester` holds the rule, paying what the orb is worth,
  and each kind has a subclass that only names its kind: `NegDigester`, `PosDigester`.
- **Chain:** the tiers eaten in order so far. It starts at tier 1 and grows one tier at a time.
- **Progress:** an eaten tier orb that is the next tier and not the last.
- **Completion:** an eaten tier orb that is the next tier and the last. The chain is over.
- **Break:** an eaten tier orb that is any other tier. It leaves no chain, whatever its tier.
- **Chain worth:** the summed reward of the orbs in the running chain, `chain_worth`, kept in every mode. Whether
  it has been paid depends on the mode.
- **Undigested:** what a digester has gathered and not paid, and still pays if the episode ends now. Every
  digester states it through `undigested()`, and the droid settles the sum when the episode ends. Max-tier holds an
  unpaid chain and returns 0, since an unfinished chain is lost. Of the digesters, only threshold holds anything
  undigested.
- **Scoring mode:** when a tier orb's worth is paid. An orb is worth the same in every mode.
- **Max-tier:** nothing is paid until completion, which pays the whole chain.
- **Step-wise:** each orb is paid as it is eaten.
- **Threshold:** nothing is paid on progress; a break or a completion pays what the chain had gathered.
- **Reward ladder:** what each tier is worth: `base_reward * tier ** growth_factor`.
