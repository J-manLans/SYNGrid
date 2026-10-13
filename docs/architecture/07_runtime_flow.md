# 07 — Runtime flow

Startup, as `app.main()` is written, for the current YAMLs
(`goal_tier_chain_spatial`, RPPO, training):

```text
app.main()
  ├─ register_env()
  ├─ load_experiment_configs()      3 YAMLs -> GlobalConf, TierScenarioConf, RunnerConf
  ├─ build_scenario()               -> Scenario                         (once)
  ├─ build_runner()                 -> LstmPPO
  └─ dispatch() -> runner.train()
        ├─ _build_env()             16 × SYNGridEnv
        ├─ create_model()           SB3 RecurrentPPO on VecNormalize(DummyVecEnv)
        └─ model.learn()            SB3 loop -> SYNGridEnv.step
```

With `human_control` set, `main` runs `HumanRunner(scenario).play()` after
`build_scenario()` and returns, so no agent runner is built and nothing is
dispatched.

One environment step, as `SYNGridEnv.step` is written:

```text
SYNGridEnv.step(action)
  ├─▶ GridWorld.perform_droid_action
  │     ├─▶ SynergyDroid.perform_action            move, step/boundary penalty
  │     ├─  tick orb timers, despawn expired orbs
  │     ├─▶ SynergyDroid.consume_orb               if the droid is on an orb
  │     │      └─▶ DigestionEngine.digest          reward
  │     └─  cool down, reactivate or refill        by the world's config flags
  │   ◀── step penalty + orb reward
  ├─  steps_left -= 1
  ├─▶ check_episode_end(world, steps_left, delay_mode, timeout_penalty, reward)
  │   ◀── terminated, truncated=False, reward
  ├─▶ ObservationHandler.get_observation
  └── return obs, reward, terminated, truncated, info
```

Neither flow runs today. `python3 -m syn_grid` stops at import, because
`gymnasium/`, `core/` and `rendering/` name config classes that no longer exist.
Startup is on the new structure as far as the environment: config is read once,
the scenario is built once, and `env_factory.make` passes that `Scenario` to
`gym.make`. `SYNGridEnv` still takes `world_conf` and `obs_conf`, and builds its
world from them.

The step is the one from before the rebuild. The world resolves movement and
consumption and returns a raw reward. `check_episode_end` in
`gymnasium/utils/episode_termination.py` then decides whether the episode ends
and may replace that reward. The observation is encoded last, from the world as
it stands after the step.

In single-chain mode that function ends an episode when the droid's score
reaches zero, when the chain is broken (outside delay mode), when the steps run
out, when the chain is completed, or in delay mode when no orb is left on the
field. `truncated` is always `False`. The scenario's `TerminationRules` are
written to replace it and are not called yet.
