# 01 — Runners

```text
runners/
├── agent_runners/
│   ├── sb3/
│   │   ├── artifact_manager.py
│   │   ├── base_sb3_runner.py
│   │   ├── execution_strategy.py
│   │   ├── frame_stack_ppo.py
│   │   ├── lstm_ppo.py
│   │   ├── policy_resolver.py
│   │   └── stateless_ppo.py
│   ├── utils/
│   │   └── extractors.py
│   ├── base_agent_runner.py
│   ├── runner_bundle.py
│   └── runner_registry.py
└── human_runner/
    └── human_runner.py
```

```text
BaseAgentRunner (ABC)
└── BaseSB3Runner[T: BaseAlgorithm]
    ├── StatelessPPO   (T = PPO)
    ├── FrameStackPPO  (T = PPO)
    └── LstmPPO        (T = RecurrentPPO)

HumanRunner
```

An agent runner takes a `RunnerBundle` (the built `Scenario` plus `RunnerConf`)
and drives environments with it. `runner_registry.build_runner` picks the class
from the `RUNNER` table, keyed by the `Algorithm` enum that `common_conf.alg`
is read as (`PPO`, `FSPPO`, `RPPO`). `app.dispatch` then calls `train()` or
`eval()` on it.

`BaseAgentRunner` owns what every run needs regardless of algorithm: the run
id, the output directories, and the per-environment wrappers (episode
statistics, CSV logging, video). `BaseSB3Runner` holds the whole train and eval
flow once. It builds a `DummyVecEnv` of `SYNGridEnv`s wrapped in
`VecNormalize`, and delegates model and normalisation-stat persistence to
`ArtifactManager`. It also names the policy: `policy_resolver` takes the
scenario's perception and the algorithm, and returns the one policy that
algorithm offers for it. The three concrete runners only supply an algorithm,
hyperparameters and an `ExecutionStrategy` for eval-time action selection;
`FrameStackPPO` also adds frame stacking to the environment.

`HumanRunner` is not an agent runner. When `human_control` is set, `app.main`
runs it and returns before any agent runner is built, so the snapshot and the
runner config are ignored. It takes the `Scenario`, builds one bare `SYNGridEnv`
in `human` render mode and loops on keyboard input until the episode ends.

The runners already take a `Scenario`, and the agent runners import.
`HumanRunner` imports `SYNGridEnv`, which does not import yet (see
`08_gymnasium.md`), so the app fails with it.
