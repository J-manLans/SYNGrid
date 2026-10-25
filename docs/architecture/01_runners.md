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
│   │   ├── run_controls_callback.py
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
id, the output directories, the per-environment wrappers (episode
statistics, CSV logging, video), the run controls for a run with a human
renderer (Space pauses, Escape or the close button quits) and the sound when
training stops. `BaseSB3Runner` holds the whole train and eval
flow once. It builds a `DummyVecEnv` of `SYNGridEnv`s wrapped in
`VecNormalize`, and delegates model and normalisation-stat persistence to
`ArtifactManager`. It also names the policy: `policy_resolver` takes the
scenario's perception and the algorithm, and returns the one policy that
algorithm offers for it. The three concrete runners only supply an algorithm,
hyperparameters and an `ExecutionStrategy` for eval-time action selection;
`FrameStackPPO` also adds frame stacking to the environment. Eval reads the run
controls before each step; training reaches them through
`RunControlsCallback`, which is added only when it renders to a window.

`HumanRunner` is not an agent runner. When `human_control` is set, `app.main`
runs it and returns before any agent runner is built, so the snapshot and the
runner config are ignored. It takes the `Scenario`, builds one bare `SYNGridEnv`
in `human` render mode, waits for an arrow key for each move until the episode
ends, and then shows a summary of the episode until a key is pressed.

Every runner reads its own keys, and the renderer reads none. Quitting stops
the runner's loop the way the loop provides, SB3's callback return in
training, and the env's `close()` in each runner's `finally` closes the
window.
