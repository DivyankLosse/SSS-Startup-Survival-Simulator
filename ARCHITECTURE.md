# Startup Survival Simulator (SSS) - Evaluation Architecture

## Phase 1: System Design

SSS is a deterministic, seed-driven startup simulation environment where an agent selects one action per step and receives a layered reward. The system is optimized for verifiable improvement and fast demo execution.

Core loop:
1. `reset(seed)` initializes a startup state.
2. Agent chooses action from discrete action space.
3. `step(action)` updates state under constraints.
4. Reward is computed with growth/survival/penalty layers.
5. Episode ends on bankruptcy or max steps.
6. Verifier checks deterministic success conditions.

## Module Breakdown

- `advanced_env.py`
  - OpenEnv-style environment (`reset`, `step`, `state`)
  - State transitions, anti-cheat constraints, deterministic market dynamics
- `reward_verifier.py`
  - Layered reward function
  - Deterministic verification checks
- `train_policy.py`
  - Tabular Q-learning loop
  - Random baseline and trained policy evaluation
- `evaluate_policy.py`
  - End-to-end reproducible baseline vs trained comparison
  - Same-seed replay and JSON artifact generation

## Data Flow

1. `evaluate_policy.py` starts training/evaluation.
2. `train_policy.py` calls `StartupSurvivalEnv.reset/step`.
3. `advanced_env.py` updates state and calls `compute_reward`.
4. Trajectories are passed to `verify_episode`.
5. Metrics are aggregated and written to `evaluation_outputs/evaluation_results.json`.

## Key Verifiability Hooks

- Fixed scenario seeds for baseline and trained policies.
- Same-seed replay for side-by-side behavioral comparison.
- Deterministic checks:
  - survived required steps
  - revenue > burn
  - zero constraint violations
