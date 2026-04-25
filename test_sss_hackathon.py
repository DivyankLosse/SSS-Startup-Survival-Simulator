"""Smoke tests for hackathon-specific SSS environment and training flow."""

from sss_hackathon_env import ACTIONS, StartupSurvivalEnv
from sss_reward_verifier import verify_episode
from sss_training import build_greedy_policy, evaluate_policy, train_q_learning


def test_required_actions_and_state_fields() -> None:
    env = StartupSurvivalEnv(seed=42)
    state = env.reset(seed=42)

    assert ACTIONS == ["hire", "fire", "build_feature", "pivot", "marketing_spend", "do_nothing"]
    for field in ("funding", "team_size", "burn_rate", "market_demand", "runway"):
        assert field in state


def test_step_and_verifier_execute() -> None:
    env = StartupSurvivalEnv(seed=42)
    env.reset(seed=42)
    trajectory = []
    done = False
    while not done:
        out = env.step("do_nothing")
        trajectory.append(out)
        done = out["done"]

    verdict = verify_episode(trajectory)
    assert "checks" in verdict
    assert "survived_x_steps" in verdict["checks"]


def test_training_improves_survival_rate() -> None:
    seeds = [7, 11, 19, 23, 31, 43]
    random_baseline = evaluate_policy(lambda _: "do_nothing", seeds)
    artifacts = train_q_learning(episodes=120, seed=2026)
    trained = evaluate_policy(build_greedy_policy(artifacts), seeds)
    assert trained["survival_rate"] >= random_baseline["survival_rate"]
