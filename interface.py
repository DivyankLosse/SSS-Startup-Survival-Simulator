"""Repository-root environment interface for submission validators."""

from __future__ import annotations

from typing import Any, Dict, Optional

from env import StartupEnv


class StartupSurvivalInterface:
    """Thin wrapper exposing reset/step/state around the core environment."""

    def __init__(self, seed: Optional[int] = 42):
        self._env = StartupEnv(seed=seed)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._env.reset(seed=seed).model_dump()

    def step(self, action: str) -> Dict[str, Any]:
        return self._env.step(action)

    def state(self) -> Dict[str, Any]:
        return self._env.state().model_dump()


def create_interface(seed: Optional[int] = 42) -> StartupSurvivalInterface:
    """Factory helper for validators expecting a module-level constructor."""
    return StartupSurvivalInterface(seed=seed)
