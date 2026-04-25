"""Compatibility interface expected by submission validators."""

from __future__ import annotations

from typing import Optional

from env import StartupEnv
from models import Action


class StartupSurvivalInterface:
    """Thin wrapper exposing reset/step/state as plain dict-returning methods."""

    def __init__(self, seed: int = 42) -> None:
        self._env = StartupEnv(seed=seed)

    def reset(self, seed: Optional[int] = None) -> dict:
        return self._env.reset(seed=seed).model_dump()

    def step(self, action: str) -> dict:
        return self._env.step(action)

    def state(self) -> dict:
        return self._env.state().model_dump()


_default_interface = StartupSurvivalInterface()


def reset(seed: Optional[int] = None) -> dict:
    """Module-level reset hook for simple validators."""
    return _default_interface.reset(seed=seed)


def step(action: str) -> dict:
    """Module-level step hook for simple validators."""
    return _default_interface.step(action=action)


def state() -> dict:
    """Module-level state hook for simple validators."""
    return _default_interface.state()


def actions() -> list[str]:
    """Return the published action list."""
    return [action.value for action in Action]
