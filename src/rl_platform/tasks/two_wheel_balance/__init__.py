"""Isolated two-wheel balance task with lazy simulator imports."""

from typing import Any


__all__ = ["RecomoTwoWheelBalanceEnv", "RecomoTwoWheelBalanceEnvCfg"]


def __getattr__(name: str) -> Any:
    if name == "RecomoTwoWheelBalanceEnvCfg":
        from .config import RecomoTwoWheelBalanceEnvCfg

        return RecomoTwoWheelBalanceEnvCfg
    if name == "RecomoTwoWheelBalanceEnv":
        from .env import RecomoTwoWheelBalanceEnv

        return RecomoTwoWheelBalanceEnv
    raise AttributeError(name)
