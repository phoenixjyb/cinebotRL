import abc
import numpy as np
from typing import Tuple, Optional


class TargetGenerator(abc.ABC):
    """Abstract base for generating targets over time/steps/episodes."""

    @abc.abstractmethod
    def reset(self, rng: Optional[np.random.Generator] = None):
        """Called at env.reset(); return initial target (x,y,z) as numpy array."""

    @abc.abstractmethod
    def step(self, step_count: int) -> np.ndarray:
        """Called every env.step(); returns current target (x,y,z)."""


class FixedTarget(TargetGenerator):
    def __init__(self, target: Tuple[float, float, float]):
        self._target = np.array(target, dtype=np.float32)

    def reset(self, rng: Optional[np.random.Generator] = None):
        return self._target

    def step(self, step_count: int) -> np.ndarray:
        return self._target

class RandomTargetForEpisode(TargetGenerator):
    def __init__(self, low: Tuple[float, float, float], high: Tuple[float, float, float], seed: Optional[int] = None):
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        if seed is None:
            import secrets
            seed = secrets.randbits(63)
        self.rng = np.random.default_rng(int(seed))
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
        return self._current

    def step(self, step_count: int) -> np.ndarray:
        if self._current is None:
            return self.reset()
        return self._current

class RandomTarget(TargetGenerator):
    def __init__(self, low: Tuple[float, float, float], high: Tuple[float, float, float], seed: Optional[int] = None,
                 respawn_interval: int = 200):
        self.low = np.array(low, dtype=float)
        self.high = np.array(high, dtype=float)
        self.respawn_interval = int(respawn_interval)
        self.rng = np.random.default_rng(seed)
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        if rng is not None:
            self.rng = rng
        self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
        self._last_change = 0
        return self._current

    def step(self, step_count: int) -> np.ndarray:
        if self._current is None:
            self.reset()
        if (step_count - getattr(self, '_last_change', 0)) >= self.respawn_interval:
            self._current = self.rng.uniform(self.low, self.high).astype(np.float32)
            self._last_change = step_count
        return self._current


class CurriculumTarget(TargetGenerator):
    def __init__(self, start: Tuple[float, float, float], end: Tuple[float, float, float], duration_steps: int):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)
        self.duration_steps = max(1, int(duration_steps))
        self._current = None

    def reset(self, rng: Optional[np.random.Generator] = None):
        self._current = self.start.astype(np.float32)
        return self._current

    def step(self, step_count: int) -> np.ndarray:
        t = float(min(step_count, self.duration_steps)) / float(self.duration_steps)
        cur = (1.0 - t) * self.start + t * self.end
        return cur.astype(np.float32)
