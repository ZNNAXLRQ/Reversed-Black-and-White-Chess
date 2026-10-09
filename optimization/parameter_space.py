"""连续参数空间：维度、上下界、采样、校验、修复。

复制/所有权规则：
- 构造时复制 lower/upper；
- sample / repair 都返回新数组，绝不共享可变内存；
- validate 只读，不修改输入。
"""
import numpy as np


class ParameterSpace:
    """连续参数空间 [lower, upper]（逐维）。"""

    def __init__(self, lower, upper):
        self._lower = np.asarray(lower, dtype=np.float64).copy()
        self._upper = np.asarray(upper, dtype=np.float64).copy()
        if self._lower.ndim != 1 or self._upper.ndim != 1:
            raise ValueError("lower and upper must be one-dimensional")
        if self._lower.shape != self._upper.shape:
            raise ValueError(
                f"lower and upper must have the same shape, "
                f"got {self._lower.shape} and {self._upper.shape}"
            )
        if self._lower.size == 0:
            raise ValueError("parameter space must have dimension >= 1")
        if not np.all(np.isfinite(self._lower)) or not np.all(np.isfinite(self._upper)):
            raise ValueError("bounds must be finite")
        if np.any(self._lower > self._upper):
            raise ValueError("each lower bound must be <= the corresponding upper bound")

    @property
    def dimension(self) -> int:
        return int(self._lower.size)

    @property
    def lower(self) -> np.ndarray:
        return self._lower.copy()

    @property
    def upper(self) -> np.ndarray:
        return self._upper.copy()

    def sample(self, rng) -> np.ndarray:
        """在 [lower, upper] 内均匀采样，返回新数组。"""
        return rng.uniform(self._lower, self._upper)

    def validate(self, candidate) -> bool:
        """形状 (D,)、数值有限且在界内返回 True。"""
        x = np.asarray(candidate)
        if x.ndim != 1 or x.shape[0] != self.dimension:
            return False
        try:
            xf = x.astype(np.float64)
        except (TypeError, ValueError):
            return False
        if not np.all(np.isfinite(xf)):
            return False
        return bool(np.all(xf >= self._lower) and np.all(xf <= self._upper))

    def repair(self, candidate) -> np.ndarray:
        """越界坐标用 np.clip 投影回 [lower, upper]，返回新数组（不修改输入）。"""
        x = np.asarray(candidate, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != self.dimension:
            raise ValueError(
                f"candidate must be one-dimensional with size {self.dimension}, "
                f"got shape {x.shape}"
            )
        return np.clip(x, self._lower, self._upper)

