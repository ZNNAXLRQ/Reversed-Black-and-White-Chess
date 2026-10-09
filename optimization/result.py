"""优化结果：最佳解、统计与停止原因。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class OptimizationResult:
    """一次优化运行的完整结果。"""

    best_candidate: np.ndarray
    best_cost: float
    num_evaluations: int
    num_iterations: int
    stop_reason: str
    history: List[float] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.best_candidate = np.asarray(self.best_candidate, dtype=np.float64).copy()

