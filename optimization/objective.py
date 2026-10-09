"""评价协议：结构化评价结果与目标函数接口。

目标函数向优化器统一提供「越小越好」的有限 cost；需要最大化的任务
应由任务适配器转换成最小化目标，不要在优化器内部混用最大化/最小化分支。
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Protocol

import numpy as np


@dataclass
class EvaluationResult:
    """单次评价结果：cost 越小越好；metrics 为可选辅助指标。"""

    cost: float
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not np.isfinite(self.cost):
            raise ValueError(f"cost must be finite, got {self.cost!r}")


class ObjectiveFunction(Protocol):
    """目标函数协议：接收一维候选解，返回 EvaluationResult。"""

    def __call__(self, candidate: np.ndarray) -> EvaluationResult:
        ...

