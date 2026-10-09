"""通用运行配置：随机种子与公共停止条件。"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class OptimizationConfig:
    """优化运行公共配置；NCS 专属参数放在 NCS 专属配置中，不塞进这里。"""

    seed: int
    max_iterations: Optional[int] = None
    max_evaluations: Optional[int] = None

    def __post_init__(self):
        if self.max_iterations is not None and self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative or None")
        if self.max_evaluations is not None and self.max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative or None")

