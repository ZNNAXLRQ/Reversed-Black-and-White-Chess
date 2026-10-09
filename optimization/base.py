"""通用优化器接口。"""
from abc import ABC, abstractmethod

from .config import OptimizationConfig
from .objective import ObjectiveFunction
from .parameter_space import ParameterSpace
from .result import OptimizationResult


class Optimizer(ABC):
    """通用优化器抽象基类。

    公共接口只规定输入、输出与运行约定（越小越好的 cost、显式 RNG、
    遵守预算与停止条件），不规定算法内部状态。
    """

    @abstractmethod
    def optimize(
        self,
        objective: ObjectiveFunction,
        parameter_space: ParameterSpace,
        config: OptimizationConfig,
    ) -> OptimizationResult:
        """运行优化并返回 OptimizationResult。"""
        raise NotImplementedError

