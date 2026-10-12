"""NCS 搜索过程的最小数据结构。"""
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchProcess:
    """一个 NCS 搜索过程：当前解、cost、步长与周期内接受计数。"""

    x: np.ndarray
    cost: float
    sigma: float
    accepted: int = 0

