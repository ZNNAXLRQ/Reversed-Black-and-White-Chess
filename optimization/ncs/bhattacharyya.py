"""Bhattacharyya 距离与相关性（论文 IV-A，Eq.(7)）。"""
import numpy as np


def bhattacharyya_distance(x_i, sigma_i, x_j, sigma_j) -> float:
    """两个球面高斯 N(x_i, sigma_i^2 I) 与 N(x_j, sigma_j^2 I) 的 Bhattacharyya 距离。

    D_B = ||x_i - x_j||^2 / (4 (sigma_i^2 + sigma_j^2))
          + D/2 * ln((sigma_i^2 + sigma_j^2) / (2 sigma_i sigma_j))
    其中 D 为参数维度，要求 sigma_i, sigma_j > 0。
    """
    x_i = np.asarray(x_i, dtype=np.float64)
    x_j = np.asarray(x_j, dtype=np.float64)
    if x_i.ndim != 1 or x_j.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    if x_i.shape != x_j.shape:
        raise ValueError(f"centers must have the same shape, got {x_i.shape} and {x_j.shape}")
    if not np.all(np.isfinite(x_i)) or not np.all(np.isfinite(x_j)):
        raise ValueError("centers must be finite")
    if not (np.isfinite(sigma_i) and sigma_i > 0) or not (np.isfinite(sigma_j) and sigma_j > 0):
        raise ValueError("sigmas must be finite and positive")

    diff = x_i - x_j
    s2_sum = sigma_i * sigma_i + sigma_j * sigma_j
    term1 = float(np.dot(diff, diff)) / (4.0 * s2_sum)
    term2 = (x_i.shape[0] / 2.0) * np.log(s2_sum / (2.0 * sigma_i * sigma_j))
    result = term1 + term2
    if not np.isfinite(result):
        raise ValueError("Bhattacharyya distance is not finite")
    return float(result)


def correlation(centers, sigmas) -> np.ndarray:
    """Corr(p_i) = min_{j != i} D_B(p_i, p_j)。

    centers 形状 (N, D)，sigmas 形状 (N,)；返回 (N,)。要求 N > 1。
    """
    centers = np.asarray(centers, dtype=np.float64)
    sigmas = np.asarray(sigmas, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("centers must be two-dimensional (N, D)")
    n, _ = centers.shape
    if sigmas.shape != (n,):
        raise ValueError(f"sigmas must have shape ({n},), got {sigmas.shape}")
    if n <= 1:
        raise ValueError("population size N > 1 is required to compute correlation")
    corr = np.empty(n, dtype=np.float64)
    for i in range(n):
        corr[i] = min(
            bhattacharyya_distance(centers[i], sigmas[i], centers[j], sigmas[j])
            for j in range(n)
            if j != i
        )
    return corr

