"""ReplayPreferenceObjective：强对手走法偏好目标（最小化单步静态评分的负对数似然）。

只读取 Plan 11 预计算的 action_features（phi(s,a)），不运行任何黑白棋对局，
不调用 MatchRunner，不调用深度搜索。复用 optimization 的通用 ObjectiveFunction 协议。

固定数学定义（Plan 12 第三节）：
- w = -10 + 20*z，z ∈ [0,1]^5；
- q_w(s,a) = dot(w, phi(s,a))，temperature=1.0；
- logit_a = q_w(s,a) / temperature；
- loss(s) = logsumexp(logits_all) - logit_chosen；
- cost = 按 game 等权平均（先对每局样本平均，再对 train split 各 game 等权平均）。
"""
import json
from typing import Dict, List

import numpy as np

from optimization.objective import EvaluationResult

DECODE_SCALE = 20.0
DECODE_OFFSET = -10.0
TEMPERATURE = 1.0
WEIGHT_DIM = 5


def logsumexp(x) -> float:
    """稳定 logsumexp：max(x) + log(sum(exp(x - max(x))))。"""
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def compute_preference_loss(samples_by_game: Dict[str, List[Dict]], weights, temperature: float = TEMPERATURE) -> float:
    """按 game 等权平均的偏好损失（每个 game 先对样本损失求平均）。"""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (WEIGHT_DIM,):
        raise ValueError(f"weights must have shape ({WEIGHT_DIM},), got {weights.shape}")
    if not samples_by_game:
        raise ValueError("samples_by_game is empty")
    total = 0.0
    for samples in samples_by_game.values():
        if not samples:
            raise ValueError("each game must have at least one sample")
        game_sum = 0.0
        for s in samples:
            feats = s["action_features"]  # (K, 5)
            logits = feats @ weights / temperature
            loss = logsumexp(logits) - float(logits[int(s["chosen_idx"])])
            game_sum += loss
        total += game_sum / len(samples)
    return total / len(samples_by_game)


def load_samples_by_game(path: str) -> Dict[str, List[Dict]]:
    """从 split jsonl 读取样本，按 game_id 分组并转成 numpy 特征。"""
    samples_by_game: Dict[str, List[Dict]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            game_id = s["game_id"]
            feat = np.asarray(s["action_features"], dtype=np.float64)
            if feat.ndim != 2 or feat.shape[1] != WEIGHT_DIM:
                raise ValueError(
                    f"{game_id} ply={s.get('ply')} action_features shape {feat.shape} 非法"
                )
            if not np.all(np.isfinite(feat)):
                raise ValueError(f"{game_id} ply={s.get('ply')} action_features 含非有限值")
            samples_by_game.setdefault(game_id, []).append({
                "action_features": feat,
                "chosen_idx": int(s["chosen_move_index"]),
            })
    return samples_by_game


class ReplayPreferenceObjective:
    """仅基于 train split 预计算特征的偏好损失目标（NCS 最小化 cost）。"""

    def __init__(self, samples_by_game: Dict[str, List[Dict]], temperature: float = TEMPERATURE):
        self.samples_by_game = samples_by_game
        self.temperature = temperature

    def decode(self, z) -> np.ndarray:
        """归一化 z ∈ [0,1]^5 -> 权重 w = -10 + 20*z ∈ [-10,10]^5。"""
        z = np.asarray(z, dtype=np.float64)
        if z.shape != (WEIGHT_DIM,):
            raise ValueError(f"candidate must have shape ({WEIGHT_DIM},), got {z.shape}")
        if not np.all(np.isfinite(z)):
            raise ValueError("candidate must be finite")
        if np.any(z < 0.0) or np.any(z > 1.0):
            raise ValueError("candidate must be within [0, 1]^5")
        return DECODE_OFFSET + DECODE_SCALE * z

    def loss(self, weights, samples_by_game=None) -> float:
        sg = samples_by_game if samples_by_game is not None else self.samples_by_game
        return compute_preference_loss(sg, weights, self.temperature)

    def __call__(self, candidate) -> EvaluationResult:
        weights = self.decode(candidate)
        cost = self.loss(weights)
        return EvaluationResult(cost=cost, metrics={"weights": weights.copy()})


def is_eligible(candidate_validation_loss: float, baseline_validation_loss: float) -> bool:
    """validation loss 严格低于 baseline 才合格；相等或更差均不合格。"""
    return candidate_validation_loss < baseline_validation_loss
