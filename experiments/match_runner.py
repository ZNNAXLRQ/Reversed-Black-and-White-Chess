"""MatchRunner：固定对手对局运行器。

复用 reversi 的规则与搜索，提供：
- play_game：两个 agent 跑一局；
- evaluate_candidate：候选权重 vs 固定基准 AI + 反转贪心 AI 的 6 局训练评价；
- head_to_head：挑战者 vs 当前冠军的独立 2 局比较（不写入训练 cost）。
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from reversi.board import COLOR_BLACK, COLOR_WHITE, count, initial_board, opponent
from reversi.rules import apply_move, is_terminal, legal_moves
from reversi.search import best_move


class ObjectiveEvaluationError(RuntimeError):
    """对局发生异常或非法走法时抛出，终止本次优化运行。"""


@dataclass
class GameResult:
    """单局结果。"""

    black_count: int
    white_count: int
    steps: int
    elapsed: float
    seed: Optional[int] = None


@dataclass
class MatchSummary:
    """固定对手 6 局训练评价的汇总统计（候选视角）。"""

    wins: int
    draws: int
    losses: int
    mean_u: float
    u_values: List[float]
    games: List[GameResult]


@dataclass
class H2HResult:
    """挑战者 vs 当前冠军的 2 局独立比较统计。"""

    games: List[GameResult]
    u_values: List[float]
    mean_u: float
    challenger_wins: bool


class WeightedAgent:
    """权重 AI：使用给定权重与搜索深度（确定性走法顺序）。"""

    def __init__(self, weights, depth=4):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.depth = depth

    def select_move(self, board, color, moves, seed=None):
        return best_move(board, color, self.depth, self.weights)


class GreedyAgent:
    """反转贪心 AI：模拟每个合法走法，选择落子翻转后己方棋子最少的走法。"""

    def select_move(self, board, color, moves, seed=None):
        rng = np.random.default_rng(0 if seed is None else seed)
        best_moves = []
        best_count = None
        for move in moves:
            new_board = apply_move(board, color, move)
            own = count(new_board, color)
            if best_count is None or own < best_count:
                best_count = own
                best_moves = [move]
            elif own == best_count:
                best_moves.append(move)
        idx = int(rng.integers(len(best_moves)))
        return best_moves[idx]


class MatchRunner:
    """对局运行器：play_game / evaluate_candidate / head_to_head。"""

    def __init__(self, depth=4):
        self.depth = depth

    def play_game(self, black_agent, white_agent, seed=None) -> GameResult:
        board = initial_board()
        color = COLOR_BLACK
        steps = 0
        start = time.time()
        try:
            while not is_terminal(board):
                moves = legal_moves(board, color)
                if moves:
                    agent = black_agent if color == COLOR_BLACK else white_agent
                    move = agent.select_move(board, color, moves, seed)
                    if move not in moves:
                        raise ObjectiveEvaluationError(
                            f"illegal move {move} by color {color}; "
                            f"legal_moves={moves}; board=\n{board}; steps={steps}"
                        )
                    board = apply_move(board, color, move)
                    steps += 1
                color = opponent(color)
                if steps > 200:
                    raise ObjectiveEvaluationError(f"game exceeded 200 steps; board=\n{board}")
        except ObjectiveEvaluationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ObjectiveEvaluationError(
                f"game error: {exc!r}; board=\n{board}; color={color}; steps={steps}"
            )
        elapsed = time.time() - start
        return GameResult(
            black_count=count(board, COLOR_BLACK),
            white_count=count(board, COLOR_WHITE),
            steps=steps,
            elapsed=elapsed,
            seed=seed,
        )

    def evaluate_candidate(self, weights, baseline_weights, greedy_seeds=(101, 202),
                           baseline_weight=1.0, greedy_weight=1.0) -> MatchSummary:
        """候选权重 vs 固定基准 AI + 反转贪心 AI 的固定 6 局训练评价。

        局序：候选黑 vs 基准白；候选白 vs 基准黑；
              候选黑 vs 贪心(101)；候选白 vs 贪心(101)；
              候选黑 vs 贪心(202)；候选白 vs 贪心(202)。
        u_g 以候选自身为视角：(opp_count - own_count) / 64。

        baseline_weight / greedy_weight 仅作用于 mean_u 的加权平均（wins/draws/losses 仍按
        每局 sign 计数）。self-play 目标用 baseline_weight=2、greedy_weight=1 实现「4 冠军 + 2 贪心」。
        """
        games = []
        u_values = []

        r = self.play_game(
            WeightedAgent(weights, self.depth), WeightedAgent(baseline_weights, self.depth)
        )
        games.append(r)
        u_values.append((r.white_count - r.black_count) / 64.0)  # 候选黑

        r = self.play_game(
            WeightedAgent(baseline_weights, self.depth), WeightedAgent(weights, self.depth)
        )
        games.append(r)
        u_values.append((r.black_count - r.white_count) / 64.0)  # 候选白

        for gs in greedy_seeds:
            r = self.play_game(WeightedAgent(weights, self.depth), GreedyAgent(), seed=gs)
            games.append(r)
            u_values.append((r.white_count - r.black_count) / 64.0)  # 候选黑

            r = self.play_game(GreedyAgent(), WeightedAgent(weights, self.depth), seed=gs)
            games.append(r)
            u_values.append((r.black_count - r.white_count) / 64.0)  # 候选白

        wins = draws = losses = 0
        for u in u_values:
            if u > 0:
                wins += 1
            elif u < 0:
                losses += 1
            else:
                draws += 1

        n_baseline = 2
        n_greedy = 2 * len(greedy_seeds)
        numerator = (
            baseline_weight * sum(u_values[:n_baseline])
            + greedy_weight * sum(u_values[n_baseline:])
        )
        denominator = baseline_weight * n_baseline + greedy_weight * n_greedy
        mean_u = float(numerator / denominator) if denominator > 0 else 0.0
        return MatchSummary(
            wins=wins, draws=draws, losses=losses,
            mean_u=mean_u, u_values=u_values, games=games,
        )

    def head_to_head(self, challenger_weights, incumbent_weights, seeds=(501, 502)) -> H2HResult:
        """挑战者 vs 当前冠军，2 局：第 1 局挑战者黑，第 2 局挑战者白。

        u = (incumbent_count - challenger_count) / 64（挑战者视角）。
        仅 mean_u > 0 判挑战者胜。
        """
        r1 = self.play_game(
            WeightedAgent(challenger_weights, self.depth),
            WeightedAgent(incumbent_weights, self.depth),
            seed=seeds[0],
        )
        u1 = (r1.white_count - r1.black_count) / 64.0  # 挑战者黑，incumbent 白
        r2 = self.play_game(
            WeightedAgent(incumbent_weights, self.depth),
            WeightedAgent(challenger_weights, self.depth),
            seed=seeds[1],
        )
        u2 = (r2.black_count - r2.white_count) / 64.0  # 挑战者白，incumbent 黑
        mean_u = (u1 + u2) / 2.0
        return H2HResult(
            games=[r1, r2], u_values=[u1, u2],
            mean_u=mean_u, challenger_wins=(mean_u > 0),
        )

