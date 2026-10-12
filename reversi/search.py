"""Negamax + Alpha-Beta 搜索（确定性走法排序）。

视角约定：
- 每个搜索节点以「当前行动方」为 perspective；
- 正常落子递归：score = -search(child, opponent, depth-1, -beta, -alpha)；
- 无合法落子但对方有 → 棋盘不变、切换行动方并翻转分数，不消耗深度；
- 仅双方均无合法走法时判定终局，终局效用始终从当前行动方视角计算。
"""
import math

from .board import opponent
from .evaluator import POSITION_TABLE, evaluate, terminal_utility
from .rules import apply_move, has_legal_move, legal_moves


def _order_moves(moves):
    """确定性走法排序：按位置表值降序（角优先），提升 Alpha-Beta 剪枝效率。"""
    return sorted(moves, key=lambda m: POSITION_TABLE[m[0], m[1]], reverse=True)


def _negamax(board, color, depth, alpha, beta, weights):
    moves = legal_moves(board, color)
    if not moves:
        if has_legal_move(board, opponent(color)):
            # pass：不消耗深度，切换行动方并翻转分数
            return -_negamax(board, opponent(color), depth, -beta, -alpha, weights)
        return terminal_utility(board, color)  # 双方均无合法走法，终局
    if depth <= 0:
        return evaluate(board, color, weights)

    best = -math.inf
    for move in _order_moves(moves):
        child = apply_move(board, color, move)
        score = -_negamax(child, opponent(color), depth - 1, -beta, -alpha, weights)
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def best_move(board, color, depth, weights):
    """返回 color 在当前局面下的最佳走法 (row, col)；无合法走法返回 None。"""
    moves = legal_moves(board, color)
    if not moves:
        return None

    best = None
    best_score = -math.inf
    alpha = -math.inf
    beta = math.inf
    for move in _order_moves(moves):
        child = apply_move(board, color, move)
        score = -_negamax(child, opponent(color), depth - 1, -beta, -alpha, weights)
        if score > best_score:
            best_score = score
            best = move
        if score > alpha:
            alpha = score
    return best

