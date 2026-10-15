"""Negamax + Alpha-Beta 搜索：置换表、走法排序、迭代加深与时间管理。

视角约定（与旧版一致，语义不变）：
- 每个搜索节点以「当前行动方」为 perspective；
- 正常落子递归：score = -search(child, opponent, depth-1, -beta, -alpha)；
- 无合法落子但对方有 → 棋盘不变、切换行动方并翻转分数，不消耗深度；
- 仅双方均无合法走法时判定终局，终局效用始终从当前行动方视角计算。

增强点（不改变 alpha-beta 的精确语义）：
- 置换表（transposition table）：以 (color, board_bytes) 为 key，避免重复局面搜索；
- 走法排序：位置表 + 历史启发 + 杀手走法，提升剪枝效率；
- 迭代加深 + 节点粒度时间预算，超时返回上一层完整结果。
"""
import math
import time

from .board import opponent
from .evaluator import POSITION_TABLE, evaluate, terminal_utility
from .rules import apply_move, has_legal_move, legal_moves

# 置换表 bound 类型
EXACT = 0
LOWER = 1
UPPER = 2

# 时间检查节流：每搜索这么多节点才检查一次 deadline，降低 perf_counter 开销。
_TIME_CHECK_MASK = 2047


class _SearchTimeout(Exception):
    """时间预算耗尽（内部信号，不向外暴露）。"""


class _SearchContext:
    """单次搜索的共享状态（置换表 / 历史 / 杀手），保证单次调用内的确定性。"""

    __slots__ = ("weights", "tt", "history", "killers", "deadline", "nodes", "timed_out")

    def __init__(self, weights, time_budget=None):
        self.weights = weights
        self.tt = {}
        self.history = {}
        self.killers = {}
        self.deadline = (
            time.perf_counter() + time_budget
            if (time_budget is not None and time_budget > 0)
            else None
        )
        self.nodes = 0
        self.timed_out = False


def _board_key(board):
    return board.tobytes()


def _check_time(ctx):
    if ctx.deadline is not None and time.perf_counter() >= ctx.deadline:
        ctx.timed_out = True
        raise _SearchTimeout()


def _order_moves(ctx, color, moves, depth):
    """确定性走法排序：杀手走法 → 历史启发 → 位置表（角优先）。"""
    killers = ctx.killers.get(depth)
    scored = []
    for move in moves:
        r, c = move
        score = POSITION_TABLE[r, c] * 0.01  # 位置表基线（-0.4 ~ 1.2）
        score += ctx.history.get((color, move), 0.0)
        if killers is not None and move in killers:
            score += 1000.0
        scored.append((score, move))
    scored.sort(key=lambda item: -item[0])
    return [m for _, m in scored]


def _negamax(ctx, board, color, depth, alpha, beta):
    ctx.nodes += 1
    if (ctx.nodes & _TIME_CHECK_MASK) == 0:
        _check_time(ctx)

    key = (color, _board_key(board))
    tt_move = None
    entry = ctx.tt.get(key)
    if entry is not None:
        tt_depth, tt_flag, tt_val, tt_best = entry
        if tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_val
            if tt_flag == LOWER and tt_val >= beta:
                return tt_val
            if tt_flag == UPPER and tt_val <= alpha:
                return tt_val
        tt_move = tt_best

    moves = legal_moves(board, color)
    if not moves:
        if has_legal_move(board, opponent(color)):
            # pass：不消耗深度，切换行动方并翻转分数
            return -_negamax(ctx, board, opponent(color), depth, -beta, -alpha)
        return terminal_utility(board, color)  # 双方均无合法走法，终局
    if depth <= 0:
        return evaluate(board, color, ctx.weights)

    if tt_move is not None and tt_move in moves:
        ordered = [tt_move] + [m for m in moves if m != tt_move]
    else:
        ordered = _order_moves(ctx, color, moves, depth)

    orig_alpha = alpha
    best = -math.inf
    best_move = None
    for move in ordered:
        child = apply_move(board, color, move)
        score = -_negamax(ctx, child, opponent(color), depth - 1, -beta, -alpha)
        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            # beta 剪枝：更新杀手走法 + 历史启发
            if depth in ctx.killers:
                if ctx.killers[depth][0] != move:
                    ctx.killers[depth] = [move, ctx.killers[depth][0]]
            else:
                ctx.killers[depth] = [move]
            ctx.history[(color, move)] = ctx.history.get((color, move), 0.0) + depth * depth
            break

    if best_move is None:
        best_move = ordered[0]
    if best <= orig_alpha:
        flag = UPPER
    elif best >= beta:
        flag = LOWER
    else:
        flag = EXACT
    ctx.tt[key] = (depth, flag, best, best_move)
    return best


def _best_move_with_context(ctx, board, color, depth):
    """在给定上下文内做一次固定深度根搜索，返回最佳走法（不剪枝根节点）。"""
    moves = legal_moves(board, color)
    if not moves:
        return None

    best = None
    best_score = -math.inf
    alpha = -math.inf
    beta = math.inf
    ordered = _order_moves(ctx, color, moves, depth)
    for move in ordered:
        child = apply_move(board, color, move)
        score = -_negamax(ctx, child, opponent(color), depth - 1, -beta, -alpha)
        if score > best_score:
            best_score = score
            best = move
        if score > alpha:
            alpha = score
    return best


def best_move(board, color, depth, weights):
    """固定深度搜索，返回最佳走法 (row, col)；无合法走法返回 None。

    结果与旧版完全一致（alpha-beta 精确搜索），仅因置换表/排序而更快。
    """
    ctx = _SearchContext(weights, time_budget=None)
    return _best_move_with_context(ctx, board, color, depth)


def best_move_iterative(board, color, max_depth, weights, time_budget=None):
    """迭代加深搜索：从 depth 1 到 max_depth，在 time_budget 秒内尽量搜深。

    - 每层复用同一上下文（置换表/历史/杀手），加速后续层；
    - 节点粒度超时：一旦超时立即中断当前层，返回上一层完整结果；
    - 最坏情况（连 depth 1 都超时）返回第一个合法走法兜底。
    """
    moves = legal_moves(board, color)
    if not moves:
        return None

    ctx = _SearchContext(weights, time_budget=time_budget)
    best = moves[0]
    try:
        for d in range(1, max_depth + 1):
            b = _best_move_with_context(ctx, board, color, d)
            if b is not None:
                best = b
            # 单层完成即视为「安全返回点」，继续加深
    except _SearchTimeout:
        pass
    return best

