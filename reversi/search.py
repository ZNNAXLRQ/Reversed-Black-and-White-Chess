"""Negamax + Alpha-Beta 搜索（位棋盘 + Numba 搜索核心）。

视角约定（语义不变）：
- 每个搜索节点以「当前行动方」为 perspective；
- 正常落子递归：score = -search(child, opponent, depth-1, -beta, -alpha)；
- 无合法落子但对方有 → 棋盘不变、切换行动方并翻转分数，不消耗深度；
- 仅双方均无合法走法时判定终局，终局效用始终从当前行动方视角计算。

搜索核心 `_negamax_nb` 用 @njit 编译；置换表/杀手/历史启发均为 numpy 数组（不用 Python dict）。
numba 不能写模块级全局数组，因此这些状态数组由 Python 层创建、作为参数传入。
时间预算在 Python 层的迭代加深循环中检查；节点上限在 numba 核心内检查。
"""
import time

import numba
import numpy as np

from .board import COLOR_BLACK, COLOR_WHITE
from .evaluator import POSITION_TABLE, _evaluate_nb, _popcount
from .rules import _apply_move_nb, _legal_mask_nb, legal_moves

EXACT = 0
LOWER = 1
UPPER = 2

# 终局目标阈值：迭代搜索逐层尝试解出；保留的固定深度接口可强制求解。
ENDGAME_EMPTY_THRESHOLD = 14

_TT_SIZE = 1 << 16
_MAX_DEPTH = 64

# 走法排序基线：位置表（角优先）。
_POSITION_SCORE = POSITION_TABLE.ravel() * 0.01


def _new_state():
    return {
        "tt_key": np.zeros(_TT_SIZE, dtype=np.uint64),
        "tt_depth": np.full(_TT_SIZE, -1, dtype=np.int8),
        "tt_flag": np.zeros(_TT_SIZE, dtype=np.int8),
        "tt_val": np.zeros(_TT_SIZE, dtype=np.float64),
        "tt_move": np.full(_TT_SIZE, -1, dtype=np.int8),
        "killers": np.full((_MAX_DEPTH + 1, 2), -1, dtype=np.int8),
        "history": np.zeros((2, 64), dtype=np.float64),
        "nodes": np.zeros(1, dtype=np.int64),
        "abort": np.zeros(1, dtype=np.int8),
        "deadline": np.zeros(1, dtype=np.float64),
        "node_limit": np.zeros(1, dtype=np.int64),
        "weights": np.zeros(5, dtype=np.float64),
    }


@numba.njit
def _hash(color, me, opp):
    h = me ^ (opp * np.uint64(0x9E3779B97F4A7C15))
    if color < 0:
        h ^= np.uint64(0x517CC1B727220A95)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return h


@numba.njit
def _terminal_utility_nb(me, opp):
    own = _popcount(me)
    opp_n = _popcount(opp)
    if own < opp_n:
        return 1000.0 + (opp_n - own)
    if own > opp_n:
        return -1000.0 + (opp_n - own)
    return 0.0


@numba.njit
def _order_moves_nb(moves_mask, depth, color, killers, history):
    """按 位置表 + 历史 + 杀手 降序排序，返回 (moves, cnt)。moves 为 bit 索引 0..63。"""
    moves = np.zeros(64, dtype=np.int8)
    scores = np.zeros(64, dtype=np.float64)
    cnt = 0
    ci = 0 if color < 0 else 1
    k0 = killers[depth, 0]
    k1 = killers[depth, 1]
    for i in range(64):
        bit = np.uint64(1) << np.uint64(i)
        if (moves_mask & bit) != np.uint64(0):
            moves[cnt] = i
            sc = _POSITION_SCORE[i] + history[ci, i]
            if i == k0:
                sc += 1000.0
            elif i == k1:
                sc += 900.0
            scores[cnt] = sc
            cnt += 1
    for a in range(1, cnt):
        km = moves[a]
        ks = scores[a]
        b = a - 1
        while b >= 0 and scores[b] < ks:
            moves[b + 1] = moves[b]
            scores[b + 1] = scores[b]
            b -= 1
        moves[b + 1] = km
        scores[b + 1] = ks
    return moves, cnt


@numba.njit
def _negamax_nb(me, opp, color, depth, alpha, beta, weights,
                tt_key, tt_depth, tt_flag, tt_val, tt_move,
                killers, history, nodes, abort, node_limit, solve_endgame=True):
    if abort[0] != 0:
        return 0.0
    if node_limit[0] > 0 and nodes[0] >= node_limit[0]:
        abort[0] = 1
        return 0.0
    nodes[0] += 1

    key = _hash(color, me, opp)
    idx = int(key & np.uint64(_TT_SIZE - 1))
    if tt_depth[idx] >= 0 and tt_key[idx] == key:
        td = tt_depth[idx]
        if td >= depth:
            tf = tt_flag[idx]
            tv = tt_val[idx]
            if tf == EXACT:
                return tv
            if tf == LOWER and tv >= beta:
                return tv
            if tf == UPPER and tv <= alpha:
                return tv

    moves_mask = _legal_mask_nb(me, opp)
    if moves_mask == np.uint64(0):
        if _legal_mask_nb(opp, me) != np.uint64(0):
            return -_negamax_nb(opp, me, -color, depth, -beta, -alpha,
                                weights, tt_key, tt_depth, tt_flag, tt_val, tt_move,
                                killers, history, nodes, abort, node_limit, solve_endgame)
        return _terminal_utility_nb(me, opp)

    empty = 64 - _popcount(me | opp)
    if solve_endgame and empty <= ENDGAME_EMPTY_THRESHOLD:
        depth = max(depth, ENDGAME_EMPTY_THRESHOLD)
    elif depth <= 0:
        return _evaluate_nb(me, opp, weights)

    moves, cnt = _order_moves_nb(moves_mask, depth, color, killers, history)

    orig_alpha = alpha
    best = -1e300
    best_move = -1
    ci = 0 if color < 0 else 1
    for k in range(cnt):
        mi = moves[k]
        bit = np.uint64(1) << np.uint64(mi)
        cme, copp = _apply_move_nb(me, opp, bit)
        score = -_negamax_nb(copp, cme, -color, depth - 1, -beta, -alpha,
                             weights, tt_key, tt_depth, tt_flag, tt_val, tt_move,
                             killers, history, nodes, abort, node_limit, solve_endgame)
        # 中断值不是评分或界，不能污染 TT、杀手或历史表。
        if abort[0] != 0:
            return 0.0
        if score > best:
            best = score
            best_move = mi
        if best > alpha:
            alpha = best
        if alpha >= beta:
            if killers[depth, 0] != mi:
                killers[depth, 1] = killers[depth, 0]
                killers[depth, 0] = mi
            history[ci, mi] += depth * depth
            break

    if best_move < 0:
        best_move = moves[0]

    if best <= orig_alpha:
        flag = UPPER
    elif best >= beta:
        flag = LOWER
    else:
        flag = EXACT

    tt_key[idx] = key
    tt_depth[idx] = depth
    tt_flag[idx] = flag
    tt_val[idx] = best
    tt_move[idx] = best_move
    return best


@numba.njit
def _best_move_nb(me, opp, color, depth, weights,
                  tt_key, tt_depth, tt_flag, tt_val, tt_move,
                  killers, history, nodes, abort, node_limit, solve_endgame=True):
    moves_mask = _legal_mask_nb(me, opp)
    if moves_mask == np.uint64(0):
        return -1
    moves, cnt = _order_moves_nb(moves_mask, depth, color, killers, history)
    best = moves[0]
    best_score = -1e300
    alpha = -1e300
    beta = 1e300
    for k in range(cnt):
        mi = moves[k]
        bit = np.uint64(1) << np.uint64(mi)
        cme, copp = _apply_move_nb(me, opp, bit)
        score = -_negamax_nb(copp, cme, -color, depth - 1, -beta, -alpha,
                             weights, tt_key, tt_depth, tt_flag, tt_val, tt_move,
                             killers, history, nodes, abort, node_limit, solve_endgame)
        if abort[0] != 0:
            break
        if score > best_score:
            best_score = score
            best = mi
        if score > alpha:
            alpha = score
    return best


def _masks(bb, color):
    if color == COLOR_BLACK:
        return np.uint64(bb[0]), np.uint64(bb[1])
    return np.uint64(bb[1]), np.uint64(bb[0])


def _to_coord(mi):
    return (int(mi) // 8, int(mi) % 8)


def _prepare(state, weights, time_budget, node_limit):
    state["tt_depth"].fill(-1)
    state["tt_move"].fill(-1)
    state["killers"].fill(-1)
    state["history"].fill(0.0)
    state["nodes"][0] = 0
    state["abort"][0] = 0
    state["deadline"][0] = (time.time() + time_budget) if (time_budget is not None and time_budget > 0) else 0.0
    # numba 核心内无法做墙钟时间检查；若只给了 time_budget，用节点上限兜底。
    if node_limit is None and time_budget is not None and time_budget > 0:
        node_limit = max(1, int(time_budget * 500000))
    state["node_limit"][0] = node_limit if (node_limit is not None and node_limit > 0) else 0
    state["weights"][:] = np.asarray(weights, dtype=np.float64)


def _run(state, me, opp, color, depth, solve_endgame=True):
    return _best_move_nb(
        me, opp, color, depth, state["weights"],
        state["tt_key"], state["tt_depth"], state["tt_flag"], state["tt_val"], state["tt_move"],
        state["killers"], state["history"], state["nodes"], state["abort"],
        state["node_limit"], solve_endgame,
    )


def best_move(bb, color, depth, weights, time_budget=None, node_limit=None):
    """固定深度搜索，返回最佳走法 (row, col)；无合法走法返回 None。"""
    state = _new_state()
    _prepare(state, weights, time_budget, node_limit)
    me, opp = _masks(bb, color)
    mi = _run(state, me, opp, color, depth)
    if mi < 0:
        return None
    return _to_coord(mi)


def best_move_iterative(bb, color, max_depth, weights, time_budget=None, node_limit=None,
                        stats=None):
    """确定性总节点预算、只提交完整层；终局逐层加深，避免首次求解吃光预算。

    <=14 空时目标深度为剩余空格数（pass 不耗深度），完整达到才标记精确解出。
    stats 为可选输出字典；墙钟仍只在层间检查，不是硬实时保证。
    """
    if not 1 <= max_depth <= _MAX_DEPTH:
        raise ValueError("max_depth must be between 1 and 64")
    state = _new_state()
    _prepare(state, weights, time_budget, node_limit)
    me, opp = _masks(bb, color)
    moves = legal_moves(bb, color)
    empties = 64 - int(me | opp).bit_count()
    target = max(max_depth, empties) if empties <= ENDGAME_EMPTY_THRESHOLD else max_depth
    result = dict(completed_depth=0, target_depth=target, nodes=0, interrupted=False,
                  terminal_solved=False, fallback_used=False, stop_reason="max_depth")
    if not moves:
        result.update(terminal_solved=not bool(legal_moves(bb, -color)), stop_reason="no_legal_move")
        if stats is not None:
            stats.clear()
            stats.update(result)
        return None
    best = moves[0]
    for d in range(1, target + 1):
        if time_budget is not None and time_budget > 0 and time.time() >= state["deadline"][0]:
            result.update(interrupted=True, stop_reason="time_budget")
            break
        # TT 只在同一种叶节点评估语义下复用：不在浅层偷偷切换到强制解终局。
        mi = _run(state, me, opp, color, d, solve_endgame=False)
        if state["abort"][0] != 0:
            result.update(interrupted=True, stop_reason="node_limit")
            break
        if mi >= 0:
            best = _to_coord(mi)
            result["completed_depth"] = d
        if d >= empties:
            result.update(terminal_solved=True, stop_reason="terminal_solved")
            break
    result.update(nodes=int(state["nodes"][0]), fallback_used=result["completed_depth"] == 0)
    if stats is not None:
        stats.clear()
        stats.update(result)
    return best
