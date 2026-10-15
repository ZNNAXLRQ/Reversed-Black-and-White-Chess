import math
import time

import numpy as np
import numba

# 游戏规则与基本下棋
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
BOARD_SIZE = 8
DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

def opponent(color):
    return -color

def in_bounds(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def count(board, color):
    return int(np.count_nonzero(board == color))

def shift(board, dr, dc):
    n = board.shape[0]
    s = np.zeros_like(board)
    r0 = max(0, -dr)
    r1 = min(n, n - dr)
    c0 = max(0, -dc)
    c1 = min(n, n - dc)
    s[r0:r1, c0:c1] = board[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return s

_DIRECTIONS_NB = np.array(DIRECTIONS, dtype=np.int8)

@numba.njit
def _legal_mask_nb(board, color):
    n = board.shape[0]
    opp = -color
    legal = np.zeros((n, n), dtype=np.bool_)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            for d in range(8):
                dr = _DIRECTIONS_NB[d, 0]
                dc = _DIRECTIONS_NB[d, 1]
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= n or cc < 0 or cc >= n:
                    continue
                if board[rr, cc] != opp:
                    continue
                rr += dr
                cc += dc
                found = False
                while 0 <= rr < n and 0 <= cc < n:
                    if board[rr, cc] == color:
                        found = True
                        break
                    if board[rr, cc] != opp:
                        break
                    rr += dr
                    cc += dc
                if found:
                    legal[r, c] = True
                    break
    return legal

def flips(board, color, row, col):
    if board[row, col] != COLOR_NONE:
        return []
    captured = []
    for dr, dc in DIRECTIONS:
        r, c = row + dr, col + dc
        line = []
        while in_bounds(r, c) and board[r, c] == opponent(color):
            line.append((r, c))
            r += dr
            c += dc
        if line and in_bounds(r, c) and board[r, c] == color:
            captured.extend(line)
    return captured

def _legal_mask(board, color):
    return _legal_mask_nb(board, color)

def legal_moves(board, color):
    return [(int(r), int(c)) for r, c in zip(*np.nonzero(_legal_mask(board, color)))]

def has_legal_move(board, color):
    return bool(_legal_mask(board, color).any())

def count_legal_moves(board, color):
    return int(np.count_nonzero(_legal_mask(board, color)))

def apply_move(board, color, move):
    row, col = move
    new_board = board.copy()
    new_board[row, col] = color
    for r, c in flips(board, color, row, col):
        new_board[r, c] = color
    return new_board

# 评估标准, 博弈辅助

POSITION_TABLE = np.array([
    [120, -20,  20,   5,   5,  20, -20, 120],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [  5,  -5,   3,   3,   3,   3,  -5,   5],
    [ 20,  -5,  15,   3,   3,  15,  -5,  20],
    [-20, -40,  -5,  -5,  -5,  -5, -40, -20],
    [120, -20,  20,   5,   5,  20, -20, 120],
], dtype=np.float64)

_CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

def _count_frontier(board, color):
    disc = board == color
    empty = board == COLOR_NONE
    frontier = np.zeros_like(board, dtype=bool)
    for dr, dc in DIRECTIONS:
        frontier |= disc & shift(empty, dr, dc)
    return int(np.count_nonzero(frontier))

def features(board, perspective):
    opp = opponent(perspective)
    own_mask = (board == perspective)
    opp_mask = (board == opp)
    own_count = int(np.count_nonzero(own_mask))
    opp_count = int(np.count_nonzero(opp_mask))

    # 1. disc_diff = (own - opp) / 64
    disc_diff = (own_count - opp_count) / 64.0

    # 2. position_diff = (sum(P[own]) - sum(P[opp])) / (64 * 120)
    pos_own = float(np.sum(POSITION_TABLE[own_mask]))
    pos_opp = float(np.sum(POSITION_TABLE[opp_mask]))
    position_diff = (pos_own - pos_opp) / (64.0 * 120.0)

    # 3. mobility_diff = (own_legal - opp_legal) / max(1, own_legal + opp_legal)
    own_moves = count_legal_moves(board, perspective)
    opp_moves = count_legal_moves(board, opp)
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)

    # 4. corner_diff = (own_corner - opp_corner) / 4
    own_corners = sum(1 for r, c in _CORNERS if board[r, c] == perspective)
    opp_corners = sum(1 for r, c in _CORNERS if board[r, c] == opp)
    corner_diff = (own_corners - opp_corners) / 4.0

    # 5. frontier_diff = (own_frontier - opp_frontier) / 64
    frontier_diff = (_count_frontier(board, perspective) - _count_frontier(board, opp)) / 64.0

    return np.array(
        [disc_diff, position_diff, mobility_diff, corner_diff, frontier_diff],
        dtype=np.float64,
    )

def evaluate(board, perspective, weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (5,):
        raise ValueError(f"weights must have shape (5,), got {weights.shape}")
    return float(np.dot(weights, features(board, perspective)))

def terminal_utility(board, perspective):
    own = count(board, perspective)
    opp = count(board, opponent(perspective))
    if own < opp:
        return 1000.0 + (opp - own)
    if own > opp:
        return -1000.0 + (opp - own)
    return 0.0

# Negamax 搜索与 Alpha-Beta 剪枝（置换表 + 走法排序 + 迭代加深）
EXACT = 0
LOWER = 1
UPPER = 2

_TIME_CHECK_MASK = 2047


class _SearchTimeout(Exception):
    pass


class _SearchContext(object):
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
    killers = ctx.killers.get(depth)
    scored = []
    for move in moves:
        r, c = move
        score = POSITION_TABLE[r, c] * 0.01
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
            return -_negamax(ctx, board, opponent(color), depth, -beta, -alpha)
        return terminal_utility(board, color)
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
    ctx = _SearchContext(weights, time_budget=None)
    return _best_move_with_context(ctx, board, color, depth)


def best_move_iterative(board, color, max_depth, weights, time_budget=None):
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
    except _SearchTimeout:
        pass
    return best

# AI 接口与权重参数
_SEARCH_DEPTH = 4

# BEGIN AUTO-UPDATED FINAL WEIGHTS
_FIXED_WEIGHTS = np.array(
    [-7.564687797463808, 9.019126725871288, 0.26414041262129473, -5.71691508627605, 0.5947836619489912],
    dtype=np.float64,
)
# END AUTO-UPDATED FINAL WEIGHTS

class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.weights = _FIXED_WEIGHTS.copy()
        self.depth = _SEARCH_DEPTH

    # The input is the current chessboard (a numpy array).
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()

        moves = legal_moves(chessboard, self.color)
        if not moves:
            return []

        # 先加入全部合法走法（元素均为 (row, col) 元组）。
        for move in moves:
            self.candidate_list.append(move)

        # 迭代加深搜索选中的合法走法追加到最后，平台取最后一个元素作为决策。
        budget = self.time_out if (self.time_out and self.time_out > 0) else None
        chosen = best_move_iterative(chessboard, self.color, self.depth, self.weights, time_budget=budget)
        if chosen is None: 
            chosen = moves[0]
        self.candidate_list.append(chosen)

        return self.candidate_list
