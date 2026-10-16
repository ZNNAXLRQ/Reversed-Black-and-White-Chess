import time
import numba
import numpy as np

# 游戏规则与基本常量（位棋盘表示：black / white 两个 np.uint64 位掩码）
# bit 索引 = row * 8 + col，row 0 在顶部、col 0 在左
COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
BOARD_SIZE = 8

FULL_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
COL0_MASK = np.uint64(0x0101010101010101)
COL7_MASK = np.uint64(0x8080808080808080)
ROW0_MASK = np.uint64(0x00000000000000FF)
ROW7_MASK = np.uint64(0xFF00000000000000)
CORNER_MASK = np.uint64(0x8100000000000081)

def opponent(color):
    return -color

def board_from_array(arr):
    black = np.uint64(0)
    white = np.uint64(0)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = int(arr[r, c])
            if v == COLOR_BLACK:
                black |= np.uint64(1) << np.uint64(r * 8 + c)
            elif v == COLOR_WHITE:
                white |= np.uint64(1) << np.uint64(r * 8 + c)
    return np.array([black, white], dtype=np.uint64)

# 8 个方向的 bit 位移量（E/W/S/N/SE/NW/SW/NE）与「源位」前置掩码。
_DIR_SHIFTS = np.array([1, -1, 8, -8, 9, -9, 7, -7], dtype=np.int64)
_DIR_PRE = np.array([
    FULL_MASK ^ COL7_MASK,
    FULL_MASK ^ COL0_MASK,
    FULL_MASK ^ ROW7_MASK,
    FULL_MASK ^ ROW0_MASK,
    FULL_MASK ^ (ROW7_MASK | COL7_MASK),
    FULL_MASK ^ (ROW0_MASK | COL0_MASK),
    FULL_MASK ^ (ROW7_MASK | COL0_MASK),
    FULL_MASK ^ (ROW0_MASK | COL7_MASK),
], dtype=np.uint64)

@numba.njit
def _sh(x, s):
    if s > 0:
        return x << np.uint64(s)
    return x >> np.uint64(-s)

@numba.njit
def _legal_mask_nb(me, opp):
    empty = ~(me | opp) & FULL_MASK
    legal = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        candidates = _sh(me & pre, s) & opp
        while candidates != np.uint64(0):
            candidates = _sh(candidates & pre, s)
            legal |= candidates & empty
            candidates &= opp
    return legal

@numba.njit
def _flips_mask_nb(me, opp, pos):
    flipped = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        m = _sh(pos & pre, s)
        line = np.uint64(0)
        while (m & opp) != np.uint64(0):
            line |= m
            m = _sh(m & pre, s)
        if (m & me) != np.uint64(0):
            flipped |= line
    return flipped

@numba.njit
def _apply_move_nb(me, opp, pos):
    flipped = _flips_mask_nb(me, opp, pos)
    me_new = me | pos | flipped
    opp_new = opp & ~flipped
    return me_new, opp_new

def _color_mask(bb, color):
    if color == COLOR_BLACK:
        return np.uint64(bb[0])
    return np.uint64(bb[1])

def legal_moves(bb, color):
    me = _color_mask(bb, color)
    opp = _color_mask(bb, -color)
    mask = np.uint64(_legal_mask_nb(me, opp))
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if mask & (np.uint64(1) << np.uint64(r * 8 + c)):
                moves.append((r, c))
    return moves

# 评估标准（5 特征 + 终局效用，位棋盘 + Numba）
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

_POSITION_WEIGHTS = POSITION_TABLE.ravel()
_POSITION_SCORE = POSITION_TABLE.ravel() * 0.01

@numba.njit
def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = x + (x >> np.uint64(8))
    x = x + (x >> np.uint64(16))
    x = x + (x >> np.uint64(32))
    return np.int64(x & np.uint64(0x7F))

@numba.njit
def _frontier_count_nb(disc, empty):
    frontier = np.uint64(0)
    for d in range(8):
        s = _DIR_SHIFTS[d]
        pre = _DIR_PRE[d]
        frontier |= _sh(empty & pre, s) & disc
    return _popcount(frontier)

@numba.njit
def _evaluate_nb(me, opp, weights):
    own = _popcount(me)
    opp_n = _popcount(opp)
    disc_diff = (own - opp_n) / 64.0

    pos_own = 0.0
    pos_opp = 0.0
    for i in range(64):
        bit = np.uint64(1) << np.uint64(i)
        if (me & bit) != np.uint64(0):
            pos_own += _POSITION_WEIGHTS[i]
        if (opp & bit) != np.uint64(0):
            pos_opp += _POSITION_WEIGHTS[i]
    position_diff = (pos_own - pos_opp) / (64.0 * 120.0)

    own_moves = _popcount(_legal_mask_nb(me, opp))
    opp_moves = _popcount(_legal_mask_nb(opp, me))
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)

    own_corners = _popcount(me & CORNER_MASK)
    opp_corners = _popcount(opp & CORNER_MASK)
    corner_diff = (own_corners - opp_corners) / 4.0

    empty = ~(me | opp) & FULL_MASK
    frontier_diff = (_frontier_count_nb(me, empty) - _frontier_count_nb(opp, empty)) / 64.0

    return weights[0] * disc_diff + weights[1] * position_diff + weights[2] * mobility_diff + weights[3] * corner_diff + weights[4] * frontier_diff


@numba.njit
def _terminal_utility_nb(me, opp):
    own = _popcount(me)
    opp_n = _popcount(opp)
    if own < opp_n:
        return 1000.0 + (opp_n - own)
    if own > opp_n:
        return -1000.0 + (opp_n - own)
    return 0.0

# Negamax + Alpha-Beta 搜索核心（位棋盘 + Numba）
EXACT = 0
LOWER = 1
UPPER = 2
ENDGAME_EMPTY_THRESHOLD = 14
_TT_SIZE = 1 << 16
_MAX_DEPTH = 64

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
def _order_moves_nb(moves_mask, depth, color, killers, history):
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
def _negamax_nb(me, opp, color, depth, alpha, beta, weights, tt_key, tt_depth, tt_flag, tt_val, tt_move, killers, history, nodes, abort, node_limit):
    nodes[0] += 1
    if abort[0] != 0:
        return 0.0
    if (nodes[0] & 2047) == 0:
        if node_limit[0] > 0 and nodes[0] >= node_limit[0]:
            abort[0] = 1
            return 0.0

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
            return -_negamax_nb(opp, me, -color, depth, -beta, -alpha, weights, tt_key, tt_depth, tt_flag, tt_val, tt_move, killers, history, nodes, abort, node_limit)
        return _terminal_utility_nb(me, opp)

    empty = 64 - _popcount(me | opp)
    if empty <= ENDGAME_EMPTY_THRESHOLD:
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
        score = -_negamax_nb(copp, cme, -color, depth - 1, -beta, -alpha, weights, tt_key, tt_depth, tt_flag, tt_val, tt_move, killers, history, nodes, abort, node_limit)
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
def _best_move_nb(me, opp, color, depth, weights, tt_key, tt_depth, tt_flag, tt_val, tt_move, killers, history, nodes, abort, node_limit):
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
        score = -_negamax_nb(copp, cme, -color, depth - 1, -beta, -alpha, weights, tt_key, tt_depth, tt_flag, tt_val, tt_move, killers, history, nodes, abort, node_limit)
        if abort[0] != 0:
            break
        if score > best_score:
            best_score = score
            best = mi
        if score > alpha:
            alpha = score
    return best


def best_move_iterative(bb, color, max_depth, weights, time_budget=None):
    tt_key = np.zeros(_TT_SIZE, dtype=np.uint64)
    tt_depth = np.full(_TT_SIZE, -1, dtype=np.int8)
    tt_flag = np.zeros(_TT_SIZE, dtype=np.int8)
    tt_val = np.zeros(_TT_SIZE, dtype=np.float64)
    tt_move = np.full(_TT_SIZE, -1, dtype=np.int8)
    killers = np.full((_MAX_DEPTH + 1, 2), -1, dtype=np.int8)
    history = np.zeros((2, 64), dtype=np.float64)
    nodes = np.zeros(1, dtype=np.int64)
    abort = np.zeros(1, dtype=np.int8)
    node_limit = np.zeros(1, dtype=np.int64)
    if time_budget is not None and time_budget > 0:
        # numba 核心内无法做墙钟时间检查，用节点上限兜底（约 50 万节点/秒）。
        node_limit[0] = max(1, int(time_budget * 500000))
    weights_arr = np.asarray(weights, dtype=np.float64)

    me = np.uint64(bb[0]) if color == COLOR_BLACK else np.uint64(bb[1])
    opp = np.uint64(bb[1]) if color == COLOR_BLACK else np.uint64(bb[0])
    moves = legal_moves(bb, color)
    if not moves:
        return None

    deadline = (time.time() + time_budget) if (time_budget and time_budget > 0) else 0.0
    best = moves[0]
    for d in range(1, max_depth + 1):
        if deadline > 0.0 and time.time() >= deadline:
            break
        mi = _best_move_nb(me, opp, color, d, weights_arr,
                           tt_key, tt_depth, tt_flag, tt_val, tt_move,
                           killers, history, nodes, abort, node_limit)
        if abort[0] != 0:
            break
        if mi >= 0:
            best = (int(mi) // 8, int(mi) % 8)
    return best

# AI 接口与权重参数
_SEARCH_DEPTH = 10

# BEGIN AUTO-UPDATED FINAL WEIGHTS
_FIXED_WEIGHTS = np.array(
    [-6.33247672352715, -3.629403162945928, 0.9349258943961338, -9.421310195876254, -0.6242989774127423],
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

    # 输入棋盘序列
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()

        # 平台传入的 chessboard 是 numpy 8x8 int8 数组；内部转换为位棋盘。
        bb = board_from_array(chessboard)
        moves = legal_moves(bb, self.color)
        if not moves:
            return []

        # 先加入全部合法走法（元素均为 (row, col) 元组）。
        for move in moves:
            self.candidate_list.append(move)

        # 迭代加深搜索选中的合法走法追加到最后，平台取最后一个元素作为决策。
        budget = self.time_out if (self.time_out and self.time_out > 0) else None
        chosen = best_move_iterative(bb, self.color, self.depth, self.weights, time_budget=budget)
        if chosen is None:
            chosen = moves[0]
        self.candidate_list.append(chosen)

        return self.candidate_list
