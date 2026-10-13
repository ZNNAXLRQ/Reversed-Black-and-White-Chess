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

# Negamax 搜索与 Alpha-Beta 剪枝

def _order_moves(moves):
    return sorted(moves, key=lambda m: POSITION_TABLE[m[0], m[1]], reverse=True)

def _negamax(board, color, depth, alpha, beta, weights):
    moves = legal_moves(board, color)
    if not moves:
        if has_legal_move(board, opponent(color)):
            # pass：不消耗深度，切换行动方并翻转分数
            return -_negamax(board, opponent(color), depth, -beta, -alpha, weights)
        # 双方均无合法走法，终局
        return terminal_utility(board, color)  
    if depth <= 0:
        return evaluate(board, color, weights)

    best = -float("inf")
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
    moves = legal_moves(board, color)
    if not moves:
        return None

    best = None
    best_score = -float("inf")
    alpha = -float("inf")
    beta = float("inf")
    for move in _order_moves(moves):
        child = apply_move(board, color, move)
        score = -_negamax(child, opponent(color), depth - 1, -beta, -alpha, weights)
        if score > best_score:
            best_score = score
            best = move
        if score > alpha:
            alpha = score
    return best

# AI 接口与权重参数

_SEARCH_DEPTH = 4

# BEGIN AUTO-UPDATED FINAL WEIGHTS
_FIXED_WEIGHTS = np.array(
    [-3.0861110771421476, -10.0, -0.3857156832111528, -7.651743487832784, 0.48587137914320344],
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

        # 搜索选中的合法走法追加到最后，平台取最后一个元素作为决策。
        chosen = best_move(chessboard, self.color, self.depth, self.weights)
        if chosen is None: 
            chosen = moves[0]
        self.candidate_list.append(chosen)

        return self.candidate_list
