"""Single-file, bounded-memory reversed Reversi agent.

Only the standard library and NumPy are required. No JIT or external data.
The *_nb helper names remain for compatibility with the project's rule tests;
their implementation now uses Python integer bitboards, not Numba.
"""
from array import array
import time

import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
BOARD_SIZE = 8

FULL_MASK = (1 << 64) - 1
COL0_MASK = 0x0101010101010101
COL7_MASK = 0x8080808080808080
ROW0_MASK = 0x00000000000000FF
ROW7_MASK = 0xFF00000000000000
CORNER_MASK = 0x8100000000000081
_INNER_COLUMNS = 0x7E7E7E7E7E7E7E7E
_DIR_SHIFTS = (1, -1, 8, -8, 9, -9, 7, -7)
_DIR_PRE = (
    FULL_MASK ^ COL7_MASK, FULL_MASK ^ COL0_MASK,
    FULL_MASK ^ ROW7_MASK, FULL_MASK ^ ROW0_MASK,
    FULL_MASK ^ (ROW7_MASK | COL7_MASK),
    FULL_MASK ^ (ROW0_MASK | COL0_MASK),
    FULL_MASK ^ (ROW7_MASK | COL0_MASK),
    FULL_MASK ^ (ROW0_MASK | COL7_MASK),
)


def _clock():
    return time.perf_counter()


def opponent(color):
    return -color


def board_from_array(arr):
    black = 0
    white = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            value = int(arr[r, c])
            if value == COLOR_BLACK:
                black |= 1 << (r * 8 + c)
            elif value == COLOR_WHITE:
                white |= 1 << (r * 8 + c)
    return np.array([black, white], dtype=np.uint64)


def _sh(x, shift):
    x = int(x)
    return ((x << shift) & FULL_MASK) if shift > 0 else x >> -shift


def _legal_mask_nb(me, opp):
    me, opp = int(me), int(opp)
    empty = FULL_MASK ^ (me | opp)
    legal = 0
    # At most six opposing discs can occur between a disc and an empty square.
    for shift in (1, 7, 8, 9):
        middle = opp if shift == 8 else opp & _INNER_COLUMNS
        line = middle & (me << shift)
        line |= middle & (line << shift)
        pairs = middle & (middle << shift)
        double = shift * 2
        line |= pairs & (line << double)
        line |= pairs & (line << double)
        legal |= line << shift
        line = middle & (me >> shift)
        line |= middle & (line >> shift)
        pairs = middle & (middle >> shift)
        line |= pairs & (line >> double)
        line |= pairs & (line >> double)
        legal |= line >> shift
    return legal & empty


def _flips_mask_nb(me, opp, pos):
    me, opp, pos = int(me), int(opp), int(pos)
    flipped = 0
    for shift in (1, 7, 8, 9):
        middle = opp if shift == 8 else opp & _INNER_COLUMNS
        cursor = pos << shift
        line = 0
        while cursor & middle:
            line |= cursor
            cursor <<= shift
        if cursor & me:
            flipped |= line
        cursor = pos >> shift
        line = 0
        while cursor & middle:
            line |= cursor
            cursor >>= shift
        if cursor & me:
            flipped |= line
    return flipped


def _apply_move_nb(me, opp, pos):
    me, opp, pos = int(me), int(opp), int(pos)
    flipped = _flips_mask_nb(me, opp, pos)
    return me | pos | flipped, opp & ~flipped


def _color_mask(bb, color):
    return int(bb[0 if color == COLOR_BLACK else 1])


def _indices(mask):
    result = []
    while mask:
        bit = mask & -mask
        result.append(bit.bit_length() - 1)
        mask ^= bit
    return result


def legal_moves(bb, color):
    me, opp = _masks(bb, color)
    return [_to_coord(index) for index in _indices(_legal_mask_nb(me, opp))]


POSITION_TABLE = np.array([
    [120, -20, 20, 5, 5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [5, -5, 3, 3, 3, 3, -5, 5],
    [20, -5, 15, 3, 3, 15, -5, 20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20, 5, 5, 20, -20, 120],
], dtype=np.float64)
_POSITION_WEIGHTS = tuple(float(value) for value in POSITION_TABLE.ravel())
_POSITION_SCORE = tuple(value * 0.01 for value in _POSITION_WEIGHTS)
_POSITION_GROUPS = tuple(
    (value, sum(1 << index for index, weight in enumerate(_POSITION_WEIGHTS) if weight == value))
    for value in sorted(set(_POSITION_WEIGHTS))
)


def _popcount(value):
    return int(value).bit_count()


def _frontier_count_nb(disc, empty):
    disc, empty = int(disc), int(empty)
    return (disc & _neighbors(empty)).bit_count()


def _neighbors(mask):
    east = mask & (FULL_MASK ^ COL7_MASK)
    west = mask & (FULL_MASK ^ COL0_MASK)
    return ((east << 1) | (west >> 1) | (mask << 8) | (mask >> 8)
            | (east << 9) | (west << 7) | (east >> 7) | (west >> 9)) & FULL_MASK


def _position_sum(mask):
    total = 0.0
    for weight, squares in _POSITION_GROUPS:
        total += weight * (mask & squares).bit_count()
    return total


def _evaluate_nb(me, opp, weights, moves_mask=None):
    me, opp = int(me), int(opp)
    disc_diff = (me.bit_count() - opp.bit_count()) / 64.0
    position_diff = (_position_sum(me) - _position_sum(opp)) / (64.0 * 120.0)
    own_moves = (_legal_mask_nb(me, opp) if moves_mask is None else moves_mask).bit_count()
    opp_moves = _legal_mask_nb(opp, me).bit_count()
    mobility_diff = (own_moves - opp_moves) / max(1, own_moves + opp_moves)
    corner_diff = ((me & CORNER_MASK).bit_count() - (opp & CORNER_MASK).bit_count()) / 4.0
    empty = FULL_MASK ^ (me | opp)
    frontier = _neighbors(empty)
    frontier_diff = ((me & frontier).bit_count() - (opp & frontier).bit_count()) / 64.0
    return (weights[0] * disc_diff + weights[1] * position_diff
            + weights[2] * mobility_diff + weights[3] * corner_diff
            + weights[4] * frontier_diff)


def _terminal_utility_nb(me, opp):
    difference = int(opp).bit_count() - int(me).bit_count()
    if difference > 0:
        return 1000.0 + difference
    if difference < 0:
        return -1000.0 + difference
    return 0.0


EXACT = 0
LOWER = 1
UPPER = 2
ENDGAME_EMPTY_THRESHOLD = 14
_TT_SIZE = 1 << 16
_MAX_DEPTH = 64
SEARCH_POLICY = "iterative_wallclock_low_memory_v4"


class _Buffer(array):
    """Fixed-size native numbers, without NumPy scalars in the search loop."""

    def fill(self, value):
        self[:] = array(self.typecode, [value]) * len(self)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return np.asarray(self) == other
        return super().__eq__(other)


class _Matrix(_Buffer):
    def __new__(cls, typecode, values, columns):
        return super().__new__(cls, typecode, values)

    def __init__(self, typecode, values, columns):
        self.columns = columns

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return super().__getitem__(index[0] * self.columns + index[1])
        if isinstance(index, int):
            return super().__getitem__(slice(index * self.columns, (index + 1) * self.columns))
        return super().__getitem__(index)

    def __setitem__(self, index, value):
        if isinstance(index, tuple):
            index = index[0] * self.columns + index[1]
        super().__setitem__(index, value)


def _new_state():
    # Fixed-size arrays; no unbounded dictionary of positions or game history.
    return {
        "tt_key": _Buffer("Q", [0] * _TT_SIZE),
        "tt_depth": _Buffer("b", [-1] * _TT_SIZE),
        "tt_flag": _Buffer("b", [0] * _TT_SIZE),
        "tt_val": _Buffer("d", [0.0] * _TT_SIZE),
        "tt_move": _Buffer("b", [-1] * _TT_SIZE),
        "killers": _Matrix("b", [-1] * ((_MAX_DEPTH + 1) * 2), 2),
        "history": _Matrix("d", [0.0] * 128, 64),
        "nodes": _Buffer("q", [0]),
        "abort": _Buffer("b", [0]),
        "deadline": _Buffer("d", [0.0]),
        "node_limit": _Buffer("q", [0]),
        "weights": _Buffer("d", [0.0] * 5),
        "root_pv": _Buffer("b", [-1]),
    }


def _hash(color, me, opp):
    key = int(me) ^ ((int(opp) * 0x9E3779B97F4A7C15) & FULL_MASK)
    if color < 0:
        key ^= 0x517CC1B727220A95
    key ^= key >> 29
    key = (key * 0xBF58476D1CE4E5B9) & FULL_MASK
    return key ^ (key >> 32)


def _order_moves_nb(moves_mask, depth, color, killers, history):
    moves = _indices(int(moves_mask))
    ci = 0 if color < 0 else 1
    k0, k1 = killers[depth]
    history_row = history[ci]

    def score(index):
        bonus = 1000.0 if index == k0 else 900.0 if index == k1 else 0.0
        return _POSITION_SCORE[index] + history_row[index] + bonus

    moves.sort(key=score, reverse=True)
    return moves, len(moves)


def _promote_hint(moves, count, hint):
    hint = int(hint)
    if not 0 <= hint < 64:
        return
    for index in range(count):
        if moves[index] == hint:
            for j in range(index, 0, -1):
                moves[j] = moves[j - 1]
            moves[0] = hint
            return


def _negamax_nb(me, opp, color, depth, alpha, beta, weights,
                tt_key, tt_depth, tt_flag, tt_val, tt_move,
                killers, history, nodes, abort, node_limit, solve_endgame=True,
                deadline=0.0):
    if abort[0] != 0:
        return 0.0
    if 0 < int(node_limit[0]) <= int(nodes[0]):
        abort[0] = 1
        return 0.0
    # Check inside recursion, not merely between completed depth iterations.
    if deadline > 0.0 and (int(nodes[0]) & 15) == 0 and _clock() >= deadline:
        abort[0] = 2
        return 0.0
    nodes[0] = int(nodes[0]) + 1
    me, opp = int(me), int(opp)
    key = _hash(color, me, opp)
    idx = key & (_TT_SIZE - 1)
    if tt_depth[idx] >= 0 and int(tt_key[idx]) == key:
        if tt_depth[idx] >= depth:
            flag, value = int(tt_flag[idx]), float(tt_val[idx])
            if flag == EXACT or (flag == LOWER and value >= beta) or (flag == UPPER and value <= alpha):
                return value

    moves_mask = _legal_mask_nb(me, opp)
    if not moves_mask:
        if _legal_mask_nb(opp, me):
            return -_negamax_nb(
                opp, me, -color, depth, -beta, -alpha, weights,
                tt_key, tt_depth, tt_flag, tt_val, tt_move,
                killers, history, nodes, abort, node_limit, solve_endgame, deadline,
            )
        return _terminal_utility_nb(me, opp)
    empty = 64 - (me | opp).bit_count()
    if solve_endgame and empty <= ENDGAME_EMPTY_THRESHOLD:
        depth = max(depth, ENDGAME_EMPTY_THRESHOLD)
    elif depth <= 0:
        return _evaluate_nb(me, opp, weights, moves_mask)

    moves, count = _order_moves_nb(moves_mask, depth, color, killers, history)
    original_alpha = alpha
    if tt_depth[idx] >= 0 and int(tt_key[idx]) == key:
        _promote_hint(moves, count, int(tt_move[idx]))
    value = -1e300
    selected = -1
    ci = 0 if color < 0 else 1
    for index in moves:
        child_me, child_opp = _apply_move_nb(me, opp, 1 << index)
        score = -_negamax_nb(
            child_opp, child_me, -color, depth - 1, -beta, -alpha, weights,
            tt_key, tt_depth, tt_flag, tt_val, tt_move,
            killers, history, nodes, abort, node_limit, solve_endgame, deadline,
        )
        if abort[0] != 0:
            return 0.0
        if score > value:
            value, selected = score, index
        alpha = max(alpha, value)
        if alpha >= beta:
            if killers[depth, 0] != index:
                killers[depth, 1] = killers[depth, 0]
                killers[depth, 0] = index
            history[ci, index] = float(history[ci, index]) + depth * depth
            break

    flag = UPPER if value <= original_alpha else LOWER if value >= beta else EXACT
    tt_key[idx], tt_depth[idx], tt_flag[idx] = key, depth, flag
    tt_val[idx], tt_move[idx] = value, selected
    return value


def _best_move_nb(me, opp, color, depth, weights,
                  tt_key, tt_depth, tt_flag, tt_val, tt_move,
                  killers, history, nodes, abort, node_limit, solve_endgame=True,
                  root_pv=-1, deadline=0.0):
    moves_mask = _legal_mask_nb(me, opp)
    if not moves_mask:
        return -1
    moves, count = _order_moves_nb(moves_mask, depth, color, killers, history)
    _promote_hint(moves, count, root_pv)
    selected = moves[0]
    best_score = alpha = -1e300
    beta = 1e300
    for index in moves:
        if deadline > 0.0 and _clock() >= deadline:
            abort[0] = 2
            break
        child_me, child_opp = _apply_move_nb(me, opp, 1 << index)
        score = -_negamax_nb(
            child_opp, child_me, -color, depth - 1, -beta, -alpha, weights,
            tt_key, tt_depth, tt_flag, tt_val, tt_move,
            killers, history, nodes, abort, node_limit, solve_endgame, deadline,
        )
        if abort[0] != 0:
            break
        if score > best_score:
            best_score, selected = score, index
        alpha = max(alpha, score)
    return selected


def _masks(bb, color):
    return (int(bb[0]), int(bb[1])) if color == COLOR_BLACK else (int(bb[1]), int(bb[0]))


def _to_coord(index):
    return int(index) // 8, int(index) % 8


def _prepare(state, weights, time_budget, node_limit):
    state["root_pv"].fill(-1)
    state["tt_depth"].fill(-1)
    state["tt_move"].fill(-1)
    state["killers"].fill(-1)
    state["history"].fill(0.0)
    state["nodes"][0] = state["abort"][0] = 0
    state["deadline"][0] = 0.0 if time_budget is None else _clock() + max(0.0, time_budget)
    # Node limits are independent, deterministic limits, never clock estimates.
    state["node_limit"][0] = max(0, int(node_limit)) if node_limit is not None else 0
    state["weights"][:] = array("d", (float(value) for value in weights))


def _run(state, me, opp, color, depth, solve_endgame=True):
    return _best_move_nb(
        me, opp, color, depth, tuple(float(value) for value in state["weights"]),
        state["tt_key"], state["tt_depth"], state["tt_flag"], state["tt_val"], state["tt_move"],
        state["killers"], state["history"], state["nodes"], state["abort"],
        state["node_limit"], solve_endgame, state["root_pv"][0], float(state["deadline"][0]),
    )


def best_move(bb, color, depth, weights, time_budget=None, node_limit=None):
    if not 1 <= depth <= _MAX_DEPTH:
        raise ValueError("depth must be between 1 and 64")
    started = _clock()
    state = _new_state()
    _prepare(state, weights, time_budget, node_limit)
    if time_budget is not None:
        state["deadline"][0] = started + max(0.0, time_budget)
    me, opp = _masks(bb, color)
    index = _run(state, me, opp, color, depth)
    return None if index < 0 else _to_coord(index)


def best_move_iterative(bb, color, max_depth, weights, time_budget=None, node_limit=None,
                        stats=None):
    if not 1 <= max_depth <= _MAX_DEPTH:
        raise ValueError("max_depth must be between 1 and 64")
    started = _clock()
    state = _new_state()
    _prepare(state, weights, time_budget, node_limit)
    if time_budget is not None:
        state["deadline"][0] = started + max(0.0, time_budget)
    me, opp = _masks(bb, color)
    moves = legal_moves(bb, color)
    empties = 64 - (me | opp).bit_count()
    target = max(max_depth, empties) if empties <= ENDGAME_EMPTY_THRESHOLD else max_depth
    result = dict(completed_depth=0, target_depth=target, nodes=0, interrupted=False,
                  terminal_solved=False, fallback_used=False, stop_reason="max_depth")
    best = moves[0] if moves else None
    if not moves:
        result.update(terminal_solved=not bool(legal_moves(bb, -color)), stop_reason="no_legal_move")
    else:
        for depth in range(1, target + 1):
            if time_budget is not None and _clock() >= state["deadline"][0]:
                result.update(interrupted=True, stop_reason="time_budget")
                break
            index = _run(state, me, opp, color, depth, solve_endgame=False)
            if state["abort"][0] != 0:
                reason = "time_budget" if state["abort"][0] == 2 else "node_limit"
                result.update(interrupted=True, stop_reason=reason)
                break
            if index >= 0:
                best = _to_coord(index)
                state["root_pv"][0] = index
                result["completed_depth"] = depth
            if depth >= empties:
                result.update(terminal_solved=True, stop_reason="terminal_solved")
                break
        result.update(nodes=int(state["nodes"][0]), fallback_used=result["completed_depth"] == 0)
    if stats is not None:
        stats.clear()
        stats.update(result)
    return best


_SEARCH_DEPTH = 10
# Conservative shared-game allowance: two copies together search for at most
# about 150 seconds, leaving headroom under a 180-second whole-game limit.
_GAME_BUDGET_SECONDS = 75.0
_MOVE_BUDGET_SECONDS = 4.4

# BEGIN AUTO-UPDATED FINAL WEIGHTS
_FIXED_WEIGHTS = np.array(
    [-6.334815803928683, -5.959005380883541, 2.133013254455287, -10.0, -8.291851370348118],
    dtype=np.float64,
)
# END AUTO-UPDATED FINAL WEIGHTS


class AI(object):
    def __init__(self, chessboard_size, color, time_out):
        started = _clock()
        self.chessboard_size = chessboard_size
        self.color = color
        self.time_out = time_out
        self.candidate_list = []
        self.weights = _FIXED_WEIGHTS.copy()
        self.depth = _SEARCH_DEPTH
        self.last_search_stats = {}
        self._time_used = max(0.0, _clock() - started)

    def go(self, chessboard):
        started = _clock()
        self.candidate_list.clear()
        self.last_search_stats.clear()
        try:
            bb = board_from_array(chessboard)
            moves = legal_moves(bb, self.color)
            if not moves:
                return self.candidate_list
            self.candidate_list.extend(moves)
            chosen = moves[0]
            try:
                requested = float(self.time_out)
            except (TypeError, ValueError, OverflowError):
                requested = 5.0
            if not requested > 0.0:  # Also handles NaN; never create an unlimited search.
                requested = 5.0
            turn_limit = min(_MOVE_BUDGET_SECONDS, requested)
            remaining = max(0.0, _GAME_BUDGET_SECONDS - self._time_used)
            empties = 64 - (int(bb[0]) | int(bb[1])).bit_count()
            # Estimate our turns, with a reserve for passes. The cumulative cap
            # still applies if we play more often than this estimate predicts.
            expected_turns = max(1, (empties + 1) // 2 + 2)
            allowance = min(turn_limit, remaining / expected_turns)
            margin = min(0.25, max(0.002, allowance * 0.15))
            deadline = started + max(0.0, allowance - margin)
            budget = deadline - _clock()
            if budget > 0.0 and len(moves) > 1:
                selected = best_move_iterative(
                    bb, self.color, self.depth, self.weights,
                    time_budget=budget, stats=self.last_search_stats,
                )
                if selected in moves:
                    chosen = selected
            self.candidate_list.append(chosen)
            return self.candidate_list
        finally:
            self._time_used += max(0.0, _clock() - started)
