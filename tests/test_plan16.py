"""Plan 16 测试：评估等价性、终局精确搜索、24 局协议、候选池选择、位棋盘等价性、
搜索核心 numba 等价性、部署深度基准。"""
import os
import time

import numpy as np
import pytest

from reversi.board import (
    COLOR_BLACK, COLOR_NONE, COLOR_WHITE, board_from_array, board_to_array,
    empty_count, initial_board, initial_board_array,
)
from reversi.evaluator import evaluate, features, terminal_utility
from reversi.rules import apply_move, has_legal_move, is_terminal, legal_moves
from reversi.search import ENDGAME_EMPTY_THRESHOLD, best_move

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def log(self, msg):
        self.messages.append(msg)


# ---------------------------------------------------------------------------
# numpy 参照实现（位棋盘交叉验证用）
# ---------------------------------------------------------------------------
def _ref_legal(board, color):
    opp = -color
    res = []
    for r in range(8):
        for c in range(8):
            if board[r, c] != 0:
                continue
            for dr, dc in _DIRECTIONS:
                rr, cc = r + dr, c + dc
                if not (0 <= rr < 8 and 0 <= cc < 8) or board[rr, cc] != opp:
                    continue
                rr += dr
                cc += dc
                while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == opp:
                    rr += dr
                    cc += dc
                if 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == color:
                    res.append((r, c))
                    break
    return sorted(res)


def _ref_apply(board, color, move):
    r, c = move
    nb = board.copy()
    nb[r, c] = color
    opp = -color
    for dr, dc in _DIRECTIONS:
        rr, cc = r + dr, c + dc
        line = []
        while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == opp:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == color:
            for x, y in line:
                nb[x, y] = color
    return nb


def _ref_frontier(board, color):
    from reversi.evaluator import POSITION_TABLE
    disc = board == color
    empty = board == COLOR_NONE
    fr = np.zeros((8, 8), np.int8)
    for dr, dc in _DIRECTIONS:
        s = np.zeros((8, 8), np.int8)
        r0 = max(0, -dr)
        r1 = min(8, 8 - dr)
        c0 = max(0, -dc)
        c1 = min(8, 8 - dc)
        s[r0:r1, c0:c1] = empty[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        fr = fr | (disc & s)
    return int(fr.sum())


def _ref_features(board, persp):
    from reversi.evaluator import POSITION_TABLE
    opp = -persp
    own_mask = board == persp
    opp_mask = board == opp
    disc = (int(own_mask.sum()) - int(opp_mask.sum())) / 64.0
    pos = (float((POSITION_TABLE * own_mask).sum()) - float((POSITION_TABLE * opp_mask).sum())) / (64.0 * 120.0)
    lm = len(_ref_legal(board, persp))
    om = len(_ref_legal(board, opp))
    mob = (lm - om) / max(1, lm + om)
    corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
    oc = sum(1 for r, c in corners if board[r, c] == persp)
    opc = sum(1 for r, c in corners if board[r, c] == opp)
    corner = (oc - opc) / 4.0
    fdiff = (_ref_frontier(board, persp) - _ref_frontier(board, opp)) / 64.0
    return np.array([disc, pos, mob, corner, fdiff])


def _random_board(rng, moves=12):
    board = initial_board_array()
    color = COLOR_BLACK
    for _ in range(moves):
        lm = _ref_legal(board, color)
        if not lm:
            color = -color
            continue
        m = lm[rng.integers(len(lm))]
        board = _ref_apply(board, color, m)
        color = -color
    return board


def _find_near_endgame(max_empty=8):
    for seed in range(50):
        rng = np.random.default_rng(seed)
        board = initial_board()
        color = COLOR_BLACK
        last = None
        steps = 0
        while steps < 64:
            if is_terminal(board):
                break
            e = empty_count(board)
            moves = legal_moves(board, color)
            if moves and e <= max_empty:
                last = (board.copy(), color)
            if moves:
                board = apply_move(board, color, moves[rng.integers(len(moves))])
            color = -color
            steps += 1
        if last is not None:
            return last
    raise RuntimeError("未能构造近终局局面")


def _brute_score(bb, color, memo):
    key = (color, int(bb[0]), int(bb[1]))
    if key in memo:
        return memo[key]
    moves = legal_moves(bb, color)
    if not moves:
        if has_legal_move(bb, -color):
            val = -_brute_score(bb, -color, memo)
        else:
            val = terminal_utility(bb, color)
    else:
        val = -1e300
        for m in moves:
            val = max(val, -_brute_score(apply_move(bb, color, m), -color, memo))
    memo[key] = val
    return val


# ---------------------------------------------------------------------------
# 1. 评估函数等价性（numba 位棋盘 vs Python 位棋盘 features）
# ---------------------------------------------------------------------------
def test_evaluate_numba_equivalence():
    rng = np.random.default_rng(0)
    weights = np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
    for _ in range(20):
        arr = _random_board(rng)
        bb = board_from_array(arr)
        for persp in (COLOR_BLACK, COLOR_WHITE):
            expected = float(np.dot(weights, features(bb, persp)))
            assert np.allclose(evaluate(bb, persp, weights), expected, rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. 位棋盘与 numpy 原实现逐项等价（走法/翻转/评估交叉验证）
# ---------------------------------------------------------------------------
def test_bitboard_rules_equivalence():
    rng = np.random.default_rng(1)
    for _ in range(100):
        arr = _random_board(rng)
        bb = board_from_array(arr)
        color = COLOR_BLACK if rng.integers(2) == 0 else COLOR_WHITE
        assert sorted(legal_moves(bb, color)) == _ref_legal(arr, color)
        lm = _ref_legal(arr, color)
        if lm:
            m = lm[rng.integers(len(lm))]
            nb_ref = _ref_apply(arr, color, m)
            nb_bb = board_to_array(apply_move(bb, color, m))
            assert np.array_equal(nb_bb, nb_ref)


def test_bitboard_evaluation_equivalence():
    rng = np.random.default_rng(2)
    weights = np.array([-1.0, 2.0, -3.0, 4.0, -5.0])
    for _ in range(50):
        arr = _random_board(rng)
        bb = board_from_array(arr)
        for persp in (COLOR_BLACK, COLOR_WHITE):
            ref = _ref_features(arr, persp)
            got = features(bb, persp)
            assert np.allclose(got, ref, rtol=1e-9, atol=1e-9)
            assert np.allclose(evaluate(bb, persp, weights), float(np.dot(weights, ref)), rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. 搜索核心 numba 版与纯 Python 版结果一致
# ---------------------------------------------------------------------------
def _py_negamax(bb, color, depth, weights):
    if depth == 0:
        return evaluate(bb, color, weights)
    moves = legal_moves(bb, color)
    if not moves:
        if has_legal_move(bb, -color):
            return -_py_negamax(bb, -color, depth, weights)
        return terminal_utility(bb, color)
    best = -1e300
    for m in moves:
        best = max(best, -_py_negamax(apply_move(bb, color, m), -color, depth - 1, weights))
    return best


def test_search_core_numba_matches_python():
    from reversi.search import _negamax_nb, _masks, _new_state, _prepare

    rng = np.random.default_rng(3)
    weights = np.array([-1.0, 1.0, 1.0, 1.0, -1.0])
    for depth in (1, 2, 3):
        for _ in range(5):
            arr = _random_board(rng, moves=8)
            bb = board_from_array(arr)
            color = COLOR_BLACK if rng.integers(2) == 0 else COLOR_WHITE
            if not legal_moves(bb, color):
                continue
            ref = _py_negamax(bb, color, depth, weights)
            state = _new_state()
            _prepare(state, weights, None, None)
            me, opp = _masks(bb, color)
            got = _negamax_nb(me, opp, color, depth, -1e300, 1e300, state["weights"],
                              state["tt_key"], state["tt_depth"], state["tt_flag"],
                              state["tt_val"], state["tt_move"], state["killers"],
                              state["history"], state["nodes"], state["abort"],
                              state["node_limit"])
            assert abs(got - ref) < 1e-6, f"depth={depth} got={got} ref={ref}"


# ---------------------------------------------------------------------------
# 4. 终局精确搜索
# ---------------------------------------------------------------------------
def test_endgame_exact_search_weight_independent():
    assert ENDGAME_EMPTY_THRESHOLD == 14
    board, color = _find_near_endgame(max_empty=8)
    move_a = best_move(board, color, 0, np.array([-10.0] * 5))
    move_b = best_move(board, color, 0, np.array([10.0] * 5))
    assert move_a is not None and move_b is not None
    assert move_a in legal_moves(board, color)
    assert move_a == move_b


def test_endgame_exact_search_optimal():
    board, color = _find_near_endgame(max_empty=8)
    moves = legal_moves(board, color)
    assert moves
    chosen = best_move(board, color, 0, np.zeros(5))
    assert chosen in moves
    memo = {}
    chosen_score = -_brute_score(apply_move(board, color, chosen), -color, memo)
    best_score = max(-_brute_score(apply_move(board, color, m), -color, memo) for m in moves)
    assert chosen_score == pytest.approx(best_score)


# ---------------------------------------------------------------------------
# 5. 24 局协议与颜色平衡
# ---------------------------------------------------------------------------
def test_validate_protocol_color_balance(monkeypatch):
    from experiments.match_runner import GreedyAgent, MatchRunner, WeightedAgent, GameResult

    calls = []

    def fake_play_game(self, black_agent, white_agent, seed=None):
        calls.append((black_agent, white_agent, seed))
        return GameResult(black_count=3, white_count=2, steps=1, elapsed=0.0, seed=seed)

    monkeypatch.setattr(MatchRunner, "play_game", fake_play_game)

    runner = MatchRunner(depth=1)
    weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    opponent_weights = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    ]
    result = runner.validate_protocol(weights, weights, opponent_weights, greedy_seed=999)

    assert len(calls) == 24
    assert len(result.games) == 24
    assert len(result.u_values) == 24
    assert len(result.self_play_u_values) == 8

    for i in range(4):
        black, white, _ = calls[i]
        assert white.weights.tolist() == weights.tolist()
    for i in range(4, 8):
        black, white, _ = calls[i]
        assert black.weights.tolist() == weights.tolist()

    idx = 8
    for opp_w in opponent_weights:
        for _ in range(2):
            black, white, _ = calls[idx]
            assert black.weights.tolist() == opp_w.tolist()
            assert white.weights.tolist() == weights.tolist()
            idx += 1
        for _ in range(2):
            black, white, _ = calls[idx]
            assert black.weights.tolist() == weights.tolist()
            assert white.weights.tolist() == opp_w.tolist()
            idx += 1
    for _ in range(2):
        black, white, _ = calls[idx]
        assert isinstance(black, GreedyAgent)
        assert white.weights.tolist() == weights.tolist()
        idx += 1
    for _ in range(2):
        black, white, _ = calls[idx]
        assert black.weights.tolist() == weights.tolist()
        assert isinstance(white, GreedyAgent)
        idx += 1
    assert idx == 24


# ---------------------------------------------------------------------------
# 6. 候选池选择
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cycles", [(1, 4, 6, 7), (1, 4, 6, 7, 10)])
def test_champion_pool_loads_history(tmp_path, monkeypatch, cycles):
    import json
    from experiments import plan9

    history = [{"cycle_index": cycle, "weights": [float(cycle)] * 5,
                "validation_cost": 0.4, "promoted_at": "2026-10-17 00:00:00"}
               for cycle in cycles]
    path = tmp_path / "champion_pool.json"
    path.write_text(json.dumps({"champions": history}), encoding="utf-8")
    monkeypatch.setattr(plan9, "_CHAMPION_POOL_PATH", str(path))
    champions = plan9.load_champion_pool()
    assert champions == history
    for c in champions:
        assert len(c["weights"]) == 5
        assert "validation_cost" in c
        assert "promoted_at" in c


def test_champion_pool_selection():
    from experiments.plan9 import select_fixed_opponents

    champions = [
        {"cycle_index": 1, "weights": [1, 0, 0, 0, 0], "validation_cost": 0.5},
        {"cycle_index": 2, "weights": [0, 1, 0, 0, 0], "validation_cost": 0.2},
        {"cycle_index": 3, "weights": [0, 0, 1, 0, 0], "validation_cost": 0.4},
        {"cycle_index": 4, "weights": [0, 0, 0, 1, 0], "validation_cost": 0.3},
        {"cycle_index": 5, "weights": [9, 9, 9, 9, 9], "validation_cost": 0.1},
    ]
    incumbent = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
    selected = select_fixed_opponents(champions, incumbent, _FakeLogger())
    assert selected == [[0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0]]


# ---------------------------------------------------------------------------
# 7. 部署深度动态基准
# ---------------------------------------------------------------------------
def test_deployment_search_depth_is_10():
    import sys
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "submission"))
    import agent
    assert agent._SEARCH_DEPTH == 10


def test_deployment_single_move_within_budget():
    import sys
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "submission"))
    import agent
    ai = agent.AI(8, agent.COLOR_BLACK, 5.0)
    ai.go(initial_board_array())  # 预热 numba 编译
    board = initial_board_array()
    start = time.time()
    cand = ai.go(board)
    elapsed = time.time() - start
    moves = agent.legal_moves(agent.board_from_array(board), agent.COLOR_BLACK)
    assert len(cand) == len(moves) + 1
    assert cand[-1] in moves
    assert elapsed < 5.0


def test_deployment_full_game_within_budget():
    import sys
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "submission"))
    import agent
    black = agent.AI(8, agent.COLOR_BLACK, 0.2)
    white = agent.AI(8, agent.COLOR_WHITE, 0.2)
    black.go(initial_board_array())
    white.go(initial_board_array())

    board = initial_board()
    color = agent.COLOR_BLACK
    steps = 0
    start = time.time()
    while True:
        if not agent._legal_mask_nb(board[0], board[1]) and not agent._legal_mask_nb(board[1], board[0]):
            break
        ai = black if color == agent.COLOR_BLACK else white
        moves = agent.legal_moves(board, color)
        if moves:
            cand = ai.go(board_to_array(board))
            chosen = cand[-1]
            assert chosen in moves
            me = board[0] if color == agent.COLOR_BLACK else board[1]
            opp = board[1] if color == agent.COLOR_BLACK else board[0]
            pos = np.uint64(1) << np.uint64(chosen[0] * 8 + chosen[1])
            me_new, opp_new = agent._apply_move_nb(me, opp, pos)
            if color == agent.COLOR_BLACK:
                board = np.array([me_new, opp_new], dtype=np.uint64)
            else:
                board = np.array([opp_new, me_new], dtype=np.uint64)
        color = -color
        steps += 1
        assert steps < 200
    elapsed = time.time() - start
    assert elapsed < 180.0
