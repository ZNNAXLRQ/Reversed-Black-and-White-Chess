"""诊断脚本：定位 Plan 9 训练中出现的原生崩溃（0xc0000005 访问违例）。

依次压力测试 numba 合法走法掩码与 numpy frontier 计数，观察哪个路径触发崩溃。
仅用于诊断，不进入正式 Agent 或提交。
"""
import sys
import time

import numpy as np

from reversi.board import initial_board
from reversi.evaluator import _count_frontier
from reversi.rules import _legal_mask
from reversi.rules import apply_move, is_terminal, legal_moves
from reversi.search import best_move

rng = np.random.default_rng(0)


def hammer_legal(n_iters=2_000_000):
    print("[probe] hammering _legal_mask (numba) ...", flush=True)
    boards = [initial_board().copy() for _ in range(4)]
    colors = [-1, 1]
    t0 = time.time()
    for i in range(n_iters):
        b = boards[i % 4]
        c = colors[i % 2]
        _legal_mask(b, c)
        if i % 200_000 == 0 and i:
            print(f"  legal_mask calls={i} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[probe] legal_mask OK: {n_iters} calls in {time.time()-t0:.1f}s", flush=True)


def hammer_frontier(n_iters=500_000):
    boards = []
    for _ in range(8):
        b = initial_board().copy()
        for _ in range(20):
            idx = rng.integers(0, 8, size=2)
            b[idx[0], idx[1]] = rng.choice([-1, 1])
        boards.append(b)
    colors = [-1, 1]
    t0 = time.time()
    for i in range(n_iters):
        _count_frontier(boards[i % 8], colors[i % 2])
        if i % 100_000 == 0 and i:
            print(f"  frontier calls={i} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[probe] frontier OK: {n_iters} calls in {time.time()-t0:.1f}s", flush=True)


def hammer_games(n_games=800, depth=3):
    """跑完整深度-3 对局，观察在哪个对局发生原生崩溃。"""
    print(f"[probe] playing {n_games} full depth-{depth} games ...", flush=True)
    w = np.array([-3.0861110771421476, -10.0, -0.3857156832111528, -7.651743487832784, 0.48587137914320344])
    t0 = time.time()
    for g in range(1, n_games + 1):
        board = initial_board()
        color = -1
        steps = 0
        while not is_terminal(board) and steps < 200:
            moves = legal_moves(board, color)
            if moves:
                mv = best_move(board, color, depth, w)
                if mv is None:
                    mv = moves[0]
                board = apply_move(board, color, mv)
                steps += 1
            color = -color
        if g % 10 == 0 or g <= 3:
            print(f"  game {g}/{n_games} steps={steps} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"[probe] games OK: {n_games} games in {time.time()-t0:.1f}s", flush=True)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "legal"
    if which == "legal":
        hammer_legal()
    elif which == "frontier":
        hammer_frontier()
    elif which == "games":
        hammer_games()
    else:
        hammer_legal()
        hammer_frontier()
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
