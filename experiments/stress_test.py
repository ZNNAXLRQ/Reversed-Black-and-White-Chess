"""Plan 7 本地可用性与压力测试（离线脚本，不进入正式 Agent）。

检查项：
- 完整深度 3 对局闭环（走法合法性 + 终局判定）；
- candidate_list 结构（全部合法走法 + 选中走法在末尾）；
- 无合法走法返回空列表；
- 单步耗时 / 整局耗时（<5s / <180s）；
- 进程内存 RSS（<100MB）。
"""
import ctypes
import os
import time
from ctypes import wintypes

import numpy as np

from reversi.agent import AI, BASELINE_WEIGHTS, SEARCH_DEPTH
from reversi.board import COLOR_BLACK, COLOR_NONE, COLOR_WHITE, initial_board
from reversi.rules import apply_move, is_terminal, legal_moves


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def current_rss_mb():
    """返回当前进程工作集大小（MB）；失败返回 None。"""
    try:
        psapi = ctypes.WinDLL("psapi.dll")
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        handle = kernel32.GetCurrentProcess()
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def load_champion_weights():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "experiments", "outputs", "plan7_champion_weights.npy")
    if os.path.exists(path):
        return np.load(path)
    return BASELINE_WEIGHTS.copy()


def main():
    champion = load_champion_weights()
    print(f"champion weights = {np.round(champion, 4).tolist()}")

    rss0 = current_rss_mb()

    black = AI(8, COLOR_BLACK, 5.0)
    white = AI(8, COLOR_WHITE, 5.0)
    black.weights = champion.copy()
    white.weights = BASELINE_WEIGHTS.copy()
    black.depth = SEARCH_DEPTH
    white.depth = SEARCH_DEPTH

    # 预热：触发 Numba JIT 编译（不计入正式走法计时）。
    t0 = time.time()
    black.go(initial_board())
    warmup_s = time.time() - t0
    print(f"warmup (JIT compile) = {warmup_s:.3f}s")

    # 完整深度 3 对局。
    board = initial_board()
    color = COLOR_BLACK
    steps = 0
    move_times = []
    t_game_start = time.time()

    while not is_terminal(board) and steps < 200:
        ai = black if color == COLOR_BLACK else white
        moves = legal_moves(board, color)
        if moves:
            t0 = time.time()
            cand = ai.go(board)
            dt = time.time() - t0
            move_times.append(dt)

            assert cand, "有合法走法时不应返回空列表"
            assert set(cand[:-1]) == set(moves), "candidate_list 前部应包含全部合法走法"
            chosen = cand[-1]
            assert chosen in moves, "选中走法必须合法"
            board = apply_move(board, color, chosen)
            steps += 1
        color = -color

    t_game_end = time.time()
    rss1 = current_rss_mb()

    print(f"terminal = {is_terminal(board)}")
    print(f"steps = {steps}")
    print(f"total_game_time = {t_game_end - t_game_start:.3f}s")
    if move_times:
        print(f"max_move_time = {max(move_times):.3f}s")
        print(f"avg_move_time = {sum(move_times) / len(move_times):.3f}s")
        print(f"num_timed_moves = {len(move_times)}")

    # 无合法走法检查。
    no_move_board = np.full((8, 8), COLOR_WHITE, dtype=np.int8)
    no_move_board[0, 0] = COLOR_NONE
    no_move_board[1, 1] = COLOR_BLACK
    result = black.go(no_move_board)
    assert result == [], "无合法走法应返回空列表"
    assert black.candidate_list == []
    print("no-legal-move returns [] = OK")

    print(f"rss_before = {rss0:.2f} MB" if rss0 else "rss_before = N/A")
    print(f"rss_after  = {rss1:.2f} MB" if rss1 else "rss_after  = N/A")


if __name__ == "__main__":
    main()
