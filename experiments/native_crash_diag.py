"""Plan 13 诊断脚本（仅定位原因，不实施修复，不改任何正式实现）。

用于定位 Plan 9 长时间 NCS-C 对局训练中的 Windows 原生崩溃（0xc0000005）。
本文件是独立诊断入口，只复用 reversi/、optimization/、experiments/ 的既有实现，
不修改正式规则、搜索、评估器、NCS、Objective、MatchRunner 或 submission/agent.py。

子命令：
  env        环境快照（Python/numpy/numba/llvmlite/平台 + numba 有效配置）
  timeline   带「逐候选 + 逐局 + 逐阶段」时间线的 NCS 复现（有 max-evals 上限，不跑完 1260 局）
  repeat     同进程内反复跑同一确定性对局（累积状态判定）
  singlegame 独立进程跑单局（给定权重/对手/颜色/种子）
  boundcheck 对 _legal_mask_nb 做 boundscheck=True 的边界核对与随机棋盘模糊测试
"""
import argparse
import ctypes
import hashlib
import os
import sys
import time
from ctypes import wintypes

import numpy as np

from optimization.config import OptimizationConfig
from optimization.ncs.optimizer import NCSConfig, NCSOptimizer
from optimization.parameter_space import ParameterSpace
from reversi.agent import BASELINE_WEIGHTS
from reversi.board import COLOR_BLACK, COLOR_WHITE, count, initial_board, opponent
from reversi.rules import _legal_mask_nb, apply_move, is_terminal, legal_moves
from reversi.search import best_move

from .match_runner import (
    GameResult,
    GreedyAgent,
    MatchRunner,
    ObjectiveEvaluationError,
    WeightedAgent,
)
from .reversi_objective import ReversiObjective

# 冠军权重（与 submission/agent.py 的 _FIXED_WEIGHTS 一致；此处仅用于诊断复现，不修改提交文件）。
CHAMPION_WEIGHTS = np.array(
    [-3.0861110771421476, -10.0, -0.3857156832111528, -7.651743487832784, 0.48587137914320344],
    dtype=np.float64,
)

_DIAG_DIRECTIONS_NB = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int8,
)


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
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return pmc.WorkingSetSize / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def board_hash(board):
    return hashlib.md5(np.ascontiguousarray(board).tobytes()).hexdigest()[:8]


def rss_str():
    r = current_rss_mb()
    return f"{r:.1f}MB" if r is not None else "N/A"


class TimelineLogger:
    """同时写 stdout 与日志文件，逐行 flush，便于崩溃前保留最后一条完整记录。"""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.t0 = time.time()
        self.game = 0
        self.cand = 0

    def _stamp(self):
        return f"[{time.strftime('%H:%M:%S')} +{time.time() - self.t0:7.1f}s]"

    def line(self, msg):
        text = f"{self._stamp()} {msg}"
        self.f.write(text + "\n")
        self.f.flush()
        print(text, flush=True)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class DiagMatchRunner(MatchRunner):
    """镜像 MatchRunner.play_game 的确定性对局循环，仅增加逐阶段时间线，不改语义。"""

    def __init__(self, depth, log, verbose=True):
        super().__init__(depth=depth)
        self._log = log
        self._verbose = verbose

    def play_game(self, black_agent, white_agent, seed=None):
        self._log.game += 1
        g = self._log.game
        board = initial_board()
        color = COLOR_BLACK
        steps = 0
        start = time.time()
        ba = type(black_agent).__name__
        wa = type(white_agent).__name__
        self._log.line(
            f"[game {g} start] black={ba} white={wa} seed={seed} rss={rss_str()} "
            f"cand={self._log.cand}"
        )
        try:
            while not is_terminal(board):
                t0 = time.time()
                moves = legal_moves(board, color)
                dt = time.time() - t0
                if self._verbose:
                    self._log.line(
                        f"[game {g} step {steps}] phase=legal_moves color={color} "
                        f"n={len(moves)} t={dt:.4f}s board={board_hash(board)}"
                    )
                if moves:
                    agent = black_agent if color == COLOR_BLACK else white_agent
                    t0 = time.time()
                    move = agent.select_move(board, color, moves, seed)
                    dt = time.time() - t0
                    if self._verbose:
                        self._log.line(
                            f"[game {g} step {steps}] phase=select_move color={color} "
                            f"move={move} t={dt:.4f}s"
                        )
                    if move not in moves:
                        raise ObjectiveEvaluationError(
                            f"illegal move {move} by color {color}; "
                            f"legal_moves={moves}; board=\n{board}; steps={steps}"
                        )
                    t0 = time.time()
                    board = apply_move(board, color, move)
                    dt = time.time() - t0
                    if self._verbose:
                        self._log.line(
                            f"[game {g} step {steps}] phase=apply_move color={color} "
                            f"t={dt:.4f}s board={board_hash(board)}"
                        )
                    steps += 1
                color = opponent(color)
                if steps > 200:
                    raise ObjectiveEvaluationError(f"game exceeded 200 steps; board=\n{board}")
        except ObjectiveEvaluationError:
            self._log.line(f"[game {g} ERROR] ObjectiveEvaluationError (re-raise)")
            raise
        except Exception as exc:  # noqa: BLE001
            self._log.line(f"[game {g} EXC] {exc!r}")
            raise ObjectiveEvaluationError(
                f"game error: {exc!r}; board=\n{board}; color={color}; steps={steps}"
            )
        elapsed = time.time() - start
        self._log.line(
            f"[game {g} end] steps={steps} black={count(board, COLOR_BLACK)} "
            f"white={count(board, COLOR_WHITE)} elapsed={elapsed:.3f}s "
            f"rss={rss_str()} board={board_hash(board)}"
        )
        return GameResult(
            black_count=count(board, COLOR_BLACK),
            white_count=count(board, COLOR_WHITE),
            steps=steps,
            elapsed=elapsed,
            seed=seed,
        )


class DiagObjective(ReversiObjective):
    """包装真实 Objective，逐候选记录，并把 runner 换成 DiagMatchRunner。"""

    def __init__(self, baseline_weights, depth, greedy_seeds, log, verbose=True):
        self._log = log
        super().__init__(baseline_weights, depth=depth, greedy_seeds=greedy_seeds)
        self.runner = DiagMatchRunner(depth=depth, log=log, verbose=verbose)
        self._cache = {}
        self._eval = 0

    def __call__(self, candidate):
        self._eval += 1
        self._log.cand = self._eval
        w = self.decode(candidate)
        whash = hashlib.md5(np.ascontiguousarray(w).tobytes()).hexdigest()[:12]
        self._log.line(
            f"[cand {self._eval} start] weights={np.round(w, 9).tolist()} whash={whash}"
        )
        result = super().__call__(candidate)
        self._log.line(
            f"[cand {self._eval} done] cost={result.cost:.9f} rss={rss_str()}"
        )
        return result


def cmd_env(args):
    import numba

    lines = []
    lines.append(f"python_version = {sys.version}")
    lines.append(f"python_executable = {sys.executable}")
    lines.append(f"platform = {sys.platform}")
    lines.append(f"numpy = {np.__version__}")
    lines.append(f"numba = {numba.__version__}")
    try:
        import llvmlite

        lines.append(f"llvmlite = {llvmlite.__version__}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"llvmlite = <unavailable: {exc!r}>")
    for key in (
        "NUMBA_DISABLE_JIT",
        "NUMBA_BOUNDSCHECK",
        "NUMBA_THREADING_LAYER",
        "NUMBA_NUM_THREADS",
        "NUMBA_OPT",
        "NUMBA_FASTMATH",
        "NUMBA_CACHE_DIR",
        "NUMBA_PARALLEL_DIAGNOSTICS",
    ):
        try:
            lines.append(f"numba.config.{key} = {getattr(numba.config, key)!r}")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"numba.config.{key} = <unavailable: {exc!r}>")
    lines.append(f"NUMBA_DISABLE_JIT env = {os.environ.get('NUMBA_DISABLE_JIT')!r}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


def cmd_timeline(args):
    log = TimelineLogger(args.out)
    log.line(
        f"[timeline] seed={args.seed} N=10 T_max=20 sigma0=0.1 r=0.99 epoch=10 "
        f"max_evals={args.max_evals} depth=3 greedy=(101,202) (历史崩溃复现配置)"
    )
    log.line(f"[timeline] python={sys.version.split()[0]} numpy={np.__version__} rss={rss_str()}")

    space = ParameterSpace([0.0] * 5, [1.0] * 5)
    objective = DiagObjective(
        BASELINE_WEIGHTS, depth=3, greedy_seeds=(101, 202), log=log, verbose=args.phase_log  # 历史 depth=3 复现配置
    )
    optimizer = NCSOptimizer(
        NCSConfig(T_max=20, population_size=10, sigma0=0.1, r=0.99, epoch=10)
    )
    cfg = OptimizationConfig(seed=args.seed, max_evaluations=args.max_evals)

    log.line("[timeline] NCS optimize start")
    t0 = time.time()
    try:
        result = optimizer.optimize(objective, space, cfg)
        log.line(
            f"[timeline] DONE stop_reason={result.stop_reason} evals={result.num_evaluations} "
            f"games={result.num_evaluations * 6} best_cost={result.best_cost:.9f} "
            f"elapsed={time.time() - t0:.1f}s rss={rss_str()}"
        )
    except BaseException as exc:  # noqa: BLE001
        log.line(
            f"[timeline] PYTHON-EXIT {type(exc).__name__}: {exc!r} "
            f"elapsed={time.time() - t0:.1f}s rss={rss_str()}"
        )
        log.close()
        raise
    finally:
        log.close()
    return 0


def _parse_weights(spec):
    if spec == "champion":
        return CHAMPION_WEIGHTS.copy()
    return np.array([float(x) for x in spec.split(",")], dtype=np.float64)


def _build_agents(weights, opponent_spec, color, depth):
    w = np.asarray(weights, dtype=np.float64)
    if opponent_spec == "baseline":
        opp = WeightedAgent(BASELINE_WEIGHTS.copy(), depth)
    elif opponent_spec == "greedy":
        opp = GreedyAgent()
    else:
        raise ValueError(f"unknown opponent {opponent_spec!r}")
    cand = WeightedAgent(w, depth)
    if color == "black":
        return cand, opp
    return opp, cand


def cmd_singlegame(args):
    w = _parse_weights(args.weights)
    black, white = _build_agents(w, args.opponent, args.color, args.depth)
    runner = MatchRunner(depth=args.depth)
    r = runner.play_game(black, white, seed=args.greedy_seed)
    print(
        f"singlegame OK black={r.black_count} white={r.white_count} "
        f"steps={r.steps} elapsed={r.elapsed:.3f}s seed={r.seed}"
    )
    return 0


def cmd_repeat(args):
    log = TimelineLogger(args.out)
    w = _parse_weights(args.weights)
    black, white = _build_agents(w, args.opponent, args.color, args.depth)
    log.line(
        f"[repeat] weights={np.round(w, 9).tolist()} opponent={args.opponent} "
        f"color={args.color} depth={args.depth} seed={args.greedy_seed} count={args.count}"
    )
    runner = MatchRunner(depth=args.depth)
    t0 = time.time()
    try:
        for i in range(1, args.count + 1):
            r = runner.play_game(black, white, seed=args.greedy_seed)
            log.line(
                f"[repeat game {i}/{args.count}] black={r.black_count} white={r.white_count} "
                f"steps={r.steps} elapsed={r.elapsed:.3f}s cum={time.time() - t0:.1f}s "
                f"rss={rss_str()}"
            )
    except BaseException as exc:  # noqa: BLE001
        log.line(f"[repeat] PYTHON-EXIT {type(exc).__name__}: {exc!r}")
        log.close()
        raise
    log.line(f"[repeat] DONE count={args.count} cum={time.time() - t0:.1f}s")
    log.close()
    return 0


import numba as _numba


@_numba.njit(boundscheck=True)
def _legal_mask_nb_bc(board, color):
    n = board.shape[0]
    opp = -color
    legal = np.zeros((n, n), dtype=np.bool_)
    for r in range(n):
        for c in range(n):
            if board[r, c] != 0:
                continue
            for d in range(8):
                dr = _DIAG_DIRECTIONS_NB[d, 0]
                dc = _DIAG_DIRECTIONS_NB[d, 1]
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


def cmd_boundcheck(args):
    rng = np.random.default_rng(20261012)
    mismatches = 0
    raised = 0
    for i in range(args.iters):
        board = rng.integers(-1, 2, size=(8, 8)).astype(np.int8)
        color = -1 if i % 2 == 0 else 1
        try:
            ref = _legal_mask_nb(board, color)
            bc = _legal_mask_nb_bc(board, color)
        except Exception as exc:  # noqa: BLE001
            raised += 1
            print(f"[boundcheck] RAISED iter={i} color={color} {type(exc).__name__}: {exc!r}")
            if raised <= 5:
                print(board)
            continue
        if not np.array_equal(ref, bc):
            mismatches += 1
            print(f"[boundcheck] MISMATCH iter={i} color={color}")
            if mismatches <= 5:
                print(board)
                print("ref=", ref.astype(int))
                print("bc =", bc.astype(int))
    # 额外用真实对局产生的棋盘核对。
    real = initial_board()
    color = COLOR_BLACK
    w = CHAMPION_WEIGHTS
    for _ in range(60):
        try:
            ref = _legal_mask_nb(real, color)
            bc = _legal_mask_nb_bc(real, color)
        except Exception as exc:  # noqa: BLE001
            raised += 1
            print(f"[boundcheck] REAL-RAISED {type(exc).__name__}: {exc!r}")
            break
        if not np.array_equal(ref, bc):
            mismatches += 1
            print("[boundcheck] REAL-MISMATCH")
        moves = legal_moves(real, color)
        if not moves:
            color = opponent(color)
            continue
        real = apply_move(real, color, best_move(real, color, 3, w) or moves[0])
        color = opponent(color)
    print(f"[boundcheck] DONE iters={args.iters} mismatches={mismatches} raised={raised}")
    return 0


def main(argv=None):
    import faulthandler

    faulthandler.enable()
    parser = argparse.ArgumentParser(description="Plan 13 原生崩溃诊断入口")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_env = sub.add_parser("env")
    p_env.add_argument("--out", default=None)
    p_env.set_defaults(func=cmd_env)

    p_tl = sub.add_parser("timeline")
    p_tl.add_argument("--seed", type=int, default=45)
    p_tl.add_argument("--max-evals", type=int, default=115)
    p_tl.add_argument("--out", required=True)
    p_tl.add_argument("--phase-log", dest="phase_log", action=argparse.BooleanOptionalAction, default=True)
    p_tl.set_defaults(func=cmd_timeline)

    p_sg = sub.add_parser("singlegame")
    p_sg.add_argument("--weights", default="champion")
    p_sg.add_argument("--opponent", default="baseline")
    p_sg.add_argument("--color", default="black")
    p_sg.add_argument("--depth", type=int, default=3)
    p_sg.add_argument("--greedy-seed", type=int, default=101)
    p_sg.set_defaults(func=cmd_singlegame)

    p_rp = sub.add_parser("repeat")
    p_rp.add_argument("--weights", default="champion")
    p_rp.add_argument("--opponent", default="baseline")
    p_rp.add_argument("--color", default="black")
    p_rp.add_argument("--depth", type=int, default=3)
    p_rp.add_argument("--greedy-seed", type=int, default=101)
    p_rp.add_argument("--count", type=int, default=700)
    p_rp.add_argument("--out", required=True)
    p_rp.set_defaults(func=cmd_repeat)

    p_bc = sub.add_parser("boundcheck")
    p_bc.add_argument("--iters", type=int, default=20000)
    p_bc.set_defaults(func=cmd_boundcheck)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())




