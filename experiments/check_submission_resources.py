"""Check an isolated AI.py in a fresh process, without importing training code.

This local utility is NOT part of the submission. Windows measures the OS peak
working set; Linux/macOS use getrusage(RUSAGE_SELF).ru_maxrss. No psutil needed.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time


def memory_snapshot():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "method": "Windows GetProcessMemoryInfo / peak working set",
            "rss_bytes": counters.WorkingSetSize,
            "peak_rss_bytes": counters.PeakWorkingSetSize,
        }
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "method": "getrusage(RUSAGE_SELF).ru_maxrss",
        "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
    }


def reference_moves(board, color):
    """Independent square-by-square oracle, not the agent's own legal mask."""
    cells = board.tolist()
    moves = {}
    for r in range(8):
        for c in range(8):
            if cells[r][c] != 0:
                continue
            flips = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    line = []
                    while 0 <= rr < 8 and 0 <= cc < 8 and cells[rr][cc] == -color:
                        line.append((rr, cc))
                        rr, cc = rr + dr, cc + dc
                    if line and 0 <= rr < 8 and 0 <= cc < 8 and cells[rr][cc] == color:
                        flips.extend(line)
            if flips:
                moves[(r, c)] = flips
    return moves


def initial_board(np):
    board = np.zeros((8, 8), dtype=np.int8)
    board[3, 3] = board[4, 4] = 1
    board[3, 4] = board[4, 3] = -1
    return board


def apply_reference(board, color, move, moves):
    board[move] = color
    for point in moves[move]:
        board[point] = color


def checked_go(ai, board, np):
    before = board.copy()
    expected = reference_moves(board, ai.color)
    started = time.perf_counter()
    candidates = ai.go(board)
    elapsed = time.perf_counter() - started
    assert np.array_equal(board, before), "go modified its input board"
    assert isinstance(candidates, list) and candidates == ai.candidate_list
    assert all(type(move) is tuple and len(move) == 2
               and all(type(value) is int for value in move) for move in candidates)
    assert set(candidates) == set(expected), "candidate_list omitted or added legal moves"
    if expected:
        assert len(candidates) == len(expected) + 1
        assert set(candidates[:-1]) == set(expected) and candidates[-1] in expected
    else:
        assert candidates == []
    assert elapsed < 5.0, "single decision exceeded 5 seconds: %.6f" % elapsed
    return candidates, elapsed


def check_source(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
            assert node.level == 0, "relative imports are not single-file-safe"
        else:
            continue
        assert "os" not in names, "forbidden module import"
        imports.extend(names)
    return sorted(set(imports))


def child_check(args):
    def progress(message):
        print(message, file=sys.stderr, flush=True)

    progress("Checking isolated AI.py...")
    report = {
        "python": sys.version,
        "platform": sys.platform,
        "source_sha256": hashlib.sha256(args.agent.read_bytes()).hexdigest(),
        "source_imports": check_source(args.agent),
        "memory": {"before_import": memory_snapshot()},
    }
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("AI", str(args.agent))
    agent = importlib.util.module_from_spec(spec)
    sys.modules["AI"] = agent
    spec.loader.exec_module(agent)
    report["import_seconds"] = time.perf_counter() - started
    report["memory"]["after_import"] = memory_snapshot()
    import numpy as np

    assert tuple(inspect.signature(agent.AI.__init__).parameters) == (
        "self", "chessboard_size", "color", "time_out",
    )
    assert tuple(inspect.signature(agent.AI.go).parameters) == ("self", "chessboard")
    started = time.perf_counter()
    ai = agent.AI(8, -1, args.time_out)
    report["init_seconds"] = time.perf_counter() - started
    board = initial_board(np)
    candidates, elapsed = checked_go(ai, board, np)
    report["first_go_seconds"] = elapsed
    report["first_go_candidates"] = list(candidates)
    report["memory"]["after_first_go"] = memory_snapshot()
    report["numba_loaded"] = "numba" in sys.modules
    progress("First go: %.4fs" % elapsed)
    report["position_checks"] = []
    rng = random.Random(20261102)
    targets = (40, 32, 24, 14, 13, 8)
    for index in range(args.positions):
        board = initial_board(np)
        color = -1
        target = targets[index % len(targets)]
        for _ in range(125):
            moves = reference_moves(board, color)
            if np.count_nonzero(board == 0) <= target:
                break
            if moves:
                apply_reference(board, color, rng.choice(list(moves)), moves)
            elif not reference_moves(board, -color):
                break
            color = -color
        ai.color = color
        _, elapsed = checked_go(ai, board, np)
        report["position_checks"].append({
            "empties": int(np.count_nonzero(board == 0)), "seconds": elapsed,
        })
        progress("Position %d/%d: %.4fs" % (index + 1, args.positions, elapsed))
    # Check pass with a deliberately dirty previous candidate list.
    board = np.ones((8, 8), dtype=np.int8)
    board[0, 0] = 0
    board[1, 1] = -1
    ai.color = -1
    ai.candidate_list[:] = [(-9, -9)]
    checked_go(ai, board, np)

    report["games"] = []
    for index in range(args.games):
        players = {-1: agent.AI(8, -1, args.time_out), 1: agent.AI(8, 1, args.time_out)}
        board = initial_board(np)
        color = -1
        durations = {-1: [], 1: []}
        wall = time.perf_counter()
        for ply in range(125):
            moves = reference_moves(board, color)
            if not moves and not reference_moves(board, -color):
                break
            candidates, elapsed = checked_go(players[color], board, np)
            durations[color].append(elapsed)
            progress("Game %d, ply %d: %.4fs" % (index + 1, ply, elapsed))
            if candidates:
                apply_reference(board, color, candidates[-1], moves)
            color = -color
        else:
            raise AssertionError("game did not terminate")
        wall = time.perf_counter() - wall
        assert wall < 180.0, "whole game exceeded 180 seconds"
        report["games"].append({
            "game": index + 1, "plies": ply, "wall_seconds": wall,
            "black_seconds": sum(durations[-1]), "white_seconds": sum(durations[1]),
            "max_go_seconds": max(durations[-1] + durations[1], default=0.0),
            "black_discs": int(np.count_nonzero(board == -1)),
            "white_discs": int(np.count_nonzero(board == 1)),
        })
    report["memory"]["final"] = memory_snapshot()
    peak = max(item["peak_rss_bytes"] for item in report["memory"].values())
    report["peak_rss_bytes"] = peak
    report["peak_rss_mb"] = peak / 1000000
    report["peak_rss_mib"] = peak / 1048576
    report["memory_limit_mb"] = args.memory_mb
    report["passed"] = peak < args.memory_mb * 1000000
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "submission" / "agent.py")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--positions", type=int, default=6)
    parser.add_argument("--time-out", type=float, default=5.0)
    parser.add_argument("--memory-mb", type=float, default=100.0)
    parser.add_argument("--process-timeout", type=float, default=300.0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.agent = args.agent.resolve()
    if args.games < 0 or args.positions < 0 or args.memory_mb <= 0 or args.process_timeout <= 0:
        parser.error("counts must be nonnegative and limits must be positive")
    if args.child:
        return child_check(args)
    with tempfile.TemporaryDirectory(prefix="reversi_resource_check_") as directory:
        isolated = Path(directory) / "AI.py"
        shutil.copyfile(args.agent, isolated)
        command = [
            sys.executable, "-I", "-B", "-X", "faulthandler",
            str(Path(__file__).resolve()), str(isolated), "--child",
            "--games", str(args.games), "--positions", str(args.positions),
            "--time-out", str(args.time_out), "--memory-mb", str(args.memory_mb),
        ]
        try:
            result = subprocess.run(command, cwd=directory, stdout=subprocess.PIPE, text=True,
                                    timeout=args.process_timeout)
        except subprocess.TimeoutExpired:
            print("FAIL: isolated process timed out", file=sys.stderr)
            return 1
    if not result.stdout:
        print("FAIL: isolated check produced no report (exit code %s)" % result.returncode,
              file=sys.stderr)
        return 1
    print(result.stdout)
    if args.report:
        output = args.report.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.stdout, encoding="utf-8")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())