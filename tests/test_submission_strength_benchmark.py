"""The local strength report must be standalone and source-hash attributable."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_strength_benchmark_runs_isolated_and_records_both_sources(tmp_path):
    source = ROOT / "submission" / "agent.py"
    report_path = tmp_path / "comparison.json"
    result = subprocess.run([
        sys.executable, "-I", "-B", str(ROOT / "experiments" / "compare_submission_strength.py"),
        str(source), str(source), "--pairs", "0", "--nodes", "73",
        "--position-seconds", "0.005", "--report", str(report_path),
    ], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(item["sha256"] == digest for item in report["sources"].values())
    assert len(report["positions"]) == 8
    assert report["summary"]["games"] == report["summary"]["candidate_points"] == 0
    for fixture in report["positions"]:
        baseline = fixture["engines"]["baseline"]["fixed_nodes"]
        candidate = fixture["engines"]["candidate"]["fixed_nodes"]
        assert candidate["move"] == baseline["move"]
        assert candidate["nodes"] == baseline["nodes"] <= 73
        assert candidate["completed_depth"] == baseline["completed_depth"]