"""Plan 9 受保护写回机制的自动化测试。

覆盖：标记缺失、重复、顺序错误、非法权重（形状/非有限/越界）、
写入失败恢复、成功写回。

全部测试在临时目录进行（monkeypatch 覆盖 _SUBMISSION_PATH / _SUBMISSION_VERSIONS_DIR），
绝不动正式 submission/agent.py。
"""
import codecs

import numpy as np
import pytest

from experiments import plan9

OLD_WEIGHTS = [-3.0861110771421476, -10.0, -0.3857156832111528, -7.651743487832784, 0.48587137914320344]
NEW_WEIGHTS = [-9.79726314898437, -4.302298754340315, 1.685966608200344, -8.507084729322496, 5.230027020280524]

BEGIN = "# BEGIN AUTO-UPDATED FINAL WEIGHTS"
END = "# END AUTO-UPDATED FINAL WEIGHTS"


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def log(self, msg):
        self.messages.append(msg)


def _template(weights_list):
    weights_repr = ", ".join(repr(float(x)) for x in weights_list)
    return (
        "import numpy as np\n"
        "\n"
        "_SEARCH_DEPTH = 4\n"
        "\n"
        f"{BEGIN}\n"
        "_FIXED_WEIGHTS = np.array(\n"
        f"    [{weights_repr}],\n"
        "    dtype=np.float64,\n"
        ")\n"
        f"{END}\n"
    )


def _setup(tmp_path, monkeypatch, content):
    from experiments import plan9 as mod

    agent_path = tmp_path / "agent.py"
    agent_path.write_text(content, encoding="utf-8")
    versions_dir = tmp_path / "versions"
    monkeypatch.setattr(mod, "_SUBMISSION_PATH", str(agent_path))
    monkeypatch.setattr(mod, "_SUBMISSION_VERSIONS_DIR", str(versions_dir))
    return mod, agent_path, versions_dir


def test_writeback_success(tmp_path, monkeypatch):
    old_content = _template(OLD_WEIGHTS)
    mod, agent_path, versions_dir = _setup(tmp_path, monkeypatch, old_content)

    info = mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())

    assert info["updated"] is True
    written = agent_path.read_text(encoding="utf-8")
    assert written.count(BEGIN) == 1
    assert written.count(END) == 1
    # 标记外内容（BEGIN 之前）未变
    assert written[: written.index(BEGIN)] == old_content[: old_content.index(BEGIN)]
    # 权重已更新
    loaded = mod.load_submission_weights(str(agent_path))
    assert np.allclose(loaded, np.asarray(NEW_WEIGHTS, dtype=np.float64))
    # 备份存在且内容为旧文件
    backups = list(versions_dir.glob("agent_pre_writeback_*.py"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == old_content


def test_writeback_missing_marker(tmp_path, monkeypatch):
    content = "import numpy as np\n_FIXED_WEIGHTS = np.array([0.0, 0, 0, 0, 0], dtype=np.float64)\n"
    mod, _, _ = _setup(tmp_path, monkeypatch, content)
    with pytest.raises(RuntimeError):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())


def test_writeback_duplicate_marker(tmp_path, monkeypatch):
    content = _template(OLD_WEIGHTS) + f"{BEGIN}\n{END}\n"
    mod, agent_path, _ = _setup(tmp_path, monkeypatch, content)
    before = agent_path.read_text(encoding="utf-8")
    with pytest.raises(RuntimeError):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_text(encoding="utf-8") == before


def test_writeback_marker_order_wrong(tmp_path, monkeypatch):
    content = (
        "import numpy as np\n"
        f"{END}\n"
        "_FIXED_WEIGHTS = np.array([0.0, 0, 0, 0, 0], dtype=np.float64)\n"
        f"{BEGIN}\n"
    )
    mod, _, _ = _setup(tmp_path, monkeypatch, content)
    with pytest.raises(RuntimeError):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())


def test_writeback_invalid_shape(tmp_path, monkeypatch):
    mod, _, _ = _setup(tmp_path, monkeypatch, _template(OLD_WEIGHTS))
    with pytest.raises(RuntimeError):
        mod.update_submission_weights([1.0, 2.0, 3.0], _FakeLogger())


def test_writeback_nonfinite(tmp_path, monkeypatch):
    mod, _, _ = _setup(tmp_path, monkeypatch, _template(OLD_WEIGHTS))
    with pytest.raises(RuntimeError):
        mod.update_submission_weights([np.nan, 0.0, 0.0, 0.0, 0.0], _FakeLogger())


def test_writeback_out_of_bounds(tmp_path, monkeypatch):
    mod, _, _ = _setup(tmp_path, monkeypatch, _template(OLD_WEIGHTS))
    with pytest.raises(RuntimeError):
        mod.update_submission_weights([11.0, 0.0, 0.0, 0.0, 0.0], _FakeLogger())


def test_writeback_replace_failure_no_change(tmp_path, monkeypatch):
    old_content = _template(OLD_WEIGHTS)
    mod, agent_path, _ = _setup(tmp_path, monkeypatch, old_content)

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_text(encoding="utf-8") == old_content


def test_writeback_verify_failure_restores(tmp_path, monkeypatch):
    old_content = _template(OLD_WEIGHTS)
    mod, agent_path, _ = _setup(tmp_path, monkeypatch, old_content)

    # 写回后验证阶段：让 load_submission_weights 返回错误权重，触发自动恢复。
    monkeypatch.setattr(mod, "load_submission_weights", lambda p: np.zeros(5))
    with pytest.raises(RuntimeError):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_text(encoding="utf-8") == old_content


@pytest.mark.parametrize("bom", [b"", codecs.BOM_UTF8], ids=["utf8", "utf8-bom"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_writeback_preserves_source_bytes(tmp_path, monkeypatch, bom, newline):
    mod, agent_path, versions_dir = _setup(tmp_path, monkeypatch, "")
    source = ("# 中文前缀\n" + _template(OLD_WEIGHTS) + "# 中文后缀\n").replace("\n", newline)
    # Include a different newline outside the block, to detect normalization.
    original = bom + source.encode("utf-8") + b"# mixed newline\n"
    agent_path.write_bytes(original)
    info = mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    written = agent_path.read_bytes()
    begin, end = BEGIN.encode(), END.encode()
    assert info["updated"] is True
    assert written[:written.index(begin)] == original[:original.index(begin)]
    assert written[written.index(end):] == original[original.index(end):]
    assert written.startswith(codecs.BOM_UTF8) == bool(bom)
    assert np.array_equal(mod.load_submission_weights(str(agent_path)), NEW_WEIGHTS)
    compile(written, str(agent_path), "exec")
    backups = list(versions_dir.glob("agent_pre_writeback_*.py"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_writeback_bom_verify_failure_restores_exact_bytes(tmp_path, monkeypatch):
    mod, agent_path, _ = _setup(tmp_path, monkeypatch, "")
    original = codecs.BOM_UTF8 + _template(OLD_WEIGHTS).replace("\n", "\r\n").encode("utf-8")
    agent_path.write_bytes(original)
    # Even a numerically close but incorrect deployment must be rejected.
    wrong = np.asarray(NEW_WEIGHTS) + 1e-9
    monkeypatch.setattr(mod, "load_submission_weights", lambda path: wrong)
    with pytest.raises(RuntimeError, match="written weights"):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_bytes() == original


@pytest.mark.parametrize("invalid_suffix", ["return 1\n", "x = \ufeff1\n"])
def test_writeback_invalid_source_never_replaced(tmp_path, monkeypatch, invalid_suffix):
    mod, agent_path, _ = _setup(tmp_path, monkeypatch, "")
    original = codecs.BOM_UTF8 + (_template(OLD_WEIGHTS) + invalid_suffix).encode("utf-8")
    agent_path.write_bytes(original)
    with pytest.raises(RuntimeError, match="does not compile"):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_bytes() == original
    assert not agent_path.with_suffix(".py.tmp").exists()


def test_writeback_reuses_identical_backup_without_overwrite(tmp_path, monkeypatch):
    mod, agent_path, versions_dir = _setup(tmp_path, monkeypatch, _template(OLD_WEIGHTS))
    original = agent_path.read_bytes()
    versions_dir.mkdir()
    backup = versions_dir / f"agent_pre_writeback_{mod.sha256_file(str(agent_path))[:12]}.py"
    backup.write_bytes(original)
    stamp = backup.stat().st_mtime_ns
    mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert backup.read_bytes() == original and backup.stat().st_mtime_ns == stamp


def test_writeback_conflicting_backup_rejected(tmp_path, monkeypatch):
    mod, agent_path, versions_dir = _setup(tmp_path, monkeypatch, _template(OLD_WEIGHTS))
    original = agent_path.read_bytes()
    versions_dir.mkdir()
    backup = versions_dir / f"agent_pre_writeback_{mod.sha256_file(str(agent_path))[:12]}.py"
    backup.write_bytes(b"historical backup must not be overwritten")
    with pytest.raises(RuntimeError, match="backup"):
        mod.update_submission_weights(NEW_WEIGHTS, _FakeLogger())
    assert agent_path.read_bytes() == original
    assert backup.read_bytes() == b"historical backup must not be overwritten"
