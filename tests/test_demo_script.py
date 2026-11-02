"""scripts/demo.py is what the demo video and README point to: every part must pass --check."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "demo.py"


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("part", ["rag", "guard", "agent"])
def test_demo_part_passes_check(demo, part, capsys):
    assert demo.run(check=True, only=part) == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out


def test_guard_showcase_covers_both_stages_and_languages(demo):
    stages = {stage for _, _, _, stage in demo.GUARD_CHECKS}
    assert {"pattern", "classifier", None} <= stages
    texts = " ".join(text for _, text, _, _ in demo.GUARD_CHECKS)
    assert any("஀" <= ch <= "௿" for ch in texts)  # Tamil
    assert any("ऀ" <= ch <= "ॿ" for ch in texts)  # Devanagari
    assert any(o == "allowed" and s is None for _, _, o, s in demo.GUARD_CHECKS)  # benign / miss


def test_guard_check_fails_on_wrong_expectation(demo, monkeypatch, tmp_path):
    from app.demo import build_demo_harness

    wrong = [("deliberately wrong", "How do I prepare ORS for a child with diarrhoea?", "blocked", "pattern")]
    monkeypatch.setattr(demo, "GUARD_CHECKS", wrong)
    assert demo.run_guard_checks(build_demo_harness(tmp_path / "c.sqlite")) == 1


def test_lfs_pointer_model_is_reported_not_loaded(tmp_path, monkeypatch, caplog):
    import app.injection_clf as ic

    pointer = tmp_path / "injection_clf.npz"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    monkeypatch.setattr(ic, "MODEL_PATH", pointer)
    ic.get_classifier.cache_clear()
    try:
        with caplog.at_level("WARNING"):
            assert ic.get_classifier() is None
        assert "git lfs pull" in caplog.text
    finally:
        ic.get_classifier.cache_clear()
