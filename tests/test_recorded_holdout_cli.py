"""CSV-only holdout 도구가 canonical provenance를 우회하지 못하게 한다."""

from __future__ import annotations

from scripts.data import make_recorded_holdout as holdout


def test_csv_only_holdout_requires_explicit_diagnostic_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(holdout, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["make_recorded_holdout.py"])

    assert holdout.main() == 2
    assert "canonical holdout을 만들 수 없습니다" in capsys.readouterr().err
    assert not list(tmp_path.rglob("*.json"))


def test_diagnostic_holdout_cannot_target_official_manifest_directory(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(holdout, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "make_recorded_holdout.py",
            "--diagnostic-only",
            "--out",
            "data/manifests/recorded_holdout.json",
        ],
    )

    assert holdout.main() == 2
    assert "results/diagnostics" in capsys.readouterr().err
    assert not (tmp_path / "data/manifests/recorded_holdout.json").exists()
