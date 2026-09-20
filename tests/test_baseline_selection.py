"""실음원/학습 없는 작은 배열·fake runner로 validation 선택 계약을 검증한다."""

import json

import numpy as np
import pytest

from deep_anc.eval.baseline_selection import load_frozen_selection, select_validation_baselines


def case(session="s", source="x", split="validation"):
    t = np.arange(256) / 16000
    d = .02 * np.cos(2 * np.pi * 2000 * t) + .01 * np.cos(2 * np.pi * 250 * t)
    return dict(split=split, session_id=session, source_id=source, source_kind="noise",
        reference=d.copy(), disturbance=d, regions={"onset": [[0, 64]], "transition": [[64, 128]], "steady": [[128, 256]]})


def candidate(name="a", algorithm="fxnlms", mu=.5):
    return dict(id=name, algorithm=algorithm, mu=mu, control_length=4, block_samples=8,
                normalization_epsilon=1e-6, weight_norm_limit=20.)


def options(tmp_path, **kwargs):
    value = dict(secondary=np.array([0., .5, -.1]), secondary_estimate=np.array([0., .5, -.1]),
        additional_delay_samples=2, estimate_delay_samples=2, sample_rate=16000, control_limit=.2,
        power_floor=1e-12, max_limited_fraction=0., max_low_amplification_db=0., max_case_runs=100,
        out=tmp_path / "new", runtime_conditions={"kind": "unit_fixture", "physical_latency_measured": False})
    value.update(kwargs)
    return value


def fake_runner(x, d, s, **kwargs):
    return dict(control=np.zeros_like(x), raw_control=np.zeros_like(x), residual=d * kwargs["mu"],
                settings={"algorithm": kwargs["algorithm"]})


def test_algorithm_specific_selection_artifacts_and_tie_policy(tmp_path):
    grid = [candidate("z", mu=.5), candidate("a", mu=.5), candidate("slow", mu=.8),
            candidate("plain", algorithm="fxlms", mu=.7)]
    report = select_validation_baselines([case(), case("s2")], grid, **options(tmp_path), runner=fake_runner)
    chosen = report["frozen_selection"]["payload"]["selected_by_algorithm"]
    assert chosen["fxnlms"]["id"] == "a"
    assert chosen["fxlms"]["id"] == "plain"
    assert len(report["case_results"]) == 8
    assert all(row["status"] == "ok" for row in report["case_results"])
    assert load_frozen_selection(tmp_path / "new/selection.json") == report["frozen_selection"]
    assert len((tmp_path / "new/case_results.jsonl").read_text().splitlines()) == 8
    assert json.loads((tmp_path / "new/report.json").read_text()) == report
    assert not report["neural_training_executed"] and not report["test_accessed"]


@pytest.mark.parametrize("bad_split", ["train", "test", "valid", None])
def test_all_split_metadata_checked_before_any_array_access(tmp_path, bad_split):
    class ForbiddenArray(dict):
        def __getitem__(self, key):
            if key in ("reference", "disturbance"):
                pytest.fail("split 검사 전 파형 접근")
            return super().__getitem__(key)
    rows = [ForbiddenArray(case()), ForbiddenArray(case("s2", split=bad_split))]
    with pytest.raises(ValueError, match="validation"):
        select_validation_baselines(rows, [candidate()], **options(tmp_path), runner=fake_runner)
    assert not (tmp_path / "new").exists()


def test_budget_overrun_refuses_entire_search_and_no_output(tmp_path):
    with pytest.raises(ValueError, match="예산"):
        select_validation_baselines([case(), case("s2")], [candidate(), candidate("b")],
            **options(tmp_path, max_case_runs=3), runner=lambda *a, **k: pytest.fail("부분 실행 금지"))
    assert not (tmp_path / "new").exists()


def test_failed_candidate_rows_are_kept_and_never_win(tmp_path):
    def execute(x, d, s, **kwargs):
        if kwargs["mu"] == .1:
            raise FloatingPointError("intentional fixture failure")
        return fake_runner(x, d, s, **kwargs)
    report = select_validation_baselines([case()], [candidate("bad", mu=.1), candidate("ok", mu=.5)],
        **options(tmp_path), runner=execute)
    assert report["case_results"][0]["status"] == "failed"
    assert report["candidate_results"][0]["failures"]
    assert report["frozen_selection"]["payload"]["selected_by_algorithm"]["fxnlms"]["id"] == "ok"


def test_nonfinite_runner_diagnostics_are_saved_as_failure(tmp_path):
    def execute(x, d, s, **kwargs):
        result = fake_runner(x, d, s, **kwargs)
        result["weight_norm"] = np.inf
        return result
    report = select_validation_baselines([case()], [candidate()], **options(tmp_path), runner=execute)
    assert report["case_results"][0]["status"] == "failed"
    assert not report["candidate_results"][0]["eligible"]
    assert report["frozen_selection"]["payload"]["selected_by_algorithm"]["fxnlms"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("violation", ["clipping", "output", "low"])
def test_high_score_with_clip_or_low_amplification_cannot_win(tmp_path, violation):
    def execute(x, d, s, **kwargs):
        result = fake_runner(x, d, s, **kwargs)
        if kwargs["mu"] == .1:
            if violation == "clipping":
                result["raw_control"][:] = .4
            elif violation == "output":
                result["control"][:] = .3
            else:
                result["residual"] += .1 * np.cos(2 * np.pi * 250 * np.arange(x.size) / 16000)
        return result
    report = select_validation_baselines([case()], [candidate("bad", mu=.1), candidate("ok", mu=.5)],
        **options(tmp_path), runner=execute)
    assert report["candidate_results"][0]["violations"]
    assert not report["candidate_results"][0]["eligible"]
    assert report["frozen_selection"]["payload"]["selected_by_algorithm"]["fxnlms"]["id"] == "ok"


def test_no_eligible_candidate_is_explicit_not_silent_fallback(tmp_path):
    row = case()
    row["disturbance"][:] = 0.
    report = select_validation_baselines([row], [candidate()], **options(tmp_path), runner=fake_runner)
    artifact = report["frozen_selection"]["payload"]
    assert artifact["selected_by_algorithm"]["fxnlms"] is None
    assert not artifact["selection_complete"]
    assert report["candidate_results"][0]["score"] is None


def test_source_order_invariance_and_session_equal_weight(tmp_path):
    first = select_validation_baselines([case("s1", "a"), case("s1", "b"), case("s2", "c")],
        [candidate()], **options(tmp_path, out=tmp_path / "one"), runner=fake_runner)
    second = select_validation_baselines([case("s2", "c"), case("s1", "b"), case("s1", "a")],
        [candidate()], **options(tmp_path, out=tmp_path / "two"), runner=fake_runner)
    assert first == second
    assert len(first["candidate_results"][0]["sessions"]) == 2


def test_short_actual_classical_runner_plumbing_only(tmp_path):
    report = select_validation_baselines([case()], [candidate("plain", "fxlms", .001), candidate("norm", mu=.001)],
        **options(tmp_path, max_low_amplification_db=100., max_limited_fraction=1.))
    assert len(report["case_results"]) == 2
    assert all(row["status"] == "ok" for row in report["case_results"])
    assert {row["runner_settings"]["normalized"] for row in report["case_results"]} == {False, True}


def test_existing_output_and_tampered_selection_rejected(tmp_path):
    opts = options(tmp_path)
    select_validation_baselines([case()], [candidate()], **opts, runner=fake_runner)
    with pytest.raises(FileExistsError):
        select_validation_baselines([case()], [candidate()], **opts,
            runner=lambda *a, **k: pytest.fail("기존 결과면 실행 금지"))
    path = tmp_path / "new/selection.json"
    value = json.loads(path.read_text())
    value["payload"]["selected_by_algorithm"]["fxnlms"]["mu"] = .99
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="SHA"):
        load_frozen_selection(path)


def test_parent_symlink_and_existing_output_rejected_before_wave_access(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    class ForbiddenWave(dict):
        def __getitem__(self, key):
            if key in ("reference", "disturbance"):
                pytest.fail("경로 거부 전에 파형을 읽으면 안 됨")
            return super().__getitem__(key)
    with pytest.raises(ValueError, match="symlink"):
        select_validation_baselines([ForbiddenWave(case())], [candidate()],
            **options(tmp_path, out=link / "new"), runner=fake_runner)
    with pytest.raises(FileExistsError):
        select_validation_baselines([ForbiddenWave(case())], [candidate()],
            **options(tmp_path, out=real), runner=fake_runner)


@pytest.mark.parametrize("bad", [None, 0., np.nan, True, np.array([.1]), .1 + 1j])
def test_candidate_scalar_validation(tmp_path, bad):
    item = candidate()
    item["mu"] = bad
    with pytest.raises(ValueError):
        select_validation_baselines([case()], [item], **options(tmp_path), runner=fake_runner)


def test_missing_candidate_fields_and_duplicate_cases_rejected(tmp_path):
    missing = candidate()
    missing.pop("normalization_epsilon")
    with pytest.raises(ValueError):
        select_validation_baselines([case()], [missing], **options(tmp_path), runner=fake_runner)
    with pytest.raises(ValueError, match="중복"):
        select_validation_baselines([case(), case()], [candidate()], **options(tmp_path), runner=fake_runner)
