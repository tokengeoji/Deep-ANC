"""실시간 런타임의 안전한 초기 상태 규약."""

import numpy as np
import pytest

from deep_anc.realtime import run_realtime
from deep_anc.realtime.engines import FxLMSEngine
from deep_anc.realtime.run_realtime import RealtimeANC, fxlms_adaptation_allowed


def test_start_on_true_is_rejected_before_hardware_initialization():
    """설정 오버라이드로 ANC ON 시작을 우회할 수 없어야 한다."""
    with pytest.raises(ValueError, match="start_on=true"):
        RealtimeANC({"start_on": True})


def test_fxlms_engine_includes_thread_handoff_and_starts_adaptation_off(tmp_path):
    secondary = tmp_path / "secondary.npz"
    np.savez(
        secondary,
        fir=np.array([1.0], dtype=np.float32),
        delay_samples=np.array(3),
        sample_rate=np.array(48_000),
    )
    engine = FxLMSEngine(
        str(secondary),
        {"control_length": 4, "mu": 0.01, "leakage": 0.0},
        hop=4,
        handoff_extra_samples=4,
    )
    assert engine.secondary_delay_samples == 7
    assert engine.secondary_total_length == 8
    assert engine.controller.secondary_delay_samples == 7
    assert engine.adapt is False

    ref = np.linspace(-0.1, 0.1, 4, dtype=np.float32)
    err = np.full(4, 0.02, dtype=np.float32)
    for _ in range(4):
        engine.step(ref, err)
    assert engine.controller.update_count == 0

    engine.set_adapt_enabled(True)
    engine.step(ref, err)
    assert engine.controller.update_count == 1
    engine.reset()
    assert engine.adapt is False


def test_fxlms_adaptation_gate_is_fail_closed():
    ready = {
        "requested": True,
        "full_anc_gain": True,
        "full_noise_gain": True,
        "hold_samples": 0,
        "output_clip_fraction": 0.0,
        "input_clip_fraction": 0.0,
        "reference_power": 1.0e-3,
        "stream_ok": True,
    }
    assert fxlms_adaptation_allowed(**ready)

    unsafe_values = {
        "requested": False,
        "full_anc_gain": False,
        "full_noise_gain": False,
        "hold_samples": 1,
        "output_clip_fraction": 0.01,
        "input_clip_fraction": 0.01,
        "reference_power": 0.0,
        "stream_ok": False,
    }
    for key, unsafe in unsafe_values.items():
        case = dict(ready)
        case[key] = unsafe
        assert not fxlms_adaptation_allowed(**case), key


def test_runtime_input_preflight_rejects_stuck_error_channel(monkeypatch):
    def fake_probe(*_args, **_kwargs):
        base = {
            "rms_dbfs": -186.64,
            "peak": 0.0,
            "clip_ratio": 0.0,
            "unique_codes": 1,
            "raw_min": -1,
            "raw_max": -1,
        }
        return {
            "channels": [
                dict(base, channel=0, valid=False),
                dict(base, channel=1, valid=False),
            ]
        }

    monkeypatch.setattr(run_realtime, "capture_input_probe", fake_probe)
    cfg = {"reference": "digital", "hardware": {"audio": {}}}
    assert run_realtime.input_preflight(cfg, seconds=0.1) is False


# ======================================================================================
# 엔진 아티팩트 preflight — 조용히 썩은 경로를 시작 전에 잡는다
# ======================================================================================
def test_engine_preflight_rejects_a_missing_artifact_even_when_unused(tmp_path):
    """**지금 읽히지 않는** 키의 경로가 없어도 거부한다.

    2026-08-05 실측: configs/runtime_tiny.yaml 의 `plan: runs/export/tiny_fp16.plan`
    은 존재하지 않는 파일이었는데 engine.type=ort 라 한 번도 읽히지 않아 조용히
    썩어 있었다. "지금 안 쓰니까 괜찮다"가 이 결함이 살아남은 이유다.
    """

    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"x")
    cfg = {
        "engine": {
            "type": "ort",
            "onnx": str(onnx),
            "plan": str(tmp_path / "does_not_exist.plan"),
        }
    }

    problems = run_realtime.engine_artifact_preflight(cfg)

    assert len(problems) == 1
    assert "does_not_exist.plan" in problems[0].detail
    assert "미사용" in problems[0].detail
    assert problems[0].missing_file is True
    # 저장소 위생 검사는 이것도 막는다 …
    with pytest.raises(FileNotFoundError, match="preflight 실패"):
        run_realtime.require_engine_artifacts(cfg)
    # … 그러나 **시작**은 막지 않는다 (읽히지도 않는 파일이다). 경고만 남는다.
    warnings = run_realtime.require_engine_artifacts_to_start(cfg)
    assert len(warnings) == 1 and "does_not_exist.plan" in warnings[0]


def test_runtime_start_is_not_blocked_by_an_unused_missing_artifact(tmp_path):
    """**시작 fail-closed 회귀 방지.**

    2026-08-06 실측 반증: ``engine.type=ort`` 인 배포가 *읽히지도 않는* ckpt/plan 이
    없다는 이유로 ``exit 2`` 로 하드 중단됐다. 모델은 GitHub Release 로 배포되고
    (``git ls-files runs/`` = 0) ort 배포가 onnx 하나만 받는 것은 정상이다.
    활성 아티팩트가 없을 때만 시작을 막는다.
    """

    onnx = tmp_path / "tiny.onnx"
    onnx.write_bytes(b"x")
    cfg = {
        "engine": {
            "type": "ort",
            "ckpt": str(tmp_path / "not_fetched.pt"),
            "onnx": str(onnx),
            "plan": str(tmp_path / "not_fetched.plan"),
        }
    }

    warnings = run_realtime.require_engine_artifacts_to_start(cfg)
    assert len(warnings) == 2
    assert all("미사용" in item for item in warnings)

    # 활성 아티팩트가 없으면 그때는 막는다.
    cfg["engine"]["onnx"] = str(tmp_path / "absent.onnx")
    with pytest.raises(FileNotFoundError, match="활성"):
        run_realtime.require_engine_artifacts_to_start(cfg)


def test_engine_preflight_rejects_a_missing_active_artifact(tmp_path):
    """활성 엔진의 아티팩트가 없으면 당연히 거부한다."""

    cfg = {"engine": {"type": "trt", "plan": str(tmp_path / "absent.plan")}}

    problems = run_realtime.engine_artifact_preflight(cfg)

    assert len(problems) == 1
    assert "활성" in problems[0].detail
    assert problems[0].fatal is True


def test_engine_preflight_rejects_a_wrong_extension_everywhere(tmp_path):
    """확장자 오류는 '아직 안 받았다'가 아니라 **잘못 적은 것**이다 — 어디서나 치명적.

    파일 시스템을 보지 않고 판정할 수 있는 부패이므로, 아티팩트를 받지 않은 새 클론
    에서도 잡힌다. (존재 검사는 그 트리에서 의미가 없다.)
    """

    cfg = {"engine": {"type": "ort", "onnx": "runs/export/tiny_corrected.plan"}}

    problems = run_realtime.engine_artifact_preflight(cfg)

    assert len(problems) == 1
    assert problems[0].fatal is True and problems[0].missing_file is False
    assert "확장자" in problems[0].detail
    with pytest.raises(FileNotFoundError):
        run_realtime.require_engine_artifacts_to_start(cfg)


def test_engine_preflight_rejects_an_empty_active_key(tmp_path):
    """활성 키가 비어 있으면 로드할 것이 없다 — 실패 폐쇄."""

    problems = run_realtime.engine_artifact_preflight({"engine": {"type": "ort"}})

    assert any("engine.onnx 가 비었습니다" in item.detail for item in problems)
    assert all(item.fatal for item in problems)


def test_engine_preflight_rejects_an_unknown_engine_type():
    problems = run_realtime.engine_artifact_preflight({"engine": {"type": "quantum"}})

    assert any("알 수 없는 engine.type" in item.detail for item in problems)
    assert all(item.fatal for item in problems)


_SHIPPED_RUNTIME_CONFIGS = ("configs/runtime.yaml", "configs/runtime_tiny.yaml")

# ``runs/`` 전체는 runtime artifact cohort가 아니다. A100 bootstrap/학습은 log,
# config snapshot, queue state를 이 아래에 정상적으로 만든다. 반면 아래 세 경로는
# Jetson의 legacy diagnostic runtime을 실제로 배포했을 때만 채워지는 cohort다.
# 한 파일이라도 있으면 config typo/deletion을 계속 fail-closed로 잡아야 한다.
_RUNTIME_ARTIFACT_COHORTS = {
    "runs/export": {".onnx", ".plan", ".engine"},
    "runs/pretrain_base_corrected/ckpt": {".pt", ".pth"},
    "runs/pretrain_tiny_corrected/ckpt": {".pt", ".pth"},
}


def _has_fetched_runtime_artifact_cohort(repo_root):
    """배포용 artifact가 실제로 내려온 환경인지, 일반 학습 ``runs``인지 구분한다."""

    for relative, suffixes in _RUNTIME_ARTIFACT_COHORTS.items():
        directory = repo_root / relative
        if not directory.is_dir():
            continue
        if any(
            item.is_file() and item.suffix.casefold() in suffixes
            for item in directory.iterdir()
        ):
            return True
    return False


def test_runtime_artifact_cohort_ignores_training_only_runs_directory(tmp_path):
    """A100의 log/snapshot만 있는 ``runs``는 Jetson artifact fetch가 아니다."""

    training = tmp_path / "runs/loss_pilot_deadbeef/telemetry"
    training.mkdir(parents=True)
    (training / "000000_000100.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runs/loss_pilot_deadbeef/config_snapshot.yaml").write_text(
        "seed: 20260803\n", encoding="utf-8"
    )
    assert not _has_fetched_runtime_artifact_cohort(tmp_path)

    export = tmp_path / "runs/export"
    export.mkdir()
    (export / "readme.txt").write_text("not an engine", encoding="utf-8")
    assert not _has_fetched_runtime_artifact_cohort(tmp_path)

    (export / "tiny_corrected.onnx").write_bytes(b"onnx")
    assert _has_fetched_runtime_artifact_cohort(tmp_path)


def test_shipped_runtime_configs_declare_a_loadable_engine():
    """설정 자체의 결함(빈 키·알 수 없는 type·확장자 부패)은 **어느 환경에서나** 잡힌다.

    아티팩트 파일의 존재와 달리 이것은 저장소에 커밋된 텍스트만으로 판정할 수 있으므로
    호스트에 무엇이 받아져 있든 항상 돈다.
    """

    from deep_anc.config import load_runtime_config

    for name in _SHIPPED_RUNTIME_CONFIGS:
        problems = run_realtime.engine_artifact_preflight(load_runtime_config(name))
        structural = [item.detail for item in problems if not item.missing_file]
        assert structural == [], f"{name}: {structural}"


def test_engine_preflight_accepts_the_shipped_runtime_configs():
    """저장소의 runtime 설정이 실제로 존재하는 파일만 가리키는지 못 박는다.

    이 테스트가 깨지면 누군가 설정에 없는 경로를 다시 적었거나 아티팩트를 지운 것이다.

    ⚠ 2026-08-06 반증 #12: ``runs/`` 는 ``.gitignore`` 대상이고 모델은 GitHub Release 로
    배포된다 — ``git ls-files runs/ | wc -l`` = 0. 따라서 **아티팩트를 받지 않은
    트리**(새 클론, CI, 원격 학습 환경)에서 파일 부재는 저장소 결함이 아니라 "아직
    안 받았다"를 뜻한다. 그 트리에서 스위트를 빨간불로 만들면 "pytest 전부 통과"
    규칙이 이 기기 전용이 되고, 규칙이 무의미해지면 다음 사람이 진짜 실패도 무시한다.

    ⚠⚠ 2026-08-06 통합 검증에서 **이 완화 자체가 반증됐다.** 직전 판정은 "파일 부재면
    무조건 skip" 이었는데, 그러면 ``engine.ckpt`` 를 존재하지 않는 경로로 오타 내도
    (``sed`` 로 실제 재현: ``ckpt: ...THIS_DOES_NOT_EXIST.pt`` → ``11 passed, 1 skipped``)
    어떤 트리에서도 빨간불이 되지 않는다. 이 게이트가 생긴 이유가 바로 ``runtime.yaml``
    이 4개월간 존재한 적 없는 ``model.onnx`` 를 가리킨 것이었으므로, 그 결함 유형이
    통째로 무검출로 돌아간 셈이다. docstring 첫 줄의 약속("없는 경로를 다시 적으면
    깨진다")이 거짓이 됐다.

    그래서 판정 축을 **부재냐 아니냐**가 아니라 :attr:`fatal` (= 그 파일을 실제로
    읽는가)로 되돌린다:
      · **활성** 아티팩트 부재 → 실패. 오타든 삭제든 여기서 잡힌다.
      · 미사용 아티팩트 부재 → 허용. ``engine.type=ort`` 배포가 onnx 만 받는 것은
        정상이고, "runs/ 는 있는데 일부만 있는" 트리가 실제로 존재한다.
      · 그 밖의 부패(빈 키·확장자·알 수 없는 type) → 어디서나 실패.
    아티팩트를 하나도 받지 않은 새 클론과 A100 학습 환경은 모두 runtime artifact
    cohort가 없다. A100은 ``runs/`` 아래에 log/config snapshot을 만들지만 Jetson의
    legacy runtime artifact를 받지는 않으므로, 단순히 ``runs/`` 디렉터리가 있다는
    사실로 fetched tree라고 판정하면 Elice의 정상 bootstrap이 가짜 실패가 된다.
    cohort 안에 하나라도 있으면 누락/오타는 여전히 즉시 실패한다.
    """

    from pathlib import Path

    from deep_anc.config import load_runtime_config

    repo_root = Path(__file__).resolve().parents[1]
    unfetched_runtime_artifacts = not _has_fetched_runtime_artifact_cohort(repo_root)

    unused: list[str] = []
    active_missing: list[str] = []
    for name in _SHIPPED_RUNTIME_CONFIGS:
        problems = run_realtime.engine_artifact_preflight(load_runtime_config(name))
        corrupt = [item.detail for item in problems if not item.missing_file]
        assert corrupt == [], f"{name}: {corrupt}"
        for item in problems:
            bucket = active_missing if item.fatal else unused
            bucket.append(f"{name}: {item.detail}")

    if active_missing and unfetched_runtime_artifacts:
        pytest.skip(
            "Jetson runtime artifact cohort를 받지 않은 트리 — 존재 검사는 실제 배포 "
            "artifact를 받은 환경에서만 의미가 있다: " + "; ".join(active_missing)
        )
    assert active_missing == [], (
        "출하 runtime 설정이 실제로 읽는 아티팩트가 없습니다 — 설정에 없는 경로를 "
        "적었거나 아티팩트를 지운 것입니다: " + "; ".join(active_missing)
    )
