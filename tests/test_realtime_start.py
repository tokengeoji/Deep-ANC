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
    assert "does_not_exist.plan" in problems[0]
    assert "미사용" in problems[0]
    with pytest.raises(FileNotFoundError, match="preflight 실패"):
        run_realtime.require_engine_artifacts(cfg)


def test_engine_preflight_rejects_a_missing_active_artifact(tmp_path):
    """활성 엔진의 아티팩트가 없으면 당연히 거부한다."""

    cfg = {"engine": {"type": "trt", "plan": str(tmp_path / "absent.plan")}}

    problems = run_realtime.engine_artifact_preflight(cfg)

    assert len(problems) == 1
    assert "활성" in problems[0]


def test_engine_preflight_rejects_an_empty_active_key(tmp_path):
    """활성 키가 비어 있으면 로드할 것이 없다 — 실패 폐쇄."""

    problems = run_realtime.engine_artifact_preflight({"engine": {"type": "ort"}})

    assert any("engine.onnx 가 비었습니다" in item for item in problems)


def test_engine_preflight_rejects_an_unknown_engine_type():
    problems = run_realtime.engine_artifact_preflight({"engine": {"type": "quantum"}})

    assert any("알 수 없는 engine.type" in item for item in problems)


_SHIPPED_RUNTIME_CONFIGS = ("configs/runtime.yaml", "configs/runtime_tiny.yaml")


def test_shipped_runtime_configs_declare_a_loadable_engine():
    """설정 자체의 결함(빈 키·알 수 없는 type)은 **어느 환경에서나** 잡힌다.

    아티팩트 파일의 존재와 달리 이것은 저장소에 커밋된 텍스트만으로 판정할 수 있으므로
    호스트에 무엇이 받아져 있든 항상 돈다.
    """

    from deep_anc.config import load_runtime_config

    for name in _SHIPPED_RUNTIME_CONFIGS:
        problems = run_realtime.engine_artifact_preflight(load_runtime_config(name))
        structural = [p for p in problems if "아티팩트가 없습니다" not in p]
        assert structural == [], f"{name}: {structural}"


def test_engine_preflight_accepts_the_shipped_runtime_configs():
    """저장소의 runtime 설정이 실제로 존재하는 파일만 가리키는지 못 박는다.

    이 테스트가 깨지면 누군가 설정에 없는 경로를 다시 적었거나 아티팩트를 지운 것이다.

    ⚠ 2026-08-06: ``runs/`` 는 ``.gitignore`` 대상이고 모델은 GitHub Release 로 배포된다
    (``git ls-files runs/`` = 0). 따라서 **아티팩트를 받지 않은 트리**(새 클론, CI,
    원격 학습 환경)에서 이 단언은 저장소 결함이 아니라 "아직 안 받았다"를 뜻한다.
    그 트리에서 스위트를 빨간불로 만들면 "pytest 전부 통과" 규칙이 이 기기 전용이 되고,
    규칙이 무의미해지면 다음 사람이 진짜 실패도 무시하게 된다. 그래서 아티팩트 트리에
    한해 검사하고, 설정 자체의 결함은 위 테스트가 항상 잡는다.
    """

    from deep_anc.config import load_runtime_config

    if not (run_realtime.REPO_ROOT / "runs").is_dir():
        pytest.skip(
            "runs/ 가 없는 트리 — 엔진 아티팩트는 .gitignore 대상이라 존재 검사는 "
            "받아 놓은 환경에서만 의미가 있다"
        )

    for name in _SHIPPED_RUNTIME_CONFIGS:
        assert run_realtime.engine_artifact_preflight(load_runtime_config(name)) == [], name
