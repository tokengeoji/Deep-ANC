"""YAML 설정 로드/병합/검증.

모든 스크립트는 이 모듈을 통해 설정을 읽는다. 설정 파일 간 참조
(train_*.yaml 의 model_config / data_config / duct_config)는 여기서 해석한다.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# 저장소 루트 (src/deep_anc/config.py 기준 두 단계 위)
REPO_ROOT = Path(__file__).resolve().parents[2]

# 3-스레드 런타임의 콜백↔추론 핸드오프(1 hop) — 학습 플랜트 지연에 가산되는 기본값.
# duct.yaml secondary_path.handoff_extra_samples 가 명시되면 그 값을 쓰고,
# 모든 소비처(.get 기본값)는 이 상수를 공유한다 (감사 L10 — 기본값 분기 금지).
DEFAULT_HANDOFF_SAMPLES = 256


def _resolve_path(path: str | Path) -> Path:
    """상대 경로는 저장소 루트 기준으로 해석한다 (실행 위치와 무관하게 동작)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        candidate = Path.cwd() / p
        p = candidate if candidate.exists() else REPO_ROOT / p
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: 최상위는 매핑(dict)이어야 합니다")
    return data


def deep_merge(base: dict, override: dict) -> dict:
    """중첩 dict 병합 — override 우선. 리스트는 통째로 교체."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """'a.b.c=value' 형태의 CLI 오버라이드 적용."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"오버라이드 형식 오류 (key=value): {item}")
        key, _, raw = item.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def load_train_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """학습 설정 로드 + 참조된 model/data/duct 설정을 함께 해석.

    오버라이드는 두 번 적용한다: 참조 경로 자체(model_config 등)를 바꿀 수 있도록
    로드 전에 한 번, 로드된 서브 설정의 내부 키(data.* 등)를 바꿀 수 있도록 후에 한 번.
    """
    cfg = load_yaml(path)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg["model"] = load_yaml(cfg["model_config"])
    cfg["data"] = load_yaml(cfg["data_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate_duct(cfg["duct"])
    _propagate_d_noise_delay(cfg)
    return cfg


def load_runtime_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = load_yaml(path)
    if overrides:
        # 참조 경로 자체(hardware_config/duct_config)를 바꿀 수 있도록 먼저 적용한다.
        cfg = apply_overrides(cfg, overrides)
    cfg["hardware"] = load_yaml(cfg["hardware_config"])
    cfg["duct"] = load_yaml(cfg["duct_config"])
    if overrides:
        # 로드된 하위 설정도 CLI에서 재현 가능하게 바꿀 수 있어야 한다.
        # 이 두 번째 적용이 없으면 ``--set hardware.audio.block_size=512`` 같은
        # 런타임 조정은 위의 참조 파일 로드에서 조용히 사라진다.
        cfg = apply_overrides(cfg, overrides)
    _propagate_d_noise_delay(cfg)
    return cfg


def _propagate_d_noise_delay(cfg: dict) -> None:
    """duct 의 ``d_noise_delay_samples`` 를 ``cfg["data"]`` 로 **통과시킨다** (유도 아님).

    왜 필요한가
    ----------
    실측 브랜치의 lead 는 ``K' = (D_noise + K) − d_recorded`` 로 유도되어야 두 브랜치가
    모델에게 주는 총 선행량이 같아진다. 그런데 ``RecordedANCDataset`` 은 ``data_cfg`` 만
    받고, ``d_noise_delay_samples`` 는 ``duct.yaml`` 에 있어서 그 값이 도달하지 못했다.
    그래서 ``recorded_lead_mode=timeline`` 을 켜면 "d_noise_delay_samples 가 필요합니다"
    로 실패했고, 결국 ``constant`` 로 남아 있었다.

    ``constant`` 의 대가 (2026-08-06 실측):

        합성  x_ref 가 d 보다  1602 + 116 = 1718 샘플 앞선다
        실측  x_ref 가 d 보다   142.5 + 116 =  258 샘플 앞선다
        → 어긋남 1460 샘플 (30.4 ms). 같은 모델이 두 브랜치에서 다른 과제를 배운다.

    여기서 하는 것은 **복사**이지 유도가 아니다. 값의 단일 출처는 여전히 duct.yaml 이고,
    두 브랜치가 같은 숫자를 읽게 만드는 것이 목적이다.
    """

    data = cfg.get("data")
    duct = cfg.get("duct")
    if not isinstance(data, dict) or not isinstance(duct, dict):
        return
    value = (duct.get("digital_reference") or {}).get("d_noise_delay_samples")
    if value is None:
        return
    declared = data.get("d_noise_delay_samples")
    if declared is not None and int(declared) != int(value):
        raise ValueError(
            f"d_noise_delay_samples 가 두 곳에서 다릅니다: data_sim {declared} vs "
            f"duct {value} — 같은 물리량을 두 곳에서 정하지 마세요 (duct.yaml 이 출처)"
        )
    data["d_noise_delay_samples"] = int(value)


def validate_duct(duct: dict) -> list[str]:
    """duct.yaml 의 미기입(null) 항목을 경고 목록으로 반환 (치명 오류는 예외)."""
    warnings: list[str] = []
    positions = duct.get("positions_m", {})
    for name in ("noise_speaker", "reference_mic", "cancel_speaker", "error_mic"):
        if positions.get(name) is None:
            warnings.append(f"duct.yaml positions_m.{name} 이 비어 있습니다 — 시뮬레이션 정확도에 영향")
    digital = duct.get("digital_reference", {})
    if digital.get("d_noise_delay_samples") is None:
        warnings.append(
            "duct.yaml digital_reference.d_noise_delay_samples 미실측 — "
            "덕트 기하로부터의 추정값을 사용합니다 (scripts/data/calibrate_wideband.py 로 실측 권장)"
        )
    for w in warnings:
        print(f"[duct.yaml 경고] {w}")
    return warnings


def duct_distance_samples(duct: dict, a: str, b: str, sample_rate: int) -> int:
    """두 장비 위치 간 음향 전파 지연(샘플). a, b는 positions_m 키."""
    pos = duct["positions_m"]
    if pos.get(a) is None or pos.get(b) is None:
        raise ValueError(f"duct.yaml positions_m 에 {a}/{b} 값이 필요합니다")
    c = float(duct["duct"]["speed_of_sound_mps"])
    dist = abs(float(pos[a]) - float(pos[b]))
    return int(round(sample_rate * dist / c))


def default_d_noise_delay(duct: dict, sample_rate: int, s_path_delay: int) -> int:
    """digital-ref 1차경로 순수지연 기본값 (미실측 시).

    소음(ch0)과 상쇄(ch1)는 같은 USB 출력 장치를 쓰므로 전기/버퍼 지연이 공통이다.
    측정된 S(z) 지연 = 공통지연 + t_ac(CS→ERR) 이므로,
        D_noise ≈ s_path_delay − t_ac(CS→ERR) + t_ac(NS→ERR)
    (근거: docs/01_physics_limits.md, 교차검증 C2)
    """
    t_cs_err = duct_distance_samples(duct, "cancel_speaker", "error_mic", sample_rate)
    t_ns_err = duct_distance_samples(duct, "noise_speaker", "error_mic", sample_rate)
    return int(s_path_delay - t_cs_err + t_ns_err)
