"""실측 녹음 파인튜닝 데이터셋.

scripts/data/record_duct.py 가 저장한 세션 구조::

    data/recorded/<session_id>/
      mics.wav            : 2ch PCM_32 (ch0=err mic, ch1=ref mic)
      source.wav          : 재생한 디지털 소스 (1ch) — **원본 provenance**
      source_aligned.wav  : ADC 시간축으로 되감은 소스 (1ch) — **학습이 쓰는 것**
      session.json        : 프로그램/레벨/설정/timeline 메타

ANC OFF 상태로 녹음했으므로 err 마이크 신호가 곧 d(t)이다.
digital-ref 모드에서는 소스를, acoustic-ref 모드에서는 ref 마이크(ch1)를 x_ref 로 쓴다.

왜 ``source_aligned.wav`` 인가
-----------------------------
``source.wav`` 는 재생 **배열**이지 방출 **시각**이 아니다. DAC PLL 헌팅(4~5초 주기,
259~407 샘플) 때문에 ``source[t]`` 와 ``mics[t]`` 는 같은 물리 시각이 아니고, 그
결과 실측 80 세션은 ``coh²(source→ERR, 150-600Hz) = 0.021~0.126`` 이었다. 학습이
배워야 할 관계 자체가 없는 데이터였다. ``source_aligned.wav`` 는 REF 마이크를 시간축
증인으로 써서 되감은 것이고, 홀드아웃(ERR) 검증에서 coh² 0.87~0.96 이 나온다.

lead 부기
--------
예전에는 ``lead = data_sim.yaml 의 상수 하나`` 를 **모든 세션·모든 세그먼트**에
적용했다. 실측 source→ERR 지연은 세션별로 1603~1682(79 샘플 산포)라 상수가 맞을 수
없었다. 재정렬 후에는 세션간 산포가 0.4 샘플로 줄어 상수가 성립하지만, 그래도
**측정값에서 유도**해야 두 번째 유도가 생기지 않는다 — :class:`RecordedLeadPlan` 을 보라.
"""

from __future__ import annotations

import io
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch.utils.data import IterableDataset, get_worker_info

from ..dsp.timing import TrainingTimingContract
from .broadband_batch_sampler import (
    MIN_TARGET_D_DENSITY_RATIO,
    QUALIFIED_SAMPLING_MODE,
    BroadbandQualifiedBatchPlanner,
    target_d_density_ratios,
)
from .manifest import read_manifest, read_manifest_bytes
from .recorded_level_calibration import (
    CURRENT_DOMAIN,
    HISTORICAL_DOMAIN,
    RecordedLevelCalibration,
    validate_recorded_level_calibration_receipt,
)
from .resumable_stream import indexed_rng, worker_global_item_indices
from .synth_dataset import _delay_np
from .transfer_contract import (
    RecordedTrainingSnapshot,
    TransferContractError,
    validate_recorded_training_snapshot,
)

__all__ = [
    "RecordedANCDataset",
    "RecordedAugmentConfig",
    "RecordedLeadPlan",
    "make_recorded_eval_batch",
]


_FROZEN = ConfigDict(frozen=True, extra="forbid")
_COMMON_EQ_HALF_HISTORY_SAMPLES = 64  # 129-tap common EQ의 왼쪽 경계 영향
PLANT_DOMAIN_SAMPLING_MODE = "family_plant_domain_component_session_balanced"


class RecordedLeadPlan(BaseModel):
    """세션 하나의 digital-reference lead 부기. **여기서만 유도된다.**

    두 브랜치가 모델에게 주는 "총 선행량"(x_ref 가 d 보다 얼마나 앞서는가)이 같아야
    한다. 아니면 같은 모델이 두 브랜치에서 서로 다른 물리를 배운다 — 실제로 실측
    1716~1795 vs 합성 1602 로 114~193 샘플(600Hz 에서 86°~145°) 어긋나 있었다.

    ::

        합성:  x_ref 가 d 보다  D_noise + K        만큼 앞선다
        실측:  x_ref 가 d 보다  d_recorded + K'    만큼 앞선다
        ⇒ K' = (D_noise + K) − d_recorded

    ``d_recorded`` 는 재정렬 후 ``source_aligned → ERR`` 잔여 지연의 세션 중앙값이고
    ``session.json`` 의 ``timeline.aligned_lag_median_samples`` 에 박혀 있다.

    ``timeline`` 의 합성 총 선행량은 resolved :class:`TrainingTimingContract`에서만
    받는다. primary NPZ ``delay_samples``/compact FIR peak/S delay/handoff를 다시
    숫자로 복사하거나 합의값으로 주입하는 경로는 허용하지 않는다.
    """

    model_config = _FROZEN

    mode: str
    """``"constant"`` = 설정 상수 그대로, ``"timeline"`` = 측정값에서 유도."""

    lead_samples: int
    constant_lead_samples: int
    total_advance_samples: int | None = None
    recorded_delay_samples: float | None = None
    jitter_sigma_samples: float = 0.0

    @model_validator(mode="after")
    def _validate(self) -> "RecordedLeadPlan":
        if self.mode not in {"constant", "timeline"}:
            raise ValueError(f"지원하지 않는 lead mode: {self.mode!r}")
        if self.lead_samples < 0:
            raise ValueError(f"lead 는 0 이상이어야 합니다: {self.lead_samples}")
        if self.jitter_sigma_samples < 0.0:
            raise ValueError("lead 지터 표준편차는 0 이상이어야 합니다")
        if self.mode == "timeline":
            if self.total_advance_samples is None or self.recorded_delay_samples is None:
                raise ValueError(
                    "timeline 모드는 total_advance_samples 와 recorded_delay_samples 가 "
                    "모두 필요합니다 — 없으면 유도가 아니라 추측입니다"
                )
            expected = int(
                round(float(self.total_advance_samples) - float(self.recorded_delay_samples))
            )
            if self.lead_samples != max(0, expected):
                raise ValueError(
                    f"lead 유도 관계 위반: {self.lead_samples} != max(0, "
                    f"{self.total_advance_samples} − {self.recorded_delay_samples})"
                )
        return self

    @classmethod
    def constant(
        cls, lead_samples: int, *, jitter_sigma_samples: float = 0.0
    ) -> "RecordedLeadPlan":
        return cls(
            mode="constant",
            lead_samples=int(lead_samples),
            constant_lead_samples=int(lead_samples),
            jitter_sigma_samples=float(jitter_sigma_samples),
        )

    @classmethod
    def from_timeline(
        cls,
        *,
        total_advance_samples: int,
        recorded_delay_samples: float,
        constant_lead_samples: int,
        jitter_sigma_samples: float = 0.0,
    ) -> "RecordedLeadPlan":
        raw = int(round(float(total_advance_samples) - float(recorded_delay_samples)))
        return cls(
            mode="timeline",
            lead_samples=max(0, raw),
            constant_lead_samples=int(constant_lead_samples),
            total_advance_samples=int(total_advance_samples),
            recorded_delay_samples=float(recorded_delay_samples),
            jitter_sigma_samples=float(jitter_sigma_samples),
        )


class RecordedAugmentConfig(BaseModel):
    """실측 브랜치 증강. 기본 off — 켜는 것은 설정의 명시적 선택이다.

    유효한 증강의 조건
    ------------------
    덕트 플랜트를 LTI 로 보면 **플랜트와 교환 가능한 연산만** 유효하다.
    ``H·(P·n) = P·(H·n)`` 이 성립해야 하므로 x_ref 와 d 에 **같은** 연산을 걸어야 한다.
    소스만 필터링하면 플랜트 불일치를 학습시키는 것이고, 피치/타임스트레치는 정렬을
    깨뜨려 **결함 2 를 다시 만든다.** 시간역전은 인과성을 깬다. 전부 금지다.

    왜 필요한가
    ----------
    실측 train 은 64세션 × 70s = 4480 s 인데 50k step × 배치 16 × 1.4987 s ×
    recorded_ratio 0.7 = 839,272 s 를 소비한다. **187.3 회 반복**이다. 합성 브랜치는
    level_dbfs[-45,-20] / snr_mic_noise_db[5,30] / RIR 추첨을 다 갖는데 실측은 단일
    레벨(amplitude 0.06) 고정이라 모델이 레벨 단축경로를 배우고 두 브랜치의 통계가
    비대칭해진다. 절대목표 2(최악값)는 분포 꼬리를 봐야 하므로 특히 치명적이다.
    """

    model_config = _FROZEN

    enabled: bool = False

    level_db_range: tuple[float, float] = (-12.0, 6.0)
    """(A) x_ref 와 d 에 **같은 이득**. 순수 이득이라 SNR 보존 = 플랜트와 교환 가능."""

    polarity_flip: bool = True
    """(B) 극성 반전. 선형계에서 완전 유효, 위험 0."""

    mic_noise_snr_db: tuple[float, float] = (12.0, 40.0)
    """(C) 마이크 자기잡음. **입력에만** 넣는다 — d(타깃)에 넣으면 '잡음까지 상쇄하라'가
    되어 배울 수 없는 것을 요구한다."""

    eq_tilt_db: float = 6.0
    eq_band_db: float = 4.0
    eq_bands_hz: tuple[float, ...] = (100.0, 300.0, 700.0, 1400.0)
    """(D) 공통 스펙트럼 성형. x_ref 와 d 에 **같은** 최소위상 FIR. 4480초의 스펙트럼
    다양성 부족을 직접 보완하는, 비용 대비 효과가 가장 큰 항목."""

    mix_probability: float = 0.0
    mix_weight_range: tuple[float, float] = (0.0, 0.7)
    """(E) 세션 중첩. 같은 split·같은 덕트라 중첩이 성립한다. 64세션이면 C(64,2)=2016 쌍.
    ⚠ 플랜트 LTI 를 전제하므로 도입 전 '두 세션을 합친 (x,d) 의 coh² 가 단일 세션 대비
    떨어지지 않는가' 를 실측으로 확인해야 한다. 미검증이라 기본 확률 0 이다."""

    lead_jitter_samples: float = 0.0
    """재정렬 잔여 오차(창별 robust-std 0.8~4.3 샘플)를 흡수한다. 모델이 특정 위상을
    암기하지 않고 ±수샘플 불확실성에 강건해진다."""

    @model_validator(mode="after")
    def _validate(self) -> "RecordedAugmentConfig":
        lo, hi = self.level_db_range
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError(f"level_db_range 가 유효하지 않습니다: {self.level_db_range!r}")
        lo, hi = self.mic_noise_snr_db
        if not (0.0 < lo <= hi):
            raise ValueError(f"mic_noise_snr_db 가 유효하지 않습니다: {self.mic_noise_snr_db!r}")
        if not 0.0 <= self.mix_probability <= 1.0:
            raise ValueError(f"mix_probability 는 0..1 이어야 합니다: {self.mix_probability}")
        lo, hi = self.mix_weight_range
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"mix_weight_range 가 유효하지 않습니다: {self.mix_weight_range!r}")
        if self.eq_tilt_db < 0.0 or self.eq_band_db < 0.0:
            raise ValueError("EQ 진폭은 0 이상이어야 합니다")
        if self.lead_jitter_samples < 0.0:
            raise ValueError("lead_jitter_samples 는 0 이상이어야 합니다")
        return self

    @classmethod
    def from_data_config(cls, data_cfg: dict) -> "RecordedAugmentConfig":
        raw = (data_cfg or {}).get("recorded_augment")
        if not raw:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("recorded_augment 는 매핑이어야 합니다")
        payload = dict(raw)
        for key in ("level_db_range", "mic_noise_snr_db", "mix_weight_range"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(float(value) for value in payload[key])
        if "eq_bands_hz" in payload and payload["eq_bands_hz"] is not None:
            payload["eq_bands_hz"] = tuple(float(value) for value in payload["eq_bands_hz"])
        return cls(**payload)


def common_eq_kernel(
    rng: np.random.Generator, cfg: RecordedAugmentConfig, sample_rate: int
) -> np.ndarray | None:
    """x_ref 와 d 에 **똑같이** 걸 선형위상 FIR 을 뽑는다.

    ``H·(P·n) = P·(H·n)`` 이므로 LTI 플랜트와 교환 가능하다. 반드시 양쪽에 걸어야
    한다 — 한쪽만 걸면 플랜트를 바꾸는 것이고 그건 증강이 아니라 오염이다.
    """

    if cfg.eq_tilt_db <= 0.0 and cfg.eq_band_db <= 0.0:
        return None
    taps = 129
    freqs = np.fft.rfftfreq(taps, d=1.0 / float(sample_rate))
    gain_db = np.zeros_like(freqs)
    if cfg.eq_tilt_db > 0.0:
        tilt = float(rng.uniform(-cfg.eq_tilt_db, cfg.eq_tilt_db))
        gain_db += tilt * np.log10(np.maximum(freqs, 20.0) / 100.0)
    if cfg.eq_band_db > 0.0:
        for centre in cfg.eq_bands_hz:
            amount = float(rng.uniform(-cfg.eq_band_db, cfg.eq_band_db))
            width = max(1.0, float(centre) * 0.5)
            gain_db += amount * np.exp(-0.5 * ((freqs - float(centre)) / width) ** 2)
    magnitude = 10.0 ** (gain_db / 20.0)
    kernel = np.fft.irfft(magnitude, n=taps)
    kernel = np.roll(kernel, taps // 2) * np.hanning(taps)
    if not np.all(np.isfinite(kernel)) or float(np.sum(np.abs(kernel))) <= 0.0:
        return None
    return kernel.astype(np.float32)


def apply_same_fir(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """선형위상 FIR 을 군지연 보정과 함께 적용한다 (``mode="same"`` 이 그 역할)."""

    return np.convolve(signal, kernel, mode="same").astype(np.float32)


class RecordedANCDataset(IterableDataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_cfg: dict,
        split: str = "train",
        seed: int = 20260803,
        timing_contract: TrainingTimingContract | None = None,
        training_batch_size: int = 1,
        resume_batch_index: int = 0,
        transfer_repo_root: str | Path | None = None,
        broadband_valid_prefix_samples: int | None = None,
    ) -> None:
        super().__init__()
        transfer_keys = {
            "bootstrap_receipt",
            "bootstrap_receipt_sha256",
            "transfer_manifest",
            "transfer_manifest_sha256",
            "recorded_transfer_aggregate_sha256",
        }
        self._recorded_transfer: RecordedTrainingSnapshot | None = None
        if any(data_cfg.get(key) is not None for key in transfer_keys):
            if transfer_repo_root is None:
                from ..config import REPO_ROOT

                transfer_repo_root = REPO_ROOT
            self._recorded_transfer = validate_recorded_training_snapshot(
                data_cfg,
                repo_root=transfer_repo_root,
            )
            manifest_absolute = Path(manifest_path).expanduser()
            if not manifest_absolute.is_absolute():
                manifest_absolute = Path(transfer_repo_root) / manifest_absolute
            manifest_absolute = Path(str(manifest_absolute.absolute()))
            if manifest_absolute != self._recorded_transfer.recorded_manifest.path:
                raise TransferContractError(
                    "RecordedANCDataset manifest가 transfer recorded_manifest와 다릅니다: "
                    f"configured={manifest_absolute}, "
                    f"validated={self._recorded_transfer.recorded_manifest.path}"
                )
            manifest_bytes = self._recorded_transfer.recorded_manifest.data
            assert manifest_bytes is not None
            self.entries = read_manifest_bytes(
                manifest_bytes,
                manifest_path=manifest_absolute,
                split=split,
            )
        else:
            self.entries = read_manifest(manifest_path, split=split)
        if not self.entries:
            raise ValueError(f"'{split}' split 세션이 없습니다: {manifest_path}")
        self.fs = int(data_cfg["sample_rate"])
        raw_segment = int(round(float(data_cfg["segment_seconds"]) * self.fs))
        self.segment = max(256, (raw_segment // 256) * 256)
        self.reference_mode = str(data_cfg.get("reference_mode", "digital"))
        self._recorded_level_calibration: RecordedLevelCalibration | None = None
        calibration_path = data_cfg.get("recorded_level_calibration")
        calibration_sha = data_cfg.get("recorded_level_calibration_sha256")
        if calibration_path is not None or calibration_sha is not None:
            if self.reference_mode != "digital":
                raise ValueError(
                    "recorded level calibration은 digital-reference ERR/d에만 "
                    "적용할 수 있습니다 (acoustic-reference 금지)"
                )
            if not calibration_path or not calibration_sha:
                raise ValueError(
                    "recorded_level_calibration path와 외부 SHA는 함께 필요합니다"
                )
            if transfer_repo_root is None:
                from ..config import REPO_ROOT

                transfer_repo_root = REPO_ROOT
            self._recorded_level_calibration = (
                validate_recorded_level_calibration_receipt(
                    calibration_path,
                    expected_sha256=str(calibration_sha),
                    repo_root=transfer_repo_root,
                    # transfer snapshot이 mics/source_aligned를 실제 SHA로 이미
                    # 검증한다. 여기서 4 GiB를 두 번 읽지 않고 각 로드도 snapshot
                    # inode/content 집합 안에서만 허용한다.
                    verify_bound_audio=self._recorded_transfer is None,
                )
            )
        self.digital_reference_lead = int(
            data_cfg.get("digital_reference_lead_samples", 0)
        )
        if self.digital_reference_lead < 0:
            raise ValueError("digital_reference_lead_samples는 0 이상이어야 합니다")
        if self.reference_mode != "digital" and self.digital_reference_lead:
            raise ValueError(
                "digital_reference_lead_samples는 reference_mode=digital에서만 "
                "사용할 수 있습니다"
            )
        fb = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
        self.feedback_delay_range = (int(fb[0]), int(fb[1]))
        self.broadband_valid_prefix_samples = (
            0
            if broadband_valid_prefix_samples is None
            else int(broadband_valid_prefix_samples)
        )
        if self.broadband_valid_prefix_samples < 0:
            raise ValueError("broadband valid prefix는 0 이상이어야 합니다")
        self.seed = int(seed)
        self.training_batch_size = int(training_batch_size)
        self.resume_batch_index = int(resume_batch_index)
        if self.training_batch_size < 1 or self.resume_batch_index < 0:
            raise ValueError("training_batch_size는 1 이상, resume_batch_index는 0 이상")

        self.sampling_mode = str(data_cfg.get("recorded_sampling", "uniform_session"))
        if self.sampling_mode not in {
            "uniform_session",
            "family_group_session_balanced",
            "family_lineage_session_balanced",
            PLANT_DOMAIN_SAMPLING_MODE,
            QUALIFIED_SAMPLING_MODE,
        }:
            raise ValueError(
                f"지원하지 않는 recorded_sampling: {self.sampling_mode!r}"
            )
        self._sampling_hierarchy: dict[str, dict[str, tuple[int, ...]]] = {}
        self._plant_sampling_hierarchy: dict[
            str, dict[str, dict[str, tuple[int, ...]]]
        ] = {}
        self.current_strict_item_fraction = 0.0
        if self.sampling_mode in {
            "family_group_session_balanced",
            "family_lineage_session_balanced",
            PLANT_DOMAIN_SAMPLING_MODE,
            QUALIFIED_SAMPLING_MODE,
        }:
            hierarchy: dict[str, dict[str, list[int]]] = {}
            for index, entry in enumerate(self.entries):
                family = str(entry.get("source_family") or "").strip()
                group = str(entry.get("group_id") or "").strip()
                if not family or not group:
                    raise ValueError(
                        "recorded_sampling=family_group_session_balanced에는 모든 manifest "
                        f"entry의 source_family/group_id가 필요합니다: index={index}, "
                        f"session_id={entry.get('session_id')!r}"
                    )
                if self.sampling_mode in {
                    "family_lineage_session_balanced",
                    PLANT_DOMAIN_SAMPLING_MODE,
                    QUALIFIED_SAMPLING_MODE,
                }:
                    source_pool_group = str(
                        entry.get("source_pool_group_id") or ""
                    ).strip()
                    if not source_pool_group or source_pool_group == group:
                        raise ValueError(
                            "family_lineage_session_balanced에는 lineage regroup 증거인 "
                            "source_pool_group_id와 서로 다른 component group_id가 "
                            f"필요합니다: session_id={entry.get('session_id')!r}"
                        )
                hierarchy.setdefault(family, {}).setdefault(group, []).append(index)
            self._sampling_hierarchy = {
                family: {
                    group: tuple(indices)
                    for group, indices in sorted(groups.items())
                }
                for family, groups in sorted(hierarchy.items())
            }
        if self.sampling_mode == PLANT_DOMAIN_SAMPLING_MODE:
            if self._recorded_level_calibration is None:
                raise ValueError(
                    f"{PLANT_DOMAIN_SAMPLING_MODE}에는 검증된 recorded level "
                    "calibration receipt가 필요합니다"
                )
            plant_hierarchy: dict[str, dict[str, dict[str, list[int]]]] = {}
            for index, entry in enumerate(self.entries):
                family = str(entry.get("source_family") or "").strip()
                group = str(entry.get("group_id") or "").strip()
                domain = self._plant_domain(entry)
                plant_hierarchy.setdefault(family, {}).setdefault(domain, {}).setdefault(
                    group, []
                ).append(index)
            self._plant_sampling_hierarchy = {
                family: {
                    domain: {
                        group: tuple(indices)
                        for group, indices in sorted(groups.items())
                    }
                    for domain, groups in sorted(domains.items())
                }
                for family, domains in sorted(plant_hierarchy.items())
            }
            families = tuple(self._plant_sampling_hierarchy)
            incomplete_families = {
                family: sorted(self._plant_sampling_hierarchy[family])
                for family in families
                if set(self._plant_sampling_hierarchy[family])
                != {HISTORICAL_DOMAIN, CURRENT_DOMAIN}
            }
            if incomplete_families:
                raise ValueError(
                    "canonical family→plant_domain sampler에는 모든 family의 train "
                    "split에 historical_calibrated/current_strict가 모두 필요합니다: "
                    f"{incomplete_families}"
                )
            # 아래 2×family exact cycle에서 각 family가 두 domain을 한 번씩 쓴다.
            self.current_strict_item_fraction = 0.5
            required_fraction = float(
                data_cfg.get("recorded_current_strict_min_fraction", 0.5)
            )
            if not math.isclose(required_fraction, 0.5, abs_tol=0.0):
                raise ValueError(
                    "recorded_current_strict_min_fraction은 공식 계약 0.5로 고정됩니다"
                )
            if self.current_strict_item_fraction < required_fraction:
                raise ValueError(
                    "family→plant_domain sampler가 current_strict item 50%를 만들 수 "
                    f"없습니다: fraction={self.current_strict_item_fraction:.3f}"
                )

        # 재정렬본 강제 여부. 기본은 폴백 허용(기존 세션/픽스처 호환)이고,
        # 파인튜닝 설정에서 true 로 켠다.
        self.require_aligned_source = bool(data_cfg.get("require_aligned_source", False))
        self.lead_mode = str(data_cfg.get("recorded_lead_mode", "constant"))
        if self.lead_mode not in {"constant", "timeline"}:
            raise ValueError(f"지원하지 않는 recorded_lead_mode: {self.lead_mode!r}")
        # timeline 은 재정렬된 시간축 위에서 K' 를 유도한다. 재정렬본이 없으면 그 유도가
        # 원본 재생 배열 위에서 일어나 조용히 틀린다 — 그런데 기본값은 폴백 허용이었고
        # 어느 설정에도 require_aligned_source 가 없었다 (2026-08-07 발견). 모드가
        # 이미 그 요구를 함의하므로 설정에 맡기지 않는다.
        if self.lead_mode == "timeline" and self.reference_mode == "digital":
            self.require_aligned_source = True
        self.timing_contract = timing_contract
        if self.timing_contract is None and data_cfg.get("training_timing_contract"):
            self.timing_contract = TrainingTimingContract.from_data_config(data_cfg)
        if self.lead_mode == "timeline" and self.reference_mode == "digital":
            if self.timing_contract is None:
                raise ValueError(
                    "recorded_lead_mode=timeline에는 실제 P(z) FIR에서 유도한 "
                    "training_timing_contract가 필요합니다"
                )
            if (
                int(self.timing_contract.digital_reference_lead_samples)
                != self.digital_reference_lead
            ):
                raise ValueError(
                    "training timing의 lead와 data 설정이 다릅니다: "
                    f"{self.timing_contract.digital_reference_lead_samples} != "
                    f"{self.digital_reference_lead}"
                )
        self.total_advance_samples = (
            None
            if self.timing_contract is None
            else int(self.timing_contract.synthetic_total_advance_samples)
        )
        self.augment = RecordedAugmentConfig.from_data_config(data_cfg)
        if (
            self.sampling_mode == PLANT_DOMAIN_SAMPLING_MODE
            and self.augment.mix_probability > 0.0
        ):
            raise ValueError(
                "plant-domain sampler는 서로 다른 물리 domain의 session mix를 "
                "허용하지 않습니다"
            )
        self._broadband_batch_planner: BroadbandQualifiedBatchPlanner | None = None
        self._broadband_eq_suffix_samples = 0
        self.broadband_reference_dropout_probability = 0.0
        self.broadband_error_dropout_probability = 0.0
        if self.sampling_mode == QUALIFIED_SAMPLING_MODE:
            if (
                self.broadband_valid_prefix_samples <= 0
                or self.broadband_valid_prefix_samples % 256 != 0
                or self.broadband_valid_prefix_samples
                < max(self.feedback_delay_range)
            ):
                raise ValueError(
                    "BLOCKED_MISSING_PREFIX_OR_STATE: recorded broadband session은 "
                    "256-aligned prefix를 실제 연속 session에서 제공해야 하며 prefix가 "
                    "최대 feedback delay 이상이어야 합니다"
                )
            if (
                self.augment.enabled
                and (self.augment.eq_tilt_db > 0.0 or self.augment.eq_band_db > 0.0)
                and self.broadband_valid_prefix_samples
                < max(self.feedback_delay_range) + _COMMON_EQ_HALF_HISTORY_SAMPLES
            ):
                raise ValueError(
                    "recorded broadband prefix가 common EQ history와 feedback delay를 "
                    "모두 덮지 못합니다"
                )
            dropout = data_cfg.get("broadband_channel_dropout")
            if not isinstance(dropout, dict) or set(dropout) != {
                "reference_probability", "error_probability"
            }:
                raise ValueError(
                    "qualified recorded broadband은 reference/error dropout exact "
                    "mapping이 필요합니다"
                )
            self.broadband_reference_dropout_probability = float(
                dropout["reference_probability"]
            )
            self.broadband_error_dropout_probability = float(
                dropout["error_probability"]
            )
            if (
                self.broadband_reference_dropout_probability != 0.0
                or not 0.0 <= self.broadband_error_dropout_probability <= 1.0
            ):
                raise ValueError(
                    "qualified digital-reference recorded의 x_ref dropout은 exact 0, "
                    "error dropout은 [0,1]이어야 합니다"
                )
            receipt_key = (
                "recorded_broadband_batch_receipt"
                if split == "train"
                else "recorded_broadband_val_batch_receipt"
            )
            receipt_sha_key = f"{receipt_key}_sha256"
            receipt_value = data_cfg.get(receipt_key)
            if not receipt_value:
                raise ValueError(
                    f"subband-qualified {split} sampler에는 data.{receipt_key}가 "
                    "필요합니다"
                )
            receipt_path = Path(str(receipt_value)).expanduser()
            if not receipt_path.is_absolute() and transfer_repo_root is not None:
                receipt_path = Path(transfer_repo_root) / receipt_path
            receipt_bytes = receipt_path.read_bytes()
            expected_sha = str(
                data_cfg.get(receipt_sha_key) or ""
            ).lower()
            import hashlib

            actual_sha = hashlib.sha256(receipt_bytes).hexdigest()
            if expected_sha != actual_sha:
                raise ValueError(
                    "recorded broadband batch receipt 외부 SHA가 없거나 다릅니다: "
                    f"configured={expected_sha!r}, actual={actual_sha}"
                )
            try:
                receipt_payload = json.loads(receipt_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("recorded broadband batch receipt JSON이 잘못됐습니다") from exc
            self._broadband_batch_planner = BroadbandQualifiedBatchPlanner(
                receipt_payload,
                verify_raw=True,
                expected_split=split,
                expected_valid_prefix_samples=self.broadband_valid_prefix_samples,
            )
            if self._broadband_batch_planner.batch_size != self.training_batch_size:
                raise ValueError(
                    "recorded broadband receipt batch_size와 training batch_size가 다릅니다"
                )
            if self._broadband_batch_planner.segment_samples != self.segment:
                raise ValueError(
                    "recorded broadband receipt segment_samples와 dataset segment가 다릅니다"
                )
            receipt_sessions = self._broadband_batch_planner.session_ids
            manifest_sessions = {str(entry.get("session_id")) for entry in self.entries}
            if not receipt_sessions.issubset(manifest_sessions):
                raise ValueError("broadband batch receipt session이 현재 train manifest 밖입니다")
            # raw d에서 통과한 segment를 common EQ/mix로 다시 바꾸면 loss 입력에서
            # density>=0.25 보장이 사라진다. gain/polarity와 입력 전용 mic noise만 허용.
            if self.augment.mix_probability > 0.0:
                raise ValueError(
                    "subband-qualified sampler는 target density를 임의로 합치는 mix "
                    "증강을 허용하지 않습니다"
                )
            if self.augment.enabled and (
                self.augment.eq_tilt_db > 0.0 or self.augment.eq_band_db > 0.0
            ):
                self._broadband_eq_suffix_samples = (
                    _COMMON_EQ_HALF_HISTORY_SAMPLES
                )
            self._qualified_session_indices = {
                str(entry.get("session_id")): index
                for index, entry in enumerate(self.entries)
            }

        # 워커마다 전 세션을 메모리에 올리면 64세션 × 70s × 48000 × 4ch × 4B ≈ 3.4 GB 다
        # (source_aligned 추가로 더 늘어난다). Jetson 에서 OOM 이 실재하므로 LRU 로 바꾼다.
        cache_size = int(data_cfg.get("recorded_session_cache", 8))
        floor = 2 if self.augment.mix_probability > 0.0 else 1
        self.cache_size = max(floor, cache_size)
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self._cache_files: dict[int, tuple[Path, ...]] = {}
        self._lead_plans: dict[int, RecordedLeadPlan] = {}

    # ------------------------------------------------------------------ 세션 I/O
    def _session_metadata(self, entry: dict) -> dict:
        path = Path(entry["path"], "session.json")
        try:
            if self._recorded_transfer is not None:
                raw = self._recorded_transfer.read_verified_recorded_file(path)
                value = json.loads(raw.decode("utf-8"))
            else:
                if not path.is_file():
                    return {}
                value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _plant_domain(self, entry: dict) -> str:
        """manifest/session evidence에서 plant domain을 유도한다."""

        session_id = str(entry.get("session_id") or "")
        calibration = self._recorded_level_calibration
        if calibration is not None and session_id in calibration.plant_domain_by_session:
            return HISTORICAL_DOMAIN
        domain = str(entry.get("plant_domain") or "").strip()
        if not domain:
            metadata = self._session_metadata(entry)
            binding = metadata.get("recording_level_campaign")
            if isinstance(binding, dict):
                domain = str(binding.get("plant_domain") or "").strip()
            if not domain:
                binding = metadata.get("recording_level")
                if isinstance(binding, dict):
                    domain = str(binding.get("plant_domain") or "").strip()
        if domain != CURRENT_DOMAIN:
            raise ValueError(
                f"{session_id}: historical receipt 밖 세션에는 검증된 "
                f"plant_domain={CURRENT_DOMAIN!r} binding이 필요합니다"
            )
        return domain

    def _has_session_file(self, path: Path) -> bool:
        if self._recorded_transfer is not None:
            return self._recorded_transfer.has_recorded_file(path)
        return path.is_file()

    def _read_audio(self, path: Path) -> tuple[np.ndarray, int]:
        if self._recorded_transfer is None:
            return sf.read(path, dtype="float32", always_2d=True)
        raw = self._recorded_transfer.read_verified_recorded_file(path)
        return sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

    def lead_plan(self, index: int) -> RecordedLeadPlan:
        """세션 하나의 lead 부기. 실측 lead 의 **유일한** 유도 지점이다."""

        cached = self._lead_plans.get(index)
        if cached is not None:
            return cached
        jitter = float(self.augment.lead_jitter_samples) if self.augment.enabled else 0.0
        if self.lead_mode == "constant":
            plan = RecordedLeadPlan.constant(
                self.digital_reference_lead, jitter_sigma_samples=jitter
            )
        else:
            meta = self._session_metadata(self.entries[index])
            timeline = meta.get("timeline") if isinstance(meta, dict) else None
            if not isinstance(timeline, dict) or "aligned_lag_median_samples" not in timeline:
                raise ValueError(
                    f"{self.entries[index]['path']}: recorded_lead_mode=timeline 인데 "
                    "session.json 에 timeline.aligned_lag_median_samples 가 없습니다 — "
                    "lead 를 추측으로 채우지 않습니다"
                )
            if self.timing_contract is None or self.total_advance_samples is None:
                raise ValueError(
                    "recorded_lead_mode=timeline 에는 training_timing_contract가 "
                    "필요합니다 (합성 브랜치의 총 선행량과 맞춰야 합니다)"
                )
            observed_sigma = float(
                timeline.get("aligned_lag_robust_std_samples") or 0.0
            )
            if not np.isfinite(observed_sigma) or observed_sigma < 0.0:
                raise ValueError(
                    "timeline.aligned_lag_robust_std_samples는 유한한 0 이상이어야 "
                    "합니다"
                )
            recorded_delay = float(timeline["aligned_lag_median_samples"])
            plan = RecordedLeadPlan.from_timeline(
                total_advance_samples=self.total_advance_samples,
                recorded_delay_samples=recorded_delay,
                constant_lead_samples=self.digital_reference_lead,
                # 측정 robust std는 QA/evidence이지 증강 지시가 아니다. 공식 1차
                # 실행의 config jitter=0을 세션 불확도로 몰래 다시 켜지 않는다.
                jitter_sigma_samples=jitter,
            )
            expected_lead = self.timing_contract.recorded_lead_samples(recorded_delay)
            if int(plan.lead_samples) != expected_lead:  # pragma: no cover - 방어
                raise ValueError(
                    f"RecordedLeadPlan이 training timing과 다릅니다: "
                    f"{plan.lead_samples} != {expected_lead}"
                )
        self._lead_plans[index] = plan
        return plan

    def _load_session(self, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        session_dir = Path(entry["path"])
        metadata_path = session_dir / "session.json"
        mics_path = session_dir / "mics.wav"
        mics, sr = self._read_audio(mics_path)
        if sr != self.fs:
            raise ValueError(f"{session_dir}: 샘플레이트 {sr} != {self.fs}")
        if mics.shape[1] < 2:
            raise ValueError(f"{session_dir}: mics.wav는 err/ref 2채널이어야 합니다")
        if not np.all(np.isfinite(mics)):
            raise ValueError(f"{session_dir}: mics.wav에 NaN/Inf가 있습니다")
        err = mics[:, 0]
        ref = mics[:, 1]

        # 재정렬본 우선. source.wav 는 재생 배열일 뿐 방출 시각이 아니다.
        aligned_path = session_dir / "source_aligned.wav"
        has_aligned = self._has_session_file(aligned_path)
        if (
            self.require_aligned_source
            and self.reference_mode == "digital"
            and not has_aligned
        ):
            raise FileNotFoundError(
                f"{session_dir}: source_aligned.wav 가 필요합니다 — source.wav 는 재생 "
                "배열이지 ADC 시간축이 아닙니다 "
                "(scripts/data/realign_recorded_sessions.py 를 먼저 도세요)"
            )
        source_path = aligned_path if has_aligned else session_dir / "source.wav"
        has_source = self._has_session_file(source_path)
        if has_source:
            source, source_sr = self._read_audio(source_path)
            if source_sr != self.fs:
                raise ValueError(
                    f"{session_dir}: {source_path.name} 샘플레이트 {source_sr} != {self.fs}"
                )
            source = source[:, 0]
            if not np.all(np.isfinite(source)):
                raise ValueError(f"{session_dir}: {source_path.name}에 NaN/Inf가 있습니다")
        elif self.reference_mode == "digital":
            raise FileNotFoundError(
                f"{session_dir}: digital-reference 학습에 source.wav가 필요합니다"
            )
        else:
            source = np.zeros_like(err)
        n = min(err.size, ref.size, source.size)
        if self._recorded_level_calibration is not None:
            session_id = str(entry.get("session_id") or "")
            gain = self._recorded_level_calibration.err_gain_by_session.get(session_id)
            if gain is None:
                if self._plant_domain(entry) != CURRENT_DOMAIN:
                    raise ValueError(f"{session_id}: ERR level gain/domain이 없습니다")
                gain = 1.0
            # 원본 WAV와 x_ref/REF는 불변이다. strict P와 단위가 다른 historical
            # ERR(=d)만 train-only receipt의 scalar로 맞춘다.
            err = (err * float(gain)).astype(np.float32)
        if self._recorded_transfer is not None:
            # session.json은 lead/lineage 의미를 결정하므로 audio cache와 같은
            # generation에 계속 고정한다.
            if not self._recorded_transfer.has_recorded_file(metadata_path):
                raise TransferContractError(
                    f"session.json이 transfer exact 집합에 없습니다: {metadata_path}"
                )
            self._recorded_transfer.assert_recorded_file_unchanged(metadata_path)
        return err[:n], ref[:n], source[:n]

    def _transferred_session_paths(self, entry: dict) -> tuple[Path, ...]:
        if self._recorded_transfer is None:
            return ()
        session_dir = Path(entry["path"])
        paths = [session_dir / "mics.wav", session_dir / "session.json"]
        aligned = session_dir / "source_aligned.wav"
        source = (
            aligned
            if self._recorded_transfer.has_recorded_file(aligned)
            else session_dir / "source.wav"
        )
        if self._recorded_transfer.has_recorded_file(source):
            paths.append(source)
        return tuple(paths)

    def _session(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cached = self._cache.get(index)
        if cached is not None:
            if self._recorded_transfer is not None:
                for path in self._cache_files[index]:
                    self._recorded_transfer.assert_recorded_file_unchanged(path)
            self._cache.move_to_end(index)
            return cached
        loaded = self._load_session(self.entries[index])
        used_paths = self._transferred_session_paths(self.entries[index])
        self._cache[index] = loaded
        self._cache_files[index] = used_paths
        while len(self._cache) > self.cache_size:
            evicted_index, _ = self._cache.popitem(last=False)
            self._cache_files.pop(evicted_index, None)
        return loaded

    # ------------------------------------------------------------------ 표본 추출
    def _worker_rng(self, worker_id: int) -> np.random.Generator:
        """worker마다 독립이면서 재시작 시 재현되는 sampler RNG."""

        return np.random.default_rng(self.seed + int(worker_id) * 1013)

    def _sample_session_index(
        self, rng: np.random.Generator, *, global_index: int | None = None
    ) -> int:
        if self.sampling_mode == "uniform_session":
            return int(rng.integers(len(self.entries)))
        if self.sampling_mode == PLANT_DOMAIN_SAMPLING_MODE:
            if global_index is None:
                raise ValueError("plant-domain sampler에는 global_index가 필요합니다")
            # 2×family cycle: 각 family가 historical/current를 정확히 한 번씩 쓴다.
            # worker/resume에서도 global item index가 같으면 동일해 long-run current
            # fraction은 추정값이 아니라 exact 0.5다.
            families = tuple(self._plant_sampling_hierarchy)
            family = families[(int(global_index) // 2) % len(families)]
            domains = self._plant_sampling_hierarchy[family]
            domain = CURRENT_DOMAIN if int(global_index) % 2 else HISTORICAL_DOMAIN
            groups = tuple(domains[domain])
            group = groups[int(rng.integers(len(groups)))]
            sessions = domains[domain][group]
            return int(sessions[int(rng.integers(len(sessions)))])
        families = tuple(self._sampling_hierarchy)
        family = families[int(rng.integers(len(families)))]
        groups = tuple(self._sampling_hierarchy[family])
        group = groups[int(rng.integers(len(groups)))]
        sessions = self._sampling_hierarchy[family][group]
        return int(sessions[int(rng.integers(len(sessions)))])

    def _draw_pair(
        self,
        index: int,
        rng: np.random.Generator,
        *,
        exact_start: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """세션 하나에서 (x_ref, d) 한 쌍을 뽑는다. 길이가 모자라면 None."""

        err, ref, source = self._session(index)
        if self.reference_mode == "digital":
            plan = self.lead_plan(index)
            lead = int(plan.lead_samples)
            if self.augment.enabled and plan.jitter_sigma_samples > 0.0:
                sigma = float(plan.jitter_sigma_samples)
                lead = max(
                    0,
                    lead + int(np.clip(rng.normal(0.0, sigma), -3.0 * sigma, 3.0 * sigma)),
                )
        else:
            lead = 0
        prefix = int(self.broadband_valid_prefix_samples)
        suffix = int(self._broadband_eq_suffix_samples)
        model_samples = prefix + int(self.segment)
        read_samples = model_samples + suffix
        if err.size < read_samples + lead:
            return None
        if exact_start is None:
            stop_exclusive = err.size - self.segment - suffix - lead + 1
            if stop_exclusive <= prefix:
                return None
            start = int(rng.integers(prefix, stop_exclusive))
        else:
            start = int(exact_start)
            if start < prefix or start + self.segment + suffix + lead > err.size:
                raise ValueError(
                    "qualified segment에 실제 session prefix가 없거나 lead 범위 "
                    f"밖입니다: index={index}, start={start}, prefix={prefix}, lead={lead}"
                )
        begin = start - prefix
        end = start + self.segment + suffix
        d = err[begin:end].copy()
        if self.reference_mode == "digital":
            x_ref = source[begin + lead : end + lead].copy()
        else:
            x_ref = ref[begin:end].copy()
        if d.size != read_samples or x_ref.size != read_samples:
            raise RuntimeError("recorded continuous prefix exact crop 길이가 다릅니다")
        return x_ref, d

    def _remove_broadband_eq_suffix(
        self, values: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """common-EQ right history를 사용한 뒤 model 입력 길이로 exact crop."""

        suffix = int(self._broadband_eq_suffix_samples)
        if suffix <= 0:
            return values
        cropped = tuple(np.asarray(value[:-suffix], dtype=np.float32) for value in values)
        expected = self.broadband_valid_prefix_samples + self.segment
        if any(value.size != expected for value in cropped):
            raise RuntimeError("recorded common-EQ suffix crop 길이가 다릅니다")
        return cropped

    def _augment(
        self,
        x_ref: np.ndarray,
        d: np.ndarray,
        rng: np.random.Generator,
        *,
        apply_eq: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """적용 순서: (D) 공통 EQ → (A) 레벨 → (B) 극성 → (C) 입력 잡음.

        (C) 가 마지막인 이유: 잡음에 EQ/레벨을 걸면 그것은 마이크 자기잡음이 아니라
        신호의 일부가 된다. 그리고 d(타깃)에는 절대 넣지 않는다.
        """

        cfg = self.augment
        peak_before = float(max(np.max(np.abs(x_ref)), np.max(np.abs(d)), 1e-12))

        kernel = common_eq_kernel(rng, cfg, self.fs) if apply_eq else None
        if kernel is not None:
            x_ref = apply_same_fir(x_ref, kernel)
            d = apply_same_fir(d, kernel)

        gain = float(10.0 ** (rng.uniform(*cfg.level_db_range) / 20.0))
        x_ref = (x_ref * gain).astype(np.float32)
        d = (d * gain).astype(np.float32)

        if cfg.polarity_flip and rng.random() < 0.5:
            x_ref = -x_ref
            d = -d

        # 중첩/EQ 로 peak 가 녹음 레벨을 크게 넘지 않게 클램프. 넘기면 실제로 측정한 적
        # 없는 비선형 영역의 통계를 학습시키게 된다.
        peak_after = float(max(np.max(np.abs(x_ref)), np.max(np.abs(d)), 1e-12))
        limit = peak_before * 2.0
        if peak_after > limit:
            scale = limit / peak_after
            x_ref = (x_ref * scale).astype(np.float32)
            d = (d * scale).astype(np.float32)

        snr_db = float(rng.uniform(*cfg.mic_noise_snr_db))
        noisy_x = x_ref.copy()
        noisy_err = d.copy()
        for signal in (noisy_x, noisy_err):
            power = float(np.mean(np.square(signal)))
            if power <= 0.0:
                continue
            sigma = math.sqrt(power / (10.0 ** (snr_db / 10.0)))
            signal += rng.normal(0.0, sigma, size=signal.shape).astype(np.float32)
        return noisy_x, noisy_err, d

    # ------------------------------------------------------------------ 반복
    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        count = len(self.entries)
        indices = worker_global_item_indices(
            start_batch_index=self.resume_batch_index,
            batch_size=self.training_batch_size,
            worker_id=worker_id,
            num_workers=num_workers,
        )

        for global_index in indices:
            if self._broadband_batch_planner is not None:
                planned = self._broadband_batch_planner.item(
                    global_index, seed=self.seed
                )
                session_index = self._qualified_session_indices[planned.session_id]
                rng = indexed_rng(self.seed, 0x524543, global_index)
                pair = self._draw_pair(
                    session_index, rng, exact_start=planned.start_frame
                )
                if pair is None:  # pragma: no cover - exact_start 검증이 먼저 예외로 닫는다
                    raise RuntimeError("qualified segment를 읽을 수 없습니다")
            else:
                pair = None
                attempt = 0
                while pair is None:
                    rng = indexed_rng(self.seed, 0x524543, global_index, attempt)
                    pair = self._draw_pair(
                        self._sample_session_index(rng, global_index=global_index), rng
                    )
                    attempt += 1
                    if attempt > max(32, len(self.entries) * 2):
                        raise RuntimeError(
                            f"recorded global item {global_index}에 유효한 segment가 없습니다"
                        )
            x_ref, d = pair

            if (
                self.augment.enabled
                and self.augment.mix_probability > 0.0
                and count > 1
                and rng.random() < self.augment.mix_probability
            ):
                other = self._draw_pair(
                    self._sample_session_index(rng, global_index=global_index), rng
                )
                if other is not None:
                    a = float(rng.uniform(0.3, 1.0))
                    b = float(rng.uniform(*self.augment.mix_weight_range))
                    x_ref = (a * x_ref + b * other[0]).astype(np.float32)
                    d = (a * d + b * other[1]).astype(np.float32)

            if self.augment.enabled and self._broadband_batch_planner is not None:
                native_x = x_ref
                native_d = d
                accepted = False
                # 예약 item은 자신에게 배정된 대역만 density>=0.25를 보존한다.
                # native receipt가 먼저 PASS하므로 EQ가 부족 corpus를 승격할 수 없다.
                required_bands = tuple(planned.required_post_augment_bands)
                attempts = 32 if required_bands else 1
                for augment_attempt in range(attempts):
                    candidate_rng = indexed_rng(
                        self.seed, 0x524541, global_index, augment_attempt
                    )
                    candidate = self._augment(
                        native_x.copy(), native_d.copy(), candidate_rng
                    )
                    candidate = self._remove_broadband_eq_suffix(candidate)
                    ratios = target_d_density_ratios(
                        candidate[2][-self.segment :],
                        sample_rate=self.fs,
                        bands_hz=self._broadband_batch_planner.bands,
                    )
                    if not required_bands or all(
                        ratios[index] >= MIN_TARGET_D_DENSITY_RATIO
                        for index in required_bands
                    ):
                        x_ref, err_source, d = candidate
                        rng = candidate_rng
                        accepted = True
                        break
                if not accepted:
                    # bounded rejection이 모두 실패해도 batch invariant를 깨지 않는다.
                    # EQ만 identity로 두고 gain/polarity/input-only mic noise는 같은
                    # global-index RNG에서 계속 적용한다.
                    rng = indexed_rng(self.seed, 0x524546, global_index)
                    x_ref, err_source, d = self._augment(
                        native_x.copy(), native_d.copy(), rng, apply_eq=False
                    )
                    x_ref, err_source, d = self._remove_broadband_eq_suffix(
                        (x_ref, err_source, d)
                    )
                    fallback_ratios = target_d_density_ratios(
                        d[-self.segment :],
                        sample_rate=self.fs,
                        bands_hz=self._broadband_batch_planner.bands,
                    )
                    if any(
                        fallback_ratios[index] < MIN_TARGET_D_DENSITY_RATIO
                        for index in required_bands
                    ):
                        raise RuntimeError(
                            "identity-EQ fallback가 assigned recorded subband 자격을 "
                            "보존하지 못했습니다"
                        )
            elif self.augment.enabled:
                x_ref, err_source, d = self._augment(x_ref, d, rng)
            else:
                err_source = d

            fb_delay = int(rng.integers(*self.feedback_delay_range))
            err_in = _delay_np(err_source, fb_delay)
            if self._broadband_batch_planner is not None:
                dropout_rng = indexed_rng(
                    self.seed, 0x524544, int(global_index)
                )
                if (
                    dropout_rng.random()
                    < self.broadband_error_dropout_probability
                ):
                    err_in = np.zeros_like(err_in)
            x = np.stack([x_ref, err_in]).astype(np.float32)
            yield {
                "x": torch.from_numpy(x),
                "d": torch.from_numpy(d.astype(np.float32)).unsqueeze(0),
                **(
                    {
                        "valid_start_sample": torch.tensor(
                            self.broadband_valid_prefix_samples,
                            dtype=torch.int64,
                        )
                    }
                    if self.broadband_valid_prefix_samples > 0
                    else {}
                ),
            }

    # ------------------------------------------------------------------ 진단
    def describe(self) -> dict[str, Any]:
        """설정 요약. 학습 로그/리포트에 박아 두면 lead 부기가 추적된다."""

        return {
            "sessions": len(self.entries),
            "segment_samples": self.segment,
            "broadband_valid_prefix_samples": self.broadband_valid_prefix_samples,
            "reference_mode": self.reference_mode,
            "require_aligned_source": self.require_aligned_source,
            "lead_mode": self.lead_mode,
            "sampling_mode": self.sampling_mode,
            "current_strict_item_fraction": self.current_strict_item_fraction,
            "recorded_level_calibration_sha256": (
                None
                if self._recorded_level_calibration is None
                else self._recorded_level_calibration.sha256
            ),
            "broadband_batch_qualified": self._broadband_batch_planner is not None,
            "constant_lead_samples": self.digital_reference_lead,
            "total_advance_samples": self.total_advance_samples,
            "training_timing_contract": (
                None
                if self.timing_contract is None
                else self.timing_contract.model_dump()
            ),
            "session_cache": self.cache_size,
            "augment": self.augment.model_dump(),
            "transfer_manifest_sha256": (
                None
                if self._recorded_transfer is None
                else self._recorded_transfer.transfer_manifest.sha256
            ),
            "recorded_transfer_aggregate_sha256": (
                None
                if self._recorded_transfer is None
                else self._recorded_transfer.recorded_aggregate_sha256
            ),
        }


def make_recorded_eval_batch(
    dataset: RecordedANCDataset, n_items: int
) -> dict[str, torch.Tensor]:
    """global index 0에서 시작하는 고정 recorded validation batch."""

    count = int(n_items)
    if count < 1 or count != dataset.training_batch_size:
        raise ValueError(
            "recorded eval batch 크기는 dataset training_batch_size와 같은 양수여야 합니다"
        )
    iterator = iter(dataset)
    items = [next(iterator) for _ in range(count)]
    batch = {
        "x": torch.stack([item["x"] for item in items]),
        "d": torch.stack([item["d"] for item in items]),
    }
    prefix_values = [item.get("valid_start_sample") for item in items]
    if any(value is not None for value in prefix_values):
        if not all(isinstance(value, torch.Tensor) for value in prefix_values):
            raise ValueError("recorded eval batch valid prefix metadata가 일부만 있습니다")
        batch["valid_start_sample"] = torch.stack(prefix_values)  # type: ignore[arg-type]
    return batch
