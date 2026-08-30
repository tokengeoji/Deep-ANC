#!/usr/bin/env python3
"""자동 실기 평가 — 시나리오별 OFF(베이스라인)→ON→OFF 프로토콜, md 리포트 생성.

  .venv/bin/python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 band
⚠ 스피커에서 소음이 재생된다. TPA3116D2 볼륨을 낮춘 상태에서, 사용자 입회 하에 실행.
"""

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.audio_io import capture_input_probe                         # noqa: E402
from deep_anc.config import REPO_ROOT, load_runtime_config, load_yaml     # noqa: E402
from deep_anc.dsp.do_no_harm import octave_band_edges_hz                  # noqa: E402
from deep_anc.dsp.secondary_path import load_secondary_path                # noqa: E402
from deep_anc.dsp.timing import BandPlan                                   # noqa: E402
from deep_anc.eval.artifacts import (                                      # noqa: E402
    atomic_write_text,
    run_directory,
    write_csv,
    write_wav_pair,
)
from deep_anc.eval.metrics import (                                        # noqa: E402
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)
from deep_anc.realtime.run_realtime import RealtimeANC                  # noqa: E402


# ``source`` 는 ANC ON/OFF와 무관하게 실제 noise 출력 ch0으로 보낸 재생 파형이다.
# 단순 ERR 대역 전력은 덕트 배경잡음도 포함하므로, "그 octave를 실제로 자극했는가"의
# 판정에는 이 OFF 재생 source를 쓴다. octave 폭으로 정규화한 25% PSD 비율은 48 kHz
# white/broadband source의 125 Hz octave는 보존하면서, 80–1000 Hz source의 2 kHz
# filter tail처럼 총 전력 비율만 보면 남는 성분은 배제한다.
OCTAVE_SOURCE_ENERGY_GATE_SCHEMA = "off_playback_source_density_v2"
DEFAULT_MIN_SOURCE_ENERGY_DENSITY_RATIO_IN_OCTAVE_BAND = 0.25
_SOURCE_ENERGY_EPS = 1e-30
LIVE_PROTOCOL_SCHEMA = "anc_off_on_with_continuous_noise_source_v1"


def run_scenario(cfg: dict, protocol: dict) -> dict:
    base_s = float(protocol.get("baseline_seconds", 10))
    on_s = float(protocol.get("on_seconds", 30))
    tail_s = float(protocol.get("tail_seconds", 5))
    total = base_s + on_s + tail_s

    anc = RealtimeANC(cfg, record_seconds=total + 2.0)
    try:
        anc.start()
        # OFF/ON은 **ANC control의 상태만** 뜻한다. noise source까지 OFF이면
        # baseline은 음향 입력 없는 방 floor가 되어 ANC 성능을 측정할 수 없다.
        # 이 명시 설정은 RuntimeState의 안전 기본값(False)을 live 평가 프로토콜에서만
        # 의도적으로 전환하는 지점이다.
        anc.state.noise_enabled = True
        time.sleep(base_s)
        anc.state.anc_enabled = True
        time.sleep(on_s)
        anc.state.anc_enabled = False
        time.sleep(tail_s)
        stats = dict(anc.state.latest_stats)
    finally:
        anc.stop()

    if anc.state.fatal_error is not None:
        raise RuntimeError("실기 평가 오디오 콜백이 실패했습니다") from anc.state.fatal_error

    data = anc.session_data()
    fs = anc.fs
    err = data["err"]
    source = data.get("source")
    if source is None:
        raise RuntimeError(
            "실기 평가 raw에 재생 source 트랙이 없습니다. 옥타브별 source-energy "
            "유효성을 검증할 수 없으므로 감쇠를 발행하지 않습니다."
        )
    source = np.asarray(source)
    if source.shape != err.shape:
        raise RuntimeError(
            "실기 평가 raw의 source/ERR 길이가 다릅니다. 옥타브별 source-energy "
            "유효성을 검증할 수 없으므로 감쇠를 발행하지 않습니다."
        )
    # 게이트 램프를 피해서 구간 절단 (경계 1초 여유)
    off_seg = err[int(1.0 * fs) : int((base_s - 1.0) * fs)]
    off_source_seg = source[int(1.0 * fs) : int((base_s - 1.0) * fs)]
    on_seg = err[int((base_s + 2.0) * fs) : int((base_s + on_s - 1.0) * fs)]
    # 후행 OFF 를 반드시 함께 쓴다. 앞뒤 베이스라인이 최대 38초 떨어져 있어 소스·온도·
    # 환경 드리프트가 감쇠로 오인된다 — 실측에서 multitone 이 선행 기준 +6.05 dB,
    # 후행 기준 +3.67 dB 로 2.39 dB 갈렸다. 둘 중 **나쁜 쪽**을 headline 으로 쓴다.
    tail_seg = err[int((base_s + on_s + 1.0) * fs) : int((base_s + on_s + tail_s - 0.5) * fs)]
    on_gain = data["anc_gain"][
        int((base_s + 2.0) * fs) : int((base_s + on_s - 1.0) * fs)
    ]
    on_duty = float(np.mean(on_gain >= 0.999)) if on_gain.size else 0.0
    if on_duty < 0.95:
        raise RuntimeError(
            "ANC ON 유효 구간이 95% 미만입니다 "
            f"({100.0 * on_duty:.1f}%). 자동 mute/언더런 결과는 성능 측정으로 인정하지 않습니다."
        )
    return {
        "fs": fs,
        "off": off_seg,
        "off_source": off_source_seg,
        "on": on_seg,
        "tail": tail_seg,
        "on_duty": on_duty,
        "stats": stats,
        "protocol_state": {
            "noise_enabled_requested": True,
            "anc_enabled_baseline_requested": False,
            "anc_enabled_on_requested": True,
            "anc_enabled_tail_requested": False,
        },
        "data": data,
    }


def band_power(signal: np.ndarray, sample_rate: int, band: tuple[float, float]) -> float:
    from scipy import signal as sps

    values = np.asarray(signal, dtype=np.float64)
    nperseg = min(8192, max(256, values.size // 8))
    freqs, psd = sps.welch(values, sample_rate, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask]))


def validate_octave_source_energy_density_ratio(value: float) -> float:
    """출력 장치를 열기 전에 octave source-energy 정책값을 fail-fast 검증한다."""

    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError(
            "min_source_energy_density_ratio_in_octave_band은 0보다 크고 1 이하인 "
            "유한값이어야 합니다"
        )
    return threshold


def qualify_octave_attenuation_by_off_source(
    off: np.ndarray,
    on: np.ndarray,
    off_source: np.ndarray,
    sample_rate: int,
    centers_hz: list[float],
    trusted_band_hz: tuple[float, float],
    *,
    min_source_energy_density_ratio: float,
) -> list[dict]:
    """OFF 재생 source가 실제로 있는 octave만 감쇠값을 발행한다.

    실기 ``ERR``의 같은 대역 전력만 보면 소스가 없는 octave에서도 마이크 floor나
    다른 환경음의 OFF/ON 차이를 감쇠처럼 읽을 수 있다. source 출력 트랙의
    ``20 Hz–Nyquist`` 전체 평균 PSD 대비 octave PSD 비율이 충분한 경우에만 공개
    ``attenuation_db``를 채운다. raw에는 전력 비율과 폭 정규화 PSD 비율을 모두
    보존한다. 원 계산값은 ``unqualified_attenuation_db``에 보존해 사후 진단은
    가능하지만, 성능 수치로 해석해서는 안 된다.

    이것은 재생 source 존재 gate이지 실제 ERR의 SNR/coherence 판정은 아니다.
    따라서 유효 octave도 P/S 신뢰도, xrun/deadline, OFF/ON drift와 함께 해석한다.
    """

    threshold = validate_octave_source_energy_density_ratio(
        min_source_energy_density_ratio
    )
    source = np.asarray(off_source, dtype=np.float64).reshape(-1)
    if source.size < 2 or not np.all(np.isfinite(source)):
        raise ValueError("OFF source-energy gate에는 유한한 2샘플 이상 source가 필요합니다")

    unqualified = octave_band_attenuation(
        off, on, sample_rate, centers_hz, trusted_band_hz
    )
    # Nyquist 밖 center는 octave_band_attenuation이 의도적으로 생략한다. source gate도
    # 정확히 같은 center만 다뤄 raw mask/CSV/표의 길이가 어긋나지 않게 한다.
    retained_centers = [float(row["center_hz"]) for row in unqualified]
    source_band_low_hz = 20.0
    source_band_high_hz = sample_rate / 2.0
    source_band_width_hz = source_band_high_hz - source_band_low_hz
    total_power = band_power(
        source, sample_rate, (source_band_low_hz, source_band_high_hz)
    )
    if not np.isfinite(total_power) or total_power < 0.0:
        raise ValueError("OFF source의 전체 대역 전력이 유한한 0 이상이어야 합니다")

    rows: list[dict] = []
    for raw_row, center in zip(unqualified, retained_centers, strict=True):
        lo, hi = octave_band_edges_hz(center)
        source_power = band_power(source, sample_rate, (lo, hi))
        if not np.isfinite(source_power) or source_power < 0.0:
            raise ValueError(f"OFF source의 {center:g}Hz octave 전력이 유효하지 않습니다")
        fraction = (
            float(source_power / total_power)
            if total_power > _SOURCE_ENERGY_EPS
            else 0.0
        )
        flat_fraction = float((hi - lo) / source_band_width_hz)
        density_ratio = fraction / flat_fraction
        valid = bool(total_power > _SOURCE_ENERGY_EPS and density_ratio >= threshold)
        raw_att = float(raw_row["attenuation_db"])
        rows.append(
            {
                **raw_row,
                # published 값은 invalid octave에서 NaN이다. 기존 이름의 수치를
                # 무심코 집계해 고역 증폭이라고 주장하는 fail-open을 막는다.
                "attenuation_db": raw_att if valid else float("nan"),
                "unqualified_attenuation_db": raw_att,
                "off_source_band_power": float(source_power),
                "source_energy_fraction": fraction,
                "source_energy_flat_fraction": flat_fraction,
                "source_energy_density_ratio": density_ratio,
                "source_energy_valid": valid,
            }
        )
    return rows


def measurement_band(scenario: dict, trusted: tuple[float, float]) -> tuple[float, float]:
    """시나리오가 실제로 에너지를 갖는 대역. 없으면 trusted 로 폴백한다."""

    band = scenario.get("measure_band_hz")
    if not band:
        return trusted
    return (float(band[0]), float(band[1]))


def input_preflight(cfg: dict, seconds: float = 2.0) -> bool:
    """출력 장치를 열기 전에 ERR/REF I2S 입력의 생존 여부를 확인한다."""
    report = capture_input_probe(cfg["hardware"]["audio"], seconds=seconds)
    names = ("ERR", "REF")
    for item in report["channels"][:2]:
        index = int(item["channel"])
        verdict = "PASS" if item["valid"] else "FAIL"
        print(
            f"[{verdict}] {names[index]} ch{index}: RMS {item['rms_dbfs']:.2f}dBFS, "
            f"peak {item['peak']:.6f}, clip {item['clip_ratio']:.3%}, "
            f"unique {item['unique_codes']}, raw [{item['raw_min']}, {item['raw_max']}]"
        )

    required_channels = (0, 1) if cfg.get("reference") == "mic" else (0,)
    failed = [index for index in required_channels if not report["channels"][index]["valid"]]
    if failed:
        labels = ", ".join(f"{names[index]} ch{index}" for index in failed)
        print(
            f"[중단] 필수 입력({labels})이 무효입니다. 스피커 출력과 실기 평가를 시작하지 않습니다.",
            file=sys.stderr,
        )
        return False
    if not report["channels"][1]["valid"]:
        print(
            "[경고] REF ch1이 무효입니다. digital-reference 평가는 가능하지만 "
            "acoustic-reference 수집/평가는 금지합니다.",
            file=sys.stderr,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--controllers", nargs="+", default=["fxlms", "dl"])
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--input-probe-seconds",
        type=float,
        default=2.0,
        help="스피커 출력 전에 수행할 무출력 마이크 사전점검 길이",
    )
    args = parser.parse_args()

    eval_cfg = load_yaml(REPO_ROOT / args.eval_config)
    scenarios = {s["name"]: s for s in eval_cfg["scenarios"]}
    chosen = []
    for name in args.scenarios or list(scenarios.keys()):
        if name in scenarios:
            chosen.append(name)
        else:
            print(f"[skip] 알 수 없는 시나리오 '{name}' — eval.yaml scenarios: {list(scenarios)}")
    protocol = eval_cfg.get("protocol", {})
    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])
    # 잘못된 정책값 때문에 입력 probe를 통과한 뒤 스피커를 여는 일을 막는다.
    octave_source_energy_density_threshold = validate_octave_source_energy_density_ratio(
        eval_cfg.get(
            "min_source_energy_density_ratio_in_octave_band",
            DEFAULT_MIN_SOURCE_ENERGY_DENSITY_RATIO_IN_OCTAVE_BAND,
        )
    )
    initial_cfg = load_runtime_config(args.config)
    try:
        if not input_preflight(initial_cfg, seconds=args.input_probe_seconds):
            return 2
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[중단] 입력 사전점검 실패: {exc}", file=sys.stderr)
        return 2
    fs_config = int(initial_cfg["hardware"]["audio"]["sample_rate"])
    sp = load_secondary_path(REPO_ROOT / initial_cfg["duct"]["secondary_path"]["npz"])
    if sp.sample_rate != fs_config:
        raise ValueError(
            f"S(z) sample_rate={sp.sample_rate}Hz != runtime sample_rate={fs_config}Hz"
        )
    trusted = BandPlan.resolve(
        plant_trusted_band_hz=sp.trusted_band_hz(),
        duct_cfg=initial_cfg["duct"],
        sample_rate=fs_config,
    ).optimize.as_tuple()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        Path(args.out) if args.out
        else run_directory(REPO_ROOT / eval_cfg.get("report_dir", "results"), "session", stamp)
    )
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "wav").mkdir(parents=True, exist_ok=True)
    print(f"산출물 디렉터리: {run_dir}")

    band_centers = [int(b) for b in bands]
    rows: list[dict] = []

    for name in chosen:
        for controller in args.controllers:
            cfg = load_runtime_config(args.config)
            cfg["controller"] = controller
            cfg["noise"] = dict(scenarios[name]["noise"])
            print(f"\n=== {name} × {controller} ===")
            result = run_scenario(cfg, protocol)
            if result["fs"] != sp.sample_rate:
                raise ValueError(
                    f"세션 sample_rate={result['fs']}Hz != S(z) sample_rate={sp.sample_rate}Hz"
                )
            fs = int(result["fs"])
            nmse_fullband = nmse_db(result["off"], result["on"])
            nmse_trusted = band_nmse_db(result["off"], result["on"], fs, trusted)
            band_att = qualify_octave_attenuation_by_off_source(
                result["off"],
                result["on"],
                result["off_source"],
                fs,
                bands,
                trusted,
                min_source_energy_density_ratio=octave_source_energy_density_threshold,
            )

            # 소스가 실제로 있는 대역에서의 감쇠. 앞/뒤 베이스라인 중 **나쁜 쪽**을 쓴다.
            measure = measurement_band(scenarios[name], trusted)
            power_on = band_power(result["on"], fs, measure)
            att_vs_initial = 10.0 * np.log10(
                band_power(result["off"], fs, measure) / max(power_on, 1e-30)
            )
            tail = result.get("tail")
            att_vs_tail = (
                10.0 * np.log10(band_power(tail, fs, measure) / max(power_on, 1e-30))
                if tail is not None and tail.size > fs // 2
                else float("nan")
            )
            headline_att = (
                min(att_vs_initial, att_vs_tail)
                if np.isfinite(att_vs_tail)
                else att_vs_initial
            )
            # 측정 대역에 소스 에너지가 없으면 그 수치는 잡음이다 — 발행하지 않는다.
            energy_fraction = band_power(result["off"], fs, measure) / max(
                band_power(result["off"], fs, (20.0, fs / 2.0)), 1e-30
            )
            band_has_source = energy_fraction >= float(
                eval_cfg.get("min_source_energy_fraction_in_measure_band", 0.5)
            )
            band_txt = " ".join(
                (
                    f"{b['center_hz']:.0f}:{b['attenuation_db']:+.1f}"
                    f"{'' if b['trusted'] else '*'}"
                    if b["source_energy_valid"]
                    else f"{b['center_hz']:.0f}:무효(PSD {b['source_energy_density_ratio']:.2f})"
                )
                for b in band_att
            )
            stats = result["stats"]
            verdict = (
                f"소스대역 {measure[0]:.0f}-{measure[1]:.0f}Hz 감쇠 **{headline_att:+.2f} dB**"
                if band_has_source
                else f"[무효] 측정대역에 소스 에너지 {energy_fraction:.2%} — 감쇠를 발행하지 않음"
            )
            print(
                f"  {verdict} (선행기준 {att_vs_initial:+.2f} / 후행기준 {att_vs_tail:+.2f})"
            )
            print(
                f"  trusted NMSE {nmse_trusted:+.2f} dB | fullband {nmse_fullband:+.2f} dB | "
                f"{band_txt}"
            )

            key = f"{name}_{controller}"
            # 판정에 쓴 바로 그 두 구간을 같은 배율로 WAV 로 남긴다. 크기 차이가 곧 감쇠다.
            wavs = write_wav_pair(
                run_dir / "wav", key, result["off"], result["on"], fs
            )
            np.savez_compressed(
                run_dir / "raw" / f"{key}.npz",
                fs=fs,
                trusted_band_hz=np.asarray(trusted, dtype=np.float64),
                nmse_trusted_db=nmse_trusted,
                nmse_fullband_db=nmse_fullband,
                nmse_gap_trusted_minus_fullband_db=nmse_trusted - nmse_fullband,
                off_segment=result["off"],
                off_source_segment=result["off_source"],
                on_segment=result["on"],
                live_protocol_schema=np.asarray(LIVE_PROTOCOL_SCHEMA),
                # requested state는 source/anc_gain raw 파형으로 교차검증할 수 있다.
                # OFF/ON이 noise 재생 자체를 끈 실험으로 오해되지 않게 immutable raw에
                # 명시한다.
                noise_enabled_requested=np.asarray(
                    result["protocol_state"]["noise_enabled_requested"], dtype=np.bool_
                ),
                anc_enabled_baseline_requested=np.asarray(
                    result["protocol_state"]["anc_enabled_baseline_requested"],
                    dtype=np.bool_,
                ),
                anc_enabled_on_requested=np.asarray(
                    result["protocol_state"]["anc_enabled_on_requested"], dtype=np.bool_
                ),
                anc_enabled_tail_requested=np.asarray(
                    result["protocol_state"]["anc_enabled_tail_requested"], dtype=np.bool_
                ),
                octave_source_energy_gate_schema=np.asarray(
                    OCTAVE_SOURCE_ENERGY_GATE_SCHEMA
                ),
                octave_source_energy_min_density_ratio=np.asarray(
                    octave_source_energy_density_threshold, dtype=np.float64
                ),
                octave_center_hz=np.asarray(
                    [band["center_hz"] for band in band_att], dtype=np.float64
                ),
                # ``octave_attenuation_db``는 source-energy gate를 통과한 공개값이다.
                # 진단용 원 계산값은 별도 key에 남겨 historical 숫자와 혼동하지 않는다.
                octave_attenuation_db=np.asarray(
                    [band["attenuation_db"] for band in band_att], dtype=np.float64
                ),
                octave_attenuation_unqualified_db=np.asarray(
                    [band["unqualified_attenuation_db"] for band in band_att],
                    dtype=np.float64,
                ),
                octave_trusted=np.asarray(
                    [band["trusted"] for band in band_att], dtype=np.bool_
                ),
                octave_off_source_band_power=np.asarray(
                    [band["off_source_band_power"] for band in band_att],
                    dtype=np.float64,
                ),
                octave_source_energy_fraction=np.asarray(
                    [band["source_energy_fraction"] for band in band_att],
                    dtype=np.float64,
                ),
                octave_source_energy_flat_fraction=np.asarray(
                    [band["source_energy_flat_fraction"] for band in band_att],
                    dtype=np.float64,
                ),
                octave_source_energy_density_ratio=np.asarray(
                    [band["source_energy_density_ratio"] for band in band_att],
                    dtype=np.float64,
                ),
                octave_source_energy_valid=np.asarray(
                    [band["source_energy_valid"] for band in band_att],
                    dtype=np.bool_,
                ),
                **result["data"],
            )

            row = {
                "scenario": name,
                "controller": controller,
                "sample_rate_hz": fs,
                "noise_type": str(cfg["noise"].get("type", "")),
                "noise_amplitude": float(cfg["noise"].get("amplitude", float("nan"))),
                "measure_low_hz": measure[0],
                "measure_high_hz": measure[1],
                "measure_band_source_fraction": energy_fraction,
                "measure_band_has_source": band_has_source,
                "octave_source_energy_gate_schema": OCTAVE_SOURCE_ENERGY_GATE_SCHEMA,
                "octave_source_energy_min_density_ratio": octave_source_energy_density_threshold,
                "attenuation_db": headline_att if band_has_source else float("nan"),
                "attenuation_vs_initial_db": att_vs_initial,
                "attenuation_vs_tail_db": att_vs_tail,
                "baseline_drift_db": att_vs_initial - att_vs_tail,
                "trusted_low_hz": float(trusted[0]),
                "trusted_high_hz": float(trusted[1]),
                "trusted_attenuation_db": -nmse_trusted,
                "fullband_attenuation_db": -nmse_fullband,
                "gap_trusted_minus_fullband_db": nmse_trusted - nmse_fullband,
                "on_duty_fraction": result["on_duty"],
                "underruns": int(stats.get("underruns", 0)),
                "xruns": int(stats.get("xruns", 0)),
                "step_ms_mean": float(stats.get("step_ms", float("nan"))),
            }
            for band in band_att:
                center = int(band["center_hz"])
                row[f"band_{center}_att_db"] = float(band["attenuation_db"])
                row[f"band_{center}_unqualified_att_db"] = float(
                    band["unqualified_attenuation_db"]
                )
                row[f"band_{center}_trusted"] = bool(band["trusted"])
                row[f"band_{center}_off_source_power"] = float(
                    band["off_source_band_power"]
                )
                row[f"band_{center}_source_energy_fraction"] = float(
                    band["source_energy_fraction"]
                )
                row[f"band_{center}_source_energy_flat_fraction"] = float(
                    band["source_energy_flat_fraction"]
                )
                row[f"band_{center}_source_energy_density_ratio"] = float(
                    band["source_energy_density_ratio"]
                )
                row[f"band_{center}_source_energy_valid"] = bool(
                    band["source_energy_valid"]
                )
            row["wav_ab"] = wavs["ab"].name
            rows.append(row)

    if not rows:
        print("[중단] 측정된 시나리오가 없습니다.", file=sys.stderr)
        return 2

    csv_path = write_csv(run_dir / "metrics.csv", rows)

    table_columns = [
        "scenario", "controller", "trusted_attenuation_db",
        "fullband_attenuation_db", "on_duty_fraction", "xruns",
    ]
    lines = [
        f"# 실기 평가 리포트 ({stamp})",
        "",
        f"- Trusted 대역: **{trusted[0]:.0f}–{trusted[1]:.0f} Hz** "
        "(S(z) 실측 대역 ∩ 덕트 목표 대역)",
        f"- 프로토콜: OFF {protocol.get('baseline_seconds', 10)}s → "
        f"ON {protocol.get('on_seconds', 30)}s → OFF {protocol.get('tail_seconds', 5)}s",
        f"- 기계가 읽는 단일 출처는 `metrics.csv` 다. 이 표는 거기서 파생된다.",
        "",
        "| 시나리오 | 컨트롤러 | 측정대역(Hz) | 감쇠(dB) | 베이스라인 드리프트 | ON 유효구간 | xrun |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        headline = (
            f"**{row['attenuation_db']:+.2f}**"
            if row["measure_band_has_source"]
            else f"무효 (소스 {row['measure_band_source_fraction']:.1%})"
        )
        lines.append(
            f"| {row['scenario']} | {row['controller']} | "
            f"{row['measure_low_hz']:.0f}–{row['measure_high_hz']:.0f} | {headline} | "
            f"{row['baseline_drift_db']:+.2f} | "
            f"{100.0 * row['on_duty_fraction']:.1f}% | {row['xruns']} |"
        )
    lines += [
        "",
        "감쇠는 **선행 OFF 와 후행 OFF 중 나쁜 쪽** 기준이다. 두 베이스라인이 최대 38초",
        "떨어져 있어 드리프트가 감쇠로 오인될 수 있고, 실측에서 2.39 dB 까지 갈렸다.",
        "측정대역에 소스 에너지가 절반 미만이면 감쇠를 발행하지 않는다 — 잡음을 재게 되기 때문이다.",
    ]
    lines += [
        "",
        "## 옥타브밴드 감쇠 (dB, *=trusted 대역 밖, 무효=OFF 재생 source 부족)",
        "",
              "| 시나리오 | " + " | ".join(f"{c}Hz" for c in band_centers) + " |",
              "|---" * (len(band_centers) + 1) + "|"]
    for row in rows:
        cells = []
        for center in band_centers:
            mark = "" if row.get(f"band_{center}_trusted") else "*"
            if row[f"band_{center}_source_energy_valid"]:
                cells.append(f"{row[f'band_{center}_att_db']:+.1f}{mark}")
            else:
                cells.append(
                    f"무효(PSD {row[f'band_{center}_source_energy_density_ratio']:.2f}){mark}"
                )
        lines.append(f"| {row['scenario']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "옥타브별 수치는 OFF 재생 source의 20 Hz–Nyquist 전력 중 해당 octave가 "
        f"평균 PSD의 {100.0 * octave_source_energy_density_threshold:.1f}% 이상일 때만 "
        "발행한다(실제 임계값은 `metrics.csv`/raw NPZ에 저장). 무효 octave의 원 계산값은 "
        "raw NPZ의 `octave_attenuation_unqualified_db`에만 보존하며 성능 주장에 쓰지 않는다.",
        "이 gate는 source가 그 대역에 있었는지만 보장한다. ERR SNR/coherence, P/S 신뢰대역, "
        "xrun/deadline, OFF/ON drift를 별도로 통과해야 한다.",
        "## 산출물",
        "",
        "| 경로 | 내용 |",
        "|---|---|",
        "| `metrics.csv` | 한 행 = 시나리오×컨트롤러. 밴드별 감쇠 열 포함 |",
        "| `raw/*.npz` | err/ref/source/control/anc_gain 원신호, 두 구간, octave source-energy fraction/validity mask |",
        "| `wav/*_off.wav`, `*_on.wav`, `*_ab.wav` | 같은 배율로 쓴 청취용 (ab = 앞 OFF, 뒤 ON) |",
        "",
    ]
    summary_path = atomic_write_text(run_dir / "summary.md", "\n".join(lines) + "\n")

    print(f"\n산출물:")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {run_dir / 'raw'} · {run_dir / 'wav'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
