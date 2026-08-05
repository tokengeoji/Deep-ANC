#!/usr/bin/env python3
"""학습된 모델의 상쇄 결과를 **들어볼 수 있는 WAV** 로 렌더링한다 (오프라인 시뮬).

오디오 장치를 열지 않는다. 스피커도 마이크도 쓰지 않으므로 하드웨어 게이트와 무관하게
지금 실행할 수 있다. 만들어지는 것은 시나리오별 3종이다.

    <scenario>_off.wav    d(t)        — ANC 끔 (에러 마이크가 들을 소리)
    <scenario>_on.wav     e(t)=d+S·y  — ANC 켬 (에러 마이크가 들을 소리)
    <scenario>_ab.wav     앞 절반 OFF → 뒤 절반 ON (A/B 비교용 한 파일)

물리 규약은 checkpoint 에 저장된 resolved 설정을 그대로 쓴다. 즉 P(z) resolver,
digital-reference lead, S(z) FIR·순수지연·스레드 핸드오프가 학습 때와 동일하다.
여기서 규약을 다시 구현하면 학습과 어긋난 그럴듯한 소리를 만들게 되므로 하지 않는다.

**이 소리는 실제 덕트 성능이 아니다.** 현재 checkpoint 의 physics_status 는
``secondary_surrogate`` 이며 P(z)=S(z) 대용품이다. 실측 P/S 와 recorded 세션을 통과하기
전에는 "이만큼 조용해진다"는 근거로 쓸 수 없다. 렌더러는 그 사실을 파일명과 리포트에
항상 함께 적는다.

    .venv/bin/python scripts/demo/render_anc_demo.py \
      --ckpt runs/pretrain_tiny_corrected/ckpt/best.pt --seconds 6
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from deep_anc.config import load_yaml  # noqa: E402
from deep_anc.data.primary_path import resolve_digital_primary_path  # noqa: E402
from deep_anc.data.synth_dataset import _delay_np, fft_filter  # noqa: E402
from deep_anc.dsp.timing import BandPlan, handoff_samples_from_config  # noqa: E402
from deep_anc.dsp.secondary_path import (  # noqa: E402
    DifferentiableSecondaryPath,
    load_secondary_path,
)
from deep_anc.eval.metrics import (  # noqa: E402
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)
from deep_anc.models import build_model  # noqa: E402
from deep_anc.realtime.noise_gen import NoiseProgram  # noqa: E402

# 평가기와 같은 authority 규칙을 쓴다: 명시 override 가 없으면 checkpoint 의 resolved
# 설정이 권위다. 여기서 configs/ 를 다시 읽으면 학습 때와 다른 물리로 렌더링하게 된다.
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))
from evaluate_offline import resolve_checkpoint_config  # noqa: E402


def write_wav(path: Path, signal: np.ndarray, sample_rate: int, peak: float = 0.7) -> float:
    """float32 신호를 16-bit WAV 로 쓴다. 반환값은 적용한 스케일.

    OFF/ON 을 **같은 스케일**로 써야 소리 크기 차이가 곧 감쇠로 들린다.
    파일마다 정규화하면 상쇄 효과가 귀에서 사라진다.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal * peak, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return peak


def build_source(scenario: dict, fs: int, length: int) -> np.ndarray:
    """시나리오의 소음원 n(t) 를 만든다.

    ``wav`` 키가 있으면 그 파일을 반복 재생해 실제 소리(음성/음악)를 소스로 쓴다.
    ``noise`` 키가 함께 있으면 둘을 섞어 "소음이 섞인 실제 소리"를 만든다 — 기능 2
    (모든 소리 제거)를 귀로 확인하기 위한 구성이다.
    """

    parts = []
    if scenario.get("wav"):
        import soundfile as sf

        audio, sr = sf.read(REPO_ROOT / scenario["wav"], dtype="float32", always_2d=True)
        if sr != fs:
            raise ValueError(f"{scenario['wav']}: {sr}Hz != {fs}Hz — 리샘플이 필요합니다")
        mono = audio.mean(axis=1)
        peak = float(np.max(np.abs(mono))) or 1.0
        mono = mono / peak * float(scenario.get("wav_amplitude", 0.05))
        parts.append(np.resize(mono, length).astype(np.float32))
    if scenario.get("noise"):
        parts.append(NoiseProgram(scenario["noise"], fs).generate(length).astype(np.float32))
    if not parts:
        raise ValueError(f"시나리오 {scenario.get('name')}: wav 또는 noise 가 필요합니다")
    return np.sum(parts, axis=0).astype(np.float32)


def render_scenario(
    scenario: dict, *, model, plant, primary, lead: int, fs: int, seconds: float, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """한 시나리오의 d(ANC OFF) 와 e(ANC ON) 를 만든다."""

    total = int(seconds * fs)
    n_full = build_source(scenario, fs, total + lead)

    # digital-ref: 모델은 실제 출력보다 lead 샘플 앞선 소음을 본다.
    x_ref = n_full[lead : lead + total].copy()
    n = n_full[:total]
    d = _delay_np(fft_filter(n, primary.fir), int(primary.delay_samples))[:total]

    # err_in(ch1)은 open-loop 근사로 d 를 블록 지연시켜 넣는다(학습과 같은 규약).
    err_in = _delay_np(d, 512)[:total]

    x = torch.from_numpy(np.stack([x_ref, err_in])[None]).to(device)
    with torch.no_grad():
        y = model(x)
        e = torch.from_numpy(d[None, None]).to(device) + plant(y.float(), {"jitter": 0})
    return d, e.squeeze().cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    ckpt = Path(args.ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    data_cfg = resolve_checkpoint_config(state, "data", None, "configs/data_sim.yaml")
    duct_cfg = resolve_checkpoint_config(state, "duct", None, "configs/duct.yaml")
    eval_cfg = load_yaml(args.eval_config)

    physics = state["cfg"].get("physics_status", "unknown")
    fs = int(data_cfg["sample_rate"])
    lead = int(data_cfg.get("digital_reference_lead_samples", 0))

    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    sp = load_secondary_path(REPO_ROOT / duct_cfg["secondary_path"]["npz"])
    plant = DifferentiableSecondaryPath(
        sp,
        handoff_extra_samples=handoff_samples_from_config(duct_cfg),
    ).to(device)
    primary, _total = resolve_digital_primary_path(data_cfg, duct_cfg, fs, sp)
    trusted = BandPlan.resolve(
        plant_trusted_band_hz=sp.trusted_band_hz(),
        duct_cfg=duct_cfg,
        sample_rate=fs,
    ).optimize.as_tuple()
    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])

    out_dir = Path(args.out) if args.out else ckpt.parent.parent / "demo_audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# 청취용 ANC 데모 — {ckpt}",
        "",
        f"- physics_status: **{physics}**",
        f"- 모델: {state['cfg']['model']['name']} · step {state.get('step')}",
        f"- sample_rate {fs} Hz · digital-ref lead {lead} 샘플 · {args.seconds:.0f}초/시나리오",
        f"- trusted 대역: {trusted[0]:.0f}–{trusted[1]:.0f} Hz",
        "",
        "> **이 소리는 실제 덕트 성능이 아니다.** surrogate 플랜트 시뮬레이션이며,",
        "> 실측 P/S 와 recorded 세션을 통과하기 전에는 성능 근거로 쓸 수 없다.",
        "",
        "OFF/ON 은 **같은 스케일**로 렌더링했다. 파일별로 정규화하면 상쇄가 귀에서 사라진다.",
        "",
        "| 시나리오 | trusted 감쇠(dB) | fullband 감쇠(dB) | A/B 파일 |",
        "|---|---:|---:|---|",
    ]
    summary = []

    for scenario in eval_cfg.get("scenarios", []):
        name = str(scenario["name"])
        d, e = render_scenario(
            scenario, model=model, plant=plant, primary=primary, lead=lead,
            fs=fs, seconds=args.seconds, device=device,
        )
        # 두 신호 공통 스케일 — OFF 기준으로 잡아야 ON 이 실제로 작게 들린다.
        scale = 0.9 / max(float(np.max(np.abs(d))), 1e-9)
        write_wav(out_dir / f"{name}_off.wav", d * scale, fs, peak=1.0)
        write_wav(out_dir / f"{name}_on.wav", e * scale, fs, peak=1.0)
        half = len(d) // 2
        ab = np.concatenate([d[:half], e[half:]])
        write_wav(out_dir / f"{name}_ab.wav", ab * scale, fs, peak=1.0)

        att_tr = -band_nmse_db(d, e, fs, trusted)
        att_fb = -nmse_db(d, e)
        lines.append(f"| {name} | **{att_tr:+.2f}** | {att_fb:+.2f} | `{name}_ab.wav` |")
        summary.append({"scenario": name, "trusted_att_db": att_tr, "fullband_att_db": att_fb})
        print(f"[{name}] trusted {att_tr:+.2f} dB / fullband {att_fb:+.2f} dB")

    band_att = octave_band_attenuation(
        np.concatenate([s for s in [d]]), np.concatenate([s for s in [e]]), fs, bands, trusted
    )
    lines += ["", "## 마지막 시나리오의 옥타브밴드 감쇠", "",
              "| 밴드(Hz) | 감쇠(dB) | 신뢰 |", "|---|---|---|"]
    for b in band_att:
        lines.append(
            f"| {b['center_hz']:.0f} | {b['attenuation_db']:+.2f} | "
            f"{'O' if b['trusted'] else '낮음*'} |"
        )
    lines += ["", "*: S(z) 보정 유효대역 밖 — 광대역 재보정 전에는 참고용.", ""]

    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(
            {"checkpoint": str(ckpt), "physics_status": physics, "scenarios": summary},
            ensure_ascii=False, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\n산출물: {out_dir}")
    print("  *_off.wav (ANC 끔) / *_on.wav (ANC 켬) / *_ab.wav (앞 OFF → 뒤 ON)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
