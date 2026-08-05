#!/usr/bin/env python3
"""오프라인 평가 — 테스트 split 합성 데이터에서 모델 성능 일괄 산출.

  .venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt
산출: runs/<exp>/eval/{metrics.md, metrics.npz, psd_*.png, spec_*.png, band_*.png}
플랜트는 섭동 없는 결정적 S(z)(핸드오프 포함)로 계산한다.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                     # noqa: E402
from deep_anc.data.synth_dataset import SynthANCDataset, make_eval_batch  # noqa: E402
from deep_anc.dsp.timing import BandPlan, handoff_samples_from_config  # noqa: E402
from deep_anc.dsp.secondary_path import (                            # noqa: E402
    DifferentiableSecondaryPath,
    load_secondary_path,
)
from deep_anc.eval.metrics import (                                  # noqa: E402
    band_nmse_db,
    intersect_frequency_bands,
    nmse_db,
    octave_band_attenuation,
)
from deep_anc.eval.plots import band_bar, psd_overlay, spectrogram_pair  # noqa: E402
from deep_anc.models import build_model                              # noqa: E402


def resolve_checkpoint_config(
    state: dict,
    key: str,
    override_path: str | None,
    legacy_default: str,
) -> dict:
    """명시 override가 없으면 checkpoint의 resolved data/duct를 authority로 쓴다."""
    if override_path:
        return load_yaml(override_path)
    resolved = (state.get("cfg") or {}).get(key)
    if isinstance(resolved, dict) and resolved:
        return copy.deepcopy(resolved)
    print(
        f"[평가 경고] checkpoint에 resolved {key} 설정이 없어 legacy 기본값 "
        f"{legacy_default}을 사용합니다.",
        file=sys.stderr,
    )
    return load_yaml(legacy_default)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--data-config",
        default=None,
        help="명시할 때만 checkpoint의 resolved data 설정을 대체",
    )
    parser.add_argument(
        "--duct-config",
        default=None,
        help="명시할 때만 checkpoint의 resolved duct 설정을 대체",
    )
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--n-items", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    data_cfg = resolve_checkpoint_config(
        state, "data", args.data_config, "configs/data_sim.yaml"
    )
    duct_cfg = resolve_checkpoint_config(
        state, "duct", args.duct_config, "configs/duct.yaml"
    )
    eval_cfg = load_yaml(args.eval_config)

    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    fs = int(data_cfg["sample_rate"])
    sp = load_secondary_path(REPO_ROOT / duct_cfg["secondary_path"]["npz"])
    if sp.sample_rate != fs:
        raise ValueError(
            f"S(z) sample_rate={sp.sample_rate}Hz != data sample_rate={fs}Hz"
        )
    # 대역·handoff 는 BandPlan/timing 이 단일 출처다 (발생기 A — 같은 세 줄이
    # 다섯 파일에 복붙돼 있었고 S npz 를 넓혀도 따라오지 않는 곳이 생겼다).
    band_plan = BandPlan.resolve(
        plant_trusted_band_hz=sp.trusted_band_hz(),
        duct_cfg=duct_cfg,
        sample_rate=fs,
    )
    trusted = band_plan.optimize.as_tuple()
    plant = DifferentiableSecondaryPath(
        sp,
        handoff_extra_samples=handoff_samples_from_config(duct_cfg),
    ).to(device)

    ds = SynthANCDataset(data_cfg, duct_cfg, split="test", seed=999)
    batch = make_eval_batch(ds, n_items=args.n_items, seed=999)

    with torch.no_grad():
        x = batch["x"].to(device)
        d = batch["d"].to(device)
        y = model(x)
        e = d + plant(y.float(), {"jitter": 0})

    d_np = d.squeeze(1).cpu().numpy()
    e_np = e.squeeze(1).cpu().numpy()

    out_dir = Path(args.out) if args.out else Path(args.ckpt).parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = eval_cfg.get("octave_bands_hz", [125, 250, 500, 1000, 2000, 4000, 8000])
    per_item_fullband = [nmse_db(d_np[i], e_np[i]) for i in range(d_np.shape[0])]
    per_item_trusted = [
        band_nmse_db(d_np[i], e_np[i], fs, trusted) for i in range(d_np.shape[0])
    ]
    overall_fullband = float(np.mean(per_item_fullband))
    overall_trusted = float(np.mean(per_item_trusted))
    d_cat, e_cat = d_np.reshape(-1), e_np.reshape(-1)
    band_att = octave_band_attenuation(d_cat, e_cat, fs, bands, trusted)

    # [기능1 보조] held-out 비선형 강도에서의 NMSE — 학습 그리드 밖 일반화 (로드맵 A1)
    from deep_anc.dsp.nonlinear import sef_torch

    eta_h = float(eval_cfg.get("heldout_sef_eta", 0.15))
    with torch.no_grad():
        e_nl = d + plant(sef_torch(y.float(), eta_h), {"jitter": 0})
    e_nl_np = e_nl.squeeze(1).cpu().numpy()
    nmse_heldout_fullband = float(
        np.mean([nmse_db(d_np[i], e_nl_np[i]) for i in range(d_np.shape[0])])
    )
    nmse_heldout_trusted = float(
        np.mean(
            [band_nmse_db(d_np[i], e_nl_np[i], fs, trusted) for i in range(d_np.shape[0])]
        )
    )

    # [기능2] 소스 종류별 감쇠 — "모든 소리 제거" 목표의 분리 점수 (소음/음성/음악/기계음…)
    per_source: list[tuple[str, float]] = []
    for tag in ["synthetic"] + [t for t in data_cfg.get("source_mix_ratio", {}) if t != "synthetic"]:
        tag_cfg = dict(data_cfg)
        tag_cfg["source_mix_ratio"] = {tag: 1.0}
        try:
            tag_ds = SynthANCDataset(tag_cfg, duct_cfg, split="test", seed=555)
            if tag != "synthetic":
                pool_paths = tag_ds.pools.get(tag, [])
                if not pool_paths or not Path(pool_paths[0]).exists():
                    continue                  # manifest 없음 — 합성 폴백 값은 무의미하므로 제외
            tb = make_eval_batch(tag_ds, n_items=8, seed=555)
            with torch.no_grad():
                ty = model(tb["x"].to(device))
                te = tb["d"].to(device) + plant(ty.float(), {"jitter": 0})
            td, te_np = tb["d"].squeeze(1).numpy(), te.squeeze(1).cpu().numpy()
            per_source.append((tag, float(np.mean([nmse_db(td[i], te_np[i]) for i in range(td.shape[0])]))))
        except Exception as exc:
            print(f"[skip] 소스별 평가 {tag}: {exc}")

    lines = [
        f"# 오프라인 평가 — {Path(args.ckpt).name}",
        "",
        f"- 테스트 아이템: {len(per_item_fullband)}개 (reference_mode={data_cfg.get('reference_mode')})",
        f"- Trusted 대역: **{trusted[0]:.0f}–{trusted[1]:.0f} Hz** "
        f"(S(z) {sp.trusted_band_hz()[0]:.0f}–{sp.trusted_band_hz()[1]:.0f} Hz ∩ "
        f"덕트 목표 {duct_cfg['acoustics']['realistic_target_band_hz'][0]:.0f}–"
        f"{duct_cfg['acoustics']['realistic_target_band_hz'][1]:.0f} Hz)",
        f"- **Trusted 평균 NMSE: {overall_trusted:.2f} dB** (감쇠 {-overall_trusted:.2f} dB)",
        f"- **Fullband 평균 NMSE: {overall_fullband:.2f} dB** (감쇠 {-overall_fullband:.2f} dB)",
        f"- Trusted−fullband NMSE 간극: {overall_trusted - overall_fullband:+.2f} dB",
        f"- held-out 비선형(η={eta_h}) NMSE: trusted {nmse_heldout_trusted:.2f} dB / "
        f"fullband {nmse_heldout_fullband:.2f} dB — 학습 그리드 밖 일반화",
        f"- 아이템 분포(trusted): 중앙값 {np.median(per_item_trusted):.2f} dB / "
        f"최악 {np.max(per_item_trusted):.2f} dB",
        f"- 아이템 분포(fullband): 중앙값 {np.median(per_item_fullband):.2f} dB / "
        f"최악 {np.max(per_item_fullband):.2f} dB",
        "",
        "## 기능1 — 주파수 대역별 감쇠 (저주파+고주파)",
        "",
        "| 밴드(Hz) | 감쇠(dB) | 신뢰 |",
        "|---|---|---|",
    ]
    for b in band_att:
        mark = "O" if b["trusted"] else "낮음*"
        lines.append(f"| {b['center_hz']:.0f} | {b['attenuation_db']:+.2f} | {mark} |")
    lines += [
        "",
        "*: S(z) 보정 유효대역 밖 — 광대역 재보정 전에는 참고용.",
        "",
        "## 기능2 — 소스 종류별 감쇠 (모든 소리 제거)",
        "",
        "| 소스 | NMSE(dB) | 감쇠(dB) |",
        "|---|---|---|",
    ]
    for tag, v in per_source:
        lines.append(f"| {tag} | {v:+.2f} | {-v:+.2f} |")
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / "metrics.npz",
        trusted_band_hz=np.asarray(trusted, dtype=np.float64),
        nmse_trusted_db=overall_trusted,
        nmse_fullband_db=overall_fullband,
        nmse_gap_trusted_minus_fullband_db=overall_trusted - overall_fullband,
        nmse_heldout_trusted_db=nmse_heldout_trusted,
        nmse_heldout_fullband_db=nmse_heldout_fullband,
        per_item_trusted_db=np.asarray(per_item_trusted, dtype=np.float64),
        per_item_fullband_db=np.asarray(per_item_fullband, dtype=np.float64),
    )

    spectrogram_pair(d_np[0], e_np[0], fs, out_dir / "spec_item0.png", "ANC OFF vs ON (시뮬)")
    psd_overlay({"d (OFF)": d_cat, "e (ON)": e_cat}, fs, out_dir / "psd.png", "PSD 비교")
    band_bar(band_att, out_dir / "band.png", "옥타브밴드 감쇠")

    print("\n".join(lines))
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
