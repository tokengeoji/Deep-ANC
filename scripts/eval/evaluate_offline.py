#!/usr/bin/env python3
"""오프라인 평가 — 테스트 split 합성 데이터에서 모델 성능 일괄 산출.

  .venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt
산출: runs/<exp>/eval/{metrics.md, metrics.npz, psd_*.png, spec_*.png, band_*.png}
플랜트는 섭동 없는 결정적 S(z)(핸드오프 포함)로 계산한다.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                     # noqa: E402
from deep_anc.data.synth_dataset import (                            # noqa: E402
    SynthANCDataset,
    make_eval_batch,
    source_mix_for_split,
)
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


def source_evaluation_policy(data_cfg: dict, active_mix: dict) -> dict:
    """test 데이터셋의 실제 표집 분포와 제외된 학습 보조 소스를 기록한다."""
    train_mix = source_mix_for_split(data_cfg, "train")
    allowed = {tag: float(weight) for tag, weight in active_mix.items() if weight > 0}
    total = float(sum(allowed.values()))
    if not allowed or not np.isfinite(total) or total <= 0:
        raise ValueError("평가에 허용된 source mix가 비어 있거나 유효하지 않습니다")
    train_only = list(data_cfg.get("train_only_source_families", []))
    if set(allowed).intersection(train_only):
        raise ValueError("train-only 소스는 독립 평가에 포함할 수 없습니다")
    return {
        "schema_version": 1,
        "split": "test",
        "diagnostic_only": True,
        "performance_claim_allowed": False,
        "evaluation_domain": "simulated_secondary_path_not_physical_measurement",
        "configured_training_source_mix": {tag: float(w) for tag, w in train_mix.items()},
        "excluded_train_only_source_families": train_only,
        "retained_training_weight_sum": float(sum(train_mix.get(tag, 0) for tag in allowed)),
        "training_weight_sum": float(sum(train_mix.values())),
        "effective_evaluation_source_mix": {tag: weight / total for tag, weight in allowed.items()},
        "overall_aggregation": "arithmetic_mean_of_item_nmse_db_from_effective_evaluation_mix",
        "per_source_aggregation": "8_items_per_source_separate_from_overall_mean",
        "per_source_requested": list(allowed),
        "per_source_evaluated": [],
        "per_source_skipped": [],
    }


def per_source_config(data_cfg: dict, tag: str, active_mix: dict) -> dict:
    """허용 소스 하나로 고정한다. acoustic override가 혼합을 되살리면 안 된다."""
    if (active_mix.get(tag, 0) <= 0
            or tag in data_cfg.get("train_only_source_families", [])):
        raise ValueError(f"독립 평가에서 허용하지 않는 소스: {tag}")
    cfg = copy.deepcopy(data_cfg)
    cfg["source_mix_ratio"] = {tag: 1.0}
    if cfg.get("reference_mode", "digital") == "acoustic":
        cfg["source_mix_ratio_acoustic"] = {tag: 1.0}
    return cfg


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

    from deep_anc.config import DEFAULT_HANDOFF_SAMPLES

    fs = int(data_cfg["sample_rate"])
    sp = load_secondary_path(REPO_ROOT / duct_cfg["secondary_path"]["npz"])
    if sp.sample_rate != fs:
        raise ValueError(
            f"S(z) sample_rate={sp.sample_rate}Hz != data sample_rate={fs}Hz"
        )
    trusted = intersect_frequency_bands(
        sp.trusted_band_hz(),
        duct_cfg["acoustics"]["realistic_target_band_hz"],
        fs / 2.0,
    )
    plant = DifferentiableSecondaryPath(
        sp,
        handoff_extra_samples=int(
            duct_cfg["secondary_path"].get("handoff_extra_samples", DEFAULT_HANDOFF_SAMPLES)
        ),
    ).to(device)

    ds = SynthANCDataset(data_cfg, duct_cfg, split="test", seed=999)
    source_policy = source_evaluation_policy(data_cfg, ds.mix_ratio)
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
    for tag in source_policy["per_source_requested"]:
        tag_cfg = per_source_config(data_cfg, tag, ds.mix_ratio)
        try:
            tag_ds = SynthANCDataset(tag_cfg, duct_cfg, split="test", seed=555)
            if tag != "synthetic":
                pool_paths = tag_ds.pools.get(tag, [])
                if not pool_paths or not Path(pool_paths[0]).exists():
                    raise ValueError("manifest 없음: 합성 폴백은 해당 소스의 평가가 아닙니다")
            tb = make_eval_batch(tag_ds, n_items=8, seed=555)
            with torch.no_grad():
                ty = model(tb["x"].to(device))
                te = tb["d"].to(device) + plant(ty.float(), {"jitter": 0})
            td, te_np = tb["d"].squeeze(1).numpy(), te.squeeze(1).cpu().numpy()
            per_source.append((tag, float(np.mean([nmse_db(td[i], te_np[i]) for i in range(td.shape[0])]))))
            source_policy["per_source_evaluated"].append(tag)
        except Exception as exc:
            source_policy["per_source_skipped"].append({"source_family": tag, "reason": str(exc)})
            print(f"[skip] 소스별 평가 {tag}: {exc}")

    lines = [
        f"# 오프라인 평가 — {Path(args.ckpt).name}",
        "",
        "- 합성 플랜트 진단이며 실기 감쇠 또는 모든 소스의 독립 일반화 입증이 아닙니다.",
        f"- 테스트 아이템: {len(per_item_fullband)}개 (reference_mode={data_cfg.get('reference_mode')})",
        "- 독립 평가 제외(train-only): "
        + (", ".join(source_policy["excluded_train_only_source_families"]) or "없음"),
        f"- 평가 분포 정규화 분모: 남은 학습 가중치 합 "
        f"{source_policy['retained_training_weight_sum']:.6g} "
        f"(전체 학습 가중치 합 {source_policy['training_weight_sum']:.6g}); "
        "허용 소스만 재정규화한 분포로 표집합니다.",
        "- 실제 test 표집 분포: " + json.dumps(
            source_policy["effective_evaluation_source_mix"], ensure_ascii=False, sort_keys=True
        ),
        "- 전체 점수는 표집 아이템별 NMSE(dB)의 산술평균입니다. "
        "소스별 8개 점수는 별도 진단이며 전체 평균의 추가 분모가 아닙니다.",
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
    for skipped in source_policy["per_source_skipped"]:
        lines.append(f"| {skipped['source_family']} | 미평가 | 미평가 |")
    if source_policy["per_source_skipped"]:
        lines += ["", "소스별 미평가 사유(성공 또는 0 dB로 집계하지 않음):", ""]
        lines += [f"- {row['source_family']}: {row['reason']}" for row in source_policy["per_source_skipped"]]
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
        source_evaluation_policy_json=json.dumps(source_policy, ensure_ascii=False, allow_nan=False),
    )

    spectrogram_pair(d_np[0], e_np[0], fs, out_dir / "spec_item0.png", "ANC OFF vs ON (시뮬)")
    psd_overlay({"d (OFF)": d_cat, "e (ON)": e_cat}, fs, out_dir / "psd.png", "PSD 비교")
    band_bar(band_att, out_dir / "band.png", "옥타브밴드 감쇠")

    print("\n".join(lines))
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
