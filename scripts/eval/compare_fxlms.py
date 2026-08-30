#!/usr/bin/env python3
"""DL vs FxLMS vs 무제어 — 동일 시나리오·동일 S(z) 오프라인 비교표(md).

  .venv/bin/python scripts/eval/compare_fxlms.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt
시나리오: 톤 300Hz / 멀티톤 / 대역잡음 80–1k / 비선형(고조파+클립) — configs/eval.yaml 과 동일 구성.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.config import REPO_ROOT, load_yaml                     # noqa: E402
from deep_anc.data.synth_dataset import _delay_np                    # noqa: E402
from deep_anc.dsp.duct_sim import duct_paths                         # noqa: E402
from deep_anc.dsp.filters import fft_filter                          # noqa: E402
from deep_anc.dsp.secondary_path import (                            # noqa: E402
    DifferentiableSecondaryPath,
    load_secondary_path,
)
from deep_anc.config import default_d_noise_delay                    # noqa: E402
from deep_anc.eval.fxlms_baseline import run_fxlms_offline           # noqa: E402
from deep_anc.eval.metrics import attenuation_db                     # noqa: E402
from deep_anc.models import build_model                              # noqa: E402
from deep_anc.realtime.noise_gen import NoiseProgram                 # noqa: E402


def make_scenario(noise_cfg: dict, seconds: float, fs: int, duct_cfg: dict, sp_delay: int):
    """digital-ref 시나리오 신호 생성: x_ref = 소스, d = P_err·지연(소스).

    p_err RIR 에 t_ac(NS→ERR) 온셋이 포함되어 있으므로 전기/버퍼 성분만 추가 지연한다
    (synth_dataset 과 동일 규약 — 이중 계상 방지).
    """
    from deep_anc.config import duct_distance_samples

    n_samples = int(seconds * fs) // 256 * 256
    program = NoiseProgram(noise_cfg, fs)
    src = program.generate(n_samples)
    paths = duct_paths(duct_cfg, fs)
    d_noise = duct_cfg.get("digital_reference", {}).get("d_noise_delay_samples")
    if d_noise is None:
        d_noise = default_d_noise_delay(duct_cfg, fs, sp_delay)
    extra = max(0, int(d_noise) - duct_distance_samples(duct_cfg, "noise_speaker", "error_mic", fs))
    d = _delay_np(fft_filter(src, paths["p_err"]), extra)
    return src.astype(np.float32), d.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--duct-config", default="configs/duct.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", default="results/compare_fxlms.md")
    args = parser.parse_args()

    from deep_anc.dsp.timing import handoff_samples_from_config

    duct_cfg = load_yaml(args.duct_config)
    eval_cfg = load_yaml(args.eval_config)
    data_cfg = load_yaml("configs/data_sim.yaml")
    fs = int(data_cfg["sample_rate"])
    fb_range = data_cfg.get("closed_loop", {}).get("feedback_delay_samples", [512, 1024])
    fb_delay = (int(fb_range[0]) + int(fb_range[1])) // 2

    sp = load_secondary_path(REPO_ROOT / duct_cfg["secondary_path"]["npz"])
    handoff = handoff_samples_from_config(duct_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plant = DifferentiableSecondaryPath(sp, handoff_extra_samples=handoff).to(device)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # 이 스크립트는 checkpoint 의 resolved 물리(P(z) resolver, digital-reference lead)를
    # 쓰지 않는다. 그래서 corrected 규약으로 학습한 checkpoint 를 넣으면 DL 열이 조용히
    # 틀린다 — 실측으로 tone300 이 +42.86 dB 여야 하는데 -0.91 dB(증폭)로 나온다.
    # 원인은 두 가지가 겹친 것이다: (1) x_ref 에 lead 를 적용하지 않음, (2) 학습이 쓴
    # P(z)=측정 S FIR(peak 0.0019) 대신 시뮬 p_err RIR(peak 0.367, 190배)을 씀.
    #
    # 딥러닝이 이 프로젝트의 목적이므로 FxLMS 비교기를 다시 만드는 데 시간을 쓰지 않는다.
    # 대신 **틀린 숫자를 발행하지 못하게** 막는다. 잘못된 비교표가 한 번 나가면 되돌리기
    # 어렵고, 이 스크립트의 결과에 걸린 결정은 현재 하나도 없다.
    checkpoint_lead = int(state.get("cfg", {}).get("digital_reference_lead_samples", 0))
    checkpoint_mode = str(
        state.get("cfg", {}).get("data", {}).get("digital_primary_path_mode", "")
    )
    if checkpoint_lead != 0 or checkpoint_mode:
        print(
            "[중단] 이 checkpoint 는 digital-reference lead "
            f"{checkpoint_lead} 샘플 / P(z) 모드 '{checkpoint_mode}' 로 학습되었습니다.\n"
            "       compare_fxlms.py 는 두 규약을 반영하지 않아 DL 감쇠를 약 43 dB 낮게 "
            "보고합니다(실측 +42.86 → -0.91 dB).\n"
            "       DL 성능은 scripts/eval/evaluate_offline.py 를 쓰세요 — checkpoint 의 "
            "resolved 설정을 그대로 사용합니다.\n"
            "       FxLMS 실기 비교가 필요하면 scripts/demo/evaluate_fxlms_direct.py 를 쓰세요.",
            file=sys.stderr,
        )
        return 2

    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model = model.eval().to(device)

    lines = [
        "# DL vs FxLMS 오프라인 비교 (digital-ref, 동일 S(z))",
        "",
        f"- ckpt: {args.ckpt} | 시나리오 길이 {args.seconds:.0f}s | 핸드오프 +{handoff}샘플",
        "",
        "| 시나리오 | FxLMS 감쇠(dB, 후반부) | DL 감쇠(dB, 후반부) |",
        "|---|---|---|",
    ]

    for scenario in eval_cfg["scenarios"]:
        name = scenario["name"]
        x_ref, d = make_scenario(scenario["noise"], args.seconds, fs, duct_cfg, sp.delay_samples)

        # FxLMS: 실기와 동일하게 핸드오프 없는 in-callback 지연으로 구동
        fx = run_fxlms_offline(x_ref, d, sp.fir, sp.delay_samples)

        # DL: 학습과 동일한 핸드오프 포함 플랜트, 피드백 지연은 학습 범위 중앙값
        with torch.no_grad():
            err_in = torch.from_numpy(_delay_np(d, fb_delay)).view(1, 1, -1)
            x = torch.cat([torch.from_numpy(x_ref).view(1, 1, -1), err_in], dim=1).to(device)
            y = model(x)
            e_dl = (torch.from_numpy(d).view(1, 1, -1).to(device) + plant(y.float(), {"jitter": 0}))
        e_dl = e_dl.view(-1).cpu().numpy()
        tail = slice(2 * len(d) // 3, len(d))
        dl_att = attenuation_db(d[tail], e_dl[tail])

        print(f"{name:10s} | FxLMS {fx['attenuation_db_tail']:+6.2f} dB | DL {dl_att:+6.2f} dB")
        lines.append(f"| {name} | {fx['attenuation_db_tail']:+.2f} | {dl_att:+.2f} |")

    lines += [
        "",
        "해석: 후반부(수렴 후) 1/3 구간 기준. FxLMS 는 협대역·선형에서 강하고,",
        "DL 은 비선형(nonlinear)·광대역 시나리오에서의 우위를 목표로 한다 (docs/07).",
    ]
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
