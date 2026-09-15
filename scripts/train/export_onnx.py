#!/usr/bin/env python3
"""체크포인트 → 스트리밍 ONNX 내보내기 + ONNX Runtime(CPU) 등가성 검증.

규약: opset 17, 배치 1, 정적 shape, 상태 전부 명시 I/O, 블록 256(2프레임 내부 언롤).

  .venv/bin/python scripts/train/export_onnx.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt \
      --out runs/export/model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from deep_anc.models import build_model                                   # noqa: E402
from deep_anc.models.streaming import (                                   # noqa: E402
    ExportWrapper,
    flatten_states,
    state_names,
)
from deep_anc.realtime.engines import (                                   # noqa: E402
    checkpoint_digital_reference_lead_samples,
    checkpoint_reference_mode,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="runs/export/model.onnx")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_cfg = state["cfg"]["model"]
    model = build_model(model_cfg)
    model.load_state_dict(state["model"])
    model.eval()

    wrapper = ExportWrapper(model, block_samples=args.block)
    names = state_names(model)
    init_states = flatten_states(model.init_states(1, "cpu"))
    x = torch.zeros(1, model.in_channels, args.block)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = ["x"] + names
    output_names = ["y"] + [f"{n}_out" for n in names]
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (x, *init_states),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            dynamic_axes=None,          # 전부 정적 shape
        )
    print(f"ONNX 저장: {out_path}")

    # ----- ORT(CPU) 등가성 검증: 임의 블록 20개 스트리밍 -----
    import onnxruntime as ort

    # Jetson(Tegra)에서 ORT 기본 스레드 affinity 설정이 크래시를 유발하므로 명시 지정
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    torch_states = model.init_states(1, "cpu")
    ort_states = [s.numpy().copy() for s in flatten_states(model.init_states(1, "cpu"))]
    max_err = 0.0
    with torch.no_grad():
        for _ in range(20):
            blk = (rng.standard_normal((1, model.in_channels, args.block)) * 0.02).astype(np.float32)
            y_t, torch_states = model.streaming_step(torch.from_numpy(blk), torch_states)
            feeds = {"x": blk}
            feeds.update({n: s for n, s in zip(names, ort_states)})
            outs = sess.run(None, feeds)
            y_o, ort_states = outs[0], outs[1:]
            max_err = max(max_err, float(np.max(np.abs(y_t.numpy() - y_o))))
    print(f"ORT 등가성 max err = {max_err:.3e} (허용 {args.tolerance})")
    if max_err > args.tolerance:
        print("검증 실패 — export 그래프를 점검하세요", file=sys.stderr)
        return 1

    meta = {
        "model_name": model_cfg.get("name"),
        "digital_reference_lead_samples": checkpoint_digital_reference_lead_samples(state),
        "reference_mode": checkpoint_reference_mode(state),
        "block_samples": args.block,
        "hop": model.hop,
        "win": model.win,
        "in_channels": model.in_channels,
        "state_names": names,
        "ckpt": str(args.ckpt),
        "ort_max_err": max_err,
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"메타 저장: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
