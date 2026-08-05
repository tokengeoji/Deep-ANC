#!/usr/bin/env python3
"""아키텍처 도해를 논문 그림 형식으로 생성한다.

형식 규칙 — 원논문(Transformer Fig.1, ResNet Fig.3, WaveNet Fig.3)을 따른다.

* 박스 안에는 **레이어 이름과 핵심 하이퍼파라미터만** 넣는다. 설명 문장을 넣지 않는다.
* 그림 안에 제목을 넣지 않는다. 캡션은 README 본문이 담당한다.
* 텐서 shape 는 그림 오른쪽 여백에 둔다(논문 관행).

모든 수치는 ``configs/model_*.yaml`` 과 ``configs/duct.yaml`` 에서 읽는다. 손으로 적은
숫자가 없어야 설정을 바꿨을 때 그림이 조용히 거짓말을 하지 않는다.

    .venv/bin/python scripts/docs/render_architecture_figures.py

생성물 (assets/diagrams/):
  fig1_system.svg              신호 흐름
  fig2_architecture.svg        HybridANCNet 전체 스택
  fig3_tcn_block.svg           TCN 잔차 블록
  fig4_receptive_field.svg     dilated causal conv 수용영역
  fig5_glstm.svg               Grouped LSTM
  fig6_streaming.svg           스트리밍 상태 I/O
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svgkit import (  # noqa: E402
    ATTN, DANGER, ENCODER, INK, MONO, MUTED, PANEL, PHYS, RECURR, TCN, Canvas,
)

from deep_anc.config import load_yaml  # noqa: E402

OUT = REPO / "assets" / "diagrams"


def model_facts(name: str) -> dict:
    cfg = load_yaml(REPO / "configs" / f"model_{name}.yaml")
    dilations = list(cfg["tcn"]["dilations"])
    repeats = int(cfg["tcn"]["repeats"])
    kernel = int(cfg["tcn"]["kernel"])
    hop, win = int(cfg["hop"]), int(cfg["win"])
    frames = 1 + repeats * sum((kernel - 1) * d for d in dilations)
    return {
        "hop": hop, "win": win, "io_scale": float(cfg["io_scale"]),
        "enc_channels": int(cfg["encoder"]["channels"]),
        "repeats": repeats, "dilations": dilations, "kernel": kernel,
        "tcn_hidden": int(cfg["tcn"]["hidden"]),
        "glstm_groups": int(cfg["glstm"]["groups"]),
        "glstm_hidden": int(cfg["glstm"]["hidden_per_group"]),
        "attn_heads": int(cfg["attention"].get("heads", 0) or 0),
        "attn_window": int(cfg["attention"].get("window_frames", 0) or 0),
        "limit": float(cfg["limiter"]["limit"]),
        "blocks": repeats * len(dilations),
        "rf_frames": frames,
        "rf_ms": (frames - 1) * hop / 48000.0 * 1000.0 + win / 48000.0 * 1000.0,
    }


# ---------------------------------------------------------------------------
# Fig 1 — 신호 흐름
# ---------------------------------------------------------------------------


def fig_system(lead: dict) -> Path:
    c = Canvas(1060, 300)
    top, bot = 80, 210
    w, h = 156, 56

    def box(x, y, label, sub, colour, fill="#ffffff"):
        c.rect(x, y - h / 2, w, h, fill=fill, stroke=colour)
        c.text(x + w / 2, y - 9, label, size=14.5, weight="600", fill=colour)
        c.text(x + w / 2, y + 13, sub, size=11.5, family=MONO, fill=MUTED)

    c.rect(40, top - h / 2, 110, h, fill=PANEL, stroke=INK)
    c.text(95, top, "n(t)", size=15, family=MONO, weight="700")

    box(250, top, "P(z)", f"{lead['p']}", PHYS)
    c.line(150, top, 250, top, arrow=True)

    c.path(f"M 95 {top + h / 2} L 95 {bot} L 250 {bot}", arrow=True)
    box(250, bot, "delay", f"+{lead['lead']}", INK, PANEL)
    c.line(406, bot, 470, bot, arrow=True)
    c.text(438, bot - 15, "x_ref", size=11.5, family=MONO, fill=MUTED)

    c.rect(470, bot - 40, 200, 80, fill="#ffffff", stroke=ENCODER)
    c.text(570, bot - 11, "HybridANCNet", size=15, weight="600", fill=ENCODER)
    c.text(570, bot + 13, "causal", size=11.5, family=MONO, fill=MUTED)

    c.line(670, bot, 730, bot, arrow=True)
    c.text(700, bot - 15, "y(t)", size=11.5, family=MONO, fill=MUTED)
    box(730, bot, "S(z)", f"{lead['s']} + {lead['handoff']}", PHYS)

    sx, sy = 920, (top + bot) / 2
    c.path(f"M 406 {top} L {sx} {top} L {sx} {sy - 16}", arrow=True)
    c.path(f"M 886 {bot} L {sx} {bot} L {sx} {sy + 16}", arrow=True)
    c.text(sx - 14, top - 15, "d(t)", size=11.5, family=MONO, anchor="end", fill=MUTED)
    c.text(sx - 14, bot - 15, "S·y", size=11.5, family=MONO, anchor="end", fill=MUTED)
    c.sum_node(sx, sy)
    c.line(sx + 16, sy, sx + 52, sy, arrow=True)
    c.text(sx + 58, sy, "e(t)", size=15, family=MONO, weight="700",
           anchor="start", fill=DANGER)

    path = OUT / "fig1_system.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 2 — 전체 스택
# ---------------------------------------------------------------------------


def fig_architecture(base: dict, tiny: dict) -> Path:
    rows = [
        ("io", f"x   [B, 2, T]", None, INK, PANEL),
        ("op", f"÷ {tiny['io_scale']}", None, INK, "#ffffff"),
        ("op", f"Conv1d {tiny['win']}, stride {tiny['hop']}", "[B, 2C, T/hop]",
         ENCODER, "#ffffff"),
        ("op", "GLU", None, ENCODER, "#ffffff"),
        ("op", "ChannelLN", None, ENCODER, "#ffffff"),
        ("op", "1×1 conv", "[B, C, T/hop]", ENCODER, "#ffffff"),
        ("grp", None, None, TCN, "#ffffff"),
        ("op", f"GLSTM  G={tiny['glstm_groups']}, H={tiny['glstm_hidden']}",
         None, RECURR, "#ffffff"),
        ("dash", f"MHSA  {base['attn_heads']} heads, w={base['attn_window']}",
         None, ATTN, "#ffffff"),
        ("op", "1×1 conv", None, ENCODER, "#ffffff"),
        ("op", f"ConvT {tiny['win']}, stride {tiny['hop']}", None, ENCODER, "#ffffff"),
        ("op", f"{tiny['limit']} · tanh", None, DANGER, "#ffffff"),
        ("io", f"y   [B, 1, T]", None, INK, PANEL),
    ]

    x0, w = 320, 320
    gap, h = 20, 42
    group_h = 40 * len(tiny["dilations"]) + 34
    total = sum((group_h if kind == "grp" else h) + gap for kind, *_ in rows) + 46
    c = Canvas(960, total)

    y = 30
    for index, (kind, label, shape, colour, fill) in enumerate(rows):
        if kind == "grp":
            c.rect(x0 - 14, y, w + 28, group_h, fill="#fbfcfd", stroke=TCN,
                   dash="5 4", width=1.3)
            c.text(x0 + w + 26, y + group_h / 2, f"× {tiny['repeats']}",
                   size=13.5, weight="700", anchor="start", fill=TCN)
            for k, d in enumerate(tiny["dilations"]):
                by = y + 17 + k * 40
                c.rect(x0, by, w, 32, fill="#ffffff", stroke=TCN)
                c.text(x0 + w / 2, by + 16, f"TCN Block,  d = {d}",
                       size=13, weight="600", fill=TCN)
                if k < len(tiny["dilations"]) - 1:
                    c.line(x0 + w / 2, by + 32, x0 + w / 2, by + 40, arrow=True)
            c.text(x0 + w + 76, y + group_h / 2, "[B, C, T/hop]", size=12,
                   family=MONO, anchor="start", fill=MUTED)
            step = group_h
        else:
            c.rect(x0, y, w, h, fill=fill, stroke=colour,
                   dash="6 4" if kind == "dash" else None)
            c.text(x0 + w / 2, y + h / 2, label, size=13.5, weight="700",
                   family=MONO if kind == "io" else None, fill=colour)
            if shape:
                c.text(x0 + w + 76, y + h / 2, shape, size=12, family=MONO,
                       anchor="start", fill=MUTED)
            step = h
        if index < len(rows) - 1:
            c.line(x0 + w / 2, y + step, x0 + w / 2, y + step + gap, arrow=True)
        y += step + gap

    path = OUT / "fig2_architecture.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 3 — TCN 잔차 블록
# ---------------------------------------------------------------------------


def fig_tcn_block(tiny: dict) -> Path:
    """src/deep_anc/models/tcn_blocks.py TCNBlock 을 그대로 그린다.

    실제 forward:
        y = norm(act(expand(x)))          # 1x1 -> PReLU -> ChannelLN
        u = dw_main(pad(y)); g = sigmoid(dw_gate(pad(y)))
        return x + project(u * g)         # residual 하나뿐. skip 분기 없음.
    """
    stack = [
        f"1×1 conv,  {tiny['tcn_hidden']}",
        "PReLU",
        "ChannelLN",
    ]
    x, w, h, gap = 340, 280, 40, 18
    c = Canvas(900, 56 + len(stack) * (h + gap) + 300)

    y = 46
    c.text(x + w / 2, y - 22, "x", size=15, family=MONO, weight="700")
    c.circle(x + w / 2, y, 5, fill=INK, stroke=INK)
    identity_y = y
    c.line(x + w / 2, y, x + w / 2, y + gap, arrow=True)
    y += gap

    for label in stack:
        c.rect(x, y, w, h, fill="#ffffff", stroke=TCN)
        c.text(x + w / 2, y + h / 2, label, size=13, weight="600", fill=TCN)
        c.line(x + w / 2, y + h, x + w / 2, y + h + gap, arrow=True)
        y += h + gap

    # 두 갈래 depthwise dilated conv (주경로 / 게이트) — 곱해서 GLU
    c.circle(x + w / 2, y, 5, fill=INK, stroke=INK)
    split = y
    bw = 210
    left_cx, right_cx = x + w / 2 - 130, x + w / 2 + 130

    for cx, label, sub in (
        (left_cx, f"D-Conv {tiny['kernel']}, dil d", "dw_main → u"),
        (right_cx, f"D-Conv {tiny['kernel']}, dil d", "dw_gate → σ(g)"),
    ):
        c.path(f"M {x + w / 2} {split} L {cx} {split} L {cx} {split + 30}", arrow=True)
        c.rect(cx - bw / 2, split + 30, bw, h, fill="#ffffff", stroke=TCN)
        c.text(cx, split + 30 + h / 2, label, size=12.5, weight="600", fill=TCN)
        c.text(cx, split + 30 + h + 18, sub, size=11.5, family=MONO, fill=MUTED)

    mul_y = split + 30 + h + 46
    for cx in (left_cx, right_cx):
        c.path(f"M {cx} {split + 30 + h + 26} L {cx} {mul_y - 14} "
               f"L {x + w / 2} {mul_y - 14}", arrow=(cx == right_cx))
    c.circle(x + w / 2, mul_y, 15, fill="#ffffff", stroke=INK)
    c.text(x + w / 2, mul_y, "×", size=17, weight="700", fill=INK)

    proj_y = mul_y + 40
    c.line(x + w / 2, mul_y + 15, x + w / 2, proj_y, arrow=True)
    c.rect(x + w / 2 - bw / 2, proj_y, bw, h, fill="#ffffff", stroke=TCN)
    c.text(x + w / 2, proj_y + h / 2, "1×1 conv (project)", size=13,
           weight="600", fill=TCN)

    add_y = proj_y + h + 46
    c.line(x + w / 2, proj_y + h, x + w / 2, add_y - 16, arrow=True)
    c.sum_node(x + w / 2, add_y)
    c.path(f"M {x + w / 2} {identity_y} L {x + w + 150} {identity_y} "
           f"L {x + w + 150} {add_y} L {x + w / 2 + 16} {add_y}",
           dash="5 4", arrow=True)
    c.text(x + w + 162, identity_y - 16, "residual (identity)", size=11.5,
           anchor="start", fill=MUTED)
    c.line(x + w / 2, add_y + 16, x + w / 2, add_y + 40, arrow=True)
    c.text(x + w / 2, add_y + 56, "out  (skip 분기 없음)", size=13.5,
           family=MONO, fill=MUTED)

    path = OUT / "fig3_tcn_block.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 4 — dilated causal 수용영역
# ---------------------------------------------------------------------------


def fig_receptive_field(tiny: dict) -> Path:
    dilations = tiny["dilations"]
    n = 1 + sum((tiny["kernel"] - 1) * d for d in dilations)
    rows = len(dilations) + 1
    c = Canvas(1000, 40 + 88 * rows)
    left, right = 110, 950
    step = (right - left) / (n - 1)
    base_y = 40

    def px(i: float) -> float:
        return left + i * step

    def py(layer: int) -> float:
        return base_y + (rows - 1 - layer) * 88

    active = [{n - 1}]
    for d in reversed(dilations):
        previous = set()
        for node in active[-1]:
            for tap in range(tiny["kernel"]):
                previous.add(node - tap * d)
        active.append({v for v in previous if v >= 0})
    active = list(reversed(active))

    labels = ["input"] + [f"d = {d}" for d in dilations]
    for layer in range(rows):
        y = py(layer)
        c.text(92, y, labels[layer], size=13, anchor="end", family=MONO,
               fill=INK if layer else MUTED)
        for index in range(n):
            on = index in active[layer]
            c.circle(px(index), y, 5.5 if on else 3.0,
                     fill=TCN if on else "#ffffff",
                     stroke=TCN if on else "#dfe4ea", width=1.3)

    for layer, d in enumerate(dilations):
        for node in sorted(active[layer + 1]):
            for tap in range(tiny["kernel"]):
                source = node - tap * d
                if source >= 0:
                    c.line(px(source), py(layer) - 6, px(node), py(layer + 1) + 6,
                           stroke=TCN, width=1.1, opacity=0.45)

    out_y = py(rows - 1)
    c.circle(px(n - 1), out_y, 8, fill=ATTN, stroke=ATTN)
    c.line(px(n - 1) + 17, base_y - 24, px(n - 1) + 17, out_y + 24,
           stroke=DANGER, dash="4 4", width=1.4)
    c.text(px(n - 1) + 25, (base_y + out_y) / 2, "t", size=14, family=MONO,
           anchor="start", fill=DANGER, weight="700")

    path = OUT / "fig4_receptive_field.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 5 — Grouped LSTM
# ---------------------------------------------------------------------------


def fig_glstm(base: dict) -> Path:
    groups = max(2, base["glstm_groups"])
    gap, h = 76, 44
    span = gap * (groups - 1) + h
    c = Canvas(1000, 100 + span + 80)
    x_in, x_split, x_lstm, x_shuf, x_out = 60, 250, 460, 700, 880
    y0 = 50
    mid = y0 + span / 2

    c.rect(x_in, y0, 100, span, fill=PANEL, stroke=INK)
    c.text(x_in + 50, mid, "C", size=16, family=MONO, weight="700")

    for g in range(groups):
        y = y0 + g * gap
        c.line(x_in + 100, mid, x_split, y + h / 2, arrow=True)
        c.rect(x_split, y, 140, h, fill="#ffffff", stroke=RECURR)
        c.text(x_split + 70, y + h / 2, "C / G", size=13.5, family=MONO, fill=RECURR)
        c.line(x_split + 140, y + h / 2, x_lstm, y + h / 2, arrow=True)
        c.rect(x_lstm, y, 170, h, fill="#f8f3fb", stroke=RECURR)
        c.text(x_lstm + 85, y + h / 2, f"LSTM   H = {base['glstm_hidden']}",
               size=13, weight="600", fill=RECURR)
        c.line(x_lstm + 170, y + h / 2, x_shuf, y + h / 2, arrow=True)

    c.rect(x_shuf, y0, 140, span, fill="#ffffff", stroke=RECURR, dash="6 4")
    c.text(x_shuf + 70, mid - 10, "channel", size=14, weight="600", fill=RECURR)
    c.text(x_shuf + 70, mid + 11, "shuffle", size=14, weight="600", fill=RECURR)
    c.line(x_shuf + 140, mid, x_out, mid, arrow=True)
    c.rect(x_out, y0, 100, span, fill=PANEL, stroke=INK)
    c.text(x_out + 50, mid, "C", size=16, family=MONO, weight="700")

    hy = y0 + span + 30
    c.rect(x_lstm - 5, hy, 180, 38, fill="#fdfaf4", stroke=RECURR, dash="5 4")
    c.text(x_lstm + 85, hy + 19, "h, c  →  t+1", size=13, family=MONO,
           weight="600", fill=RECURR)
    for g in range(groups):
        c.path(f"M {x_lstm + 85} {y0 + g * gap + h} L {x_lstm + 85} {hy}",
               stroke=RECURR, dash="4 3", width=1.1, opacity=0.5)

    path = OUT / "fig5_glstm.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fig 6 — 스트리밍 상태 I/O
# ---------------------------------------------------------------------------


def fig_streaming(tiny: dict) -> Path:
    c = Canvas(1000, 300)
    w, h, y = 160, 46, 60

    c.rect(70, y, w, h, fill=PANEL, stroke=INK)
    c.text(70 + w / 2, y + h / 2, "x[t]", size=15, family=MONO, weight="700")
    c.rect(70, y + 74, w, h, fill=PANEL, stroke=MUTED)
    c.text(70 + w / 2, y + 74 + h / 2, "s[t−1]", size=15, family=MONO,
           weight="700", fill=MUTED)

    c.line(230, y + h / 2, 340, y + 44, arrow=True)
    c.line(230, y + 74 + h / 2, 340, y + 76, arrow=True)

    c.rect(340, y + 4, 240, 112, fill="#ffffff", stroke=ENCODER)
    c.text(460, y + 60, "HybridANCNet", size=15.5, weight="600", fill=ENCODER)

    c.line(580, y + 44, 690, y + h / 2, arrow=True)
    c.line(580, y + 76, 690, y + 74 + h / 2, arrow=True)
    c.rect(690, y, w, h, fill="#ffffff", stroke=INK)
    c.text(690 + w / 2, y + h / 2, "y[t]", size=15, family=MONO, weight="700")
    c.rect(690, y + 74, w, h, fill=PANEL, stroke=MUTED)
    c.text(690 + w / 2, y + 74 + h / 2, "s[t]", size=15, family=MONO,
           weight="700", fill=MUTED)

    c.path(f"M 850 {y + 74 + h / 2} L 920 {y + 74 + h / 2} L 920 {y + 186} "
           f"L 150 {y + 186} L 150 {y + 74 + h}", dash="5 4", arrow=True)

    names = [("st_enc", ENCODER),
             (f"st_0_tcn … st_{tiny['blocks'] - 1}_tcn", TCN),
             ("st_lstm_h,  st_lstm_c", RECURR),
             ("st_dec", ENCODER)]
    sy = y + 218
    c.text(70, sy, "s =", size=13.5, family=MONO, anchor="start", fill=MUTED)
    sx = 118
    for label, colour in names:
        width = 24 + len(label) * 7.4
        c.rect(sx, sy - 17, width, 34, fill="#ffffff", stroke=colour, width=1.3)
        c.text(sx + width / 2, sy, label, size=12, family=MONO, fill=colour)
        sx += width + 14

    path = OUT / "fig6_streaming.svg"
    path.write_text(c.render(), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base, tiny = model_facts("base"), model_facts("tiny")

    duct = load_yaml(REPO / "configs" / "duct.yaml")
    data = load_yaml(REPO / "configs" / "data_sim.yaml")
    import numpy as np

    secondary = np.load(REPO / duct["secondary_path"]["npz"], allow_pickle=False)
    s_delay = int(np.asarray(secondary["delay_samples"]).reshape(-1)[0])
    handoff = int(duct["secondary_path"]["handoff_extra_samples"])
    lead_samples = int(data["digital_reference_lead_samples"])
    lead = {"s": s_delay, "handoff": handoff, "lead": lead_samples,
            "p": s_delay + handoff - lead_samples}

    for path in (
        fig_system(lead),
        fig_architecture(base, tiny),
        fig_tcn_block(tiny),
        fig_receptive_field(tiny),
        fig_glstm(base),
        fig_streaming(tiny),
    ):
        print(f"  {path.relative_to(REPO)}  ({path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
