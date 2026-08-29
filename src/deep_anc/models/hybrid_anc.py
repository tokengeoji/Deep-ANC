"""HybridANCNet — 시간영역 인과 하이브리드 ANC 모델.

구성 (docs/04_model_architecture.md):
  Conv-TasNet 학습형 인코더/디코더(k=384, hop=128, block-acquired 인과 경로)
  + WaveNet dilated causal depthwise TCN (GLU 게이팅)
  + GCRN GLSTM 병목 (그룹 LSTM)
  + windowed causal MHSA 1층 (주기 잡음 재조회)
  + 소프트 리미터 (control limit 0.2 정합)

입력  [B, 2, T]: ch0 = 레퍼런스, ch1 = 에러(피드백) — 채널 dropout 증강으로
                 ref-only / err-only 운용도 같은 인터페이스로 커버.
출력  [B, 1, T]: 상쇄 스피커 구동 신호 y.

인과성: 인코더는 좌측 win-hop 패딩을 쓰지만 한 frame의 첫 출력 sample은 같은 hop의
마지막 입력까지 최대 +127 samples를 본다. 따라서 sample-zero-lookahead 모델은 아니다.
실시간 경로는 256-sample 입력 block 전체를 받은 뒤 출력하며, 이 block handoff를 P/S
timing contract가 정확히 한 번 포함한다. 디코더 overlap-add는 과거 frame tail만 더한다.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import WindowedCausalMHSA
from .glstm import GLSTM
from .tcn_blocks import ChannelLayerNorm, TCNBlock


class HybridANCNet(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.hop = int(cfg["hop"])
        self.win = int(cfg["win"])
        self.in_channels = int(cfg.get("in_channels", 2))
        channels = int(cfg["encoder"]["channels"])
        self.channels = channels
        self.limit = float(cfg.get("limiter", {}).get("limit", 0.2))
        self.register_buffer(
            "io_scale", torch.tensor(float(cfg.get("io_scale", 0.02)), dtype=torch.float32)
        )

        if self.win <= self.hop:
            raise ValueError("win 은 hop 보다 커야 합니다")
        self.context = self.win - self.hop  # 인코더 좌측 히스토리 / 디코더 꼬리 길이

        # 인코더: Conv1d → GLU
        self.encoder = nn.Conv1d(self.in_channels, 2 * channels, self.win, stride=self.hop, bias=False)
        self.enc_norm = ChannelLayerNorm(channels)
        self.enc_proj = nn.Conv1d(channels, channels, 1)

        # 본체: TCN 반복 사이에 GLSTM/MHSA 삽입 (순서가 스트리밍 상태 순서를 결정)
        tcn_cfg = cfg["tcn"]
        glstm_cfg = cfg.get("glstm", {}) or {}
        attn_cfg = cfg.get("attention", {}) or {}
        blocks: list[nn.Module] = []
        for repeat in range(1, int(tcn_cfg["repeats"]) + 1):
            for dilation in tcn_cfg["dilations"]:
                blocks.append(
                    TCNBlock(
                        channels=channels,
                        hidden=int(tcn_cfg["hidden"]),
                        kernel=int(tcn_cfg.get("kernel", 3)),
                        dilation=int(dilation),
                    )
                )
            if glstm_cfg and repeat == int(glstm_cfg.get("insert_after_repeat", -1)):
                blocks.append(
                    GLSTM(
                        channels=channels,
                        groups=int(glstm_cfg["groups"]),
                        hidden_per_group=int(glstm_cfg["hidden_per_group"]),
                    )
                )
            if attn_cfg.get("enabled", False) and repeat == int(attn_cfg.get("insert_after_repeat", -1)):
                blocks.append(
                    WindowedCausalMHSA(
                        channels=channels,
                        heads=int(attn_cfg["heads"]),
                        head_dim=int(attn_cfg["head_dim"]),
                        window_frames=int(attn_cfg["window_frames"]),
                    )
                )
        self.blocks = nn.ModuleList(blocks)

        # 헤드 + 디코더
        head_channels = 2 * channels
        self.head = nn.Conv1d(channels, head_channels, 1)
        self.head_act = nn.PReLU()
        self.decoder = nn.ConvTranspose1d(head_channels, 1, self.win, stride=self.hop, bias=False)

    # ---------- 공용 ----------

    def _soft_limit(self, y: torch.Tensor) -> torch.Tensor:
        return self.limit * torch.tanh(y / self.limit)

    def _encode(self, x_padded: torch.Tensor) -> torch.Tensor:
        h = F.glu(self.encoder(x_padded), dim=1)
        return self.enc_proj(self.enc_norm(h))

    # ---------- 오프라인 (학습) ----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_ch, T] (T 는 hop 배수) → y: [B, 1, T]."""
        if x.shape[-1] % self.hop != 0:
            raise ValueError(f"입력 길이 {x.shape[-1]} 는 hop {self.hop} 의 배수여야 합니다")
        x = x / self.io_scale
        h = self._encode(F.pad(x, (self.context, 0)))
        for block in self.blocks:
            h = block(h)
        h = self.head_act(self.head(h))
        y = self.decoder(h)[..., : x.shape[-1]]
        y = y * self.io_scale
        return self._soft_limit(y)

    # ---------- 스트리밍 ----------

    def init_states(self, batch: int = 1, device: torch.device | str = "cpu") -> list:
        """상태 목록 (블록 순서 고정): [enc_hist, block states..., dec_tail]."""
        device = torch.device(device)
        states: list = [torch.zeros(batch, self.in_channels, self.context, device=device)]
        for block in self.blocks:
            if isinstance(block, TCNBlock):
                states.append(block.init_state(batch, device))
            elif isinstance(block, GLSTM):
                states.append(block.init_state(batch, device))       # (h, c)
            elif isinstance(block, WindowedCausalMHSA):
                states.append(block.init_state(batch, device))       # (k, v)
            else:  # pragma: no cover
                raise TypeError(f"알 수 없는 블록: {type(block)}")
        states.append(torch.zeros(batch, 1, self.context, device=device))
        return states

    def streaming_step(self, x_block: torch.Tensor, states: list) -> tuple[torch.Tensor, list]:
        """x_block: [B, in_ch, N] (N = k·hop, N ≥ context 필요) → y: [B, 1, N].

        오프라인 forward 와 수치 등가 (GLSTM 은 수동 셀 경로 사용).
        """
        n = x_block.shape[-1]
        if n % self.hop != 0:
            raise ValueError("블록 길이는 hop 의 배수여야 합니다")
        if n < self.context:
            raise ValueError(f"블록 길이 {n} 는 context {self.context} 이상이어야 합니다")

        x_block = x_block / self.io_scale
        new_states: list = []

        enc_hist = states[0]
        enc_in = torch.cat([enc_hist, x_block], dim=-1)
        new_states.append(enc_in[..., enc_in.shape[-1] - self.context :])
        h = self._encode(enc_in)

        idx = 1
        for block in self.blocks:
            state = states[idx]
            if isinstance(block, TCNBlock):
                h, s = block.streaming_forward(h, state)
                new_states.append(s)
            elif isinstance(block, GLSTM):
                h, h_s, c_s = block.streaming_forward(h, state[0], state[1])
                new_states.append((h_s, c_s))
            else:  # WindowedCausalMHSA
                h, k_s, v_s, m_s = block.streaming_forward(h, state[0], state[1], state[2])
                new_states.append((k_s, v_s, m_s))
            idx += 1

        h = self.head_act(self.head(h))
        conv_out = self.decoder(h)                      # [B, 1, (k-1)·hop + win]
        tail = states[idx]
        out = conv_out[..., :n].clone()
        out[..., : self.context] = out[..., : self.context] + tail
        rem = conv_out[..., n:]                          # 길이 win - hop ... 다음 블록 꼬리
        new_tail = F.pad(rem, (0, self.context - rem.shape[-1]))
        new_states.append(new_tail)

        y = out * self.io_scale
        return self._soft_limit(y), new_states


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(model_cfg: dict[str, Any]) -> HybridANCNet:
    return HybridANCNet(model_cfg)
