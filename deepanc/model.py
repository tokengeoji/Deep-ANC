"""Small causal time-domain ANC baseline, not a reproduction of a paper."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def forward(self, value):
        history = (self.kernel_size[0] - 1) * self.dilation[0]
        return super().forward(F.pad(value, (history, 0)))


class CausalController(nn.Module):
    """Map past/current reference samples to a bounded digital DAC command.

    No temporal normalization or future samples are used. Streaming deployment
    still requires preserved history, measured latency, and hardware validation.
    """

    def __init__(self, channels=16, kernel_size=7, dilations=(1, 2, 4, 8), max_output=0.1):
        super().__init__()
        if channels < 1 or kernel_size < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError("Controller dimensions/dilations must be positive")
        if not 0 < max_output <= 1:
            raise ValueError("max_output must lie in (0, 1] digital full scale")
        self.max_output = float(max_output)
        self.receptive_field = kernel_size + sum(2 * dilation for dilation in dilations)
        self.input = CausalConv1d(1, channels, kernel_size, bias=False)
        self.blocks = nn.ModuleList([
            CausalConv1d(channels, channels, 3, dilation=dilation, bias=False)
            for dilation in dilations
        ])
        self.output = nn.Conv1d(channels, 1, 1, bias=False)
        # Begin at almost zero DAC output while retaining gradient to all layers.
        nn.init.normal_(self.output.weight, std=0.001)

    def forward(self, reference):
        if reference.ndim != 2:
            raise ValueError("reference must have shape [batch, time]")
        value = torch.tanh(self.input(reference.unsqueeze(1)))
        for block in self.blocks:
            value = value + torch.tanh(block(value))
        return self.max_output * torch.tanh(self.output(value).squeeze(1))
