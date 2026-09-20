"""Differentiable, causal full-length FIR, with an FFT training implementation."""

import torch
from torch import nn
from torch.nn import functional as F

from .calibration import load_secondary_path


class CausalSecondaryPath(nn.Module):
    """Map physical DAC command u to error-microphone contribution S*u.

    The measured path already contains its original delay. ``delay_samples`` is
    only ADDITIONAL, independently measured processing/transport latency.
    FFT zero padding computes linear convolution, never circular convolution.
    """

    def __init__(self, coefficients, delay_samples=0, method="fft"):
        super().__init__()
        if isinstance(delay_samples, bool) or not isinstance(delay_samples, int) or delay_samples < 0:
            raise ValueError("delay_samples must be a nonnegative integer")
        if method not in {"fft", "direct"}:
            raise ValueError("method must be fft or direct")
        h = torch.as_tensor(coefficients, dtype=torch.float64)
        if h.ndim != 1 or h.numel() == 0 or not torch.isfinite(h).all() or not torch.any(h != 0):
            raise ValueError("coefficients must be a finite nonzero one-dimensional FIR")
        self.register_buffer("coefficients", F.pad(h, (delay_samples, 0)))
        self.delay_samples = delay_samples
        self.method = method

    @property
    def history_samples(self):
        return self.coefficients.numel() - 1

    def forward(self, command):
        if command.ndim != 2 or command.shape[-1] == 0 or not command.is_floating_point():
            raise ValueError("command must be a floating tensor [batch, samples]")
        # FFT and calibrated gain should remain in at least float32 under AMP.
        value = command if command.dtype in (torch.float32, torch.float64) else command.float()
        h = self.coefficients.to(device=value.device, dtype=value.dtype)
        if self.method == "direct":
            return F.conv1d(F.pad(value.unsqueeze(1), (h.numel() - 1, 0)), h.flip(0)[None, None, :]).squeeze(1)
        full_length = value.shape[-1] + h.numel() - 1
        fft_length = 1 << (full_length - 1).bit_length()
        spectrum = torch.fft.rfft(value, n=fft_length) * torch.fft.rfft(h, n=fft_length)
        return torch.fft.irfft(spectrum, n=fft_length)[..., :value.shape[-1]]

    def filter_chunk(self, command, state=None):
        """Reference stateful FIR for validating chunk boundaries, not audio I/O.

        Returns (filtered block, preceding-command history for the next block).
        History is not detached: callers can choose their own gradient boundary.
        """
        if command.ndim != 2 or command.shape[-1] == 0 or not command.is_floating_point():
            raise ValueError("command must be a floating tensor [batch, samples]")
        count = self.history_samples
        if state is None:
            state = command.new_zeros((command.shape[0], count))
        if state.shape != (command.shape[0], count) or state.device != command.device or state.dtype != command.dtype:
            raise ValueError("Invalid FIR history shape, device or dtype")
        joined = torch.cat((state, command), dim=-1)
        result = self(joined)[..., count:]
        return result, joined[..., -count:] if count else joined[..., :0]
