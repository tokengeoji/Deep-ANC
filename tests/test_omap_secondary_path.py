import numpy as np
import pytest
import torch

from deepanc.calibration import DEFAULT_RIR, ROOT, load_secondary_path, read_coefficients
from deepanc.secondary_path import CausalSecondaryPath


def test_authoritative_measurement_matches_successful_controller():
    h = load_secondary_path(DEFAULT_RIR)
    assert len(h) == 500
    assert np.argmax(np.abs(h)) == 49
    np.testing.assert_array_equal(h, read_coefficients(ROOT / "firmware/omap_l138/FxNLMS0/ISR.c"))
    with pytest.raises(ValueError, match="16000 Hz"):
        load_secondary_path(DEFAULT_RIR, sample_rate=48000)


@pytest.mark.parametrize("method", ["direct", "fft"])
@pytest.mark.parametrize("delay", [0, 7])
def test_physical_impulse_and_no_circular_wrap(method, delay):
    h = load_secondary_path(DEFAULT_RIR)
    value = torch.zeros(1, 1024, dtype=torch.float64)
    value[0, 10] = 1
    expected = np.convolve(value[0].numpy(), np.pad(h, (delay, 0)))[:1024]
    result = CausalSecondaryPath(h, delay_samples=delay, method=method)(value)
    np.testing.assert_allclose(result[0].numpy(), expected, atol=1e-14)
    value.zero_()
    value[0, -1] = 1
    result = CausalSecondaryPath(h, delay_samples=delay, method=method)(value)
    np.testing.assert_allclose(result[0, :-1].numpy(), 0, atol=1e-14)


def test_chunked_filter_keeps_calibrated_history():
    torch.manual_seed(19)
    value = torch.randn(2, 1400, dtype=torch.float64)
    filt = CausalSecondaryPath(load_secondary_path(DEFAULT_RIR), delay_samples=3)
    expected = filt(value)
    chunks, state = [], None
    for block in value.split(127, dim=-1):
        output, state = filt.filter_chunk(block, state)
        chunks.append(output)
    torch.testing.assert_close(torch.cat(chunks, dim=-1), expected, atol=1e-12, rtol=1e-12)


def test_gradient_matches_direct_convolution_and_cancellation_sign():
    torch.manual_seed(41)
    value = torch.randn(2, 512, dtype=torch.float64, requires_grad=True)
    h = load_secondary_path(DEFAULT_RIR)
    fft = CausalSecondaryPath(h)(value)
    direct = CausalSecondaryPath(h, method="direct")(value)
    gradient_fft = torch.autograd.grad(fft.square().sum(), value, retain_graph=True)[0]
    gradient_direct = torch.autograd.grad(direct.square().sum(), value)[0]
    torch.testing.assert_close(gradient_fft, gradient_direct, atol=1e-12, rtol=1e-12)
    # If d = S*x, the physical command -x cancels; +x doubles pressure.
    torch.testing.assert_close(fft.detach() + CausalSecondaryPath(h)(-value.detach()), torch.zeros_like(fft), atol=1e-12, rtol=0)


def test_parser_does_not_read_dimensions_comments_or_other_arrays(tmp_path):
    path = tmp_path / "coeff.c"
    path.write_text("// 16000 Hz\nfloat S_hat[2] = {1.0f, -2e-2f}; float other[1] = {999};")
    np.testing.assert_array_equal(read_coefficients(path), [1.0, -0.02])
    path.write_text("float S_hat[2] = {1.0f, BROKEN};")
    with pytest.raises(ValueError, match="numeric"):
        read_coefficients(path)
