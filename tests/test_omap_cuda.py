"""OMAP 16 kHz CUDA 합성 회귀: 오디오·실측 감쇠·하드웨어 지연 검증이 아니다."""

import json
import math

import pytest
import torch

from deepanc.calibration import DEFAULT_RIR, ROOT, load_secondary_path
from deepanc.model import CausalController
from deepanc.secondary_path import CausalSecondaryPath
from deepanc.train import file_sha256, main, restore_checkpoint


RIR_SHA256 = "9ece235a68a78a7f7c702d49d75525974df2ac2a8a9ca42ced155af901bcf9e3"


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("실제 CUDA가 없어 OMAP GPU 회귀를 생략함: CPU 대체 실행 아님")
    device = torch.device("cuda", torch.cuda.current_device())
    # 비교 오차에 TF32 근사 연산이 섞이지 않게 하며, 종료 시 설정을 복원한다.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        with torch.backends.cudnn.flags(
            enabled=True, benchmark=False, deterministic=True, allow_tf32=False
        ):
            yield device


@pytest.fixture
def measured_rir():
    assert file_sha256(DEFAULT_RIR) == RIR_SHA256
    coefficients = load_secondary_path(DEFAULT_RIR, sample_rate=16000)
    assert coefficients.shape == (500,)
    return coefficients


def test_cuda_measured_rir_fft_matches_direct_output_and_input_gradient(cuda_device, measured_rir):
    generator = torch.Generator(device=cuda_device).manual_seed(109)
    reference = torch.randn(2, 769, device=cuda_device, generator=generator) * 0.05
    probe = torch.randn(reference.shape, device=cuda_device, generator=generator)
    # 7은 경계/패딩 회귀용 임의 합성 지연이며, OMAP/Jetson 측정값이 아니다.
    for synthetic_delay in (0, 7):
        outputs, gradients = [], []
        for method in ("fft", "direct"):
            secondary = CausalSecondaryPath(
                measured_rir, delay_samples=synthetic_delay, method=method
            ).to(cuda_device)
            assert secondary.coefficients.is_cuda
            torch.testing.assert_close(
                secondary.coefficients[synthetic_delay:].cpu(),
                torch.from_numpy(measured_rir), atol=0, rtol=0,
            )
            assert torch.count_nonzero(secondary.coefficients[:synthetic_delay]) == 0
            command = reference.clone().requires_grad_()
            output = secondary(command)
            gradient, = torch.autograd.grad(output, command, grad_outputs=probe)
            assert output.device == gradient.device == cuda_device
            assert output.dtype == gradient.dtype == torch.float32
            assert torch.isfinite(output).all() and torch.isfinite(gradient).all()
            outputs.append(output)
            gradients.append(gradient)
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(gradients[0], gradients[1], atol=1e-6, rtol=2e-5)


def test_cuda_chunk_state_preserves_full_output_and_cross_boundary_gradient(cuda_device, measured_rir):
    # 추가 7샘플은 합성 조건이다. 원본 RIR의 선행 지연·극성·이득은 보존한다.
    secondary = CausalSecondaryPath(measured_rir, delay_samples=7).to(cuda_device)
    generator = torch.Generator(device=cuda_device).manual_seed(211)
    reference = torch.randn(2, 768, device=cuda_device, generator=generator) * 0.05
    whole_command = reference.clone().requires_grad_()
    stream_command = reference.clone().requires_grad_()
    expected = secondary(whole_command)
    state, parts, start = None, [], 0
    zeros = reference.new_zeros(reference.shape[0], secondary.history_samples)
    # 1샘플 청크 및 FIR보다 짧은 청크를 포함해 초기 padding과 누적 state를 검사한다.
    for size in (31, 127, 1, 256, 353):
        stop = start + size
        part, state = secondary.filter_chunk(stream_command[:, start:stop], state)
        assert state.shape == (reference.shape[0], secondary.history_samples)
        assert state.device == cuda_device and state.dtype == reference.dtype
        assert state.requires_grad
        expected_state = torch.cat((zeros, stream_command[:, :stop]), dim=1)
        torch.testing.assert_close(
            state, expected_state[:, -secondary.history_samples:], atol=0, rtol=0
        )
        parts.append(part)
        start = stop
    assert start == reference.shape[1]
    actual = torch.cat(parts, dim=1)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    probe = torch.zeros_like(reference)
    # 마지막 청크 손실만 사용: state를 detach하면 앞 청크의 gradient가 유실된다.
    last_boundary = reference.shape[1] - 353
    probe[:, last_boundary:] = torch.randn(
        2, 353, device=cuda_device, generator=generator
    )
    expected_gradient, = torch.autograd.grad(expected, whole_command, grad_outputs=probe)
    actual_gradient, = torch.autograd.grad(actual, stream_command, grad_outputs=probe)
    assert torch.count_nonzero(expected_gradient[:, :last_boundary]) > 0
    assert torch.isfinite(actual_gradient).all()
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-6, rtol=2e-5)


def test_cuda_synthetic_training_checkpoint_resume_and_optimizer_devices(cuda_device, measured_rir, tmp_path):
    # 임시 config/output만 생성하며 측정 자산·공식 학습 설정은 수정하지 않는다.
    config = json.loads((ROOT / "configs/anc_train.json").read_text(encoding="utf-8"))
    config.update(secondary_path=str(DEFAULT_RIR), secondary_delay_samples=0,
                  chunk_samples=128, batch_size=2, num_workers=0)
    config["model"] = {"channels": 3, "kernel_size": 3, "dilations": [1, 2], "max_output": 0.1}
    config_path = tmp_path / "synthetic-cuda-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "synthetic-cuda-run"
    arguments = ["--config", str(config_path), "--output", str(output),
                 "--device", "cuda", "--smoke-test"]

    torch.manual_seed(config["seed"])
    initial_model = CausalController(**config["model"]).to(cuda_device)
    initial = {name: value.detach().clone() for name, value in initial_model.state_dict().items()}
    assert main(arguments) == 0
    # map_location 없이 읽어 저장 당시 model/moment의 실제 CUDA 배치를 검증한다.
    first = torch.load(output / "last.pt", weights_only=True)
    assert first["epoch"] == 1 and first["synthetic"] is True
    assert first["rir_sha256"] == RIR_SHA256
    assert first["cuda_rng_state"]
    assert all(value.device == cuda_device and torch.isfinite(value).all()
               for value in first["model"].values())
    assert any(not torch.equal(initial[name], value) for name, value in first["model"].items())

    restored_model = CausalController(**config["model"]).to(cuda_device)
    optimizer = torch.optim.Adam(restored_model.parameters(), lr=config["learning_rate"])
    epoch, best = restore_checkpoint(
        output / "last.pt", restored_model, optimizer, first["signature"], cuda_device
    )
    assert epoch == 1 and best == first["best_validation_loss"] and math.isfinite(best)
    assert len(optimizer.state) == len(list(restored_model.parameters()))
    for name, parameter in restored_model.named_parameters():
        assert parameter.device == cuda_device
        torch.testing.assert_close(parameter, first["model"][name], atol=0, rtol=0)
    for parameter, state in optimizer.state.items():
        assert parameter.device == cuda_device
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].device == cuda_device
            assert torch.isfinite(state[name]).all()
        # 非 capturable Adam의 step은 CPU가 정상이며 moment만 CUDA로 복원한다.
        assert state["step"].device.type == "cpu" and state["step"].item() == 2
    assert any(torch.count_nonzero(state["exp_avg"]) > 0 for state in optimizer.state.values())

    assert main(arguments + ["--resume", str(output / "last.pt"), "--epochs", "2"]) == 0
    second = torch.load(output / "last.pt", weights_only=True)
    assert second["epoch"] == 2 and second["synthetic"] is True
    assert second["signature"] == first["signature"]
    assert any(not torch.equal(value, second["model"][name]) for name, value in first["model"].items())
    assert all(value.device == cuda_device and torch.isfinite(value).all()
               for value in second["model"].values())
    for state in second["optimizer"]["state"].values():
        assert state["step"].device.type == "cpu" and state["step"].item() == 4
        for name in ("exp_avg", "exp_avg_sq"):
            assert state[name].device == cuda_device and torch.isfinite(state[name]).all()
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [entry["epoch"] for entry in metrics] == [1, 2]
    for entry in metrics:
        assert entry["synthetic"] is True
        for split, samples in (("train", 512), ("valid", 256)):
            assert entry[split]["samples"] == samples
            assert 0 < entry[split]["output_peak"] <= config["model"]["max_output"]
            assert all(math.isfinite(value) for value in entry[split].values())
    info = json.loads((output / "run.json").read_text())
    assert info["device"] == "cuda" and info["synthetic"] is True
    assert info["context_samples"] == initial_model.receptive_field - 1 + len(measured_rir)
    assert "NOT measured hardware ANC performance" in info["data_source"]
    assert (output / "best.pt").is_file()
    assert file_sha256(DEFAULT_RIR) == RIR_SHA256
