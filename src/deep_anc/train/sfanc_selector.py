"""완료된 REF 창으로 사전 FIR 후보를 고르는 오프라인 연구용 선택기.

SFANC의 사전 필터 선택 구상을 사용하는 최소 구현이며 원논문 재현이나 실기
배포 모델이 아니다. 후보별 비용에는 학습용 ERR을 사용할 수 있지만 추론 입력은
REF뿐이다. 창 종료 뒤의 구간에 선택을 적용하고 출처별 train/val/test를 분리하는
책임은 호출자에게 있다. 이 모듈은 테스트 자료, 플랜트, 오디오 장치에 접근하지 않는다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.signal import welch
import torch
from torch import nn
from torch.nn import functional as F


_MAX_FFT = 16_384
_MAX_ELEMENTS = 64_000_000
_MAX_WINDOWS = 1_000_000
_MAX_CANDIDATES = 256
_LOG_FLOOR = 1e-12


def _integer(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return int(value)


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not array.size:
        raise ValueError(f"{name} must be a nonempty real numeric array")
    if array.size > _MAX_ELEMENTS:
        raise ValueError(f"{name} exceeds {_MAX_ELEMENTS} elements")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def reference_features(reference: np.ndarray, n_fft: int = 512) -> np.ndarray:
    """REF [N]/[B,N] -> float32 [B,2,n_fft//2+1]. 입력은 변경하지 않는다.

    채널 0은 Hann/50% overlap Welch PSD를 주파수 합으로 나눈 뒤의 자연로그다.
    채널 1은 원 REF의 log RMS를 모든 bin에 반복하므로 실제 gain 정보가 남는다.
    두 로그의 하한은 1e-12이며 무음에서도 유한하다. 짧은 창은 nperseg만 줄이고
    FFT를 zero-pad한다. detrend하지 않으므로 DC도 보존한다. PSD 정규화는 특징
    추출일 뿐이며 필터 계산·플랜트·DAC로 전달할 원 REF 정규화가 아니다.
    """
    n_fft = _integer(n_fft, "n_fft", 2, _MAX_FFT)
    reference = _array(reference, "reference")
    if reference.ndim == 1:
        reference = reference[None, :]
    if reference.ndim != 2 or not 1 <= reference.shape[0] <= _MAX_WINDOWS:
        raise ValueError("reference must have shape [N] or [B,N]")
    if reference.shape[0] * 2 * (n_fft // 2 + 1) > _MAX_ELEMENTS:
        raise ValueError("reference feature output exceeds the element limit")
    reference = reference.astype(np.float64, copy=False)
    # float64 FFT/제곱을 쓰더라도 임의의 거대한 값을 안전한 오디오로 취급하지 않는다.
    if np.max(np.abs(reference)) > np.finfo(np.float32).max:
        raise ValueError("reference magnitude exceeds float32 range")
    nperseg = min(reference.shape[1], n_fft)
    _, psd = welch(
        reference, fs=1.0, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, nfft=n_fft, detrend=False, axis=-1,
    )
    total = psd.sum(axis=-1, keepdims=True)
    normalized = np.divide(psd, total, out=np.zeros_like(psd), where=total > 0)
    log_psd = np.log(np.maximum(normalized, _LOG_FLOOR))
    rms = np.sqrt(np.mean(reference * reference, axis=-1, keepdims=True))
    log_rms = np.broadcast_to(np.log(np.maximum(rms, _LOG_FLOOR)), psd.shape)
    features = np.stack((log_psd, log_rms), axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("reference features are not finite")
    return features


class RefSpectrumSelector(nn.Module):
    """주파수 위치를 보존하는 작은 CNN. 출력은 후보별 logits [B,K]."""

    def __init__(self, candidate_count: int, bins: int, width: int = 16):
        super().__init__()
        self.candidate_count = _integer(candidate_count, "candidate_count", 1, _MAX_CANDIDATES)
        self.bins = _integer(bins, "bins", 2, _MAX_FFT // 2 + 1)
        self.width = _integer(width, "width", 1, 256)
        pooled_bins = min(16, self.bins)
        self.network = nn.Sequential(
            nn.Conv1d(2, self.width, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(self.width, self.width, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pooled_bins),
            nn.Flatten(),
            nn.Linear(self.width * pooled_bins, self.candidate_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != (2, self.bins) or not features.shape[0]:
            raise ValueError(f"features must have nonempty shape [B,2,{self.bins}]")
        return self.network(features)


@dataclass
class SelectorTrainingResult:
    model: RefSpectrumSelector
    history: list[dict[str, int | float]]
    best_epoch: int
    initial_validation_loss: float
    best_validation_loss: float


def _features(value: np.ndarray, name: str) -> np.ndarray:
    value = _array(value, name)
    if (value.ndim != 3 or value.shape[1] != 2
            or not 1 <= value.shape[0] <= _MAX_WINDOWS
            or not 2 <= value.shape[2] <= _MAX_FFT // 2 + 1):
        raise ValueError(f"{name} must have shape [N,2,bins]")
    if np.max(np.abs(value)) > np.finfo(np.float32).max:
        raise ValueError(f"{name} exceeds float32 range")
    return np.array(value, dtype=np.float32, copy=True, order="C")


def _costs(value: np.ndarray, name: str, rows: int) -> np.ndarray:
    value = _array(value, name)
    if (value.ndim != 2 or value.shape[0] != rows
            or not 1 <= value.shape[1] <= _MAX_CANDIDATES):
        raise ValueError(f"{name} must have shape [{rows},K]")
    if np.min(value) < 0 or np.max(value) > np.finfo(np.float32).max:
        raise ValueError(f"{name} must be nonnegative and within float32 range")
    return np.array(value, dtype=np.float64, copy=True, order="C")


def _device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError("CUDA device index is unavailable")
        device = torch.device("cuda", index)
    return device


def _targets(costs: np.ndarray, temperature: float) -> torch.Tensor:
    # 큰 cost/작은 temperature의 -inf는 확률 0으로 수렴한다. 최소값은 항상 0.
    with np.errstate(over="ignore"):
        exponent = -(costs - costs.min(axis=1, keepdims=True)) / temperature
    weights = np.exp(exponent)
    weights /= weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(weights.astype(np.float32))


def _finite(tensor: torch.Tensor, what: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"non-finite {what} in selector training/inference")


def _regrets(costs: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((costs - costs.min(axis=1, keepdims=True)).astype(np.float32))


def _loss_terms(
    logits: torch.Tensor, targets: torch.Tensor, regrets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """창별 soft CE와 선택 확률의 기대 regret. 큰 비용을 자르지 않는다."""
    cross_entropy = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    expected_regret = (F.softmax(logits, dim=1) * regrets).sum(dim=1)
    _finite(cross_entropy, "cross entropy")
    _finite(expected_regret, "expected regret")
    return cross_entropy, expected_regret


def _evaluate(
    model: RefSpectrumSelector, features: torch.Tensor, targets: torch.Tensor,
    costs: np.ndarray, batch_size: int, device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    selected_sum = 0.0
    regret_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            logits = model(features[start:end].to(device))
            _finite(logits, "validation logits")
            losses, regrets = _loss_terms(
                logits, targets[start:end].to(device), _regrets(costs[start:end]).to(device),
            )
            chosen = logits.argmax(dim=1).cpu().numpy()
            selected = costs[start:end][np.arange(len(chosen)), chosen]
            loss_sum += float(losses.double().sum().item())
            selected_sum += float(selected.sum())
            regret_sum += float(regrets.double().sum().item())
    return loss_sum / len(features), selected_sum / len(features), regret_sum / len(features)


def train_selector(
    train_features: np.ndarray,
    train_costs: np.ndarray,
    validation_features: np.ndarray,
    validation_costs: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 0,
    device: str | torch.device = "cpu",
    temperature: float = 0.05,
    risk_weight: float = 1.0,
) -> SelectorTrainingResult:
    """Train/validation만 사용해 cost soft label로 학습하고 최상 상태를 복원한다.

    label = softmax(-(cost-min(cost))/temperature).
    loss = soft CE + risk_weight * mean(sum(softmax(logits)*(cost-min(cost)))).
    위험 가중치는 비음수이며 0이면 기존 CE 단독 학습이다. 비용을 clipping하지 않고
    비유한 손실/gradient이면 중단한다. history의 train_loss/validation_loss는 CE를
    유지하며 expected_regret와 risk_weight를 별도로 기록한다.
    비용은 호출자가 계산한 nonnegative residual+effort/baseline 정규화 값이다.
    validation 선택 기준 및 반환 initial/best_validation_loss는 soft CE가 아니라
    logits.argmax로 선택한 후보의 실제 평균 비용이다. 초기 모델은 epoch 0이며
    같은 비용이면 먼저 얻은 상태를 유지한다. history는 epoch별 soft CE와 실제
    선택 비용을 모두 기록한다. best state는 깊은 복사이며 반환 모델은 eval이다.

    seed와 cuDNN deterministic 설정으로 같은 장치·버전에서 반복 실행을 지원한다.
    CPU와 CUDA 간 bitwise 일치나 다른 버전 간 재현성을 보장하지 않는다.
    """
    epochs = _integer(epochs, "epochs", 1, 10_000)
    batch_size = _integer(batch_size, "batch_size", 1, 65_536)
    seed = _integer(seed, "seed", 0, 2**32 - 1)
    learning_rate = _positive(learning_rate, "learning_rate")
    temperature = _positive(temperature, "temperature")
    risk_weight = _nonnegative(risk_weight, "risk_weight")
    device = _device(device)
    train_features = _features(train_features, "train_features")
    validation_features = _features(validation_features, "validation_features")
    train_costs = _costs(train_costs, "train_costs", len(train_features))
    validation_costs = _costs(validation_costs, "validation_costs", len(validation_features))
    if train_features.shape[1:] != validation_features.shape[1:]:
        raise ValueError("train/validation feature shapes must agree")
    if train_costs.shape[1] != validation_costs.shape[1]:
        raise ValueError("train/validation candidate counts must agree")
    train_x = torch.from_numpy(train_features)
    validation_x = torch.from_numpy(validation_features)
    train_y = _targets(train_costs, temperature)
    validation_y = _targets(validation_costs, temperature)
    train_regrets = _regrets(train_costs)
    cuda_devices = [device.index] if device.type == "cuda" else []
    history: list[dict[str, int | float]] = []
    # 전역 RNG와 cuDNN 플래그는 종료/예외 시 복원한다.
    with torch.random.fork_rng(devices=cuda_devices), torch.backends.cudnn.flags(
        benchmark=False, deterministic=True,
    ), torch.autocast(device_type=device.type, enabled=False):
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        model = RefSpectrumSelector(train_costs.shape[1], train_features.shape[2]).to(
            device=device, dtype=torch.float32,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        _, initial, _ = _evaluate(model, validation_x, validation_y, validation_costs, batch_size, device)
        best_loss = initial
        best_epoch = 0
        best_state = deepcopy(model.state_dict())
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train_x), generator=generator)
            for start in range(0, len(train_x), batch_size):
                indices = order[start:start + batch_size]
                logits = model(train_x[indices].to(device))
                _finite(logits, "training logits")
                cross_entropy, expected_regret = _loss_terms(
                    logits, train_y[indices].to(device), train_regrets[indices].to(device),
                )
                loss = cross_entropy.mean()
                if risk_weight:
                    loss = loss + risk_weight * expected_regret.mean()
                _finite(loss, "training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        _finite(parameter.grad, "gradient")
                optimizer.step()
                for parameter in model.parameters():
                    _finite(parameter, "parameter")
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            _finite(value, "optimizer state")
            train_loss, train_selected, train_regret = _evaluate(model, train_x, train_y, train_costs, batch_size, device)
            validation_loss, validation_selected, validation_regret = _evaluate(
                model, validation_x, validation_y, validation_costs, batch_size, device,
            )
            history.append({
                "epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                "train_selected_cost": train_selected, "validation_selected_cost": validation_selected,
                "train_expected_regret": train_regret, "validation_expected_regret": validation_regret,
                "risk_weight": risk_weight,
            })
            if validation_selected < best_loss:
                best_loss = validation_selected
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f"SFANC epoch {epoch}/{epochs}: train CE={train_loss:.6g}, "
                    f"validation CE={validation_loss:.6g}, "
                    f"기대 regret={validation_regret:.6g}, "
                    f"선택 비용={validation_selected:.6g}, 최상={best_loss:.6g}",
                    flush=True,
                )
        model.load_state_dict(best_state)
        model.eval()
    return SelectorTrainingResult(model, history, best_epoch, initial, best_loss)


def predict_selector(
    model: RefSpectrumSelector, reference: np.ndarray, *, n_fft: int = 512,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """REF만 사용해 후보 index [B]를 반환한다. 실행 중 출력에 연결하지 않는다.

    완전히 관측한 창의 선택을 이후 구간에 적용할 책임은 호출자에게 있다.
    장치를 지정하면 모델의 실제 장치와 일치해야 하며 모델을 임의 이동하지 않는다.
    """
    features = reference_features(reference, n_fft=n_fft)
    model_device = next(model.parameters()).device
    device = model_device if device is None else _device(device)
    if device != model_device:
        raise ValueError("inference device must match the model device")
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            selected = []
            for start in range(0, len(features), 64):
                logits = model(torch.from_numpy(features[start:start + 64]).to(device))
                _finite(logits, "inference logits")
                selected.append(logits.argmax(dim=1).cpu().numpy())
            return np.concatenate(selected).astype(np.int64)
    finally:
        model.train(was_training)
