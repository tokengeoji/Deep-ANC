"""과거 REF 상태를 보존하는 SFANC FIR 교체의 오프라인 수치 코어.

u[t] = sum(w_t[k] * x[t-k])이며 추가 극성 반전·S·장치 지연은 없다.
선택기·오디오·스레드를 실행하지 않으며 실시간 callback 구현이 아니다.
전환 길이와 hard clip은 비교 실험 매개변수이지 배포 승인 정책이 아니다.
"""

from __future__ import annotations

import math

import numpy as np


MAX_BLOCK_SAMPLES = 1_000_000
MAX_TAPS = 4096
MAX_BANK_COEFFICIENTS = 1_000_000
MAX_FILTER_PRODUCTS = 16_777_216


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name}: {minimum} 이상 정수가 필요합니다")
    return int(value)


def _array(value, name, dimensions):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 실수 배열이 필요합니다") from exc
    if raw.ndim != dimensions or not raw.size or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name}: 비어 있지 않은 {dimensions}차원 실수 배열이 필요합니다")
    # 큰 입력은 복사하기 전에 거부한다.
    if name == "coefficients":
        if raw.shape[1] > MAX_TAPS or raw.size > MAX_BANK_COEFFICIENTS:
            raise ValueError("coefficients: 필터 뱅크 크기 제한 위반")
    elif raw.size > MAX_BLOCK_SAMPLES:
        raise ValueError("reference: 블록 크기 제한 위반")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: FP64 유한 값만 허용합니다")
    return array


class StreamingFIRBank:
    """공유 입력 이력으로 인과 FIR을 실행하고 계수를 샘플 단위로 전환한다.

    ``request_filter(k)`` 뒤의 첫 샘플부터 전환한다. N>0이면 마지막 실제
    계수에서 목표 계수로 α=1/N, 2/N, ..., 1 보간한다. 전환 중 다른 후보를
    요청하면 마지막 실제 계수에서 새 N샘플 전환을 시작한다. 같은 목표의
    재요청은 진행률을 초기화하지 않는다. N=0은 다음 샘플부터 즉시 바뀐다.

    ``control_limit=None``은 선형 출력 그대로, 양수면 마지막 명령만
    대칭 hard clip한다. 입력 이력·계수는 clip하지 않고 초과 표본을 기록한다.
    clip은 음향 안정성 보증이 아니다. ``process``는 빈 블록을 거부하며
    잘못된 입력/수치 overflow 때 이력·전환·통계를 변경하지 않는다.
    """

    def __init__(self, coefficients, initial_index=0, transition_samples=0,
                 control_limit=None):
        bank = _array(coefficients, "coefficients", 2)
        initial = _integer(initial_index, "initial_index")
        if initial >= len(bank):
            raise ValueError("initial_index: 후보 범위를 벗어났습니다")
        transition = _integer(transition_samples, "transition_samples")
        if transition > MAX_BLOCK_SAMPLES:
            raise ValueError("transition_samples: 전환 길이 제한 위반")
        if control_limit is not None:
            if (isinstance(control_limit, (bool, np.bool_))
                    or not isinstance(control_limit, (int, float, np.integer, np.floating))):
                raise ValueError("control_limit: 유한한 양수 또는 None이 필요합니다")
            try:
                control_limit = float(control_limit)
            except (ValueError, OverflowError) as exc:
                raise ValueError("control_limit: FP64 유한한 양수가 필요합니다") from exc
            if not math.isfinite(control_limit) or control_limit <= 0:
                raise ValueError("control_limit: 유한한 양수 또는 None이 필요합니다")
        bank.setflags(write=False)
        self._bank = bank
        self._initial_index = initial
        self._transition_samples = transition
        self._control_limit = control_limit
        self.reset()

    @property
    def current_coefficients(self):
        """마지막 표본에 적용한 계수의 독립 복사본(즉시 교체 요청은 새 계수)."""
        return self._current.copy()

    @property
    def target_index(self):
        return self._target_index

    @property
    def transition_remaining(self):
        return self._remaining

    @property
    def diagnostics(self):
        """reset 이후 누적치의 독립 사본. peak는 실제 FP64 계산 명령 기준."""
        return dict(self._statistics, target_index=self._target_index,
                    transition_remaining=self._remaining,
                    transition_samples=self._transition_samples,
                    control_limit=self._control_limit,
                    limiter_kind="none" if self._control_limit is None else "hard_clip",
                    realtime_validated=False, deployment_allowed=False)

    def reset(self):
        """입력 이력·전환·누적 통계를 지우고 지정된 최초 후보로 복원한다."""
        self._history = np.zeros(self._bank.shape[1] - 1, dtype=np.float64)
        self._target_index = self._initial_index
        self._current = self._bank[self._initial_index].copy()
        self._origin = self._current.copy()
        self._remaining = 0
        self._statistics = {"processed_samples": 0, "raw_peak": 0.0,
                            "output_peak": 0.0, "limited_samples": 0,
                            "filter_requests": 0, "transitions": 0}

    def request_filter(self, index):
        """다음 출력부터 사용할 목표 후보 요청. 동일 목표 요청은 완전 no-op."""
        index = _integer(index, "index")
        if index >= len(self._bank):
            raise ValueError("index: 후보 범위를 벗어났습니다")
        if index == self._target_index:
            return
        self._origin = self._current.copy()
        self._target_index = index
        self._remaining = self._transition_samples
        self._statistics["filter_requests"] += 1
        if self._remaining:
            self._statistics["transitions"] += 1
        else:
            self._current = self._bank[index].copy()

    def process(self, reference):
        """FP64[N] 제어 명령 반환. 청크 길이 자체는 지연·전환 규약을 바꾸지 않는다."""
        x = _array(reference, "reference", 1)
        length = self._bank.shape[1]
        if x.size * length > MAX_FILTER_PRODUCTS:
            raise ValueError("reference: 블록×FIR 계산 크기 제한 위반")
        context = np.concatenate((self._history, x))
        remaining = self._remaining
        current = self._current.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            if remaining:
                target = self._bank[self._target_index]
                # 동일 입력 이력에 두 FIR을 적용한 출력 보간은 샘플별 계수
                # 보간과 선형 등가다. 후보마다 상태를 초기화하지 않는다.
                raw = np.convolve(context, target, mode="valid")
                count = min(remaining, x.size)
                old = np.convolve(context[:count + length - 1], self._origin, mode="valid")
                elapsed = self._transition_samples - remaining
                alpha = (elapsed + np.arange(1, count + 1, dtype=np.float64)) / self._transition_samples
                raw[:count] = (1.0 - alpha) * old + alpha * raw[:count]
                remaining -= count
                if remaining:
                    last_alpha = float(alpha[-1])
                    current = (1.0 - last_alpha) * self._origin + last_alpha * target
                else:
                    current = target.copy()
            else:
                raw = np.convolve(context, current, mode="valid")
        if not np.isfinite(raw).all() or not np.isfinite(current).all():
            raise ValueError("FIR 출력의 FP64 범위를 벗어났습니다; 상태 변경 없음")
        raw_peak = float(np.max(np.abs(raw)))
        if self._control_limit is None:
            output, limited = raw, 0
        else:
            limited = int(np.count_nonzero(np.abs(raw) > self._control_limit))
            output = np.clip(raw, -self._control_limit, self._control_limit)
        # 계산 전체가 성공한 뒤에만 내부 상태를 반영한다.
        self._history = context[-(length - 1):].copy() if length > 1 else np.empty(0, dtype=np.float64)
        self._current = current
        self._remaining = remaining
        self._statistics["processed_samples"] += int(x.size)
        self._statistics["raw_peak"] = max(self._statistics["raw_peak"], raw_peak)
        self._statistics["output_peak"] = max(self._statistics["output_peak"], float(np.max(np.abs(output))))
        self._statistics["limited_samples"] += limited
        return output
