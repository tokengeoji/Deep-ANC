"""느린 계수 제안 + 인과 FIR + FxNLMS 잔차의 오프라인 연구용 제어기.

실시간 엔진 factory에 연결하지 않는다. 모델·장치·스레드를 열지 않으며 모든 메서드는
한 소비자가 호출해야 한다(스레드 안전 mailbox가 아님). 논문의 frame/sample 분리를
시험하지만 기준 FIR + 잔차 유지/crossfade는 이 프로젝트의 실험 정책이다.
계수·출력 제한은 폐루프 안정성 인증이 아니며 e=d+S*y 극성을 유지한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .fxlms_core import FxLMSController, AdaptationResult


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name}: {minimum} 이상 정수가 필요합니다")
    return int(value)


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, str, np.bool_)):
        raise ValueError(f"{name}: 유한한 양수가 필요합니다")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}: 유한한 양수가 필요합니다") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name}: 유한한 양수가 필요합니다")
    return number


@dataclass(frozen=True)
class FilterProposal:
    """기준 FIR만 담는다. 잔차를 포함한 최종 계수를 다시 넣으면 이중 합산이다.

    observed_until_sample은 관측 구간의 exclusive 끝이다. 제안 제출 시 이미 소비한
    sample_cursor보다 클 수 없다. 이것은 학습 데이터 누수까지 검증하는 메타가 아니다.
    """

    context_id: str
    generation: int
    revision: int
    observed_until_sample: int
    coefficients: tuple[float, ...]
    source_id: str

    def __post_init__(self):
        for name in ("context_id", "source_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name}: 빈 문자열은 허용하지 않습니다")
        for name in ("generation", "revision", "observed_until_sample"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        values = np.asarray(self.coefficients)
        if (values.ndim != 1 or values.size == 0 or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()):
            raise ValueError("제안 계수는 비어 있지 않은 유한 실수 1차원 배열이어야 합니다")
        # frozen dataclass 안의 ndarray도 수정될 수 있으므로 실제 불변 tuple로 고정한다.
        object.__setattr__(self, "coefficients", tuple(float(v) for v in values))


@dataclass(frozen=True)
class ProposalDecision:
    accepted: bool
    reason: str


class PreparedFxNLMSController:
    """generate(ref) → 플랜트 잔류 ERR → adapt(ERR) 순서의 연구용 API.

    fast path는 선택기/CNN을 호출하거나 기다리지 않는다. 제안이 없으면 마지막 FIR을
    유지한다. 시작/초기화는 계수 0, 적응은 adapt_block(enabled=True) 때만 가능하다.
    handoff는 실제 모사할 값으로 명시하고 S 순수지연에 정확히 한 번 더한다.
    """

    reference_mode = "acoustic"
    digital_reference_lead_samples = 0
    research_only = True

    def __init__(
        self, s_hat: np.ndarray, *, sample_rate: int, hop: int,
        secondary_delay_samples: int, handoff_extra_samples: int, control_length: int,
        reference_peak_limit: float, control_limit: float, prepared_l1_limit: float,
        residual_norm_limit: float, mu: float, leakage: float,
        transition_samples: int, max_proposal_age_samples: int,
    ) -> None:
        self.sample_rate = _integer(sample_rate, "sample_rate", 1)
        self.hop = _integer(hop, "hop", 1)
        self.control_length = _integer(control_length, "control_length", 1)
        delay = _integer(secondary_delay_samples, "secondary_delay_samples")
        handoff = _integer(handoff_extra_samples, "handoff_extra_samples")
        self.secondary_delay_samples = delay + handoff
        self.transition_samples = _integer(transition_samples, "transition_samples", 1)
        self.max_proposal_age_samples = _integer(max_proposal_age_samples, "max_proposal_age_samples")
        self.reference_peak_limit = _positive(reference_peak_limit, "reference_peak_limit")
        self.control_limit = _positive(control_limit, "control_limit")
        self.prepared_l1_limit = _positive(prepared_l1_limit, "prepared_l1_limit")
        residual_norm_limit = _positive(residual_norm_limit, "residual_norm_limit")
        mu = _positive(mu, "mu")
        if isinstance(leakage, (bool, str, np.bool_)):
            raise ValueError("leakage는 [0,1)의 유한 값이어야 합니다")
        try:
            leakage = float(leakage)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("leakage는 [0,1)의 유한 값이어야 합니다") from exc
        if not np.isfinite(leakage) or not 0 <= leakage < 1:
            raise ValueError("leakage는 [0,1)의 유한 값이어야 합니다")
        # float32 FxNLMS 코어의 곱/합이 overflow하는 터무니없는 연구 설정은 먼저 거부한다.
        limits = (self.reference_peak_limit, self.control_limit, self.prepared_l1_limit, residual_norm_limit)
        if any(v > 1e6 for v in limits):
            raise ValueError("연구 제어기의 수치 제한은 1e6 이하여야 합니다")
        secondary = np.asarray(s_hat)
        if (secondary.ndim != 1 or secondary.size == 0 or secondary.dtype.kind not in "fiu"
                or not np.isfinite(secondary).all() or not np.any(secondary != 0)
                or np.max(np.abs(secondary.astype(np.float64))) > 1e6):
            raise ValueError("S는 유한하고 0이 아닌 1차원 FIR이어야 합니다")
        # 호출자 배열과 공유하면 context hash는 그대로인데 실제 filtered-x S가 바뀐다.
        secondary = np.array(secondary, dtype=np.float32, copy=True)
        self.secondary_tail_samples = self.secondary_delay_samples + secondary.size - 1
        self._adaptive = FxLMSController(
            secondary, secondary_delay_samples=self.secondary_delay_samples,
            control_len=self.control_length, mu=mu, leakage=float(leakage),
            weight_norm_limit=residual_norm_limit,
        )
        conditions = {
            "schema": "prepared_fir_context.v1", "reference_mode": "acoustic", "sign": "e=d+S*y",
            "sample_rate": self.sample_rate, "hop": self.hop, "control_length": self.control_length,
            "secondary_delay_samples": delay, "handoff_extra_samples": handoff,
            "secondary_fir": secondary.tolist(), "reference_peak_limit": self.reference_peak_limit,
            "control_limit": self.control_limit, "prepared_l1_limit": self.prepared_l1_limit,
            "residual_norm_limit": residual_norm_limit, "mu": mu, "leakage": float(leakage),
            "transition_samples": self.transition_samples, "max_proposal_age_samples": self.max_proposal_age_samples,
        }
        self.context_id = hashlib.sha256(json.dumps(conditions, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.generation = -1
        self.reset()

    def reset(self) -> None:
        """새 제안 세대와 빈 이력으로 초기화한다. 이전 세대 제안은 재사용하지 않는다."""
        self.generation += 1
        self.sample_cursor = 0
        self._revision = -1
        self._from = np.zeros(self.control_length, dtype=np.float32)
        self._target = self._from.copy()
        self._history = np.zeros(self.control_length - 1, dtype=np.float32)
        self._fade_done = self.transition_samples
        self._adaptive.reset(reset_histories=True)
        self._guard_until = self.secondary_tail_samples + self.control_length - 1
        self._pending = None
        self.clip_samples = 0
        self.last_guard_reason = "reset_history_guard"
        self.last_adaptation = None

    @property
    def residual_weights(self) -> np.ndarray:
        return self._adaptive.w.copy()

    @property
    def prepared_coefficients(self) -> np.ndarray:
        alpha = min(1.0, self._fade_done / self.transition_samples)
        return ((1 - alpha) * self._from + alpha * self._target).copy()

    @property
    def transitioning(self) -> bool:
        return self._fade_done < self.transition_samples

    def submit(self, proposal: FilterProposal) -> ProposalDecision:
        """소비자가 완결 블록 경계에서 호출한다. 전환 중 중첩 제안은 보류 없이 거부한다."""
        if not isinstance(proposal, FilterProposal):
            raise TypeError("FilterProposal이 필요합니다")
        reason = None
        if self._pending is not None:
            reason = "pending_error_block"
        elif proposal.context_id != self.context_id:
            reason = "context_mismatch"
        elif proposal.generation != self.generation:
            reason = "generation_mismatch"
        elif proposal.revision <= self._revision:
            reason = "stale_revision"
        elif proposal.observed_until_sample > self.sample_cursor:
            reason = "future_observation"
        elif self.sample_cursor - proposal.observed_until_sample > self.max_proposal_age_samples:
            reason = "expired_observation"
        elif self.transitioning:
            reason = "transition_in_progress"
        elif len(proposal.coefficients) != self.control_length:
            reason = "coefficient_shape"
        elif sum(abs(v) for v in proposal.coefficients) > self.prepared_l1_limit:
            reason = "coefficient_l1_limit"
        if reason is not None:
            return ProposalDecision(False, reason)
        self._from = self._target.copy()
        self._target = np.asarray(proposal.coefficients, dtype=np.float32)
        self._fade_done = 0
        self._revision = proposal.revision
        self._guard_until = max(
            self._guard_until, self.sample_cursor + self.transition_samples + self.secondary_tail_samples,
        )
        return ProposalDecision(True, "accepted")

    def _block(self, values: np.ndarray, name: str, limit: float | None = None) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except (TypeError, ValueError, OverflowError) as exc:
            self.reset()
            raise ValueError(f"{name}: 배열 변환 실패 — 제어기를 초기화했습니다") from exc
        if (raw.shape != (self.hop,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all()
                or np.max(np.abs(raw.astype(np.float64))) > (limit if limit is not None else 1e6)):
            self.reset()
            raise ValueError(f"{name}: hop 크기의 유한 실수/범위 검증 실패 — 제어기를 초기화했습니다")
        return raw.astype(np.float32)

    def generate_block(self, reference_block: np.ndarray) -> np.ndarray:
        if self._pending is not None:
            raise RuntimeError("앞선 ERR 블록을 adapt_block으로 먼저 소비해야 합니다")
        reference = self._block(reference_block, "REF", self.reference_peak_limit)
        matrix, self._history = FxLMSController._tap_matrix(reference, self._history, self.control_length)
        baseline = matrix @ self._target
        if self.transitioning:
            alpha = np.minimum(1.0, (self._fade_done + np.arange(1, self.hop + 1)) / self.transition_samples)
            baseline = (1 - alpha) * (matrix @ self._from) + alpha * baseline
            self._fade_done = min(self.transition_samples, self._fade_done + self.hop)
        combined = baseline.astype(np.float64) + self._adaptive.generate_block(reference)
        if not np.isfinite(combined).all():
            self.reset()
            raise FloatingPointError("합산 출력이 유한하지 않아 초기화했습니다")
        clipped = np.abs(combined) > self.control_limit
        self.clip_samples += int(clipped.sum())
        start = self.sample_cursor
        self.sample_cursor += self.hop
        if np.any(clipped):
            self._guard_until = max(self._guard_until, self.sample_cursor + self.secondary_tail_samples)
        self._pending = {
            "eligible": start >= self._guard_until,
            "reason": "output_clipping_guard" if np.any(clipped) else "transition_or_history_guard",
        }
        return np.clip(combined, -self.control_limit, self.control_limit).astype(np.float32)

    def adapt_block(self, error_block: np.ndarray, enabled: bool = False) -> AdaptationResult:
        if self._pending is None:
            raise RuntimeError("generate_block을 먼저 호출해야 합니다")
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("enabled는 bool이어야 합니다")
        error = self._block(error_block, "ERR")
        pending, self._pending = self._pending, None
        permitted = bool(enabled and pending["eligible"])
        self.last_guard_reason = "enabled" if permitted else (pending["reason"] if enabled else "disabled")
        try:
            # guard 중에도 pending gradient를 소비하고 filtered-x 이력은 계속 전진한다.
            self.last_adaptation = self._adaptive.adapt_block(error, enabled=permitted)
            if not np.isfinite([
                self.last_adaptation.filtered_reference_power,
                self.last_adaptation.gradient_norm, self.last_adaptation.weight_norm,
            ]).all():
                raise FloatingPointError("적응 진단의 수치 범위를 벗어났습니다")
        except Exception:
            self.reset()
            raise
        return self.last_adaptation
