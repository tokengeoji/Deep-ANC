"""추론 엔진 — 공통 인터페이스 step(ref, err) → anti-noise.

마이그레이션 경로 (docs/06): torch(개발) → ort(등가성 검증) → trt(배포).
모든 엔진은 내부 상태를 보관하며 reset() 으로 제로 초기화한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceEngine(Protocol):
    hop: int
    digital_reference_lead_samples: int | None
    reference_mode: str | None

    def reset(self) -> None: ...

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        """ref/err: (hop,) float32 → anti-noise (hop,) float32."""
        ...


def checkpoint_digital_reference_lead_samples(state: dict) -> int:
    """체크포인트의 학습 lead를 반환한다 (기존 artifact는 lead=0 호환)."""
    cfg = state.get("cfg", {}) or {}
    # 개발 중 full cfg를 저장했던 임시 artifact도 읽을 수 있게 한다.
    data_cfg = cfg.get("data", {}) or {}
    if "digital_reference_lead_samples" in cfg:
        lead = int(cfg["digital_reference_lead_samples"])
        if "digital_reference_lead_samples" in data_cfg and lead != int(data_cfg["digital_reference_lead_samples"]):
            raise ValueError("체크포인트 상위/중첩 digital_reference_lead_samples가 다릅니다")
        return lead
    return int(data_cfg.get("digital_reference_lead_samples", 0))


def checkpoint_reference_mode(state: dict) -> str | None:
    """lead=0만으로 acoustic 학습을 추정하지 않고 저장된 모드를 읽는다."""
    cfg = state.get("cfg", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    if "reference_mode" in cfg and "reference_mode" in data_cfg and cfg["reference_mode"] != data_cfg["reference_mode"]:
        raise ValueError("체크포인트 상위/중첩 reference_mode가 다릅니다")
    mode = cfg.get("reference_mode", data_cfg.get("reference_mode"))
    if mode not in {None, "digital", "acoustic"}:
        raise ValueError(f"체크포인트 reference_mode가 잘못됐습니다: {mode!r}")
    return mode


def validate_secondary_calibration(runtime_cfg: dict, secondary) -> None:
    """측정 S(z)의 샘플레이트·block·latency를 실제 런타임 조건과 대조한다.

    acoustic은 두 측정 조건 메타가 모두 있어야 한다. digital의 legacy S는
    block=0/latency=unknown 또는 legacy-npy를 허용하되, 기록된 조건이 현재와
    다르면 거부한다. secondary는 baselines.fxlms_core.SecondaryPathModel이다.
    이 검사는 경로 gain·현재 음향 응답·실제 지터의 일치를 보증하지 않는다.
    """
    reference = str(runtime_cfg.get("reference", "digital"))
    if reference not in {"mic", "digital"}:
        raise ValueError(f"reference는 digital/mic이어야 합니다: {reference!r}")
    try:
        audio = runtime_cfg["hardware"]["audio"]
        sample_rate = int(audio["sample_rate"])
        block = int(audio["block_size"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("S(z) 검증에 hardware.audio.sample_rate/block_size가 필요합니다") from exc
    if sample_rate <= 0 or block <= 0:
        raise ValueError("audio.sample_rate/block_size는 양수여야 합니다")
    latency = str(audio.get("latency", "low"))
    if latency not in {"low", "high"}:
        raise ValueError("audio.latency는 low/high이어야 합니다")
    if secondary.sample_rate != sample_rate:
        raise ValueError(
            f"S(z) sample_rate가 런타임과 다릅니다: 측정={secondary.sample_rate}, 설정={sample_rate}"
        )
    measured_block = getattr(secondary, "calibration_block_size", 0)
    measured_latency = getattr(secondary, "calibration_latency", "unknown")
    if measured_block is not None and measured_block < 0:
        raise ValueError("S(z) calibration_block_size는 음수일 수 없습니다")
    if measured_block in (None, 0):
        if reference == "mic":
            raise ValueError("acoustic S(z)에 calibration_block_size 측정 메타가 필요합니다")
    elif measured_block != block:
        raise ValueError(
            f"S(z) calibration_block_size가 런타임과 다릅니다: 측정={measured_block}, 설정={block}"
        )
    if measured_latency in {None, "", "unknown", "legacy-npy"}:
        if reference == "mic":
            raise ValueError("acoustic S(z)에 calibration_latency 측정 메타가 필요합니다")
    elif measured_latency not in {"low", "high"}:
        raise ValueError(f"S(z) calibration_latency가 잘못됐습니다: {measured_latency!r}")
    elif measured_latency != latency:
        raise ValueError(
            f"S(z) calibration_latency가 런타임과 다릅니다: 측정={measured_latency}, 설정={latency}"
        )


def _load_ckpt_model(ckpt_path: str | Path):
    import torch

    from ..models import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.digital_reference_lead_samples = checkpoint_digital_reference_lead_samples(state)
    model.reference_mode = checkpoint_reference_mode(state)
    return model.eval()


class TorchEngine:
    """PyTorch eager 스트리밍 (개발/디버깅용 — 커널 런치 오버헤드 큼)."""

    def __init__(self, ckpt: str, hop: int = 256, device: str | None = None) -> None:
        import torch

        self.hop = int(hop)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = _load_ckpt_model(ckpt).to(device)
        self.digital_reference_lead_samples = int(
            self.model.digital_reference_lead_samples
        )
        self.reference_mode = self.model.reference_mode
        self._torch = torch
        self.reset()

    def reset(self) -> None:
        self.states = self.model.init_states(1, self.device)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = np.stack([ref, err]).astype(np.float32)[None]      # [1,2,hop]
        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device)
            y, self.states = self.model.streaming_step(xt, self.states)
        return y.squeeze().float().cpu().numpy()


class OrtEngine:
    """ONNX Runtime CPU — export 정합성 검증·CPU 폴백용."""

    def __init__(self, onnx_path: str, hop: int = 256) -> None:
        import json

        import onnxruntime as ort

        self.hop = int(hop)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2       # Tegra affinity 크래시 회피 (명시 지정)
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        meta_path = Path(onnx_path).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # key가 없는 기존 ONNX artifact는 기존 정렬인 lead=0으로 호환한다.
        self.digital_reference_lead_samples = int(
            meta.get("digital_reference_lead_samples", 0)
        )
        self.reference_mode = meta.get("reference_mode")
        self.state_names: list[str] = meta["state_names"]
        self._init_shapes = {
            i.name: (i.shape, np.float32) for i in self.sess.get_inputs() if i.name != "x"
        }
        self.reset()

    def reset(self) -> None:
        self.states = {
            name: np.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in self._init_shapes.items()
        }
        # attention mask 상태는 -1e4 초기화 (빈 슬롯 무효화)
        for name in self.states:
            if name.endswith("_attn_m"):
                self.states[name][:] = -1.0e4

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        x = np.stack([ref, err]).astype(np.float32)[None]
        feeds = {"x": x}
        feeds.update(self.states)
        outs = self.sess.run(None, feeds)
        y = outs[0].reshape(-1)
        for name, val in zip(self.state_names, outs[1:]):
            self.states[name] = val
        return y.astype(np.float32)


def _numpy_from_pinned(ptr: int, shape: tuple[int, ...]) -> np.ndarray:
    """cudaHostAlloc 포인터를 복사 없이 numpy 뷰로 감싼다."""

    import ctypes

    count = int(np.prod(shape))
    buffer = (ctypes.c_float * count).from_address(int(ptr))
    return np.ctypeslib.as_array(buffer).reshape(shape)


class TrtEngine:
    """TensorRT 10.x FP16 엔진 — 상태 핑퐁 + execute_async_v3 (배포 경로).

    필요: tensorrt 파이썬 바인딩 + cuda-python. 엔진 빌드는 scripts/export/build_trt.sh.
    """

    def __init__(self, plan: str, onnx_meta: str | None = None, hop: int = 256) -> None:
        import json

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "tensorrt 파이썬 바인딩이 없습니다. docs/06_deployment_jetson.md 의 "
                "TensorRT 설치 절을 참조하세요."
            ) from exc
        # cuda-python 12 부터 cudart 가 cuda.bindings.runtime 으로 옮겨졌다. 두 배치를
        # 모두 받아준다 — Jetson 이미지마다 버전이 달라 한쪽만 지원하면 배포가 막힌다.
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            try:
                from cuda import cudart
            except ImportError as exc:
                raise RuntimeError(
                    "cuda-python 바인딩이 없습니다 (cuda.bindings.runtime / cuda.cudart "
                    "둘 다 없음). docs/06_deployment_jetson.md 참조."
                ) from exc

        self.hop = int(hop)
        self._trt = trt
        self._cudart = cudart
        # 동기 대기에서 스레드를 재우면 OS 가 깨워줄 때까지 수 ms 가 날아간다. 실시간
        # 오디오 콜백에서는 그 지연이 곧 마감 초과다. 스핀 대기로 바꿔 커널 완료를
        # 즉시 회수한다 — CPU 코어 하나를 태우지만 hop 당 1ms 미만이라 감당된다.
        cudart.cudaSetDeviceFlags(cudart.cudaDeviceScheduleSpin)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        meta_path = Path(onnx_meta) if onnx_meta else Path(plan).with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.digital_reference_lead_samples = int(
            meta.get("digital_reference_lead_samples", 0)
        )
        self.reference_mode = meta.get("reference_mode")
        self.state_names: list[str] = meta["state_names"]

        err_code, self.stream = cudart.cudaStreamCreate()
        assert err_code == cudart.cudaError_t.cudaSuccess

        # 텐서별 호스트/디바이스 버퍼. 상태는 A/B 핑퐁.
        self.host: dict[str, np.ndarray] = {}
        self.dev: dict[str, int] = {}
        self.state_dev: dict[str, tuple[int, int]] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape)) * 4
            base = name[:-4] if name.endswith("_out") else name
            if base in self.state_names:
                if base not in self.state_dev:
                    a = cudart.cudaMalloc(size)[1]
                    b = cudart.cudaMalloc(size)[1]
                    self.state_dev[base] = (a, b)
                    self.host[base] = np.zeros(shape, dtype=np.float32)
            else:
                self.dev[name] = cudart.cudaMalloc(size)[1]
                self.host[name] = np.zeros(shape, dtype=np.float32)
        self._cur = 0

        # 고정(pinned) 호스트 버퍼. pageable 메모리에서의 cudaMemcpyAsync 는 드라이버가
        # 내부 스테이징 버퍼를 거치며 사실상 동기 동작이 된다 — 비동기 이득이 사라지고
        # CUDA Graph 안에도 넣을 수 없다.
        self._x_nbytes = int(self.host["x"].nbytes)
        self._y_nbytes = int(self.host["y"].nbytes)
        self._pin_x_ptr = cudart.cudaHostAlloc(
            self._x_nbytes, cudart.cudaHostAllocDefault
        )[1]
        self._pin_y_ptr = cudart.cudaHostAlloc(
            self._y_nbytes, cudart.cudaHostAllocDefault
        )[1]
        self._pin_x = _numpy_from_pinned(self._pin_x_ptr, self.host["x"].shape)
        self._pin_y = _numpy_from_pinned(self._pin_y_ptr, self.host["y"].shape)
        self._pin_x[...] = 0.0
        self._pin_y[...] = 0.0

        self.graph_exec: list = []
        self.reset()
        self.graph_captured = self._try_capture_graphs()
        # 캡처는 상태 버퍼를 워밍업 실행으로 오염시킨다. 반드시 다시 0 으로 되돌린다.
        self.reset()
        self._cur = 0

    def reset(self) -> None:
        cudart = self._cudart
        for name, (a, b) in self.state_dev.items():
            init = np.zeros_like(self.host[name])
            if name.endswith("_attn_m"):
                init[:] = -1.0e4
            for ptr in (a, b):
                cudart.cudaMemcpy(
                    ptr, init.ctypes.data, init.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )

    def _bind(self, parity: int) -> None:
        """상태 A/B 핑퐁 주소를 바인딩한다. 경우의 수는 2 뿐이라 매 스텝 부를 필요가 없다."""

        self.context.set_tensor_address("x", self.dev["x"])
        self.context.set_tensor_address("y", self.dev["y"])
        for name in self.state_names:
            a, b = self.state_dev[name]
            self.context.set_tensor_address(name, a if parity == 0 else b)
            self.context.set_tensor_address(f"{name}_out", b if parity == 0 else a)

    def _try_capture_graphs(self) -> bool:
        """스텝 전체(H2D → 추론 → D2H)를 parity 별 CUDA Graph 로 캡처한다.

        캡처하지 않으면 매 스텝 커널 수십 개를 개별 런치하게 되고, 커널 하나가 수 µs 인
        이 모델에서는 **런치 오버헤드가 연산을 압도한다**. trtexec 가 --useCudaGraph 로
        재는 값과 런타임 값이 크게 벌어졌던 원인이다.

        실패하면 조용히 폴백한다 — 그래프 캡처는 드라이버/TRT 버전에 민감하고, 여기서
        예외를 올리면 배포 경로 전체가 막힌다.
        """

        cudart = self._cudart
        try:
            self.graph_exec = []
            for parity in (0, 1):
                self._bind(parity)
                # TRT 는 첫 실행에서 내부 workspace 를 잡는다. 캡처 전에 워밍업이 필요하다.
                self.context.execute_async_v3(self.stream)
                cudart.cudaStreamSynchronize(self.stream)

                err = cudart.cudaStreamBeginCapture(
                    self.stream,
                    cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
                )[0]
                if err != cudart.cudaError_t.cudaSuccess:
                    return False
                cudart.cudaMemcpyAsync(
                    self.dev["x"], self._pin_x_ptr, self._x_nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
                )
                self.context.execute_async_v3(self.stream)
                cudart.cudaMemcpyAsync(
                    self._pin_y_ptr, self.dev["y"], self._y_nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
                )
                err, graph = cudart.cudaStreamEndCapture(self.stream)
                if err != cudart.cudaError_t.cudaSuccess:
                    return False
                err, exec_ = cudart.cudaGraphInstantiate(graph, 0)
                if err != cudart.cudaError_t.cudaSuccess:
                    return False
                self.graph_exec.append(exec_)
            return True
        except Exception:
            self.graph_exec = []
            return False

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        cudart = self._cudart
        # 고정(pinned) 버퍼에 직접 쓴다. pageable 메모리의 cudaMemcpyAsync 는 내부적으로
        # 동기 동작이라 비동기 이득이 사라지고, 매 스텝 배열을 새로 만들면 할당이 핫패스에
        # 들어온다.
        self._pin_x[0, 0, :] = ref
        self._pin_x[0, 1, :] = err

        if self.graph_exec:
            cudart.cudaGraphLaunch(self.graph_exec[self._cur], self.stream)
        else:
            self._bind(self._cur)
            cudart.cudaMemcpyAsync(
                self.dev["x"], self._pin_x_ptr, self._x_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
            )
            self.context.execute_async_v3(self.stream)
            cudart.cudaMemcpyAsync(
                self._pin_y_ptr, self.dev["y"], self._y_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            )
        cudart.cudaStreamSynchronize(self.stream)
        self._cur ^= 1
        return self._pin_y.reshape(-1).copy()


class FxLMSEngine:
    """FxLMS 폴백/베이스라인 — anc_project 검증 구현 사용."""

    def __init__(
        self,
        secondary_npz: str,
        fxlms_cfg: dict,
        hop: int = 256,
        handoff_extra_samples: int | None = None,
        *,
        runtime_cfg: dict | None = None,
    ) -> None:
        from ..baselines.fxlms_core import FxLMSController, load_secondary_path

        self.hop = int(hop)
        if self.hop <= 0:
            raise ValueError("FxLMS hop은 양수여야 합니다")
        # RealtimeANC는 입력 블록을 추론 스레드로 넘기고 다음 콜백에서 y를
        # 재생하므로 직접-callback legacy 구현보다 정확히 1 hop이 더 늦다.
        handoff = self.hop if handoff_extra_samples is None else int(handoff_extra_samples)
        if handoff < 0:
            raise ValueError("FxLMS handoff_extra_samples는 0 이상이어야 합니다")
        model = load_secondary_path(secondary_npz)
        if runtime_cfg is not None:
            validate_secondary_calibration(runtime_cfg, model)
        self.handoff_extra_samples = handoff
        self.secondary_delay_samples = int(model.delay_samples) + handoff
        self.secondary_total_length = self.secondary_delay_samples + int(model.fir.size)
        self.controller = FxLMSController(
            model.fir,
            secondary_delay_samples=self.secondary_delay_samples,
            control_len=int(fxlms_cfg.get("control_length", 256)),
            mu=float(fxlms_cfg.get("mu", 0.05)),
            leakage=float(fxlms_cfg.get("leakage", 1.0e-6)),
            weight_norm_limit=float(fxlms_cfg.get("weight_norm_limit", 20.0)),
        )
        # 학습 체크포인트가 없는 적응 필터이므로 runtime lead를 별도로 제한하지 않는다.
        self.digital_reference_lead_samples = None
        self.reference_mode = None  # 적응 필터는 특정 reference 모드로 사전학습하지 않는다.
        # ANC OFF 베이스라인 중 가중치가 몰래 누적되지 않도록 fail-closed로 시작한다.
        self.adapt = False
        self.last_adaptation = None

    def reset(self) -> None:
        self.controller.reset(reset_histories=True)
        self.adapt = False
        self.last_adaptation = None

    def set_adapt_enabled(self, enabled: bool) -> None:
        self.adapt = bool(enabled)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        y = self.controller.generate_block(ref)
        self.last_adaptation = self.controller.adapt_block(err, enabled=self.adapt)
        return y


class HybridEngine:
    """고정 신경망 출력 + FxNLMS 잔차 제어의 첫 비교군.

    두 경로는 같은 REF와 실제 합성 잔류 ERR를 사용한다. 극성은 e=d+S*y이며
    y 합산 뒤의 리미터/ANC fade는 런타임에서 한 번만 적용한다. 이것은 파형
    출력 결합이며 느린 신경망 필터 생성이나 온라인 S 식별을 구현한 것은 아니다.
    """

    def __init__(self, neural: InferenceEngine, adaptive: FxLMSEngine) -> None:
        if neural.hop != adaptive.hop:
            raise ValueError("hybrid 두 엔진의 hop이 같아야 합니다")
        self.neural = neural
        self.adaptive = adaptive
        self.hop = adaptive.hop
        self.digital_reference_lead_samples = neural.digital_reference_lead_samples
        self.reference_mode = getattr(neural, "reference_mode", None)
        self.secondary_delay_samples = adaptive.secondary_delay_samples
        self.secondary_total_length = adaptive.secondary_total_length
        self.reset()

    @property
    def last_adaptation(self):
        return self.adaptive.last_adaptation

    def reset(self) -> None:
        self.neural.reset()
        self.adaptive.reset()

    def set_adapt_enabled(self, enabled: bool) -> None:
        self.adaptive.set_adapt_enabled(enabled)

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        try:
            ref = np.asarray(ref, dtype=np.float32)
            err = np.asarray(err, dtype=np.float32)
            if ref.shape != (self.hop,) or err.shape != (self.hop,):
                raise ValueError("hybrid REF/ERR는 각각 (hop,) 모양이어야 합니다")
            if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(err)):
                raise ValueError("hybrid REF/ERR에 유한하지 않은 값이 있습니다")
            # 분기 사이에서 입력을 수정하는 엔진 때문에 적응 기준이 달라지지 않도록 복사.
            y_neural = np.asarray(self.neural.step(ref.copy(), err.copy()), dtype=np.float32)
            if y_neural.shape != ref.shape or not np.all(np.isfinite(y_neural)):
                raise ValueError("hybrid 신경망 출력의 모양 또는 유한값 검증 실패")
            y = y_neural + self.adaptive.step(ref, err)
            if not np.all(np.isfinite(y)):
                raise ValueError("hybrid 합산 출력에 유한하지 않은 값이 있습니다")
            return y
        except Exception:
            # 일부 경로만 진행된 이력을 다음 블록에 재사용하지 않는다.
            self.reset()
            raise


def secondary_path_npz(runtime_cfg: dict) -> str:
    """S(z) npz 경로 — duct.yaml secondary_path.npz 가 단일 출처 (감사 M9)."""
    from ..config import _resolve_path

    return str(_resolve_path(runtime_cfg["duct"]["secondary_path"]["npz"]))


def build_engine(runtime_cfg: dict) -> InferenceEngine:
    """dl / fxlms / hybrid를 구성하고 학습 reference 모드 혼용을 거부한다."""
    hop = int(runtime_cfg.get("hop", 256))
    controller = str(runtime_cfg.get("controller", "dl"))
    if controller not in {"dl", "fxlms", "hybrid"}:
        raise ValueError(f"알 수 없는 controller: {controller}")
    reference = str(runtime_cfg.get("reference", "digital"))
    if reference not in {"digital", "mic"}:
        raise ValueError(f"reference는 digital/mic이어야 합니다: {reference!r}")
    adaptive = None
    if controller in {"fxlms", "hybrid"}:
        handoff = int(
            runtime_cfg.get("duct", {})
            .get("secondary_path", {})
            .get("handoff_extra_samples", hop)
        )
        if handoff != hop:
            raise ValueError(
                "실시간 FxLMS의 handoff_extra_samples는 실제 1-hop 파이프라인과 "
                f"같아야 합니다: 설정={handoff}, hop={hop}"
            )
        adaptive = FxLMSEngine(
            secondary_path_npz(runtime_cfg),
            runtime_cfg.get("fxlms", {}),
            hop=hop,
            handoff_extra_samples=handoff,
            runtime_cfg=runtime_cfg,
        )
        if controller == "fxlms":
            return adaptive
    eng = runtime_cfg.get("engine", {})
    kind = str(eng.get("type", "torch"))
    if kind == "torch":
        neural = TorchEngine(eng["ckpt"], hop=hop)
    elif kind == "ort":
        neural = OrtEngine(eng["onnx"], hop=hop)
    elif kind == "trt":
        neural = TrtEngine(eng["plan"], eng.get("onnx_meta"), hop=hop)
    else:
        raise ValueError(f"알 수 없는 엔진: {kind}")
    mode = getattr(neural, "reference_mode", None)
    if mode not in {None, "digital", "acoustic"}:
        raise ValueError(f"엔진 reference_mode가 잘못됐습니다: {mode!r}")
    if reference == "mic" and mode != "acoustic":
        raise ValueError(
            "reference=mic에는 reference_mode=acoustic으로 학습한 artifact가 필요합니다. "
            "digital 모델이나 모드 메타가 없는 모델은 사용할 수 없습니다."
        )
    if reference == "digital" and mode == "acoustic":
        raise ValueError("acoustic artifact를 reference=digital에 사용할 수 없습니다")
    if controller == "hybrid":
        assert adaptive is not None
        return HybridEngine(neural, adaptive)
    return neural
