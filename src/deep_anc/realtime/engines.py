"""추론 엔진 — 공통 인터페이스 step(ref, err) → anti-noise.

마이그레이션 경로 (docs/06): torch(개발) → ort(등가성 검증) → trt(배포).
모든 엔진은 내부 상태를 보관하며 reset() 으로 제로 초기화한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np


class InferenceEngine(Protocol):
    hop: int
    digital_reference_lead_samples: int | None

    def reset(self) -> None: ...

    def step(self, ref: np.ndarray, err: np.ndarray) -> np.ndarray:
        """ref/err: (hop,) float32 → anti-noise (hop,) float32."""
        ...


def checkpoint_digital_reference_lead_samples(state: dict) -> int:
    """체크포인트의 학습 lead를 반환한다 (기존 artifact는 lead=0 호환)."""
    cfg = state.get("cfg", {}) or {}
    if "digital_reference_lead_samples" in cfg:
        return int(cfg["digital_reference_lead_samples"])
    # 개발 중 full cfg를 저장했던 임시 artifact도 읽을 수 있게 한다.
    data_cfg = cfg.get("data", {}) or {}
    return int(data_cfg.get("digital_reference_lead_samples", 0))


def engine_digital_reference_lead_samples_from_config(
    runtime_cfg: dict,
) -> int | None:
    """엔진을 생성하지 않고 artifact metadata의 digital lead만 읽는다.

    이 함수는 runtime preflight에서 sounddevice import보다 먼저 쓴다. 추론 엔진을
    실제로 열어 GPU/ORT 세션을 만들 필요 없이, runtime lead와 artifact lead의 모순을
    잡아 입력 probe조차 시작하지 않게 한다. FxLMS는 checkpoint/ONNX artifact가 없으므로
    ``None``을 반환한다.
    """

    if str(runtime_cfg.get("controller", "dl")) == "fxlms":
        return None
    engine = runtime_cfg.get("engine") or {}
    kind = str(engine.get("type", "torch"))
    if kind == "torch":
        import torch

        path = Path(str(engine.get("ckpt", "")))
        if not path.is_file():
            raise FileNotFoundError(f"torch checkpoint가 없습니다: {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise ValueError(f"torch checkpoint 형식이 아닙니다: {path}")
        return checkpoint_digital_reference_lead_samples(state)
    if kind not in {"ort", "trt"}:
        raise ValueError(f"알 수 없는 엔진: {kind}")
    if kind == "ort":
        artifact = Path(str(engine.get("onnx", "")))
        metadata = artifact.with_suffix(".json")
    else:
        artifact = Path(str(engine.get("plan", "")))
        metadata = Path(str(engine.get("onnx_meta"))) if engine.get("onnx_meta") else artifact.with_suffix(".json")
    if not artifact.is_file():
        raise FileNotFoundError(f"{kind} engine artifact가 없습니다: {artifact}")
    if not metadata.is_file():
        raise FileNotFoundError(f"{kind} engine metadata가 없습니다: {metadata}")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{kind} engine metadata를 읽을 수 없습니다: {metadata}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} engine metadata 최상위가 mapping이 아닙니다: {metadata}")
    return int(payload.get("digital_reference_lead_samples", 0))


def _load_ckpt_model(ckpt_path: str | Path):
    import torch

    from ..models import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(state["cfg"]["model"])
    model.load_state_dict(state["model"])
    model.digital_reference_lead_samples = checkpoint_digital_reference_lead_samples(state)
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


def secondary_path_npz(runtime_cfg: dict) -> str:
    """S(z) npz 경로 — duct.yaml secondary_path.npz 가 단일 출처 (감사 M9)."""
    from ..config import _resolve_path

    return str(_resolve_path(runtime_cfg["duct"]["secondary_path"]["npz"]))


def build_engine(runtime_cfg: dict) -> InferenceEngine:
    """runtime.yaml 로 엔진 구성. controller=fxlms 면 FxLMSEngine.

    핸드오프는 **다시 유도하지 않는다**. 예전에는 여기서
    ``duct.secondary_path.handoff_extra_samples`` 를 직접 읽고 hop 과 비교했는데,
    같은 값을 ``run_realtime`` 이 :class:`~deep_anc.realtime.safety.PipelineHandoffBudget`
    으로 또 유도하고 있었다 — 같은 물리량의 두 번째 유도(발생기 A)다. 이제 둘 다
    예산 타입 하나에서 나오고, 예산이 hop 불일치·입출력 비대칭을 생성 시점에 거부한다.
    """
    hop = int(runtime_cfg.get("hop", 256))
    if runtime_cfg.get("controller", "dl") == "fxlms":
        from .safety import PipelineHandoffBudget

        budget = PipelineHandoffBudget.derive(
            duct_cfg=runtime_cfg.get("duct", {}), hop=hop
        )
        return FxLMSEngine(
            secondary_path_npz(runtime_cfg),
            runtime_cfg.get("fxlms", {}),
            hop=hop,
            handoff_extra_samples=budget.handoff_samples,
        )
    eng = runtime_cfg.get("engine", {})
    kind = str(eng.get("type", "torch"))
    if kind == "torch":
        return TorchEngine(eng["ckpt"], hop=hop)
    if kind == "ort":
        return OrtEngine(eng["onnx"], hop=hop)
    if kind == "trt":
        return TrtEngine(eng["plan"], eng.get("onnx_meta"), hop=hop)
    raise ValueError(f"알 수 없는 엔진: {kind}")
