# 06. Jetson AGX Orin 배포

배포 환경은 실제 ARM64 Jetson AGX Orin의 Docker다. 현재 호스트는 JetPack 6 / L4T
R36.4.4이며, RT 커널과 30W 구성은 유지한다. 현재 작업 상태와 검증 결과는
[HANDOFF](../HANDOFF.md), 환경 생성·시작 절차는 [Docker 문서](../docker/README.md),
현장 작업의 선행조건·산출물은 [docs/14](14_pc_jetson_workplan.md)를 단일 기준으로 삼는다.

## 1. 환경 준비와 검증 범위

코드·Python·테스트는 모두 컨테이너 안에서 실행한다. 호스트에서는 Docker 환경 관리만 한다.
호스트 가상환경을 사용하거나 패키지를 설치하지 않는다. 핀/I²S, RT 커널, 전원 모드,
클록, 오디오 서비스, 시스템 권한 설정을 변경하지 않는다. `~/anc_project`는 읽기 전용이다.

| 대상 | 구성 | 사용 조건 |
|---|---|---|
| `jetson` | L4T JetPack 이미지 안에 Python 환경 설치 | 이미지·빌드 캐시·venv 볼륨의 저장공간 확보 |
| `jetson-local` | L4T CUDA 이미지 + 전용 venv 볼륨 + 동일 호스트의 cuDNN/TensorRT 읽기 전용 연결 | R36.4 계열 호스트 라이브러리와 Python 바인딩 존재 |
| `cpu` | x86 CPU 개발 이미지 | 코드·합성 회귀용; Jetson GPU 실측을 대체하지 않음 |

venv 준비와 실행 검증은 아래 명령의 실제 결과로 판정하고 HANDOFF에 기록한다.
이미지가 생성됐다는 사실만으로 PyTorch·ONNX Runtime·TensorRT 동작을 완료 처리하지 않는다.
`jetson-local`은 해당 호스트의 라이브러리에 의존하므로 다른 Jetson에서 독립 실행되는 이미지도 아니다.

ONNX Runtime은 Tegra 호환 규약에 따라 **1.18.1로 고정**한다. NVIDIA PyTorch wheel의 보조
라이브러리와 preload 훅은 Docker 환경 정의·부트스트랩이 관리한다. 기존 환경의 파일을 수동 복사하지 않는다.

새 환경을 만드는 경우의 무오디오 절차는 다음과 같다. 이미 `deep-anc-dev`가 있으면
재생성하지 않고 Docker 문서의 `status`·`start`·`exec` 절차를 따른다.

```bash
bash scripts/docker/dev.sh build jetson-local
bash scripts/docker/dev.sh up jetson-local
bash scripts/docker/dev.sh exec bash scripts/docker/bootstrap_jetson_local.sh
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_jetson_stack.py
```

`check_jetson_stack.py`는 고정 난수의 작은 Conv1d 모형으로 CUDA 행렬곱·cuDNN,
opset 17 ONNX의 ORT CPU 실행, TensorRT FP32 빌드·실행 결과를 비교한다. 버전과 오차를 JSON으로
출력하며 `--output`은 기존 파일을 덮어쓰지 않는다. 이 모형은 설치 진단용이고 ANC 학습모델이 아니다.
통과해도 프로젝트 모델의 스트리밍 등가성·추론 지연·실제 감쇠 검증은 별도로 남는다.

## 2. 프로젝트 모델의 배포 게이트

| 단계 | 확인할 내용 |
|---|---|
| 학습 산출물 | 실제 checkpoint·해결된 설정·학습 데이터 정책·플랜트 근거 확인 |
| 추론 등가성 | 오프라인↔스트리밍, PyTorch↔ONNX의 수치 일치 |
| 엔진 선택 | Torch 개발 경로, ORT CPU 경로, TensorRT 배포 경로를 해당 환경에서 각각 검증 |
| 지연 | 실제 모델·블록·전원 조건을 기록하고 `measure_inference_latency.py`로 분포 측정 |
| 실기 자격 | 경로·입출력 상태·안전 조건 확인 후 독립 OFF→ON→OFF 평가 |

block 256 / 48 kHz의 블록 시간은 5.33 ms이며 프로젝트 추론 게이트는 **P99 < 3.0 ms**다.
이 게이트는 추론 처리시간에 대한 것이고 I/O·음향 전달·콜백 마감 전체의 통과를 뜻하지 않는다.
과거 호스트 실행이나 다른 모델의 지연 수치를 현재 Docker 배포 결과로 재사용하지 않는다.

프로젝트 ONNX는 opset 17, 정적 shape, 명시적 상태 입출력을 사용한다.
학습·ONNX 메타데이터·런타임의 reference mode와 digital lead는 일치해야 한다.
현재 우선 모드인 acoustic-ref는 실제 REF 마이크를 사용하며 미래 소스나 digital preview를 공급하지 않는다.
`secondary_surrogate` 기반 체크포인트의 합성 결과를 실제 P/S 경로의 감쇠 성능으로 인용하지 않는다.
모델·스트리밍 계약은 [docs/04](04_model_architecture.md), 지연 해석은
[docs/01](01_physics_limits.md)과 [docs/13](13_acoustic_hybrid.md)을 따른다.

## 3. 오디오 현장 검증

기본 개발 컨테이너는 오디오 장치를 노출하지 않는다. 실제 장치·채널·권한과 컨테이너 접근
구성을 확인한 뒤 [docs/14의 현장 순서](14_pc_jetson_workplan.md#4-jetson에서만-수행할-단계)를 따른다.
스피커 출력은 사용자 입회·물리 볼륨 최소 상태에서만 수행하고, 런타임은 항상 ANC OFF로 시작한다.
입력 게이트, 경로 측정, I/O 지연, 실제 ANC 감쇠는 설치 진단과 구분해 기록한다.

실기에서는 유효한 REF/ERR 입력, 측정 S의 조건·극성·신뢰대역, REF 선행 시간과 피드백 경로,
xrun·클리핑·클록 안정성을 확인한다. 시작 OFF, 출력 제한·페이드, 발산·데드라인 감시를 유지한다.
실패한 측정이나 과거 경로의 메타데이터를 바꿔 통과시키지 않고 새 결과 경로를 사용한다.

성공 기준은 **ERR 한 점에서 저역·고역을 모두 감쇠하고 음성·음악을 포함한 모든 소리를 줄이는 것**이다.
Jetson의 우선 개선 대역은 **1000–1600 Hz**이며 이 대역의 개선만으로 전체 목표를 완료 처리하지 않는다.
DSP/Jetson 출력 합산·스피커 공유 방식은 미정이므로 배포 과정에서 임의의 듀얼 제어기 배선을 추가하지 않는다.
