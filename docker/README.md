# Docker 개발 환경

코드·문서 읽기와 편집, Git, Python, 테스트는 컨테이너에서 수행한다.
호스트는 아래 스크립트로 Docker 환경만 관리한다. 호스트 `.venv`는 사용하지 않는다.
현재 실행 상태·검증 결과는 [HANDOFF](../HANDOFF.md)에서 확인한다.

## 1. 환경 선택

| 대상 | 실행 위치 | 구성 |
|---|---|---|
| `cpu` | x86_64 PC | Python 3.10 / PyTorch 2.5.1 CPU |
| `jetson-local` | 기존 cuDNN 9·TensorRT 10이 있는 실제 L4T R36.4 Jetson | CUDA 런타임 이미지 + 호스트 NVIDIA 라이브러리 읽기 전용 + venv 볼륨 설치 |
| `jetson` | 실제 ARM64 Jetson, 충분한 저장공간 | L4T JetPack R36.4.0 이미지 안에 Python 의존성 포함 |

현재 저장공간이 제한된 Jetson에서는 `jetson-local`을 사용한다.
CPU 결과는 Jetson CUDA·TensorRT·지연·오디오 검증을 대체하지 않는다.
두 Jetson 대상은 기존 NVIDIA 컨테이너 런타임을 사용하며 설치·시스템 설정 변경을 하지 않는다.

## 2. 최초 생성

이미 `deep-anc-dev`가 있으면 이 절을 반복하지 말고 §3의 `start`/`exec`를 사용한다.
저장소 루트에서 실행한다. Docker 접근에 기존 sudo 인증이 필요한 호스트에서는 관리 명령에
`sudo`를 붙일 수 있다. `SUDO_UID/GID`를 반영해 컨테이너 사용자는 원래 사용자 UID/GID를 유지한다.
스크립트가 자동으로 sudo를 호출하거나 호스트 권한을 바꾸지는 않는다.

### x86 PC

```bash
bash scripts/docker/dev.sh build cpu
bash scripts/docker/dev.sh up cpu
```

### 현재 Jetson: 저장공간 절약형

```bash
bash scripts/docker/dev.sh build jetson-local
bash scripts/docker/dev.sh up jetson-local
bash scripts/docker/dev.sh exec bash scripts/docker/bootstrap_jetson_local.sh
```

이미지는 Python 도구만 포함한다. 큰 PyTorch wheel과 프로젝트 의존성은 전용 venv 볼륨에
직접 설치하여 이미지 export·unpack 시의 중복 공간을 줄인다.
설치 스크립트는 Docker/ARM64/별도 venv 마운트를 검사하고 동시 설치 잠금을 잡는다.
실패 시 같은 bootstrap 명령으로 재시도할 수 있다. 설치 중 디스크 여유 공간을 확인한다.

`up jetson-local`은 R36.4 호스트의 버전 있는 cuDNN·TensorRT·ONNX parser `.so`와
TensorRT Python 패키지만 읽기 전용 연결한다. 호스트 libc·전체 Python 환경은 연결하지 않는다.
필수 라이브러리가 없으면 중단하며 호스트 apt 설치로 우회하지 않는다.

### 독립 의존성 포함 Jetson 이미지

```bash
bash scripts/docker/dev.sh build jetson
bash scripts/docker/dev.sh up jetson
```

큰 이미지이므로 빌드 캐시·최종 이미지·venv 볼륨 공간을 함께 고려한다.
`jetson-local` 검증을 이 별도 이미지의 검증으로 표기하지 않는다.

## 3. 일상 작업·중지·재개

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh shell
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh stop
bash scripts/docker/dev.sh start
```

`shell` 안의 작업 경로는 `/workspace/Deep-ANC`다. 이 안에서 검색·편집·Git·Python을 실행한다.
셸을 나가도 컨테이너는 유지된다. `stop`/`start`는 코드·venv를 삭제하지 않는다.
`up`은 기존 컨테이너를 자동 교체하지 않는다. 새 이미지 빌드도 기존 환경을 바꾸지 않는다.

저장소는 bind mount이므로 컨테이너의 편집이 실제 저장소에 반영된다.
`.venv`는 `deep-anc-<대상>-venv-<이미지 ID 앞 12자리>` Docker 볼륨이 호스트 `.venv`를 가린다.
`cpu`/`jetson`은 이미지의 venv를 복사하고, `jetson-local`은 빈 디렉터리 소유권만 받아 bootstrap한다.
환경을 재생성할 때도 기존 실험·볼륨을 임의 삭제하지 않는다.

## 4. 무오디오 검증

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q -o addopts= -ra
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --json
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --require-band 1000 1600 --require-broadband
```

readiness의 기본 exit 0은 보고 생성이다. 요구 조건 미달의 exit 1을 설치 오류나 감쇠 성공으로 오독하지 않는다.
실제 Jetson에서만 다음 설치 진단을 추가한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_jetson_stack.py
```

고정 난수의 작은 Conv1d로 CUDA 행렬곱·cuDNN·ORT CPU·TensorRT FP32를 비교한다.
ANC 모델·실시간 마감·오디오 감쇠는 검사하지 않는다. JSON 보존 시 `--output`에 **새 경로**를 준다.

## 5. 버전·재현성·안전

- ONNX Runtime은 **1.18.1 고정**이다. NVIDIA PyTorch와 NVTX/CUPTI/cuSPARSELt 핀은
  [requirements-jetson-runtime.txt](requirements-jetson-runtime.txt)를 두 Jetson 대상이 공유한다.
- [PyTorch 24.08 공식 스택](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-08.html)과
  [JetPack 6.2.1 구성](https://docs.nvidia.com/jetson/jetpack/6.2.1/release-notes/index.html)을 기준으로 하며,
  실제 연산 검증이 필요하다. 이미지 ID·호스트 JetPack 라이브러리·venv 패키지 버전을 함께 기록한다.
- Jetson은 `--runtime nvidia --network host`를 쓴다. 현재 RT 커널의 Docker bridge/iptables raw 문제를
  호스트 방화벽·커널 변경 없이 피하기 위한 것이다. CPU 네트워크 방식은 변경하지 않는다.
- 모든 기본 대상은 **오디오 장치 미노출·비특권 컨테이너**다. `--privileged`를 쓰지 않는다.
  호스트 RT 커널·핀·전원·오디오 서비스·패키지는 변경하지 않는다.
- 호스트 JetPack 라이브러리가 외부 사유로 바뀌면 `jetson-local` 컨테이너를 재생성하고 검증을 다시 한다.
- 개인키·토큰은 이미지·저장소에 복사하지 않는다. Git 작성자·원격·인증은 실제 환경에서 확인한다.
  push는 승인된 대상만 사용하고 force push하지 않는다.

실기 장치 연결·사용자 입회·최소 볼륨·ANC OFF 규칙은 [현장 실행표](../docs/14_pc_jetson_workplan.md)를 따른다.
Docker 진단이 성공해도 실험용 오디오 접근이 준비됐다는 뜻은 아니다.
