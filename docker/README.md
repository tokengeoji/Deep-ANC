# Docker 개발 환경

개발·분석·테스트와 Python 실행은 컨테이너 안에서 수행한다.
호스트에서는 `scripts/docker/dev.sh`로 Docker 환경을 관리한다.
기존 호스트 `.venv`를 보존하며, 컨테이너는 전용 volume의 `.venv`를 사용한다.

## 1. CPU 개발과 Jetson 검증의 구분

| 항목 | `cpu` | `jetson` |
|---|---|---|
| 실행 위치 | x86_64 개발 머신 | 실제 ARM64 Jetson AGX Orin |
| 이미지 기반 | Python 3.10 / Debian Bookworm | NVIDIA L4T JetPack R36.4.0 |
| PyTorch | 2.5.1 CPU wheel | NVIDIA Jetson 2.5.0a0 wheel |
| 용도 | 코드 수정·회귀 테스트·오프라인 진단 | Jetson 라이브러리·GPU·추론 검증 |

CPU 이미지는 Jetson 에뮬레이터가 아니다. CPU 테스트가 통과해도 Jetson의 CUDA,
TensorRT, 처리 지연, 실제 오디오 경로가 검증된 것은 아니다.
Jetson Dockerfile은 준비된 빌드 정의이며 **현재 실제 Jetson에서의 이미지 빌드·CUDA·TensorRT 검증은 미완료**다.
2026-09-15 CPU 컨테이너의 전체 테스트는 523개 통과, 현장 자료 부재 2개 건너뜀이며 `pip check`도 통과했다.

환경 정의는 [Dockerfile.cpu](Dockerfile.cpu), [Dockerfile.jetson](Dockerfile.jetson),
공통 검증 패키지는 [requirements-dev.txt](requirements-dev.txt)에 있다.
ONNX Runtime은 프로젝트의 Tegra 호환 규약에 따라 **1.18.1로 고정**한다.

## 2. x86 CPU 개발 시작

기존 Docker가 현재 사용자 권한으로 사용 가능한 환경에서 저장소 루트를 기준으로 실행한다.
이 절차는 Docker 설치나 호스트 권한·시스템 설정 변경을 수행하지 않는다.

```bash
bash scripts/docker/dev.sh build cpu
bash scripts/docker/dev.sh up cpu
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
```

`build`는 이미지를 만들고, `up`은 `deep-anc-dev` 컨테이너를 처음 생성한다.
Dockerfile의 기본 사용자는 호스트 UID/GID에 맞춘 `ancdev`다.
테스트 결과는 실행한 코드와 환경의 실제 출력을 기준으로 기록한다.

대화형 작업은 다음 명령으로 컨테이너 셸에 들어간 뒤 수행한다.

```bash
bash scripts/docker/dev.sh shell
```

셸의 작업 디렉터리는 `/workspace/Deep-ANC`다. 이 안에서 `rg`, `git diff`, 파일 편집,
`.venv/bin/python` 실행을 수행한다. 셸에서 `exit`해도 개발 컨테이너는 계속 실행된다.
명령 하나만 실행하려면 `exec` 뒤에 프로그램과 인자를 전달한다.

```bash
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --json
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --require-band 1000 1600 --require-broadband
```

readiness는 설정과 저장된 S 경로만 읽는 무출력 검사다. 기본 보고의 exit 0은 광대역
준비 완료를 뜻하지 않는다. 선택한 대역·광대역 조건을 충족하지 못하면 요구 검사에서 exit 1이 나온다.
현재 acoustic 개발 순서는 [docs/13_acoustic_hybrid.md](../docs/13_acoustic_hybrid.md)를 따른다.

## 3. 중지와 재시작

```bash
bash scripts/docker/dev.sh stop
bash scripts/docker/dev.sh start
bash scripts/docker/dev.sh status
```

- `stop`: 기존 `deep-anc-dev`를 중지한다. 코드·컨테이너·volume을 삭제하지 않는다.
- `start`: 중지한 기존 컨테이너를 다시 시작한다. 이미지 빌드나 컨테이너 생성은 하지 않는다.
- `status`: 컨테이너 이름, 실행 상태, 이미지 이름을 확인한다.
- `exec`·`shell`: 실행 중인 `deep-anc-dev`에 접속한다.

`up`은 이름이 같은 컨테이너가 이미 있으면 중지 상태여도 생성을 거부한다.
중지한 환경을 다시 사용할 때는 `start`를 사용한다.
새 이미지를 `build`해도 기존 컨테이너와 가상환경은 교체되지 않는다.
새 이미지를 적용하려면 컨테이너 교체가 필요하며, 관리 스크립트는 기존 환경을 자동 삭제하지 않는다.

## 4. 코드와 Python 환경의 저장 위치

호스트 저장소는 컨테이너의 `/workspace/Deep-ANC`에 바인드 마운트된다.
따라서 컨테이너 안에서 수정한 코드·문서·생성한 결과는 원본 저장소에도 반영된다.
Docker는 코드 변경을 별도 복사본으로 격리하는 방식이 아니다.

그 아래 `.venv`에는 별도 Docker volume을 마운트하여 호스트 `.venv`를 가린다.
volume 이름은 `deep-anc-<대상>-venv-<이미지 ID 앞 12자리>` 형식이다.
처음 생성할 때 이미지에 설치된 Python 환경으로 volume을 채우므로 CPU와 Jetson,
서로 다른 이미지의 가상환경이 섞이지 않는다.
기존 호스트 `.venv`와 환경 파일을 복사해 컨테이너 Python 환경으로 사용하지 않는다.

상위 폴더가 Git 루트인 배치에서는 상위 `.git`을 읽기 전용으로 연결해
컨테이너에서 `git status`·`git diff`를 확인할 수 있게 한다.
이미지 빌드 입력에서는 `.dockerignore`로 `.venv`, 데이터·실험 결과, 모델 파일,
일반적인 키·환경 파일 패턴을 제외한다. 비밀정보를 저장소에 커밋하지 않는 규칙도 계속 적용한다.

## 5. 실제 Jetson에서 사용할 환경

아래 명령은 **기존 JetPack 6 / L4T R36.4 계열의 실제 ARM64 Jetson**에서만 실행한다.
x86에서 Jetson 이미지를 빌드하거나 실행하지 않는다.
현재 Jetson에 NVIDIA 컨테이너 런타임이 제공되어 있어야 하며 스크립트가 이를 설치하지 않는다.

```bash
bash scripts/docker/dev.sh build jetson
bash scripts/docker/dev.sh up jetson
bash scripts/docker/dev.sh exec .venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
bash scripts/docker/dev.sh exec .venv/bin/python -c 'import tensorrt; print(tensorrt.__version__)'
```

기본 베이스는 `nvcr.io/nvidia/l4t-jetpack:r36.4.0`이며 `up jetson`은 `--runtime nvidia`를 사용한다.
Jetson 가상환경은 이미지의 시스템 Python 패키지를 함께 볼 수 있고,
NVIDIA PyTorch wheel과 컨테이너 가상환경 내부의 라이브러리 preload 훅을 설치한다.
이 구성의 실제 호환성은 위 현장 검사와 후속 추론 검사로 확인해야 한다.
라이브러리 import 성공만으로 스트리밍 등가성이나 실시간 마감 충족을 판정하지 않는다.

컨테이너는 **호스트의 NVIDIA L4T 커널과 드라이버를 공유**한다.
기존 RT 커널·핀 설정·전원 모드·오디오 서비스를 그대로 유지한다.
Jetson 환경의 실패를 호스트 드라이버나 시스템 설정을 바꾸는 방식으로 우회하지 않는다.

## 6. 기본 컨테이너의 장치 접근 범위

관리 명령에는 `sudo`와 `--privileged`를 사용하지 않는다.
Dockerfile의 패키지 설치는 이미지 파일시스템 안에서만 수행한다.
기본 `cpu`와 `jetson` 컨테이너에는 `/dev/snd` 등 **오디오 장치를 노출하지 않는다**.
따라서 기본 개발 환경에서 마이크 입력 검사나 스피커 실험을 완료했다고 주장할 수 없다.

실기 오디오 단계에서는 실제 Jetson의 장치 연결과 접근 조건을 별도로 준비해야 한다.
소리가 나는 실험은 사용자 입회·볼륨 최소 상태에서만 수행하고 항상 ANC OFF로 시작한다.
`~/anc_project` 읽기 전용과 Jetson 시스템 변경 금지 규칙은 컨테이너 안에서도 유지한다.

## 7. 작업 분담과 Git 반영

[현재 PC / Jetson 실행표](../docs/14_pc_jetson_workplan.md)에 현장 전용 작업의
선행조건·명령·산출물·중단조건을 정리했다. CPU에서 가능한 코드·합성 시험·분석은 현장 준비와 병행한다.

로컬 `git status`, diff, staging 등도 컨테이너에서 수행한다. 기본 이미지에 호스트의
Git 작성자 설정이나 인증정보가 들어 있다고 가정하지 않는다. 개인키·토큰을 이미지나 저장소에 복사하지 않는다.
인증된 GitHub 연결로 반영할 경우에도 검증한 tree와 부모 커밋을 대조하고 force push하지 않는다.
현재 원격은 `tokengeoji/Deep-ANC`이며 이전 `Roka-jsj/Deep-ANC` 주소도 같은 저장소로 연결된다.
