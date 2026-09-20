# Jetson Orin에서 이어서 준비하기

이 절차는 GitHub 코드를 받은 Jetson에서 실측 2차 경로를 검증하고, 딥러닝의 순전파·역전파 smoke test까지 실행한다. 실제 소음 녹음과 음향 루프 검증은 별도 단계다. 이 저장소를 정리한 개발 환경에는 Jetson이 없었으므로 보드 검증 결과를 미리 주장하지 않는다. 아래 명령이 실제 Jetson에서 성공해야 GPU 준비가 확인된다.

## 1. 코드와 Python 환경

기존 clone에서는 작업 파일을 먼저 확인하고 업데이트한다.

```bash
git status --short
git pull --ff-only
bash tools/prepare_jetson.sh --install
```

새로 clone하는 경우:

```bash
git clone https://github.com/tokengeoji/DeepANC.git
cd DeepANC
bash tools/prepare_jetson.sh --install
```

`--install`은 Python 3.8 이상을 사용하는 `.venv`를 `--system-site-packages`로 만들고 `requirements-jetson.txt`의 NumPy, SciPy, SoundFile, pytest만 설치한다. 시스템의 NVIDIA PyTorch를 그대로 사용하기 위해 PyTorch는 설치하거나 업그레이드하지 않는다. NumPy는 오래된 NVIDIA 빌드의 ABI를 고려해 1.x로 제한한다. 기존 `.venv`가 시스템 패키지를 차단하면 자동 변경하지 않고 원인을 출력한다. `venv`나 pip가 없으면 해당 JetPack의 Python 환경에 먼저 준비해야 한다.

의존성이 이미 준비되었다면 설치 없이 아래 명령만 실행한다.

최소 OS 이미지에 `ensurepip`가 없더라도 기존 `venv`와 `pip`가 있으면 시스템 pip 모듈을 공유하여 가상환경 안에 설치한다. `venv/pip` 모듈이 없다는 메시지가 나오면 JetPack 기본 Python에서 `sudo apt install python3-venv python3-pip`로 OS 패키지를 준비한 뒤 다시 실행한다. 기본 Python이 아닌 버전이라면 그 버전의 `venv` 패키지가 필요하다. 스크립트는 OS 패키지 설치를 자동으로 수행하지 않는다. `soundfile` import에서 `libsndfile` 오류가 발생하면 `sudo apt install libsndfile1`을 사용한다.

```bash
bash tools/prepare_jetson.sh
```

다른 환경의 Python을 사용할 수 있다. `--install` 없이 `--python`을 지정하면 그 환경을 그대로 쓴다. `--install`과 함께 지정하면 새 `.venv`를 만들 때 사용하는 기본 Python이 된다.

```bash
bash tools/prepare_jetson.sh --python /path/to/environment/bin/python
```

스크립트는 자신의 위치로 저장소 경로를 찾으므로 다른 작업 디렉터리에서도 실행할 수 있다. 자동 `sudo`, JetPack 업그레이드, 전력 모드 변경, `jetson_clocks` 실행은 하지 않는다.

## 2. NVIDIA PyTorch가 없는 경우

JetPack/L4T와 PyTorch 빌드를 먼저 맞춰야 한다. `pip install torch`로 PyPI의 기본 빌드를 덮어쓰면 Jetson CUDA가 동작하지 않을 수 있다. 설치 안내는 [NVIDIA 공식 설치 문서](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html), 빌드 선택은 [JetPack/PyTorch 호환성 표](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html)를 따른다. wheel과 컨테이너 제공 범위가 JetPack 릴리스마다 다르므로 임의의 wheel URL이나 컨테이너 태그를 사용하지 않는다.

학습에는 `torch.fft.rfft`/`irfft`와 `torch.load(weights_only=...)`가 필요하다. upstream PyTorch 1.13 이상에서 지원하는 조합이며, preflight는 NVIDIA 버전 문자열 대신 실제 API 유무를 확인한다. `weights_only`는 [PyTorch 1.13 소스](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/serialization.py#L600)에 명시되어 있다. 오래된 빌드에서 CUDA가 동작하더라도 이 API가 없으면 준비 실패로 보고하므로 학습 이후 재개 단계에서 뒤늦게 오류가 나지 않게 한다. 필요한 경우 호환성 표에서 현재 JetPack에 맞는 NVIDIA 빌드를 직접 선택하고, 준비 스크립트가 자동 업그레이드하지 않도록 유지한다.

환경 점검만 하려면:

```bash
python3 tools/jetson_preflight.py --require-jetson --require-cuda
```

점검 결과는 `artifacts/jetson_preflight.json`에 저장된다. 기록 항목은 CPU 아키텍처, 보드 모델, L4T, `nvidia-jetpack` 패키지 버전, Python과 의존성 버전, PyTorch 설치 위치, CUDA/cuDNN 버전, GPU, RAM과 디스크 여유 공간이다. `torch.cuda.is_available()` 결과뿐 아니라 실제 GPU 메모리 할당, 연산, 역전파와 유한값까지 확인한다. GPU 검사가 실패하거나 요구한 Jetson 하드웨어가 확인되지 않으면 종료 코드가 0이 아니다. `nvidia-jetpack` 메타패키지가 없어도 L4T가 확인되는 설치는 있을 수 있다.

## 3. 준비 명령이 확인하는 내용

1. Jetson 하드웨어와 CUDA 연산/역전파, Python 의존성.
2. 루트 `rir.txt`와 교정 메타데이터 및 성공한 DSP 계수의 일치.
3. 실측 16 kHz, 500-tap 2차 경로의 파생 파일 생성 (`artifacts/secondary_path/`).
4. 합성 데이터로 작은 학습 실행 (`runs/smoke/run-XXXXXXXX/`). 매번 새 폴더를 만들어 이전 결과를 보존한다.

2차 경로의 측정 이득, 부호, 지연과 500개 탭을 그대로 보존한다. 학습용 파일을 생성하는 과정에서 크기 정규화, peak 정렬, tail 절삭을 적용하지 않는다. 이 경로는 측정 당시의 DAC/ADC 이득, 스피커·마이크 위치, 배선, 샘플링 주파수에 대응한다. 이 조건을 바꾸면 경로를 다시 측정하고 교정 정보를 갱신해야 한다.

`--allow-cpu`는 개발 PC에서 파일과 코드 흐름을 확인하는 옵션이다. GPU가 있어도 모델 smoke test는 CPU로 실행하며 Jetson 준비 완료를 의미하지 않는다.

```bash
bash tools/prepare_jetson.sh --allow-cpu
```

## 4. 실제 학습으로 이어가기

`docs/DATASET.md`에 따라 동기 reference/primary 녹음을 준비하고 `datasets/anc/prepared/manifest.jsonl`을 만든다. 기존 OMAP FxNLMS 제어 중의 잔류 오차를 제어하지 않은 primary target과 혼동하지 않는다. 필요한 녹음이 없는 상태에서는 smoke test까지만 가능하다.

```bash
.venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda \
  --output runs/anc
```

`.venv` 대신 기존 Python 환경을 선택했다면 같은 인터프리터로 실행한다. 다음 세션에서 “이어서 해줘”라고 요청할 때는 저장소의 `AGENTS.md`와 인수인계 문서, preflight 결과, 녹음 manifest를 함께 확인하도록 구성되어 있다. 기존 OMAP FxNLMS 성공 경로는 학습 smoke test와 별도로 실제 음향 성능의 기준으로 유지한다.
