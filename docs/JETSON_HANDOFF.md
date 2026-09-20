# Jetson Orin 인수인계

## 여기까지 확정된 내용

OMAP-L138에서 FxNLMS ANC에 성공했다는 사용자 실험 결과가 출발점이다. Jetson은 아직 그 실험에 사용하지 않았다. `rir.txt`는 성공한 `S_hat`와 동일한 16 kHz / 500탭 계수다. 계수 최대값은 index 49의 +0.01850763이며, 이 위치를 임의로 제거하거나 Jetson 지연 예산으로 해석하지 않는다.

기존 GitHub main `318cca4`의 코드와 README를 보존한 뒤 준비 기능을 추가했다. 원격 저장소 이름은 `tokengeoji/DeepANC`이며 이전 주소와 같은 저장소다. 기존 로컬 추적 파일은 내용 변경 없이 CRLF/LF 차이만 있었고, Python/셸 파일의 LF 정책을 추가했다. OMAP 프로젝트는 중복된 상위 폴더를 없애 `firmware/omap_l138/`로 이동했다. generated Debug/SYS/BIOS 파일과 참고 PDF는 로컬에 보존하고 Git에서는 제외했다.

## 준비된 기능

- 원본 checksum과 DSP `S_hat` 일치 검사, NPY/CSV/C 헤더 및 경로 분석 리포트 재생성.
- 전체 500탭을 유지하는 미분 가능한 인과적 FFT convolution과 비교용 직접 FIR.
- 과거 샘플만 사용하는 작은 시간영역 신경망, 출력 제한, 잔차 `d+S*u` 학습.
- 세션별 train/valid 분리, 동기 녹음 데이터 검사, 신호 단위 보존, 구간 과거 문맥 처리.
- CUDA 실제 연산 점검, 합성 데이터 smoke 학습, 체크포인트 저장/재개.

## Jetson에서 수행할 순서

```bash
git remote set-url origin https://github.com/tokengeoji/DeepANC.git
git pull --ff-only
bash tools/prepare_jetson.sh --install
source .venv/bin/activate
python -m pytest -q
```

기존에 작동하는 Jetson Python 환경이 있으면 먼저 설치 옵션 없이 `bash tools/prepare_jetson.sh`를 사용한다. [환경 안내](JETSON_SETUP.md)에 따라 현재 JetPack에 맞는 NVIDIA PyTorch를 유지한다. 일반 x86 Docker image나 임의의 wheel URL로 덮어쓰지 않는다.

`artifacts/jetson_preflight.json`, `artifacts/secondary_path/report.json`과 `runs/smoke/`를 검토한다. 실측 녹음이 있으면 [데이터 안내](DATASET.md)의 준비 도구로 manifest를 만들고 다음을 실행한다.

```bash
python -m deepanc.train --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --output runs/anc
```

기존 `runs/anc`가 있으면 새 출력 폴더를 사용하거나 해당 체크포인트를 `--resume`으로 명시한다. 재개 옵션은 `python -m deepanc.train --help`로 확인한다. 데이터와 경로 조건을 바꾸면서 이전 모델/optimizer를 무심코 재개하지 않는다.

## 다음 단계에 필요한 실측 정보

Jetson의 모델/RAM, JetPack/L4T/Python/PyTorch/CUDA 버전은 준비 도구가 실제 보드에서 기록한다. 동기 reference/error의 ANC-OFF 녹음, 측정 당시와 현재의 codec/amp gain, 연결 채널·위치와 성공 시 CCS 설정은 실제 실험 기록이 필요하다. 측정된 1차경로 dump는 아직 없다. `NLMS1` 소스만으로 계수를 만들지 않는다.

기본 `delay_samples=0`은 OMAP 측정 경로에 **추가 지연을 가정하지 않는 오프라인 설정**이다. Jetson→OMAP 전송, 블록 버퍼링, inference scheduling 지연이 0이라는 뜻이 아니다. 실시간 구현 전 추가 지연과 기준 마이크의 선행 시간, 출력 headroom, acoustic feedback 경로를 확인해야 한다. 입력 장치를 Jetson USB 오디오 등으로 바꾸면 그 경로의 실측/검증이 필요하다.

## 검증 범위

이 준비 작업을 실행한 PC는 x86_64 CPU 환경이다. 테스트와 합성 smoke 학습은 소프트웨어·수치 동작 확인이며, Jetson CUDA, TI CCS rebuild, 스피커 재생, 실측 소음 감쇠를 대신하지 않는다. 보드에서 이어서 실행한 결과를 이 문서에 추가할 때는 실제 장치·명령·지표·체크포인트 위치를 남긴다.

로컬 43개 테스트와 준비 명령의 실행 범위는 [VALIDATION.md](VALIDATION.md), 학습 설정과 재개 방법은 [TRAINING.md](TRAINING.md)에 기록되어 있다.
