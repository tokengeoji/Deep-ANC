# OMAP 16 kHz 경로의 Jetson 오프라인 학습 준비

이 절차는 `tokengeoji/DeepANC`에서 가져온 OMAP-L138 실측 경로를 현재 [tokengeoji/Deep-ANC](https://github.com/tokengeoji/Deep-ANC)에서 실행하는 방법이다. 준비 명령은 원본 `rir.txt`의 16 kHz·500탭을 검증하고 작은 `deepanc` 모델의 합성 순전파·역전파를 실행한다. 이 OMAP 성공 실험에는 Jetson이 사용되지 않았다. 기존 `src/deep_anc/`의 Jetson 검증 이력과 현재 통합 상태는 루트 [HANDOFF.md](../HANDOFF.md)를 따른다.

OMAP 경로와 기존 Jetson USB 오디오의 48 kHz NPZ 경로는 별도다. `configs/duct.yaml`이나 `assets/measured/`를 이 RIR로 교체하지 않는다. 데이터·입출력 gain·지연·체크포인트 역시 서로 자동 호환되지 않는다.

**현재는 실측 전·새 학습 전 준비 단계다.** 아래 `prepare_jetson.sh`·전체 pytest·학습 명령은
실제 학습/역전파를 포함하므로 현재 실행하지 않는다. 무학습 절차는
[실측 전 실행 안내](20_measurement_runbook.md), 허용 회귀 목록은 [docs/19](19_high_frequency_comparison.md)를 따른다.
아래 절차는 별도 학습 재개 승인 이후의 일반 `deepanc` 환경 검사이며 SFANC 준비 CLI가 아니다.

## 1. 기존 Docker 환경 사용

코드·Git·Python·테스트는 Docker 내부에서 수행한다. 호스트 `.venv`를 만들거나 호스트 패키지를 설치하지 않는다. 아래 명령은 저장소 루트에서 Docker 관리 스크립트를 호출한다.

```bash
bash scripts/docker/dev.sh status
# 중지된 기존 컨테이너일 때만:
bash scripts/docker/dev.sh start
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh exec git remote -v
bash scripts/docker/dev.sh exec git pull --ff-only
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
```

원격은 하이픈이 있는 `tokengeoji/Deep-ANC`인지 확인한다. 기존 로컬 작업이나 다른 원격을 자동으로 덮어쓰지 않는다. 새 환경은 [Docker 안내](../docker/README.md)에 따라 PC에서는 `cpu`, 실제 Jetson에서는 `jetson-local` 또는 `jetson`을 준비한다. 예전 소스 문서의 `--install`과 호스트 venv 설치 절차는 통합 저장소에서 사용하지 않는다.

## 2. 무오디오 준비 실행

실제 Jetson의 기존 컨테이너에서 실행한다.

```bash
bash scripts/docker/dev.sh exec bash tools/prepare_jetson.sh
```

이 명령은 다음을 확인한다.

1. Jetson 하드웨어, Python 의존성, CUDA 실제 연산·역전파 및 학습에 필요한 API.
2. 루트 `rir.txt`의 checksum과 성공한 DSP `S_hat` 일치.
3. 원본 전체 500탭의 파생 파일 생성 (`artifacts/secondary_path/`).
4. 합성 데이터의 작은 학습과 체크포인트 저장 (`runs/smoke/run-XXXXXXXX/`).

새 smoke 폴더를 매번 생성해 이전 결과를 보존한다. 정본의 크기 정규화, peak 정렬, tail 절삭, 부호 반전이나 resampling은 하지 않는다. 스피커 출력과 OMAP 펌웨어 변경도 수행하지 않는다.

PC의 CPU 컨테이너에서는 다음으로 동일한 오프라인 흐름을 확인한다. GPU가 있어도 이 옵션의 학습은 CPU로 실행하며 Jetson 준비 완료를 의미하지 않는다.

```bash
bash scripts/docker/dev.sh exec bash tools/prepare_jetson.sh --allow-cpu
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
```

환경 진단만 실행하려면:

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/jetson_preflight.py \
  --require-jetson --require-cuda
```

`artifacts/jetson_preflight.json`에서 실제 CPU 아키텍처·L4T·PyTorch/CUDA와 CUDA 연산 결과를 확인한다. 단순히 `torch.cuda.is_available()`가 참인 것만으로 통과하지 않는다. 기존 NVIDIA PyTorch를 일반 PyPI torch로 교체하지 않는다. CUDA 또는 의존성이 누락되면 [Docker 안내](../docker/README.md)의 해당 컨테이너 구성·bootstrap을 확인하며, 호스트 `apt`나 시스템 설정 변경으로 우회하지 않는다.

## 3. 실제 OMAP 녹음으로 학습

[DATASET.md](DATASET.md)에 따라 16 kHz raw reference와 ANC-OFF disturbance를 준비한다. 기존 48 kHz USB 세션이나 ANC-ON 잔류 오차를 이 학습의 target으로 바로 사용하지 않는다. OMAP 조건의 동기 녹음이 없다면 합성 smoke test 이후 실제 학습을 시작할 수 없다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --output runs/omap_anc
```

`--device cuda`가 실패하면 CPU로 조용히 대체하지 않는다. 녹음 조건과 RIR은 측정 당시의 gain·배선·위치·샘플링 주파수에 대응한다. 장치 경로를 바꾸면 새 경로와 추가 지연을 검증해야 한다. 학습·재개 옵션은 [TRAINING.md](TRAINING.md), OMAP 관련 후속 확인은 [JETSON_HANDOFF.md](JETSON_HANDOFF.md)를 따른다.
