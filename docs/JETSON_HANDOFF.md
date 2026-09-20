# OMAP 실측 경로의 통합 후 인수인계

현재 저장소와 전체 작업 상태는 [루트 HANDOFF.md](../HANDOFF.md)가 단일 출처다. 이 문서는 `tokengeoji/DeepANC`에서 `tokengeoji/Deep-ANC`로 가져온 OMAP 16 kHz 오프라인 학습 경로만 설명한다. 작업 규칙과 환경 설치는 [AGENTS.md](../AGENTS.md), [Docker 안내](../docker/README.md)를 따른다.

## 보존한 실험과 코드

사용자는 OMAP-L138 단독 FxNLMS ANC 성공을 보고했다. Jetson이 참여하지 않았다는 설명은 그 성공 실험에 한정된다. 기존 Deep-ANC의 Jetson CUDA·ORT·TensorRT 검증 이력을 부정하지 않으며, 과거 Jetson 설치 검증도 새 OMAP–Jetson 결합 성능을 입증하지 않는다.

루트 `rir.txt`는 성공한 DSP `S_hat`와 동일한 16 kHz·500탭 계수다. gain·부호·500개 탭과 선행 지연을 보존한다. peak index 49를 제거하거나 Jetson 지연 예산으로 해석하지 않는다. 원본 checksum과 실측 근거는 [HARDWARE_BASELINE.md](HARDWARE_BASELINE.md)에 있다.

| 경로 | 용도와 구분 |
| --- | --- |
| `deepanc/`, `configs/anc_train.json` | OMAP 16 kHz RIR을 사용하는 작은 오프라인 신경망 |
| `src/deep_anc/`, 기존 YAML 설정 | 기존 Jetson 48 kHz 데이터·모델·평가·실시간 경로 |
| `firmware/omap_l138/` | 성공한 FxNLMS0 및 NLMS 측정 프로젝트 원본 |
| `scripts/train.py`, `scripts/utils/` | 별도 보존한 과거 GCRN 음성 향상 코드 |

16 kHz `rir.txt`를 기존 48 kHz `assets/measured/`의 NPZ로 대체해 사용하지 않는다. 기존 NPZ의 별도 측정 delay와 handoff 규약도 OMAP FIR에 이식하지 않는다. 두 Python 경로의 데이터 형식·tensor 형태·체크포인트는 자동 호환되지 않는다.

## 이어서 실행할 순서

기존 Docker 환경에서 Git 상태와 실제 실행 장치를 먼저 확인한다. 원격이 하이픈이 있는 `tokengeoji/Deep-ANC`인지 확인하고 기존 작업을 보존한다.

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh exec git remote -v
bash scripts/docker/dev.sh exec git pull --ff-only
# 실제 Jetson 컨테이너에서:
bash scripts/docker/dev.sh exec bash tools/prepare_jetson.sh
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
```

PC 컨테이너에서는 준비 명령에 `--allow-cpu`를 붙인다. 별도 호스트 venv나 `--install`은 사용하지 않는다. 준비 결과인 `artifacts/jetson_preflight.json`, `artifacts/secondary_path/report.json`, 새 `runs/smoke/` 폴더를 확인한다.

OMAP 조건에 맞는 실측 녹음이 있으면 [데이터 안내](DATASET.md)대로 manifest를 준비한 뒤 작은 학습부터 시작한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --epochs 1 --output runs/omap_anc
```

기존 결과가 있으면 새 출력 폴더를 쓰거나 [TRAINING.md](TRAINING.md)의 같은 실행 resume 절차를 따른다. 새로운 데이터·경로 조건을 이전 모델/optimizer에 조용히 섞지 않는다.

## 다음 단계에 필요한 실측 정보

OMAP raw reference/error의 ANC-OFF 동기 녹음, 측정 당시와 현재의 codec/amp gain, 연결 채널·위치와 CCS 설정을 확인한다. 가져온 자료에는 실측 1차경로 dump가 없으며 `NLMS1` 소스만으로 계수를 만들지 않는다. 기존 Jetson 또는 Drive 자료가 있더라도 조건과 단위를 확인하기 전에는 OMAP 학습 데이터로 간주하지 않는다.

`secondary_delay_samples=0`은 OMAP RIR 외의 추가 지연을 가정하지 않는 오프라인 설정이다. Jetson→OMAP 전송·블록 버퍼링·inference 지연이 실제로 0이라는 뜻이 아니다. 실제 결합은 전송·출력 구조가 정해지고 추가 지연, 기준 마이크 선행 시간, headroom, feedback 경로를 측정한 뒤 별도 진행한다. 출력 합산이나 스피커 공유를 이 준비 단계에서 임의 구현하지 않는다.

원래 DeepANC에서 수행한 CPU 43개 테스트와 smoke 기록은 [VALIDATION.md](VALIDATION.md)에 역사적 기록으로 보존했다. 통합 후 전체 회귀 결과와 현재 장치에서 확인한 상태는 루트 HANDOFF에 기록한다. 소프트웨어 테스트는 TI CCS rebuild나 실제 음향 감쇠를 대신하지 않는다.
