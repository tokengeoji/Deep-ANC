# DeepANC: OMAP-L138 실측 경로에서 Jetson Orin 학습까지

OMAP-L138에서 성공한 FxNLMS 코드와 실제 측정한 `rir.txt`를 기준으로, Jetson Orin에서 ANC 딥러닝을 준비하는 저장소다. 현재 GitHub 주소는 **https://github.com/tokengeoji/DeepANC**이다. 이전 `Roka-jsj/DeepANC` 주소는 같은 저장소로 연결된다.

`rir.txt`는 16 kHz, 500-tap 2차경로이며 성공한 DSP `S_hat`와 모든 계수가 일치한다. 원본 gain·부호·선행 지연을 보존한다. 신경망은 실제 DAC command `u`를 출력하고 `e = d + S*u`의 잔차를 줄이도록 학습한다. FFT 선형 convolution으로 500탭을 그대로 계산한다.

## Jetson에서 시작

```bash
git remote set-url origin https://github.com/tokengeoji/DeepANC.git
git pull --ff-only
bash tools/prepare_jetson.sh --install
```

준비 명령은 Jetson/JetPack/PyTorch CUDA 확인, 실측 계수 검증과 변환, 합성 데이터 1회 학습 점검을 수행한다. 설치된 NVIDIA PyTorch를 유지하며 일반 PyPI `torch`로 교체하지 않는다. CUDA PyTorch가 없거나 JetPack과 맞지 않으면 원인과 설치 문서가 출력된다. 이 경우 [Jetson 환경 준비](docs/JETSON_SETUP.md)를 따른다.

다른 PC에서는 다음으로 수치 검증을 할 수 있다.

```bash
bash tools/prepare_jetson.sh --allow-cpu
python3 -m pytest -q
```

실측 동기 녹음 데이터가 준비되면 [데이터 준비](docs/DATASET.md)에 따라 manifest를 만들고 학습한다.
모델 구성·손실·체크포인트 재개 옵션은 [학습 안내](docs/TRAINING.md)에 정리되어 있다.

```bash
source .venv/bin/activate  # --install로 가상환경을 만든 경우
python -m deepanc.train --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --output runs/anc
```

## 디렉터리

```text
rir.txt                       실측 2차경로 정본 (원본 바이트 유지)
calibration/                  경로의 단위·샘플레이트·checksum·출처
firmware/omap_l138/            성공한 FxNLMS0 및 NLMS0/NLMS1 CCS 소스
deepanc/                      인과적 신경망, 데이터 로더, 2차경로, ANC 학습
configs/                      Jetson용 소규모 학습 기본 설정
tools/                        환경 점검, 계수 변환, 녹음 데이터 준비
tests/                        물리 경로·인과성·데이터·학습 검증
docs/                         하드웨어 근거, 데이터, Jetson 인수인계
scripts/                      기존 GCRN 음성 향상 코드 (호환 경로 보존)
datasets/                     녹음·학습 데이터 (로컬, Git 제외)
artifacts/                    계수 파생 파일·환경 점검 결과 (Git 제외)
runs/                         학습 결과·체크포인트 (Git 제외)
```

## 이어서 작업할 때

Jetson에서 “이어서 해줘”라고 요청하면 루트 [AGENTS.md](AGENTS.md)와 [JETSON_HANDOFF.md](docs/JETSON_HANDOFF.md)에 따라 환경 확인부터 시작한다. 현재 소스 기본값과 실측의 근거는 [HARDWARE_BASELINE.md](docs/HARDWARE_BASELINE.md)에 기록되어 있다. 기존 speech enhancement 실험은 [별도 문서](docs/LEGACY_SPEECH_ENHANCEMENT.md)에 보존했다.

준비된 신경망은 인과적 시간영역 ANC의 **오프라인 출발점**이다. 합성 학습 점검은 실측 감쇠 성능을 의미하지 않는다. 실제 학습에는 ANC를 끈 상태에서 동시에 수집한 reference `x`와 오류 마이크 disturbance `d`가 필요하다. Jetson 오디오 장치나 전송 방식을 바꾸면 추가 지연과 경로를 측정해야 하며, 이 저장소는 아직 Jetson 실시간 출력·OMAP 통신을 구현하지 않는다.
