# OMAP 실측 2차경로를 사용하는 오프라인 ANC 학습

`deepanc/`는 실측 `rir.txt`를 통과한 제어 신호가 오류 마이크의 소음을 줄이도록 학습하는 작은 시간영역 기준 모델이다. 현재 준비된 것은 데이터 검증, 학습, 오프라인 검증과 체크포인트 저장까지다. 실제 ANC-OFF 동기 녹음은 저장소에 포함되어 있지 않으므로 녹음을 준비하기 전에는 합성 smoke test만 실행할 수 있다.

이 모델은 첨부 논문이나 기존 GCRN의 재현 구현이 아니다. 보존한 `scripts/train.py`의 음성 향상 모델과 목적함수가 다르며, 이 `deepanc` 경로 자체에는 실시간 오디오 입출력 엔진이나 OMAP–Jetson 통신을 구현하지 않는다. CPU 테스트 통과와 모델의 인과성만으로 Jetson 실시간 ANC 성능을 판단할 수 없다.

통합 저장소의 기존 `src/deep_anc/`에는 별도의 48 kHz 모델·실시간 엔진이 있다. 이 문서의 16 kHz 모델·manifest·checkpoint와 자동 호환되지 않는다. 기존 YAML·NPZ·hand-off 지연 규약을 이 JSON·RIR 경로와 혼합하지 않는다. 전체 상태와 기존 Jetson 검증 이력은 [루트 HANDOFF.md](../HANDOFF.md)를 따른다.

## 신호와 손실의 의미

모델은 기준 마이크 `x[n]`를 받아 실제 DAC에 대응하는 디지털 제어 명령 `u[n]`를 예측한다. ANC OFF 상태에서 동시에 기록한 오류 마이크 신호가 `d[n]`이며, 오프라인 잔차는 다음과 같다.

```text
u = model(x)
e = d + S(u)
loss = mean(e²) + 0.001 × mean(u²) + 0.0001 × mean((u[n] - u[n-1])²)
```

`S`는 정본 `rir.txt`의 500개 탭을 모두 사용하는 인과적 선형 convolution이다. FFT 계산에 충분한 zero padding을 적용해 원형 convolution의 wraparound를 방지한다. 측정된 gain·부호·선행 지연은 그대로 유지한다. 기존 DSP의 `u=-y`와 달리 신경망 출력 자체가 물리적인 command `u`이므로 별도의 부호 반전을 넣지 않는다.

입력 WAV는 PCM full scale로만 나눈다. PCM16이면 `x`와 `d`는 각각 원래 ADC 값/32768이며 `u`도 같은 기준의 DAC digital full scale을 사용한다. 파일별 최대값 정규화, 채널별 gain 보정, DC 제거, peak 정렬이나 자동 resampling은 하지 않는다. 측정 당시의 ADC/DAC·amplifier gain과 실제 녹음 조건이 맞아야 이 잔차 모델이 성립한다. 데이터 준비와 필수 manifest 항목은 [DATASET.md](DATASET.md)를 따른다.

출력 에너지와 변화량 penalty는 과도한 제어 출력을 억제하기 위한 시작 설정이다. 계수를 크게 잡으면 얻을 수 있는 소음 감소량도 달라질 수 있다. 최적 체크포인트는 penalty를 포함한 검증 loss가 가장 작은 모델이며, 가장 높은 noise reduction dB만으로 선택하지 않는다.

## 기본 모델과 설정

설정 파일은 [`configs/anc_train.json`](../configs/anc_train.json)이다.

| 항목 | 기본값과 의미 |
| --- | --- |
| 샘플링 주파수 | 16,000 Hz, 측정한 RIR과 동일 |
| 모델 크기 | 학습 파라미터 3,200개 |
| 입력 convolution | 1→16 채널, kernel 7, 왼쪽 padding |
| 잔차 block | 16채널, kernel 3, dilation 1/2/4/8, tanh |
| 출력 | 1채널, `0.1 × tanh(...)` |
| 참조 범위 | 현재 샘플과 과거 36개 샘플, 총 37개 |
| chunk / batch | 유효 샘플 2,048개 / batch 1 |
| 학습 | Adam, learning rate 0.0003, 20 epochs |
| 추가 지연 | `secondary_delay_samples=0` |
| 데이터 | `datasets/anc/prepared/manifest.jsonl` |

모든 convolution은 미래 샘플을 사용하지 않으며 시간축 전체의 통계를 사용하는 normalization도 없다. bias를 사용하지 않아 입력과 과거 상태가 모두 0이면 출력도 0이다. `max_output=0.1`은 보수적인 학습 시작 hyperparameter이며 하드웨어의 안전 한계 인증이나 최적 감쇠 수준을 의미하지 않는다. 충분한 제어 크기를 제한할 수 있으므로 실제 gain과 검증 결과를 바탕으로 검토한다. 이 학습 명령은 스피커로 신호를 출력하지 않는다.

추가 지연 0은 원래 RIR 외의 Jetson 연산·전송 지연을 오프라인 시뮬레이션에서 추가하지 않는다는 뜻이다. Jetson 지연을 0으로 측정했다는 뜻이 아니다. 원래 RIR에는 측정 시스템의 지연이 이미 들어 있다. 향후 전송 방식과 입출력 경로를 정한 뒤 추가 지연을 측정하고, 중복 계산 없이 반영해야 한다. 임의의 지연을 추정해 현재 RIR에 더하지 않는다.

## chunk 경계와 검증 지표

각 chunk 앞에는 모델과 2차경로가 필요로 하는 실제 녹음의 과거 구간을 함께 읽는다. 필요한 길이는 `모델 과거 범위 + FIR 과거 범위 + 추가 지연 + 1`이며 기본값은 `36 + 499 + 0 + 1 = 536`개 샘플이다. 마지막 1개는 chunk 경계에서 출력 변화량 penalty를 정확히 계산하기 위한 문맥이다.

이 앞부분 문맥과 녹음 마지막 chunk의 padding에는 loss를 부과하지 않는다. 녹음 시작보다 앞선 문맥은 0으로 채우며, 이는 기록 시작 이전 입력과 제어 상태가 0이라는 시뮬레이션 초기 조건이다. 녹음에 실제 존재하는 유효 샘플만 학습·검증 통계에 포함한다. 각 chunk에 필요한 과거 문맥을 다시 제공하므로 chunk 순서를 섞어도 FIR과 모델의 과거를 잃지 않는다. 전체 녹음을 한 번에 처리한 잔차와 chunk를 이어 붙인 잔차가 일치하는지 테스트한다.

검증의 noise reduction은 전체 유효 샘플의 에너지를 합산해 계산한다.

```text
NR[dB] = 10 × log10(sum(d²) / sum(e²))
```

양수면 오프라인 잔차 에너지가 줄었고, 음수면 증가했다는 뜻이다. `residual_mse`, penalty를 포함한 `loss`, `output_peak`도 함께 기록한다. 완전한 무음 disturbance만 있는 데이터는 NR을 정의할 수 없어 거부한다. 세션 분리 검증에서도 실제 장치의 지연·feedback·경로 변화·clipping 등을 모두 재현하지는 않으므로 이 지표는 실측 음향 감쇠 결과와 구분한다.

## 실행과 저장 결과

저장소 루트에서 아래 Docker 실행기를 호출한다. 호스트 Python이나 호스트 venv는 사용하지 않는다. 컨테이너의 NVIDIA PyTorch/CUDA 준비는 [JETSON_SETUP.md](JETSON_SETUP.md)를 따른다. `--device cuda`는 CUDA가 없으면 실패하며 CPU로 조용히 대체하지 않는다.

```bash
# 개발 PC의 합성 데이터 실행 점검
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --smoke-test --device cpu --output runs/omap-smoke-check

# 실제 녹음으로 첫 1 epoch 실행
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --epochs 1 --output runs/omap_anc

# 같은 실행을 총 20 epochs까지 이어서 학습
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda --epochs 20 --output runs/omap_anc \
  --resume runs/omap_anc/last.pt
```

`--epochs`는 추가 epoch 수가 아니라 이미 끝난 epoch를 포함한 총 목표다. config 안의 RIR·manifest 경로는 config 파일 위치를 기준으로 해석한다. CLI의 `--manifest`, `--output`, `--resume`는 실행 디렉터리를 기준으로 한다.

출력 폴더에는 다음 파일을 저장한다.

| 파일 | 내용 |
| --- | --- |
| `run.json` | 사용 설정, 장치, 파라미터·문맥 크기, 데이터/RIR hash, 합성 여부 |
| `metrics.jsonl` | epoch별 train/valid loss·에너지·NR·출력 peak |
| `last.pt` | 가장 최근 epoch의 모델·optimizer·설정·RIR hash·난수 상태 |
| `best.pt` | penalty를 포함한 검증 loss가 가장 낮았을 때의 체크포인트 |

체크포인트는 임시 파일에 저장한 뒤 교체한다. 새 학습에서 이미 결과가 있는 폴더를 지정하면 덮어쓰기를 거부하므로 새 `--output`을 사용하거나 위와 같이 같은 실행을 resume한다. 이어서 학습할 때 모델 구조, 학습 설정, 데이터 내용, RIR, sample rate와 추가 지연의 호환성을 검사한다. epoch 수를 늘리거나 worker 수·파일 위치를 바꾸는 것은 허용하지만, 학습 의미가 달라진 설정을 같은 실험으로 조용히 이어 가지 않는다. 변경 실험은 새 출력 폴더에서 시작한다.

같은 출력 폴더에는 학습 프로세스를 하나만 실행한다. 동시 실행 잠금은 아직 없으며,
재개는 위 예처럼 **동일 폴더의 `last.pt`**를 사용한다. 새 폴더로 resume하면 과거 `best.pt`는
자동 복사되지 않으므로 이 절차에서는 사용하지 않는다.

`--smoke-test`는 고정 seed의 합성 reference와 합성 disturbance만 사용하며, 기본적으로 짧은 chunk 512개 샘플로 1 epoch 실행한다. 실제 녹음 manifest를 함께 지정할 수 없다. 출력과 체크포인트에 합성 여부를 남긴다. 성공은 forward/backward, optimizer, 실측 FIR 적용, 검증과 저장 절차가 실행된다는 의미이며, 모델 학습 품질·실제 primary path·하드웨어 소음 감소량의 증거가 아니다.

## 현재 확인한 범위

개발 환경에서 CPU 순전파·역전파와 실제 체크포인트 저장/복구를 실행했고, 모델의 인과성과 출력 범위, 녹음/채널 단위 보존, 세션·파일·동일 PCM 내용의 train/valid 누수 방지, chunk 문맥 일치 및 호환되지 않는 resume 거부를 테스트했다. 추가 지연과 FFT/DSP 계수의 수치 검증은 2차경로 테스트가 담당한다.

통합 후 실제 Jetson Docker에서도 이 OMAP 경로의 CUDA 합성 순전파·역전파·학습 및 체크포인트 저장을 확인했다. 장치·버전·로그와 최신 회귀 결과는 [루트 HANDOFF.md](../HANDOFF.md)가 단일 출처다. 실제 ANC-OFF 동기 녹음 학습, 장치의 추가 지연 및 실시간 음향 성능은 아직 미검증이며 합성 CUDA 결과와 구분한다. OMAP 후속 순서는 [JETSON_HANDOFF.md](JETSON_HANDOFF.md)를 따른다.
