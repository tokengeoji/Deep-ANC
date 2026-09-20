# 18. OMAP 원본 S 기반 SFANC 시뮬레이션 사전학습

사용자가 확정한 기준은 **16 kHz OMAP 실측 `rir.txt` 500탭**이다. 지금은 필터 준비와
선택기 학습을 우선하며, Jetson–OMAP 연결·오디오 출력은 보류한다. 저역·고역을 모두
줄이고 음성·음악까지 다루는 목표는 유지하지만, 이 실험만으로 달성했다고 판단하지 않는다.

사전 FIR bank와 과거 REF 기반 CNN 선택이라는 SFANC 구조를 구현했다.
필터 계산은 FP64 ridge 최소제곱이므로 **원논문의 FxLMS 필터 학습·하드웨어·실험을 그대로
재현한 것은 아니다.** 기존 비학습 PSD 선택 기준선은 [docs/15](15_prepared_fir_research.md)를 참고한다.

| 단계 | 현재 범위 | 이 단계에서 증명하지 않는 것 |
|---|---|---|
| 수치·소프트웨어 검증 | FIR/direct convolution, 인과 특징, 분할, CPU/CUDA 학습·저장/복원 회귀 | 실측 음향 감쇠·실시간 마감 |
| 시뮬레이션 사전학습 | 합성 소음 + 로컬 음성 원천, 합성 P + 원본 실측 S | 실제 REF/ERR 학습·현재 Jetson 출력 경로 |
| 실측 보정 학습 | 동기 raw REF/ANC-OFF ERR 확보 후 별도 준비 | 현재 미수집 자료의 성공 추정 |
| 연속 수치 진단 | FIR 상태·필터 교체·limiter·추가 지연 시뮬레이션(§7), 계산시간 진단(§8) | 실제 비동기 실행·장치 왕복 지연·음향 안정성 |
| 실시간 연결 | live binding·OMAP 전송 미구현, 연결 보류 | 실시간 마감·현장 quiet zone |

최신 실행 상태·검증 집계는 [HANDOFF](../HANDOFF.md), 실측 데이터 계약은
[DATASET](DATASET.md), 시스템 안전은 [Docker 안내](../docker/README.md)를 따른다.

## 1. 경로·단위·부호

[`configs/sfanc_omap_pretrain.json`](../configs/sfanc_omap_pretrain.json)의 기준은 다음과 같다.

- `sample_rate=16000`, `secondary_kind="omap_measured"`. S는 `rir.txt`의 전체 500탭이다.
  원본 SHA·이득·부호·선행 탭을 보존하며 정규화·자르기·피크 정렬·리샘플링하지 않는다.
- 제어 명령은 `u=W*x`, 잔차는 `e=d+S*u`다. DSP의 `u=-y`를 다시 적용하지 않는다.
- P는 **실측이 아닌 두 임펄스 합성 경로**다. `P[128]=0.05`, `P[201]=-0.02`이며
  두 번째 탭은 73샘플 뒤의 −0.4배 echo다. 따라서 `d=P*x`도 합성 disturbance다.
- `secondary_delay_samples=0`은 S 외의 추가 지연을 넣지 않는 오프라인 시나리오다.
  원본 S 안의 선행 지연은 남아 있다. **실제 Jetson 전송·버퍼·DAC 지연이 0이라는 뜻이 아니다.**
- 기존 48 kHz 측정 NPZ의 delay·handoff 규약을 여기에 가져오지 않는다.
  이 구현은 `src/deep_anc/` 아래에 있지만 현재 SFANC 실험 API는 16 kHz 전용이다.

`configs/sfanc_synthetic.json`은 S까지 합성인 별도 수치 실험이다. OMAP 원본 S 실험과
결과를 합치지 않으며, 추가 지연이 다르면 다른 플랜트 조건으로 기록한다.

## 2. FIR bank와 선택기 학습

### FIR 준비: CPU FP64

`fit_control_fir()`는 워밍업 이후 다음 목적함수를 최소화한다.

```text
J(w) = mean((d + S*(W*x))²)
     + effort_penalty * mean((W*x)²)
     + regularization * ||w||²
```

인과 Toeplitz 행렬의 augmented least-squares를 SVD로 풀고, 전체 플랜트를 적용한 뒤
워밍업을 제외한다. 현 설정은 제어 FIR 128탭, 준비 음원 16,384샘플,
`regularization=1e-7`, `effort_penalty=0.002`다. 출력 한도를 넘은 준비 필터는 실패시키며
계수 축소·정규화·성공 seed 선별로 통과시키지 않는다.

후보는 **zero + 8개 대역 FIR = 9개**다. 대역은 80–400, 400–800, 800–1000,
1000–1200, 1200–1600, 1600–3000, 3000–7500, 80–7500 Hz다.
이 대역들은 필터 준비용 합성 잡음 조건이며 측정 S의 신뢰대역을 인증하지 않는다.

### CNN: 완료된 과거 창 → 다음 창 선택

관측 창 `W=4096`샘플을 모두 받은 뒤 그 다음 창에 적용할 후보를 고른다.
기본 8192샘플 crop에서 앞 절반만 특징 입력이다. 뒤 절반의 disturbance/잔차는
오프라인 학습 라벨에만 사용하며 추론 입력에 ERR·미래 REF·음원 family 정답은 넣지 않는다.

- 입력: Hann/50% overlap Welch의 정규화 log PSD와 원 REF의 log RMS.
  `n_fft=512`이면 `[batch, 2, 257]`이며 출력은 후보별 logits `[batch, 9]`다.
- 특징용 PSD 정규화는 원 음원 파형 정규화가 아니다. log RMS에 음원 레벨 정보가 남는다.
- 작은 Conv1d CNN을 FP32 Adam으로 학습한다. `--device cuda`는 **선택기 학습**에 적용되며
  FIR 해 계산·자료 준비까지 GPU에서 수행한다는 뜻은 아니다. CUDA가 없으면 실패한다.
- 첫 관측 창은 무제어 조건으로 해석한다. 다음 창의 채점도 전체 경로 settling 이후만 한다.
  현재 128탭 W/500탭 S/추가 지연 0에서는 crop의 0-based index 4722부터 채점한다.
  이 제외 규약은 연속 FIR 교체·crossfade를 검증한 것이 아니다.

후보 `k`의 다음 창 비용은 다음과 같다. `E`는 해당 채점 구간의 평균 에너지다.

```text
C_k = (E(e_k) + 3*E_1000–1600(e_k) + 0.002*E(u_k))
      / max(E(d) + 3*E_1000–1600(d), 1e-12)
      + 10 * I(다음 창의 |u_k| 최대값 > 0.2)
q_k = softmax(-(C_k - min(C)) / temperature)
loss = soft_cross_entropy(q, logits)
       + risk_weight * sum(softmax(logits)_k * (C_k - min(C)))
```

목표 대역은 **[1000,1600) Hz**다. 추가 weight 3을 주되 전대역 잔차와 제어 에너지 항을
유지한다. 이 목표 가중은 후보 채점/선택기 라벨에 적용되며, 위 FIR bank 해 계산은 각 준비
음원의 전대역 ridge 목적함수를 사용한다. 현재 `temperature=0.05`, `risk_weight=1.0`이며
`risk_weight=0`이면 CE 단독 기준 실험이다. 큰 후보 비용을 잘라서 숨기지 않는다.

최상 checkpoint는 validation에서 **argmax로 실제 선택한 후보의 평균 비용**으로 고른다.
`initial_validation_loss`·`best_validation_loss`도 이 선택 비용이다. 반면 history의
`train_loss`·`validation_loss`는 CE이고, expected regret·선택 비용은 별도 필드다.
test는 checkpoint나 고정 대조군을 결정하는 데 사용하지 않는다.

## 3. 음원·분할·재현성

절차 생성 소음은 대역 잡음·광대역·대역 전환·무음을 포함하며 split별 seed namespace가
분리된다. `train_groups` 등은 이 합성 seed 묶음의 수이지 녹음 세션·화자 수가 아니다.
예측 불가능한 다음 창 대역 전환도 실패 사례를 포함해 평가한다.

음성 원천은 Docker 내부 `data/raw/speech/LibriSpeech`의 dev-clean이다.
**실제 사람이 녹음한 일반 음성이지만, 동기 REF/ANC-OFF ERR 실측 데이터는 아니다.**
원천 음성으로 `d=P*x`를 만들므로 이 실행은 시뮬레이션 사전학습이다.

- 로컬 `LICENSE.TXT`의 CC BY 4.0 확인을 요구하고 SPEAKERS/CHAPTERS에서 화자와 책의
  연결요소를 먼저 구성한다. 그 뒤 train/validation/test를 나누고 그룹 round-robin으로 선별한다.
- 현재 확인한 2,703개 음성은 36개 연결요소로 분리된다. 기본 seed `20260920`에서
  28/4/4개 그룹에 60/20/20파일, 파일당 최대 두 개의 겹치지 않는 8192샘플 crop을 쓴다.
  선택 원천의 화자·책·그룹 교차 누출은 0으로 확인했다. 기존 `speech.jsonl`은 재사용·수정하지 않는다.
- 파일 SHA·상대경로·화자·chapter·책·group·split·crop 시작점·라이선스·메타데이터 SHA를 남긴다.
  공식 원본 archive checksum을 새로 검증한 것은 아니며, PCM 검사는 선택 crop 범위다.
- 원 음원은 리샘플링·정규화하지 않는다. 단위는 `decoded_source_audio_not_calibrated_adc_dac`로
  기록한다. 음원 진폭을 실제 ADC/DAC 교정 단위로 간주하지 않는다.
- 같은 source/group의 crop을 서로 다른 split으로 나누지 않는다. 공식 dev-clean을 연구용으로
  다시 분할한 것이므로 LibriSpeech 공식 ASR train/test 성능을 주장하지 않는다.
- 음악은 현재 미사용이다. 로컬 음악 혼합본에는 트랙별 ND 계열 라이선스가 포함되어 있어
  통째로 일반 학습 가능 자료로 간주하지 않는다. 출처·개별 조건 확인 없이 포함하지 않는다.

동일 seed·장치·버전의 반복을 지원하며 CPU/CUDA 또는 버전 간 bitwise 일치를 보장하지 않는다.
실측 데이터 준비는 [DATASET](DATASET.md), 원본 보관·Drive receipt는
[docs/16](16_drive_acoustic_preparation.md)의 별도 절차다.

## 4. 실행과 산출물 보존

기존 Docker에서 실행한다. `NEW_RUN`은 **존재하지 않는 새 이름**으로 바꾼다.
기존 실행 폴더·로그를 지우거나 덮어써서 재시도하지 않는다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -u scripts/train/train_sfanc.py \
  --config configs/sfanc_omap_pretrain.json \
  --device cuda \
  --librispeech-root data/raw/speech/LibriSpeech \
  --out runs/sfanc/NEW_RUN
```

CLI는 stdout/stderr 로그 파일을 자동 생성하지 않는다. 로그를 보존할 경우 위 명령 대신
다음처럼 Docker 내부의 별도 새 파일에 기록한다. `noclobber`는 기존 로그 덮어쓰기도 거부한다.

```bash
bash scripts/docker/dev.sh exec bash -lc '
  set -eu
  mkdir -p results/sfanc_logs
  set -C
  exec .venv/bin/python -u scripts/train/train_sfanc.py \
    --config configs/sfanc_omap_pretrain.json --device cuda \
    --librispeech-root data/raw/speech/LibriSpeech --out runs/sfanc/NEW_RUN \
    > results/sfanc_logs/NEW_RUN.log 2>&1
'
```

산출물은 Git에 포함하지 않는다. 정상 완료는 종료 코드·완료 로그·report·저장/복원 검사를
함께 확인한다. 실패 시 부분 산출물을 보존하고 원인을 조사한다. 현재 CLI에 resume 옵션은 없다.

| 파일 | 내용 |
|---|---|
| `bank.npz` | 9개 FIR 계수, 원본 S, 합성 P, PSD 기준선 prototype |
| `selector.pt` | 선택기 tensor state, feature 규격, 결합 bank SHA, `deployment_allowed=false` |
| `report.json` | 설정·경로 출처·학습 history·split별 원천·모든 대조군/대역 평가·artifact SHA |
| `speech_manifest.json` | `--librispeech-root` 사용 시 생성되는 원천·분할·crop·라이선스 기록 |
| 별도 실행 로그 | FIR 준비/자료 수/epoch/실패/완료 기록; 위 예제의 `results/sfanc_logs/` |

`load_selector(run_directory)`는 bank/selector SHA와 연결 관계를 확인하고 CPU 모델·필터를
복원한다. SHA는 파일 무결성 검사이지 학습 출처 인증이나 배포 승인 자체가 아니다.
`selector.pt`는 optimizer/resume checkpoint도, ONNX/TensorRT 배포 artifact도 아니다.

## 5. 결과를 읽는 기준

대조군은 zero, train에서 고른 단일 고정 FIR, 비학습 nearest PSD, 학습된 선택기,
미래 비용 oracle다. **oracle는 배포 불가능한 비교 기준**이다. 추론 중 미래 잔차나 출력 peak를
보고 실패 후보를 바꾸지 않는다. 완료된 과거 REF의 평균 제곱이 `1e-12` 이하면 평가 wrapper가
zero 후보를 택한다. 이 무음 fallback은 CNN의 숨겨진 경로 추정 능력을 뜻하지 않는다.

항상 전대역·1 kHz 미만·1000–1600 Hz·1600 Hz 이상을 각각 읽고, 개별 음원·최악 창·증폭을
함께 확인한다. `reduction_db`의 음수는 증폭이며, 무신호/계산 불가는 `null`이다.
`emergent_error_energy`는 원래 거의 무음인 대역에 새 잔차가 생긴 경우다.

`control_limit_exceeded_windows`는 선형 출력이 한도를 넘은 창 수다. **실제 limiter를 통과한
클리핑 음향 실험이 아니며**, 한도 초과가 0이어도 실제 ADC/DAC 안전을 인증하지 않는다.
report의 물리 성능·실시간·배포 허용 플래그는 모두 false를 유지한다.

## 6. 실행 기록: OMAP 시뮬레이션 사전학습 002

2026-09-20 실제 Jetson Orin CUDA에서 60 epoch를 실행했고 validation 선택 비용 기준
최상 epoch는 **52**였다. 산출물은 `runs/sfanc/omap_20260920_02/`, 로그는
`results/sfanc_omap_20260920_02.log`다. bank는 9×128이며 저장된 S의 500탭이 원본과
`np.array_equal=True`임을 확인했다. 원본·기존 실행을 덮어쓰지 않았다.

학습은 600창(절차 생성 480창 중 무음 48창 + 음성 120창), validation/test는 각각
160창(절차 생성 120창 중 무음 12창 + 음성 40창)이다. 음성은 §3의 독립 그룹 분할을 썼다.
validation의 실제 선택 평균 비용은 초기 **1.855579 → 0.596088**이었다.

| Test 대조군 | 평균 비용 |
|---|---:|
| zero | 0.925000 |
| train에서 고른 고정 FIR | 0.574615 |
| nearest PSD | 1.877872 |
| 학습된 선택기 | 0.554180 |
| 미래 비용 oracle — 배포 불가 | 0.475191 |

이 수치는 **단위 없는 고역 가중 잔차 + effort + 출력 한도 초과 벌점 비용이며 dB가 아니다.**
학습된 선택기는 이 test 집합에서 고정 FIR보다 평균 비용이 약 3.56% 낮았다.
zero 평균에도 비용 0인 무음 12창이 포함되므로 평균이 1보다 작다.

중요한 한계는 학습된 선택기의 **출력 한도 초과가 test 1/160창, validation 2/160창**이라는 점이다.
실제 limiter는 적용하지 않은 선형 계산이고 초과 비용·경고를 보고서에 남겼다.
음성/소음별 전대역·목표 대역의 증폭 행과 미약한 대역 누설도 삭제하지 않았다.
평균 개선은 모든 음원·모든 대역의 감쇠나 실측 음향 성능을 뜻하지 않으며,
`deployment_allowed=false`를 유지한다.

`omap_20260920_01`의 CE 단독 60 epoch 기준 실험도 보존했다. validation에서 CE 하락과
실제 선택 비용 악화가 함께 나타난 점을 보고, 002에서 expected regret(`risk_weight=1`)를
추가했다. 002의 test 결과를 본 뒤 모델·threshold를 다시 조정하지 않았다.
이 두 실험은 시뮬레이션 사전학습 기록이다. 최종 전체 회귀 집계는 [HANDOFF](../HANDOFF.md)에 별도로 기록한다.

## 7. 연속 처리·추가 지연 진단

§6의 모델·bank·원본 S를 동결한 채 연속 신호에서 출력 제한과 경로 불일치를 살핀다.
기존 학습 결과를 덮어쓰거나 마지막 test로 모델·임계값을 다시 고르지 않는다.
`StreamingFIRBank`는 REF 이력을 보존하며 필터를 즉시 교체하거나 샘플별 계수 보간으로
전환한다. 같은 목표의 재요청은 전환을 다시 시작하지 않는다. 제한을 켜면 **최종 명령 u를
hard clip한 뒤** 추가 지연·원본 S를 통과시켜 `e=d+S*delay(u)`를 계산한다.
이는 Python/NumPy 오프라인 수치 코어이며 오디오 callback·비동기 worker는 아니다.

[`configs/sfanc_delay_diagnostic.json`](../configs/sfanc_delay_diagnostic.json)의 조건은 다음과 같다.

- 음원당 32,768샘플(2.048초), 제어 FIR 128탭, 처리 블록 32샘플, 출력 한도 0.2.
- 합성 P의 첫 탭 지연은 48/128샘플(**3/8 ms**), 73샘플 뒤 −0.4배 echo.
  3 ms는 민감도 진단점이지 OMAP 선행 시간 실측값이나 Jetson 기하의 전용값이 아니다.
- S는 원본 500탭을 보존한다. **S 외의 추가 지연**은 0/8/16/32/64/128샘플
  (0/0.5/1/2/4/8 ms)이다. 원본 S 내부 지연과 별개이며 실제 I/O 지연 측정값이 아니다.
  블록 크기를 근거로 handoff를 숨겨 추가하거나 S의 선행 탭을 제거하지 않는다.
- 선택기는 완료된 과거 4096샘플만 본다. 선택 결과는 **가상 4 ms 뒤** 적용하고
  **8 ms 동안 계수 보간**한다. 두 값은 진단용 설정이며 실제 장치·스레드 정책으로
  확정한 것이 아니다. 처음에는 zero 후보이며 최초 관측·선택 결과 도착 전에는 무제어다.
- 신호는 새 seed의 백색잡음·저역/고역 잡음·급격한 대역 전환·다중 톤·무음과 선택적 음성이다.
  다중 톤은 실제 음악이 아니며 `music_evaluated=false`를 유지한다.
  음성은 기존 test crop을 정규화·리샘플링 없이 이어 붙인 **고정 진단 재사용**이다.
  접합부는 합성이며 새로운 독립 test 집합이나 동기 REF/ERR 실측으로 주장하지 않는다.

대조군 간 정보·초기 조건 차이를 숨기지 않는다.

| 방법 | 조건과 해석 |
|---|---|
| `zero` | 무제어 |
| `frozen_train_best_fir` | 기존 train에서 선택한 단일 FIR을 그대로 사용, 시작부터 출력 |
| `frozen_sfanc` | 기존 모델·bank 동결, 첫 과거 창 관측 후 선택·전환 |
| `scenario_fitted_fir_not_frozen` | 바뀐 합성 P/지연을 알고 별도 백색잡음으로 FIR 재계산. 동결 SFANC와 정보 조건이 다르며 최적 성능 상한도 아님 |
| `cold_block_fxnlms` | 계수 0으로 시작, block=32·mu=0.05 고정. 진단 플랜트와 같은 S/추가 지연을 filtered-x에 제공하며 동결 SFANC와 정보 조건이 다름 |

FxNLMS는 clipping 이후 S의 꼬리까지 적응을 보류하고 상태를 원천 간 공유하지 않는다.
이는 성공한 OMAP 펌웨어의 동등 재현이나 최적화된 FxNLMS를 뜻하지 않는다.
펌웨어의 DC blocker·noise gate·tap clamp·step cap까지 재현하지 않는다. 한 가지 고정
step size와 짧은 cold-start 결과로 FxNLMS 전체의 성능 우열을 단정하지 않는다.

전대역·1 kHz 미만·[1000,1600)·1600 Hz 이상의 결과를 전체 스트림, 공통 초기 구간 이후,
각 4096샘플 창으로 남긴다. 공통 초기 구간 제외도 전체 플랜트 적용 **후** 수행한다.
증폭·clip 횟수·무음 대역에 새로 생긴 에너지를 보존한다. 직사각 FFT 창의 대역 누설과
미약한 분모에 주의하며 null dB를 성공으로 바꾸지 않는다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/benchmark_sfanc_stress.py \
  --run runs/sfanc/omap_20260920_02 \
  --config configs/sfanc_delay_diagnostic.json \
  --librispeech-root data/raw/speech/LibriSpeech \
  --out results/sfanc_stress/NEW_DIAGNOSTIC
```

출력은 신규 디렉터리의 `report.json`이다. 같은 OMAP 16 kHz 조건으로 검증된 HybridANCNet
학습본이 없어 **HybridANCNet 감쇠 비교는 수행하지 않는다.** legacy 48 kHz/digital artifact를
대신 넣지 않는다. 연속 수치 진단 완료와 실측 감쇠·실시간 안전·배포 승인은 별개다.
현재 jitter·feedback·비선형 실제 경로·장치 clock·음악은 검증 범위 밖이다.

### 실행 결과: 2026-09-20 연속 진단 002

`results/sfanc_stress/omap_20260920_02/report.json`에 12개 경로/지연 조건 × 7개 원천 ×
5개 방법의 **420회 실행, 16,800개 지표**를 저장했다. 실행 로그는
`results/sfanc_stress_20260920_02.log`다. 기존 CNN은 재학습하지 않았다.
첫 CLI 시도는 패키지 경로 누락으로 출력 폴더 생성 전 실패했으며 로그를 보존하고,
직접 실행 회귀를 추가한 뒤 002를 새 경로에서 실행했다.

아래는 동일한 새 고역 잡음(`band_high`)의 **1000–1600 Hz, 공통 초기 구간 이후** 결과다.
양수는 감쇠, 음수는 증폭이다. 모두 합성 P/오프라인 플랜트 수치이며 실측 dB가 아니다.

| 합성 P 지연 | 추가 제어 지연 | 동결 SFANC dB | 해당 경로로 새로 계산한 FIR dB | cold block FxNLMS dB |
|---|---|---:|---:|---:|
| 8 ms | 0 ms | 3.556 | 3.493 | 3.768 |
| 8 ms | 1 ms | −1.758 | 3.393 | 4.033 |
| 8 ms | 8 ms | −1.555 | 0.363 | 0.350 |
| 3 ms | 0 ms | −0.110 | 2.796 | 3.656 |
| 3 ms | 1 ms | −1.460 | 0.426 | 1.170 |
| 3 ms | 8 ms | −1.385 | 0.006 | −0.915 |

이 고역 잡음의 동결 SFANC 결과는 기존 train-best 단일 FIR과 같았다. 즉, 이번 사례에서
CNN 선택이 단일 필터보다 우수하다는 증거는 없다. 추가 지연으로 기존 필터의 위상 정합이
깨지는 것과, 해당 조건에 맞춰 재계산/적응한 제어기의 한계를 구분해야 한다.
SFANC 실패를 모든 재설계 필터의 불가능성으로, FxNLMS 일부 성공을 모든 소리의 성공으로
확대 해석하지 않는다. 48샘플 P 조건도 실제 보드의 선행 시간으로 해석하지 않는다.

420회 중 7회에서 FxNLMS의 제한 전 명령이 한도를 넘었다. 실제 명령에는 hard clip을
적용하고 경고·적응 보류·증폭 결과를 보존했다. 모든 제한 명령의 peak는 0.2 이하였다.
동결 SFANC는 이번 원천들에서 clip이 없었지만, §6의 기존 test 1/160창 실패를 지우거나
해결한 것은 아니다. 이번 음성은 이전 test 중 일부를 재사용한 진단이며 새로운 안전 인증도 아니다.
전체 스트림·저역·고역·음원별 실패는 원 보고서에 남기며 배포 허용은 계속 false다.

## 8. Jetson 무오디오 계산시간 진단

`results/sfanc_compute/jetson_20260920_01/report.json`은 실제 Orin Docker에서 구성요소를
**순차 실행**한 wall-time 기록이다. 각 항목 warmup 30회 뒤 200회 측정했으며
torch intra-op threads=1, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`을 사용했다.
기존 RT 커널·30W·시스템·오디오 설정은 변경하지 않았다.

| 구성요소 | 장치·입력 | 중앙값 ms | p99 ms | 관측 최댓값 ms |
|---|---|---:|---:|---:|
| SFANC FIR 유지 | CPU, 32샘플 | 0.0695 | 0.0873 | 0.1082 |
| SFANC FIR 매 블록 전환 | CPU, 32샘플 | 0.1010 | 0.1299 | 0.1465 |
| Welch 특징 + 선택기 | CPU, 4096샘플 | 1.1532 | 1.2535 | 1.3392 |
| Welch 특징 + 선택기 | CUDA, 4096샘플 | 3.7960 | 8.9363 | 9.0835 |
| 미학습 HybridANCNet tiny | CPU, 256샘플 | 50.4101 | 51.2103 | 51.4532 |
| 미학습 HybridANCNet tiny | CUDA, 256샘플 | 9.3255 | 11.6947 | 12.0119 |

FIR은 16/32/64/128/256샘플 각각 유지·전환을 검사했으며 위 표는 32샘플의 발췌다.
이 벤치의 전환 길이는 **해당 블록 길이**로, §7의 128샘플 전환 설정과 다르다.
FIR 상태 갱신·내부 유한값 검사·0.2 limiter까지 포함하지만 production native 최적화본은 아니다.
CUDA 항목은 CPU 입력에서 전송·추론·CPU 결과 회수와 전후 동기화를 포함한다.
작은 선택기는 이 실행에서 CPU가 더 빨랐으므로 GPU 사용 자체를 저지연의 근거로 삼지 않는다.

Hybrid는 기존 `model_tiny.yaml`의 **고정 seed 미학습 PyTorch eager FP32 구조**다.
ERR 입력은 0이고 상태를 호출 간 유지한다. 해당 OMAP 16 kHz 학습본·ANC 성능 검증본이
아니며 ONNX/TensorRT로 최적화한 모델의 속도도 아니다. 이를 근거로 SFANC의 고역 감쇠가
Hybrid보다 우수하다고 주장하거나 기존 48 kHz 모델을 16 kHz 배포용으로 승격하지 않는다.

32샘플은 16 kHz에서 2 ms, 256샘플은 16 kHz에서 16 ms/48 kHz에서 약 5.33 ms다.
보고서의 두 표본률 예산은 산술 비교일 뿐 모델 표본률 변경이 아니다. **4096샘플 관측창은
선택 반응시간이며 매 샘플 출력에 필수적인 256 ms 대기시간으로 더하지 않는다.**
이 벤치에는 블록 수집·스레드 handoff·ADC/DAC·스피커·음향 전파가 없고 구성요소 간
동시 실행 간섭도 측정하지 않았다. 유한 200회 p99/최댓값은 최악 실행시간 보증이 아니다.
따라서 전체 인과 여유·장치 왕복 지연·실시간 마감 통과는 여전히 미검증이다.

```bash
bash scripts/docker/dev.sh exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python scripts/bench/benchmark_sfanc_compute.py \
  --run runs/sfanc/omap_20260920_02 \
  --out results/sfanc_compute/NEW_COMPUTE \
  --devices cpu cuda --warmup 30 --trials 200 --torch-threads 1
```

신규 출력만 허용하며 `report.json`에 중앙값·p95·p99·최댓값·측정 원표본, 환경·스레드·TF32,
모델/설정/구현 SHA를 기록한다. SFANC 로드는 bank/selector SHA와 원본 S 전체 탭 일치를
검증한다. 계산시간 측정만으로 오디오를 열거나 배포 허용 플래그를 바꾸지 않는다.

## 9. 코드 진입점과 남은 검증

- [`sfanc_design.py`](../src/deep_anc/baselines/sfanc_design.py): `fit_control_fir()`, `score_control_filters()`.
- [`sfanc_selector.py`](../src/deep_anc/train/sfanc_selector.py): `reference_features()`,
  `RefSpectrumSelector`, `train_selector()`, REF-only `predict_selector()`.
- [`sfanc_sources.py`](../src/deep_anc/data/sfanc_sources.py): manifest를 반환하는
  `prepare_librispeech_manifest()`와 `(samples, metadata)`를 순차 반환하는 `iter_librispeech_crops()`.
- [`sfanc_experiment.py`](../src/deep_anc/train/sfanc_experiment.py): `ExperimentConfig`,
  `run_experiment(config, out, device=..., librispeech_root=...)`, `load_selector()`.
- CLI: [`scripts/train/train_sfanc.py`](../scripts/train/train_sfanc.py).
- 연속 FIR: [`sfanc_stream.py`](../src/deep_anc/baselines/sfanc_stream.py).
- 연속·지연 진단: [`sfanc_stress.py`](../src/deep_anc/eval/sfanc_stress.py),
  [`sfanc_fxnlms.py`](../src/deep_anc/eval/sfanc_fxnlms.py).
- 계산시간 진단: [`sfanc_compute.py`](../src/deep_anc/eval/sfanc_compute.py).

연속 FIR 교체·보간·제한의 **오프라인 수치 코어**는 추가했지만,
live callback·실제 비동기 선택기·OMAP 계수/샘플 전송은 아직 없다.
실제 경로 변화·feedback·비선형성·클록·단위·전송 지연을 확인하기 전 실시간 배포하지 않는다.
실측 보정 학습은 독립 세션의 동기 raw REF/ANC-OFF ERR와 실제 P/S 조건을 확보한 뒤 별도다.
음악까지 포함한 독립 평가와 반복 OFF/ON/OFF 음향 검증도 남아 있다.

관련 회귀는 Docker 안에서 실행한다. GPU 테스트·전체 suite의 최신 결과는 HANDOFF에 기록한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q \
  tests/test_sfanc_design.py tests/test_sfanc_selector.py \
  tests/test_sfanc_sources.py tests/test_sfanc_experiment.py \
  tests/test_sfanc_stream.py tests/test_sfanc_fxnlms.py \
  tests/test_sfanc_stress.py tests/test_sfanc_stress_review.py \
  tests/test_sfanc_compute_bench.py
```
