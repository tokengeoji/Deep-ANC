# 외부 `duct_cnn_anc` 폴더 감사 및 현행 Deep-ANC 대조

감사일: 2026-08-27 (Asia/Seoul)

대상 외부 폴더:

`/home/capston/DeepANC_CRN_n_codex/duct_cnn_anc`

이 감사는 외부 폴더를 **읽기 전용**으로 조사한 결과다. 외부 폴더의 파일을
수정·삭제·실행하지 않았고, 오디오 장치를 열거나 스피커를 구동하지 않았다.
외부 폴더는 Git 저장소가 아니며(`git status`가 repository 아님), `pyproject.toml`,
학습 스크립트, 데이터셋 manifest, checkpoint, ONNX/TensorRT artifact가 없다.

## 결론 요약

| 항목 | 외부 `duct_cnn_anc` | 현행 `Deep_ANC` | 현행에 미치는 영향 |
|---|---|---|---|
| 목적 | 논문형 raw CNN·측정 준비 도구 | strict P/S 기반 HybridANC 학습·배포 파이프라인 | 외부는 대체 구현이 아님 |
| 모델 | 10층 causal dilated CNN, 3,232 params, FIR 512 | `hybrid_anc_tiny`, 1,164,809 params | 외부 모델을 init/배포 후보로 사용하지 않음 |
| 학습 | 없음 | A100 loss pilot 진행 중, canonical pretrain 미시작 | 외부에 학습 결과 없음 |
| Primary | 100–1,000 Hz, 별도 스트림, 5회/3회 raw | 150–1,600 Hz strict interleaved P | 외부 P는 진단 교차검증만 가능 |
| Secondary | 완전한 S 없음; retry는 proxy·프레임 불일치 | strict interleaved S, 19 repeats, xrun 0 | 외부 S를 절대 가져오지 않음 |
| Feedback | 없음 | 현행 1차 open-loop 계약에는 미사용 | 외부에서 보강되지 않음 |
| 고주파 | 1,000 Hz 이상 자극·검증 없음 | 1,600 Hz까지 P/S 신뢰대역, 1,633 Hz 위는 do-no-harm | 외부 결과로 고주파 ANC를 주장할 수 없음 |
| 지연 | 모델/loopback/음향을 분리하지만 end-to-end artifact 없음 | `TrainingTimingContract`, `PlantDelays.lead()` 단일 출처 | 외부 숫자로 lead를 덮어쓰지 않음 |

감사 시점의 strict 기준선은 branch `fix/finetune-readiness-repair`, commit
`d269699`였고, 이 감사 기록을 포함한 현재 HEAD는 `6473b27`이다. strict P/S 공식
artifact는 `capture_id=5ac1313488c8434bb4d672a36503df59`,
P effective delay 1386, S effective delay 1245, handoff 256, 따라서
`PlantDelays.lead() = 115`이다. 외부 자료를 읽었다고 canonical readiness나
파인튜닝 완료로 판정하지 않는다.

## 1. [가설] 외부 폴더는 현행 학습에 바로 쓸 수 있는 모델·데이터를 포함한다

### [근거]

- `README.md`는 범위를 “학습 직전까지”로 제한하고 모델 학습, dataset 생성/증강,
  TensorRT, FPGA를 수행하지 않는다고 명시한다.
- `find` 결과에 `*.pt`, `*.pth`, `*.onnx`, `*.plan`, `*.engine`, 학습 로그, manifest가
  없다.
- `models/paper_controller.py`와 `models/streaming_controller.py`는 새로 초기화되는
  PyTorch 클래스이며 checkpoint loader가 없다.
- 외부 폴더는 Git 저장소가 아니고 패키지 빌드 메타데이터(`pyproject.toml` 등)도 없다.

### [확인 방법]

외부 폴더 전체 파일 목록·확장자·JSON top-level schema를 읽고, 모델/학습/배포 artifact의
존재 여부를 직접 확인했다. 핵심 코드 SHA256은 다음과 같다.

| 파일 | SHA256 앞부분 | 확인 내용 |
|---|---|---|
| `models/paper_controller.py` | `a05c67defd08c8b2…` | 3,232-parameter paper 추정 구현 |
| `models/streaming_controller.py` | `57cf1cca029aa0ac…` | stateful streaming 추정 구현 |
| `audio/streaming_runtime.py` | `6c65e87173f0ffbb…` | 실험용 callback adapter |
| `rir/measure_path.py` | `1dc776dc53af3679…` | 별도 스트림 raw 캡처 |
| `validation/core.py` | `48a22ce020a8549b…` | 명시 기준 없는 RIR 진단 |

### [결과]

현행 학습에 전이할 수 있는 외부 checkpoint나 고정된 실험 계약은 없다.

### [판정]

**Contradicted** — 외부 폴더는 학습 입력/모델 artifact가 아니라 준비·진단 영역이다.

### [다음 행동]

외부 폴더의 모델·RIR·raw를 현행 `assets/`, `runs/`, canonical manifest에 복사하지
않는다. 필요한 설계 아이디어만 아래 재사용 목록에 반영한다.

## 2. [가설] 외부 논문형 모델은 현행 realtime tiny의 대체 후보가 될 수 있다

### [근거]

외부 모델은 다음 구조를 고정한다.

- 10개 residual causal dilated layer
- 각 layer: `1 → 16` Conv1d, kernel 16, dilation `1,2,…,512`, ReLU,
  `16 → 1` FC
- residual 합과 global skip 합 뒤 causal FIR 512
- bias 없음, 총 `10 × (16×1×16 + 1×16) + 512 = 3,232` parameters
- 논문 원본 source/checkpoint, padding·normalization·tap 순서·output index는
  `UNKNOWN_FROM_PAPER`

외부 `audio/streaming_runtime.py`는 입력 callback 안에서 각 1-sample chunk마다
  `torch.from_numpy(...).to(device)`와 `model.process_chunk()`를 호출한다. 기본
 `model_step_size=1`이면 256-sample ALSA block마다 Python/Torch 호출을 256회 만든다.
 출력 callback은 `deque`에서 완성 block을 꺼내며 없으면 0을 내보내고 underrun만
 증가시킨다. 물리적 S(z) 보상, measured lead, 출력 enable/fade 계약은 없다.

`TRAINING_HANDOFF.md`에 적힌 model-only 진단 수치
`p50=4,838.37 us`, `p95=7,082.44 us`는 별도 JSON benchmark artifact가 없고,
transport·USB·ALSA·callback 지연을 포함하지 않는다. 1-sample budget
`20.833 us`와 직접 비교할 수 있는 현장 통과 증거가 아니다.

현행 모델은 `configs/model_tiny.yaml`의 `HybridANCNet`(1,164,809 params)이며,
strict P/S·lead 115·ONNX/ORT preflight·안전 watchdog을 포함하는 별도 계약이다.

### [확인 방법]

모델 정의, streaming state 갱신, callback queue를 정적으로 읽고 현행
`src/deep_anc/models/hybrid_anc.py`, `src/deep_anc/models/streaming.py`,
`src/deep_anc/dsp/timing.py`와 비교했다.

### [결과]

외부 모델에는 학습된 weight가 없고, 논문 구현의 미공개 결정이 남아 있다. 또한 외부
runtime은 measured physical lead 없이 callback 내부에서 계산하므로 현행의 수치 등가·
deadline 계약을 충족하지 않는다. 3,232 params가 고주파에 충분한지 또는 불충분한지는
weight와 동일 조건의 ANC raw가 없으므로 판단할 수 없다.

### [판정]

**Invalid experiment** — 외부 모델을 현행 ANC 성능 비교 대상으로 삼을 수 없다.

### [다음 행동]

논문 모델은 필요하면 나중에 “offline paper baseline”으로 별도 학습·계약·측정을 만들되,
현행 tiny의 init/resume/배포 모델로 사용하지 않는다. 고주파 판단은 canonical tiny의
150–1600 Hz 및 이후 Level-5 natural challenge에서 한다.

## 3. [가설] 외부 Primary 캡처는 현행 strict P(z)를 보완할 수 있다

### [근거]

외부 실제 파일:

- `measurements/primary_raw/run_20260827_01/recording_session.json`
  (SHA256 `ef17f155…`): 48 kHz, 256, 13 s, log sweep 100–1,000 Hz, 5 repeats
- `measurements/primary_raw/run_20260827_02_white/recording_session.json`
  (SHA256 `998ed1f3…`): 48 kHz, 256, 13 s, band-limited white 100–1,000 Hz, 3 repeats
- 모든 기록에서 xrun 0, clipping 후보 빈 배열
- input `NVIDIA Jetson AGX Orin APE: (hw:1,1)`, output `AB13X USB Audio: (hw:2,0)`
- input channel 1=error, channel 2=reference, output channel 1=noise

`run_20260827_01`의 RIR/검증은 다음과 같다.

| 산출물 | 값 | 해석 |
|---|---:|---|
| `primary_summary_run_20260827_01.json` | status `REVIEW` | 기준 미적용 |
| regularization `1e-8` | NMSE `0.21139` (`-6.75 dB`), corr `0.88879` | path fit 진단, ANC 감쇠가 아님 |
| regularization `1e-4` | NMSE `0.10861` (`-9.64 dB`), corr `0.94589` | 정규화 선택에 민감 |
| 1e-4 cross-noise | NMSE `0.16627` (`-7.79 dB`), corr `0.94188` | 별도 white 검증, 여전히 REVIEW |
| 1e-4 repeatability | pairwise corr mean `0.99872` | 반복 파형은 유사 |
| 1e-8 coherence | 100–1,000 Hz mean `0.65508`, median `0.75010` | sweep 진단 품질 낮음 |
| 1e-4 coherence | 100–1,000 Hz mean `0.93866`, median `0.97167` | 후처리 개선 흔적이지 strict 증거 아님 |

독립 계산에서도 white run의 100–1,000 Hz coherence median은 약 `0.972`, 150–600 Hz
median은 약 `0.982`였지만 1,600 Hz 이상은 자극 에너지가 없어 coherence median이
약 `0.075` 수준으로 떨어진다. 따라서 이 캡처는 고주파 ANC를 검증하지 않는다.

외부 session metadata의 `positions`, `microphone_gain`, `device_serial_or_id`는 모두
null이고, level evidence, clock witness, fractional joint-LS, raw/analysis SHA,
operator의 구체적 routing/geometry 기록이 없다. 별도 InputStream/OutputStream이라
입력·출력 clock을 공통 축으로 보정하지 않는다.

### [확인 방법]

외부 raw NPY/WAV의 shape·dtype·PCM 범위·RMS·SHA를 독립적으로 읽고, `scipy`의
cross-correlation/coherence로 재계산했다. `validation/core.py`의 결과 JSON도 원문과
대조했다.

### [결과]

외부 Primary는 “실제 마이크가 움직였고 xrun 없이 캡처되었다”는 현장 진단에는 유용하다.
그러나 100–1,000 Hz에 한정된 별도-clock P-only 측정이며 공식 strict P(z)의 계약을
만족하지 않는다. 특히 `NMSE -9.64 dB`는 RIR이 녹음 파형을 얼마나 재현하는지의 수치이지
ANC 출력 감쇠(dB)가 아니다.

### [판정]

**Likely** — 현장 Primary의 독립 교차 참고로는 유효하지만, **canonical P(z)로 승격은
불가**하다.

### [다음 행동]

외부 P raw는 보존된 진단자료로만 인용한다. 현행 학습은
`assets/measured/primary_path_il_strict_5dc06fdd.npz`만 사용한다. Primary의 독립
교차검증이 필요하면 스피커 없이 기존 raw를 오프라인으로 비교하고, 새 측정은 현재
strict protocol을 그대로 사용한다.

## 4. [가설] 외부 Secondary 캡처가 현행 S(z)를 보완한다

### [근거]

외부 `measurements/secondary_raw`에는 두 종류가 있다.

1. `run_20260827_01`: `recording_session.json`이 없고 repeat 2개만 존재한다.
2. `run_20260827_02_retry/recording_session.json` (SHA256 `d6bd9b1c…`):
   5 repeats, 48 kHz라고 기록했지만 `period_size=512`, `latency=high`,
   gain `0.25`, excitation 100–1,000 Hz이다.

retry metadata는 다음과 같은 내부 모순을 가진다.

| 항목 | 기록값 |
|---|---:|
| 출력 frames/repeat | 624,128 |
| 입력 frames/repeat | 1,247,232 또는 1,248,256 |
| 출력 callbacks | 1,219 |
| 입력 callbacks | 2,436–2,438 |
| xrun/clipping | 0 / 빈 배열 |
| 전기적 `speaker_input` capture | 없음 |
| `path_input_source` | `playback_command_waveform_proxy` |

첫 repeat의 callback timestamp도 입력 clock에서 중복/비정상 cadence를 보인다. 예를 들어
`run_20260827_02_retry`는 period 512인데 입력 callback 수가 출력의 약 2배이고,
`run_20260827_01`은 period 256에서 입력 4,876 callback 대 출력 2,438 callback이다.
capture 파일은 출력 길이로 잘렸기 때문에 metadata의 입력 frames와 저장 raw sample 축이
일치하지 않는다.

또한 외부 코드 자체가 Secondary에서 전기적 ANC-speaker 입력 채널이 없으면 command
waveform을 unverified proxy로 저장하도록 설계되어 있다. 실제 `run_20260827_02_retry`
metadata도 이 경고를 포함한다.

### [확인 방법]

session 존재 여부, repeat 수, callback/frame 수, timestamp의 시작·끝, channel mapping,
proxy warning을 직접 읽고 raw 배열 shape를 확인했다. Secondary RIR/validation bundle은
외부 폴더에 존재하지 않는다.

### [결과]

외부에는 동시 측정된 유효한 S(z)나 feedback RIR이 없다. xrun 0만으로 입력 clock
불일치와 proxy 입력 문제를 상쇄할 수 없다.

### [판정]

**Invalid experiment** — 외부 Secondary를 학습·lead·평가에 사용하면 안 된다.

### [다음 행동]

현행 strict S artifact를 계속 단일 출처로 사용한다. 후속 acoustic-ref/closed-loop를
할 때 Secondary는 반드시 전기적 speaker-input loopback과 공통 clock/clock witness를
포함한 별도 strict capture로 다시 확보한다.

## 5. [가설] 외부 덕트 geometry가 현행 덕트와 동일하다

### [근거]

외부 `configs/duct.yaml`과 `rir/metadata.py`는 `125 × 125 × 1200 mm`만 기록한다.
벽 두께, 내측 단면, 마이크/스피커 위치는 없거나 null이다.

현행 `configs/duct.yaml`은 외형 125 mm, 벽 10 mm, **내측 단면 105 × 105 mm**, 내측 길이
1.190 m를 명시한다. 현행 plane-wave cutoff는 내측 단면으로

`343 / (2 × 0.105) = 약 1,633 Hz`

이다. 외부의 125 mm를 내측 단면으로 해석하면

`343 / (2 × 0.125) = 약 1,372 Hz`

가 된다. 두 값은 1,600 Hz 주변의 물리 판정을 바꾼다.

### [확인 방법]

두 config와 공통 geometry 상수를 읽고 같은 `c/(2a)` 식으로 계산했다. 외부 session의
positions/gain/serial null도 확인했다.

### [결과]

외부 125 mm가 외형인지 내측인지 판별할 증거가 없다. 외부 RIR의 geometry provenance가
현행 duct와 일치한다고 가정할 수 없다.

### [판정]

**Inconclusive** — 문서 숫자만으로 덕트 동일성을 확정할 수 없다.

### [다음 행동]

현행 geometry와 strict measurement metadata를 유지한다. 외부 데이터를 재사용하려면
실제 내측 단면·벽 두께·마이크/스피커 위치를 사용자 입회 아래 별도 확인하고, 그 뒤에도
공식 strict P/S를 새로 만들어야 한다. 외부 RIR을 숫자만 옮기는 방식은 금지한다.

## 6. [가설] 외부 검증 도구의 기준은 현행 평가에 그대로 적용해도 된다

### [근거]

재사용 가치가 있는 부분:

- `validation/core.py`가 convolution prediction, NMSE, correlation, PSD, phase,
  group delay, coherence, repeatability를 한 보고서에 남긴다.
- 기본값으로 암묵적 time alignment를 하지 않고 `alignment_offset_samples`를 명시한다.
- `rir/excitation.py`가 seed, 실제 peak/RMS, pre/post silence, frequency range를 sidecar에
  저장한다.
- `audio/alsa_io.py`가 실제 int16 playback와 raw int32 input, PortAudio ADC/DAC/current
  timestamp, callback count, XRUN을 저장한다.
- `system/causality.py`가 `DP - DSA - DSE`를 units별로 보존하고 미측정값을
  `UNKNOWN_FROM_MEASUREMENT`로 남긴다.

재사용하면 안 되는 부분:

- `validation/rir_report.py`의 기본 status는 항상 `REVIEW`이고 기준이 비어 있다.
- `apply_explicit_criteria()`는 호출자가 준 `max_nmse` 하나만 통과해도 PASS를 낼 수
  있어 frequency/coherence/repeatability/고역 do-no-harm을 보장하지 않는다.
- 주파수 응답은 단순 FFT 비율이며 band excitation 밖의 수치에 confidence gate가 없다.
- callback trace에 dict/list append와 파일 후처리용 allocation이 있어 hard realtime path에
  적합하지 않다.
- `StreamingANCApplication`은 error channel을 읽지만 제어 입력으로 사용하지 않고,
  reference만 모델에 넣는다. queue underrun을 0 출력으로 대체할 뿐 deadline 계약이
  없다.

### [확인 방법]

검증·오디오·causality 모듈과 현행 `TrainingTimingContract`, strict raw schema,
`evaluate_session`/G4 gate를 정적으로 비교했다.

### [결과]

외부의 “raw 보존·명시적 정렬·REVIEW 상태·DP/DSA/DSE 분리”는 좋은 설계 원칙이다.
하지만 현행 G4와 canonical prerequisite를 대체할 정도의 게이트는 아니다.

### [판정]

**Confirmed** — 일부 원칙은 참고할 가치가 있으나, 판정 엔진은 현행 strict 계약을
우선해야 한다.

### [다음 행동]

현행 저장소에서는 다음 세 가지를 유지·강화한다.

1. raw playback와 raw input, callback/xrun/timestamp provenance 보존
2. 수동 offset/lead 금지 및 `TrainingTimingContract` 단일 출처
3. 단일 NMSE가 아닌 150–1600 Hz 부대역, source family, worst-case, CI, out-of-band
   do-no-harm을 함께 보는 G4

## 7. 현행 파이프라인에 대한 최종 판정

| 단계 | 현행 상태 | 외부 폴더가 바꾸는가 |
|---|---|---|
| strict P/S | **DONE** (P/S 모두 같은 capture, 19 kept, xrun 0, clock/joint-LS/compact 검증) | 아니오 |
| source/manifest 계보 | 복구·Elice 재검증 경로 진행 | 아니오 |
| public corpus/recorded QA | Elice bootstrap 및 QA 완료 | 아니오 |
| loss pilot | 4번째 20k 후보 실행 중; 앞선 3개는 diagnostic | 아니오 |
| canonical 100k pretrain | **NOT STARTED** (pilot winner·probe·deterministic resume smoke 선행) | 아니오 |
| measured 50k fine-tune | **NOT STARTED** | 아니오 |
| ONNX/TensorRT canonical export | **NOT STARTED** | 아니오 |
| Jetson canonical realtime | **NOT STARTED** | 아니오 |
| Level-5 새 자연음 challenge | **NOT STARTED** | 아니오 |

따라서 현재 결론은 다음과 같다.

- 외부 폴더의 실제 raw는 “현장 장치에서 100–1,000 Hz Primary를 녹음할 수 있었다”는
  진단 증거다.
- 외부 폴더에는 실제 덕트에서 ANC가 몇 dB 감쇠했다는 evidence가 없다. 외부 RIR fit의
  `-9.64 dB`는 ANC attenuation이 아니다.
- 1,000–1,600 Hz 및 1,600 Hz 이상, speech/music/environment/machine의 실제 ANC,
  고주파 증폭 유무, Tiny/Base 우열은 외부 폴더로 판단할 수 없다.
- 현재 고주파 최우선 목표는 외부 3232-parameter 모델을 붙이는 것이 아니라, 현행
  strict S가 검증된 150–1,600 Hz에서 canonical 학습 후 실제 덕트 G4와 Level-5
  natural challenge로 검증하는 것이다.

## 재사용 목록

### 채택할 원칙

- 측정 명령 전 dry-run과 명시적 안전 확인
- 실제 제출된 PCM과 raw input을 함께 저장
- 독립 stream의 ADC/DAC/host timestamp와 XRUN을 보존
- RIR 검증은 자동 PASS가 아니라 `REVIEW`에서 명시 기준을 적용
- 음향 지연·전기 지연·모델 지연을 단위별로 분리
- 논문에 없는 구현 선택은 `UNKNOWN_FROM_PAPER`로 표시

### 채택하지 않을 것

- 외부 3,232-parameter 모델의 weight/init/배포 전환
- 외부 100–1,000 Hz P-only RIR을 현행 P(z)로 교체
- 외부 proxy Secondary 또는 프레임 불일치 raw를 S(z)로 사용
- 외부 125 mm geometry를 현행 105 mm 내측 geometry로 간주
- 외부 model-only latency 수치를 현장 deadline 통과로 해석
- 외부 RIR fit NMSE를 ANC 감쇠 dB로 보고

이 문서 자체는 분석 기록이며, 외부 폴더의 원본을 복제하지 않는다.
