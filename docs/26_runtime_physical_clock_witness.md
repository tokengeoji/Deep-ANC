# Runtime physical clock witness 계약

> 상태 기준일: 2026-08-28
> 범위: digital-reference realtime session의 actual submitted source/control,
> ERR/REF raw, ANC gain, PortAudio callback receipt를 한 번에 결속하는
> offline 조건부 감사
> 현재 live authority: **None**. 이 문서를 작성하며 오디오 출력은 0회다.

## 1. 현재 판정

### [가설]

PortAudio callback telemetry에 152–600 Hz의 낮은 출력 연속 reserved
pilot을 더하면, callback 이전의 silent ADC-period drop과 ADC↔DAC
rate trajectory를 실제 acoustic raw에서 보완할 수 있다고 가정한다.

### [근거]

- software telemetry: `src/deep_anc/realtime/clock_telemetry.py`
- actual-int16 연속 pilot과 estimator:
  `src/deep_anc/dsp/fullband_causal_v4.py`
- runtime 전용 결속/issuer:
  `src/deep_anc/realtime/physical_clock_witness.py`
- 현 runtime NPZ는 `source`, `control`, `err`, `ref`, `anc_gain`을 float32 raw로
  저장하고 clock sidecar에 callback time/counter를 저장한다.
- v4의 fixed-LTI synthetic fixture에서 시계 기울기는 식별 가능하지만,
  acoustic-only raw는 `ADC clock motion`과 `두 마이크에 공통인
  time-varying plant delay`를 구분할 수 없는 byte-identical 반례가 있다.

### [확인 방법]

runtime issuer는 측정 P/S schema를 재사용하지 않고 별도
`runtime_physical_clock_witness_v1`을 발행한다. 감사자는 다음을 raw에서
다시 계산한다.

1. 캡처 전 no-replace plan과 session/clock target path, plan SHA
2. runtime float32 `source/control`을 콜백과 동일한
   `rint(clip(x)×32767)`로 복원한 exact submitted S16 PCM과 SHA
3. source output의 v4 primary pilot line exact 일치, control output의 같은
   line exact null
4. ERR/REF raw, ANC gain, NPZ/clock-sidecar file SHA와 embedded telemetry SHA
5. 44개×32,768 samples = **30.037333 s** 전 구간 ANC gain ≥0.999
6. 시간축을 덮는 사전 고정 fit row와 나머지 untouched validation row
7. ERR/REF별 rate, linear/cubic 차이, segment q residual/phase slope, gain
   stationarity, change-point, one-sample slip
8. xrun/deadline/fallback/watchdog/ring drop·add·overrun·underrun exact 0,
   absolute backlog `≤` predeclared one-hop 256 samples, maximum excess backlog exact 0

SPSC producer/consumer를 어느 순간 읽었는지에 따라 정상 absolute backlog가 0 또는
256 samples로 관측될 수 있다. 따라서 absolute 0은 고역 안전조건이 아니며,
허용 one-hop을 넘은 excess만 exact 0으로 판정한다.

clock fit은 v4의 `_estimate_rate_ratio`, `_validate_clock_rows`,
`_transfer_bank`, `_fractional_delay`를 그대로 사용한다. runtime은 source
pilot path만 존재하므로 v4 primitive에 실제 요청 path subset을 넘기는
backward-compatible 인자를 추가했다. 기본값은 여전히 P/S 두 path 전체이며,
없는 secondary pilot을 가짜로 넣지 않는다.

### [결과]

신규 계약의 상태는 세 가지뿐이다.

| 상태 | 의미 | 상위 runtime PASS |
|---|---|---|
| `BLOCKED` | raw/SHA/pilot/counter/30 s/stationarity 중 하나라도 실패 | 불가 |
| `FIXTURE_ONLY_PASS` | synthetic 구조 fixture의 수치 통과 | 불가 |
| `CONDITIONAL_PASS` | 실제 sounddevice raw가 fixed-LTI acoustic scope에서 통과 | 독립 clock PASS 아님 |

`CONDITIONAL_PASS`에서도 다음은 항상 `false`다.

- `independent_clock_authority_pass`
- `canonical_runtime_pass`
- `deployment_eligible`
- `independent_electrical_clock_witness_present`

기존 `realtime_clock_telemetry_v1` 자체의 authority도 `INCONCLUSIVE`로 남긴다.
acoustic fixed-LTI 가정이 더해졌다고 software telemetry를 `PASS`로 바꾸지
않는다.

### [판정]

**Likely / conditional under fixed LTI. Independent authority: Inconclusive.**

현재 저장된 일반 runtime session에는 predeclared pilot plan과 exact
reserved-line/null 증거가 없다. 따라서 현 Jetson의 실제 high-frequency
runtime clock authority는 여전히 **BLOCKED**다. 이 계약은 부족한 raw를
사후에 PASS로 바꾸는 도구가 아니다.

### [다음 행동]

physical live를 열기 전에 runtime capture가 plan SHA, `synthetic_fixture`,
capture origin, hardware/config/checkpoint/deployment SHA를 NPZ와 clock sidecar 양쪽에
직접 봉인하도록 통합해야 한다. 현 모듈은 이 필드가 없어도
수치 분석은 하지만, 독립/canonical authority로는 절대 승격하지
않는다.

## 2. 11.314 kHz timing budget과 고역 결과 독립성

### [가설]

저역 pilot으로 추정한 q residual이 8 kHz octave 상단 11.314 kHz의
20 dB-grade timing budget을 만족하면, 고역 attenuation 결과를 순환적으로
clock fit에 쓰지 않고도 timing 정밀도를 판정할 수 있다고 가정한다.

### [근거]

- 허용 residual: **0.06755189029558946 sample**
- pilot fit 대역: v4와 동일한 152–600 Hz
- `highband_target_or_attenuation_used_for_clock_fit=false`
- 출력 두 channel은 같은 AB13X DAC clock을 공유한다.

### [확인 방법]

fit/validation은 low-pilot complex line만 받는다. 2/4/8 kHz target ERR
power, DL attenuation, FxLMS 차이, model output score는 함수 인자에 없다.
ERR과 REF 두 view의 capture-end rate disagreement, leaveout, linear/cubic,
segment change-point를 모두 같은 sample budget으로 막는다.

### [결과]

timing 계약은 11.314 kHz에서 필요한 정밀도를 검사하지만,
11.314 kHz의 음향 에너지나 감쇠를 검사하지 않는다. 또한
`ADC–DAC rate`를 `NS–CS 상대 출력 위상 drift`로 재해석하지 않는다.

### [판정]

**Confirmed as a timing-resolution gate; not an attenuation result.**

### [다음 행동]

이 timing witness 통과 후에도 별도 matched OFF/DL/FxLMS raw에서
2/4/8 kHz attenuation을 재계산해야 한다.

## 3. 125 Hz octave와 150 Hz point-control union의 불일치

### [가설]

현 광대역 계약의 `octave_centers_hz=125` 표기가 125 Hz octave 전체
제어를 의미하지만, 실제 point-control union은 150 Hz에서 시작해
일부 대역이 빠졌을 가능성이 있다고 가정한다.

### [근거]

`src/deep_anc/dsp/control_band_contract.py`의 현재 값은 다음과 같다.

```text
point-control union lower = 150 Hz
octave center = 125 Hz
125-Hz octave = [125/sqrt(2), 125*sqrt(2)]
               = [88.3883476483, 176.7766952976] Hz
```

즉 **88.388–150 Hz**는 125 Hz octave에 포함되지만 현 point-control
union 밖이다. 더군다나 runtime clock pilot은 152–600 Hz이므로 125 Hz
octave 전체는 물론 150–152 Hz도 덮지 않는다.

### [확인 방법]

runtime plan/evidence에 다음을 강제했다.

- `pilot_band_is_clock_witness_not_control_or_evaluation_band=true`
- `control_attenuation_assessed=false`
- `octave_125_hz_fully_covered_by_pilot=false`
- `point_control_union_150_11314_claimed_by_witness=false`

### [결과]

152–600 Hz pilot PASS로 125 Hz octave, 150–11.314 kHz union, 저역 ANC
성능을 주장할 수 없다. 현 control contract 자체에도 “125 Hz octave
전체”와 “150 Hz 이상 point-control” 사이의 의미 불일치가 있다.

### [판정]

**Confirmed contract gap.** runtime clock witness와는 독립적으로 정리해야 한다.

### [다음 행동]

최종 제어 계약은 두 선택지 중 하나를 raw를 보기 전에 고정해야 한다.

1. 125 Hz octave 전체를 성능 주장에 남기려면 P/S·source·recorded·G4
   하한을 **88.388 Hz**까지 확장하고 현 80–150 Hz SNR/일관성
   blocker를 실측으로 복구한다.
2. 150 Hz 아래를 제어 대상에서 제외하려면 125 Hz를
   `controlled octave`가 아닌 별도 diagnostic/do-no-harm 항목으로 명시한다.

사용자의 “저역·고역 모두” 목표와 80–150 Hz의 현 물리 증거 부족을
감안하면, 이 선택은 runtime witness가 임의로 대신하지 않는다.

## 4. Output-clock-master 별도 stream 구조 감사

### [가설]

digital-reference open-loop를 AB13X `OutputStream` 시계로 독립 구동하고,
APE `InputStream`을 기록/안전용으로 분리하면 full-duplex PortAudio가
playback ready 상태 때 callback 전 capture period를 버리는 경로를 제거할
수 있다고 가정한다.

### [근거]

현 `sounddevice.Stream` combo는 다른 clock domain의 APE input과 AB13X
output을 하나의 fixed-256 callback에 묶는다. 현 PortAudio ALSA backend은
playback이 ready가 아니고 `neverDropInput=false`일 때 Python callback 이전에
input period를 버릴 수 있다. 반면 NS/CS는 원래 같은 AB13X
output stream/DAC clock을 공유한다.

그러나 현 Tiny/Base의 모델 입력은 `[reference, error-feedback]`다.
물리 APE 채널은 ERR=ch0, REF=ch1이지만, 모델 tensor에서는 ERR가
**두 번째 feature channel**이다. 따라서 “input은 기록만”으로 바꾸면 현
체크포인트의 입력 계약을 바꾸는 것이다.

### [확인 방법]

두 구조를 분리해 판정했다.

#### A. 현 2-input model을 그대로 유지

```text
APE InputStream --(rate matcher/ASRC + timestamped ring)--> ERR feature
AB13X OutputStream clock --> source/control scheduling and DAC frame index
```

- 출력 time-base는 안정되지만 ERR를 DAC 축으로 연속 ASRC해야 한다.
- hard one-sample insert/drop은 고역 위상을 깨므로 금지한다.
- ASRC 비율/knots/residual, cross-clock ring occupancy, stale ERR age를 저장한다.
- 실효 handoff/queue가 바뀌므로 old P/S·lead=115를 재사용하지
  않고 새 구조에서 다시 측정한다.

#### B. 참조만 쓰는 진정한 open-loop model로 변경

```text
AB13X OutputStream clock --> digital reference --> ref-only controller --> NS/CS
APE InputStream --> recording + independent safety watchdog only
```

- 출력 파형 생성은 ADC rate drift와 분리된다.
- 하지만 현 2-input checkpoint에 ERR=0을 넣는 것은 ref-only 모델의 증거가
  아니다. 별도 architecture/ablation, 처음부터 학습, matched G4가 필요하다.
- ADC 기록이 중단되면 출력 watchdog이 유한한 최대 시간 안에
  상쇄 출력을 fade-to-zero해야 한다. 이 최대 mute latency는 이론값이
  아니라 실측 receipt로 고정한다.

### [결과]

output-clock-master는 다음을 **실제로 줄인다**.

- NS/CS submitted sample index의 공통 DAC time-base
- combined full-duplex backend의 playback-not-ready 때 silent input discard가 output
  callback 스케줄을 끊는 경로
- digital source FIFO/control submission의 절대 DAC frame-index 불투명성

하지만 다음은 **제거하지 못한다**.

- APE ADC raw의 별도 rate, xrun, silent loss
- 현 모델의 ERR 두 번째-feature 의존성
- ERR을 출력 축으로 옮기는 ASRC의 residual/latency
- 실제 덕트의 time-varying acoustic delay와 ADC motion의 acoustic-only confounding
- ERR 녹음의 올바른 DAC-time mapping 없이 감쇠를 계산하는 문제

### [판정]

**Likely beneficial but not sufficient.** 2-input 현 모델에서는 output-master +
continuous ASRC가 필수이며, ref-only로 바꾸려면 새 학습/실측 계약이다.

### [다음 행동]

output-master prototype을 열 때 receipt에 최소 다음을 봉인한다.

- DAC absolute frame index, actual submitted stereo int16 SHA, source/control SHA
- ADC absolute callback index/time, raw ERR/REF SHA
- ASRC input/output sample index, ratio/knots/residual, filter delay, state reset count
- input/output ring absolute maximum, one-hop allowed maximum, derived maximum excess를
  함께 저장하고 absolute `≤allowed(256)`, excess `=0`을 강제한다. drop/add/slip/fallback은
  별도 event counter로 exact 0을 유지한다.
- model inference P50/P95/P99/max, deadline miss 0
- ADC stale-age maximum, watchdog detection-to-mute latency, mute completion frame
- exact hardware/config/checkpoint/deployment/plant/timing/plan SHA
- 30초 이상 연속 pilot의 q/stationarity receipt
- 새 stream topology에서 재측정한 P/S·handoff·`PlantDelays.lead()`

프로토타입은 운영 코드와 분리한 무음 fixture→device-open 없는
dry-run→사용자 승인 순서를 거친다. 현 문서는 live 명령을 승인하지
않는다.

## 5. 남은 하드웨어 한계

### [가설]

이 acoustic runtime witness가 통과하면 외부 electrical loopback이나
공통-clock I2S DAC가 불필요하다고 가정할 수 있다.

### [근거]

acoustic-only byte-identical 반례가 존재하고, 현 AB13X capture의 DAC
공통 oscillator 여부는 알려지지 않았다. 현 I2S2에는 DOUT pin과
2-channel slave DAC가 없다.

### [확인 방법]

acoustic conditional scope와 independent clock scope를 별도 boolean으로 발행한다.

### [결과]

acoustic witness는 software-only 현 구조에서 가장 강한 보완이지만,
공통 time-varying path 반례를 제거하지 못한다.

### [판정]

**Independent hardware authority remains BLOCKED.**

### [다음 행동]

가장 강한 최종 구조는 APE ADC와 공통 clock을 쓰는 I2S2 2-channel
slave DAC다. AB13X를 유지하려면 안전한 electrical loopback 또는
output-master+ASRC의 실제 장시간 receipt가 추가로 필요하다.
