# 40. Fullband causal v7 — 비 affine clock 연구 경로와 최종 권위 경로

> 상태 기준일: 2026-08-29
>
> 상태: **설계 중 / live 출력 BLOCK / canonical training BLOCK**
>
> 범위: 보존된 v6 raw가 드러낸 시간가변 clock 현상을 다음 세대에서 어떻게
> 반증할지 정의한다. 이 문서는 v7 live 실행을 승인하지 않으며, P/S·ANC 감쇠·학습
> authority를 발행하지 않는다.

## 1. 결론부터

v7은 다음 두 경로를 명시적으로 분리한다.

| 경로 | 하드웨어 | 만들 수 있는 결과 | 만들 수 없는 결과 |
|---|---|---|---|
| A. conditional research | 현 AB13X 출력 + APE ERR/REF 입력 | fixed-LTI 가정 아래의 acoustic time-map/P/S 진단 | independent clock authority, canonical training admission |
| B. canonical-authority route | 1차 조사: APE I2S2 입력 + 온보드 RT5640/I2S1 출력, 2차: 외부 shared-clock DAC 또는 독립 electrical witness | hardware frame identity/electrical gate까지 통과한 뒤의 canonical 후보 | topology·무음 smoke만으로 만든 clock/P/S authority |

경로 A가 수치 gate를 모두 통과해도 다음 값은 고정한다.

```text
clock_authority = conditional_acoustic_only
independent_clock_authority_pass = false
canonical_training_eligible = false
common_time_varying_path_counterexample_excluded = false
```

경로 B에서도 shared-clock/electrical receipt 하나만으로 학습을 열지 않는다. 정확한
actual-int16 signal lineage, P/S 절대 지연, exact condition, 비선형성, fit-a/fit-b
stationarity, untouched holdout, 8개 물리 부대역이 모두 통과해야 별도 검토를 통해
canonical candidate가 될 수 있다.

## 2. v6 raw가 실제로 반증한 것

### [가설]

v6의 실패가 단순 저 SNR이나 한 개의 잘못된 affine 비율 때문이 아니라, capture 중
time-base 또는 그와 구분되지 않는 acoustic path가 시간에 따라 변했기 때문이라고
가정한다.

### [근거]

보존해야 하는 원본은 다음과 같다.

```text
results/fullband_causal_v6/raw_capture.npz
SHA-256 f153c8664106b0c341b67db940fb2fb1d76cb7e58c2fa9a6e49558e1dba50a63

results/fullband_causal_v6/failure_232a4e53a4eaa024d54b740a01c95fe1.json
SHA-256 10856999254a8dc70c3696b02aed239db1b80f217a3dfd771442cedb2aacc75d
```

공식 v6 failure receipt는 다음을 기록한다.

- `failure_stage=global_grid_basin_search`
- `global clock objective가 multimodal ambiguous`
- preterminal clock-line 최저 SNR `24.174183 dB`
- terminal clock-line 최저 SNR `26.963488 dB`
- optimizer는 시작했지만 operator/P/S는 발행하지 않음
- `canonical_training_eligible=false`

따라서 20 dB SNR admission은 통과했지만 단일 affine q admission은 통과하지 못했다.
v6 estimator는 `±1000 ppm` 안의 단일 affine 비율을 찾으며, P/S clock block을
시간 분리하고 PE slot에는 continuous witness를 두지 않는다.

보존 raw의 short-time diagnostic-only 재검산은 약 다음 세 mode를 보였다.

```text
-4743.6 ppm 부근
 -851.6 ppm 부근
+3149.5 ppm 부근
```

더 중요한 사실은 일부 2-period clock block **내부에서도 여러 번 전환**이 관측된다는
점이다. 이 mode 요약은 admission gate가 아니며 P/S·clock·학습 authority가 아니다.
그러나 “plateau 세 개이므로 change point는 최대 두 개”라는 설계 역시 raw에 의해
지지되지 않는다. 같은 mode로 되돌아오는 여러 전환, 짧은 dwell, path switch nuisance가
모두 가능하다.

### [확인 방법]

v4/v5/v6의 식별 가정을 비교했다.

| 세대 | witness | P/S 역할 | time-map | 현장 한계 |
|---|---|---|---|---|
| v4 | 연속 periodic low-band pilot | pilot 동시, main 역할 분리 | affine `±1000 ppm` | actual-int16 support-1024 condition `>20` |
| v5 | 연속 periodic pilot + near-white PE | main P/S 시간 분리 | affine `±1000 ppm` | periodic alias와 path switch, live authority 미완성 |
| v6 | 고-SNR 2-repeat checkpoints | P/S 완전 시간 분리 | affine `±1000 ppm` | PE blind interval, 실제 raw multimodal/nonstationary |

관련 근거는 [21](21_runtime_clock_domain_audit.md),
[22](22_fullband_causal_v4.md), [26](26_runtime_physical_clock_witness.md),
[31](31_fullband_causal_v5.md), [31 offline](31_fullband_causal_v5_offline_authority.md),
[39](39_fullband_causal_v6_clock_checkpoints.md)에 있다.

### [결과]

다음 세 가지를 동시에 구분할 수 없는 상태다.

1. 실제 ADC↔DAC rate가 여러 번 변함
2. 시간 분리된 primary/secondary acoustic path의 phase/delay nuisance
3. periodic checkpoint의 cycle/basin alias

v6 raw에 새 spline이나 임의 개수의 change point를 사후 fit해도 이 구분은 생기지 않는다.

### [판정]

**v6: Invalid for clock/P/S promotion. Diagnostic evidence: Confirmed.**

v6 raw와 failure는 immutable diagnostic artifact로 보존한다. v7 코드가 생겨도 v6 raw를
v7 authority로 다시 포장하거나 같은 target을 교체하지 않는다.

## 3. v7 공통 fail-closed 원칙

### 3.1 time-map parameterization은 아직 미정이다

다음 후보는 연구 대상일 뿐 현재 계약이 아니다.

- bounded piecewise-affine monotone map
- monotone cubic spline 또는 integrated positive-slope basis
- 제한된 상태공간/rate trajectory
- hardware-observed rate를 보간하는 deterministic map

**segment 수, knot 수·위치 grid, dwell, slope bound, regularizer, model-selection rule을
현 v6 mode 세 개에 맞춰 고정하지 않는다.** 특히 `max_change_points=2` 또는
`max_plateaus=3`은 금지한다.

후보 우선순위는 `affine → monotone P-spline → integrated-TV rate`다. affine이 모든
cross-fit gate를 통과하면 더 복잡한 후보를 열지 않는 **simplest-pass**를 쓴다.
free-level semi-Markov/state 모델과 `1/256=3,906.25 ppm` lattice snap은 현재
diagnostic-only이며 live 선택 후보가 아니다. v6 mode 수·간격을 hyperparameter prior로
쓰지 않는다.

어느 non-affine parameterization을 사용할지는 다음이 끝나기 전까지 live blocker다.

1. exact dtype/shape/SHA로 봉인된 immutable synthetic raw suite 생성
2. affine, smooth drift, 반복 plateau 회귀, 여러 block 내부 전환 positive fixture
3. path별 delay/phase/gain drift, common time-varying acoustic delay, periodic alias,
   one-sample insert/drop negative fixture
4. 후보 parameterization·bounds·selection rule별 raw replay
5. actual-int16 joint condition `≤20` 확인
6. fit-a/fit-b만으로 선택하고 validation/holdout에서 no-refit 확인
7. 아래 timing hard gate와 8-band plant gate 통과
8. 선택된 signal plan, estimator, thresholds, expected artifact SHA를 live 전에 봉인

model-selection 계약은 다음을 모두 live 전에 exact bytes로 고정한다.

- float64, optimizer 초기값·순서·iteration·tolerance, random restart 금지
- 후보별 hyperparameter grid, effective-complexity 계산식, 고정 noise margin과 tie-break
- fit-a와 fit-b mask는 capture 시작/중간/끝을 모두 포함하도록 시간축에 interleave
- fit-a→fit-b와 fit-b→fit-a 양방향 cross-fit 및 네 path×mic view 전부 평가
- q-fit은 사전 선언된 low-band aperiodic witness만 읽고 high-band P/S residual을 읽지 않음
- 복잡한 후보는 단순 후보보다 worst cross-fit residual 개선이 고정 noise margin의 4배보다
  클 때만 승격
- noise-equivalent objective 안의 모든 q가 validation frame에서 서로 `≤0.050 sample`인
  ambiguity envelope
- 동일 complexity tie는 worst cross-fit residual, 고정 family 우선순위, canonical
  parameter bytes SHA 순으로 결정
- 선택 후 fit-a+fit-b final refit은 정확히 한 번만 하고 **첫 validation open 전** q/FIR/
  delay/support/formula SHA를 freeze
- validation 실패 후 같은 raw에서 다음 후보·penalty를 열지 않고 generation 전체 실패
- transition 주변 frame을 사후 제외하지 않음
- nominal actual-int16뿐 아니라 선택된 q-warp와 absolute delay가 적용된 support-1024
  operator condition도 `≤20`
- holdout bytes를 바꿔도 pre-holdout selected-model SHA가 불변인 leakage fixture
- off-lattice rate와 transition 비정렬 fixture를 포함하고 3,906.25 ppm snap을 명시적으로 거부

동일 synthetic raw를 보며 알고리즘과 threshold를 계속 고치는 개발 set과, 마지막에
한 번 여는 immutable terminal synthetic set도 분리한다. terminal fixture를 본 뒤 model
complexity를 바꾸면 새 generation으로 다시 봉인한다.

### 3.2 time-map의 공통 불변식

parameterization과 무관하게 다음은 바꿀 수 없다.

- `q(0)` gauge는 first submitted/first captured frame 계약으로 하나만 고정한다.
- q는 capture 전체에서 연속이고 strictly monotone이어야 한다.
- ERR/REF × P/S 네 view가 **동일한 q 하나**를 공유한다.
- path, microphone, fit role, amplitude role별 q/offset은 금지한다.
- q가 sample insert/drop 같은 불연속을 흡수하게 하지 않는다.
- plant FIR/gain/phase/delay를 time-map segment마다 새로 주지 않는다.
- 고주파 P/S residual이나 ANC attenuation으로 q를 repair하지 않는다.
- search boundary optimum, unidentifiable complexity, multiple equivalent optimum은 BLOCK이다.
- uncertainty와 parameter covariance를 발행하지 못하면 authority를 열지 않는다.

### 3.3 timing gate

v7은 v4의 더 엄격한 timing subgate를 유지한다.

| gate | 최대값 |
|---|---:|
| ERR/REF×P/S capture-end/view disagreement | `0.050 sample` |
| 동일 frozen q의 linear/cubic interpolation crosscheck | `0.006 sample` |
| 위 두 오차의 combined budget | `0.056 sample` |
| 모든 independent validation/terminal hard residual | `0.06755189029558946 sample` |

linear/cubic 비교에서 각 interpolation이 별도 q/knot를 fit해서는 안 된다. 동일하게
freeze된 time-map과 path nuisance에 resampler만 바꿔 계산한다. high-band 결과에 맞춘
추가 phase shift는 exact 0이어야 한다.

`0.06755189029558946 sample`은 11.314 kHz에서 필요한 timing-resolution budget이지
2/4/8 kHz ANC 감쇠를 뜻하지 않는다.

## 4. A — 현 AB13X+APE conditional research path

### [가설]

P/S를 동시에 출력하고 capture 전체에 independent aperiodic witness를 유지하면 v6의
path switch와 blind interval을 제거하여, fixed-LTI 범위에서 time-map과 P/S를 더 강하게
반증할 수 있다고 가정한다.

### [신호 계약]

- 48 kHz, block 256 유지
- primary/noise speaker와 secondary/control speaker를 capture 전 구간 동시 활성화
- 두 DAC channel에는 서로 독립인 deterministic aperiodic actual-int16 code 사용
- clock witness는 모든 frame에 존재하며 단순 반복 multisine을 authority로 사용하지 않음
- actual submitted stereo int16 전체와 그 SHA를 denominator로 사용
- intended float, pilot-only, 재생 전 배열로 바꾸면 실패
- 두 channel 합산 active-block power가 동일 amplifier setting의 v7 meter를 넘지 않음
- peak/RMS/crest/cross-correlation/8-band spectrum을 audio 전 exact audit
- transition/role boundary에서도 witness가 끊기지 않음

signal duration, code amplitude와 role별 frame 수는 아직 봉인하지 않았다. exact condition과
meter power를 통과한 builder가 생기기 전에는 audible time을 제시하거나 live를 실행하지
않는다.

### [역할과 접근 순서]

모든 역할은 서로 다른 seed/code와 disjoint frame mask를 사용한다.

```text
fit_a
fit_b
clock/time-map validation
amplitude/polarity nonlinear validation
operator holdout
terminal no-refit clock/plant validation
```

필수 실행 순서는 다음과 같다.

```text
exact plan/PCM/SHA
→ fit_a·fit_b만으로 time-map/path nuisance/delay/support 결정
→ untouched conditional acoustic time-map validation
→ nonlinear validation
→ fixed candidate formula와 SHA 봉인
→ operator holdout 첫 open
→ terminal frozen-map validation
→ captured full SHA와 receipt 발행
```

첫 validation을 연 뒤 다음 중 하나라도 바꾸면 generation 전체를
실패시킨다.

- time-map complexity, knot, coefficient, slope/intercept
- P/S FIR, absolute delay, fractional delay
- support, compact pre-roll, candidate weight
- noise estimator, band threshold, nonlinear threshold
- interpolation 종류 또는 high-band phase repair

### [joint path nuisance와 절대 P/S delay]

각 microphone은 actual two-input causal operator로만 맞춘다.

```text
y_m[n] = sum_l h_(m,P)[l] x_P(q(n)-d_(m,P)-l)
       + sum_l h_(m,S)[l] x_S(q(n)-d_(m,S)-l)
       + fixed noise/offset
```

- `h_(ERR,P)`, `h_(REF,P)`, `h_(ERR,S)`, `h_(REF,S)`는 capture 전체에서 고정
- signed gain/polarity는 FIR에 포함하며 추가 부호 반전 금지
- role/segment마다 새 FIR, gain, phase, delay를 허용하지 않음
- coarse scan은 최대 4,800 samples, compact support 1,024, pre-roll 256을 기준 후보로
  사용하되 synthetic replay 전에 자동 승격하지 않음
- fit-a/fit-b absolute bulk/fractional delay disagreement는 `≤0.15 sample`
- 네 absolute delay와 ERR 기준 `P-S` 상대 delay를 모두 발행
- compact FIR peak는 diagnostic이며 bulk delay와 중복 합산하지 않음
- fractional delay는 compact shape에 정확히 한 번만 encode
- handoff 256은 plant NPZ delay와 별도 필드로 유지
- training/runtime lead는 오직 `PlantDelays.lead()`로 계산
- config나 문서에 lead를 손으로 복사하지 않음

### [exact condition]

오디오를 열기 전에 exact actual-int16 input으로 다음을 계산한다.

- fit-a support-1024 condition `≤20`
- fit-b support-1024 condition `≤20`
- fit-a+fit-b joint condition `≤20`
- holdout input condition `≤20`
- 선택된 absolute integer delay shift를 적용한 condition `≤20`
- exact Gram과 독립 quadratic/circular crosscheck PASS

condition이 실패하면 amplitude, seed, correlation을 live raw에 맞춰 수정하지 않는다.
signal generation을 새 SHA의 새 generation으로 다시 시작한다. 2,048/4,096/8,192
support를 계산하지 않았다면 `NOT_AUDITED_NO_CLAIM`이다.

### [amplitude, polarity, harmonic/IMD gate]

두 speaker를 동시에 구동하므로 선형 superposition을 별도로 반증한다.

- 최소 두 개의 사전 봉인 amplitude level
- P/S 네 polarity 조합 `++`, `+-`, `-+`, `--`
- actual PCM peak/RMS/crest와 channel cross-correlation
- amplitude-normalized complex transfer의 gain/phase invariance
- 2차·3차 harmonic과 P/S intermodulation residual
- amplitude/polarity를 바꿔도 frozen q가 timing hard budget 안에서 동일
- 각 P/S×ERR/REF×8 physical subband를 독립 score
- low-band global energy가 high-band residual을 숨기지 못함

정확한 amplitude ratio와 harmonic/IMD threshold도 immutable synthetic/electrical replay
전에 고정해야 하며, 미정인 동안 live blocker다. clipping/xrun/status는 항상 exact 0이다.

### [정보론적 한계와 판정]

다음 두 설명은 acoustic raw만으로 byte-identical할 수 있다.

```text
A: fixed acoustic H + time-varying ADC/DAC q
B: ideal q + 두 microphone에 공통인 time-varying acoustic delay
```

따라서 경로 A가 모든 수치 검사를 통과해도 판정은 최대 다음과 같다.

```text
status = CONDITIONAL_ACOUSTIC_RESEARCH_PASS
fixed_lti_assumption_required = true
independent_clock_authority_pass = false
canonical_training_eligible = false
```

이는 failed v6를 구제하거나 canonical P/S를 만드는 경로가 아니다.

## 5. B — canonical shared-clock/electrical witness path

### 5.1 1차 후보: 온보드 RT5640/I2S1 출력

2026-08-29 실제 Jetson을 read-only로 다시 감사한 결과, 외부 DAC를 추가하기 전에
검증할 수 있는 온보드 경로가 확인됐다.

- `/sys/bus/i2c/devices/8-001c`는 `rt5640` driver에 실제 bind되어 있다.
- APE `hw:APE,0`은 ADMAIF1 playback, `hw:APE,1`은 현 I2S2 ERR/REF capture로
  열거된다.
- 현재 mixer에서 `I2S1 Mux=ADMAIF1`, `I2S2 Mux=ADMAIF2`이고 두 I2S controller는
  같은 `PLL_A_OUT0` 계열의 1,536,000 Hz clock을 사용한다.
- 현재 DT는 RT5640을 I2S1에 연결하며 codec은 slave 구성이다.
- 감사 시점 모든 APE PCM은 `closed`였다. control 장치만 PulseAudio가 점유했으며 PCM
  충돌은 없었다.

따라서 다음 경로는 **Likely**인 1차 shared-rate 후보다.

```text
PLL_A_OUT0
├─ I2S2 master → ERR/REF digital microphones → hw:APE,1 capture
└─ I2S1 master → onboard RT5640 slave → J511 stereo output → amplifier
```

그러나 공통 PLL parent를 발견한 것만으로 shared-clock 또는 canonical authority를 PASS하지
않는다. 같은 parent는 child divider·실제 BCLK/WS 비율·stream 중 reparent/retune 0·
AHUB/codec SRC 부재·DMA drop/add 0·고정 frame offset을 각각 증명하지 않는다. 아직
확인되지 않은 항목은 다음과 같다.

- RT5640 J511 출력과 현재 TPA3116D2 입력 사이의 실제 물리 배선
- RT5640 headphone/speaker route와 gain을 되돌릴 수 있게 설정하는 정확한 mixer recipe
- `hw:APE,0` playback과 `hw:APE,1` capture의 동시 48 kHz/2-channel 운용
- callback/xrun/drop/add 0과 장시간 sample-slip 0
- 두 I2S의 고정 frame offset 및 제출/수신 frame counter 결속
- 실제 출력 level, channel mapping, polarity, THD/IMD와 P/S

검증 순서는 ALSA 전체 상태 byte snapshot과 restore 검증, PCM 무점유, amplifier/speaker
분리 상태의 **all-zero simultaneous duplex**, timestamp/frame transport receipt 순이다. all-zero
payload가 아닌 sample은 writer와 callback 양쪽에서 즉시 실패시킨다. 이 단계의 audible
time은 0초이며, route 변경 전후 상태와 raw telemetry를 no-replace artifact로 보존해야
한다. 이 무음 검증이 PASS한 뒤에만 J511 배선과 짧은 level/channel/polarity 측정을 별도
실행한다.

all-zero는 drop/add/reorder가 발생해도 sample 값이 계속 0이므로 physical sample identity나
slip을 증명할 수 없다. 따라서 무음 단계의 최대 판정은 고정한다.

```text
status = ZERO_DUPLEX_TRANSPORT_SMOKE_PASS
common_clock_topology_pass = false
hardware_frame_identity_pass = false
shared_clock_authority_pass = false
physical_output_route_pass = false
sample_identity_pass = false
```

무음 smoke는 exact `hw:` endpoint/rate/format/channel/period/buffer 재조회, 모든 callback의
zero SHA, exact 제출·수신 frame, callback 연속성, xrun/status/overflow/underflow/deadline/
fallback/drop/add 0, watchdog, pre/during/post parent/divider/rate snapshot, 정상·예외 종료의
ALSA byte-exact restore까지만 증명한다. nonzero electrical sequence 또는 hardware frame
counter 없이 `slip=0` authority를 발행하지 않는다.

### 5.2 2차 후보: APE I2S2 공통-clock 외부 DAC

온보드 RT5640 경로가 물리적으로 J511에 나오지 않거나 동시 스트림/slip gate를 통과하지
못할 때의 다음 후보는 APE I2S2 ADC와 동일 BCLK/WS를 쓰는 외부 2-channel I2S slave
DAC다.

```text
APE I2S2 BCLK/WS
├─ APE ADC: ERR/REF
└─ 2-channel slave DAC: primary/secondary output
```

하드웨어 요구사항:

- 3.3 V logic, 48 kHz stereo I2S slave DAC
- 기존 pin 12 BCLK, pin 35 WS, pin 38 DIN 유지
- pin 40을 DOUT으로 추가하는 reversible DT/pinmux 계획
- absolute ADC/DAC frame counter와 submitted/captured SHA 결속
- drop/add/slip/xrun/fallback exact 0
- continuous aperiodic signal은 독립 clock 추정이 아니라 slip/stationarity negative-control로
  계속 사용

공통-clock receipt가 PASS하면 acoustic q를 자유롭게 fit해 time-base를 만드는 대신 hardware
frame identity를 사용한다. 그래도 path delay, nonlinear, condition, fit/holdout은 경로 A와
동일하거나 더 엄격하게 검사한다.

AB13X에서 얻은 meter, P/S, absolute delay, lead, latency는 새 DAC에 재사용하지 않는다.
하드웨어 변경 후 모두 새 capture-id와 새 SHA로 다시 측정한다.

### 5.3 AB13X 유지 시 electrical witness

AB13X를 유지하면서 independent authority를 주장하려면 endpoint descriptor나 acoustic
pilot만으로는 부족하다. 최소 다음이 필요하다.

- actual playback voltage를 관측하는 안전한 attenuated/DC-blocked tap
- ERR/REF와 electrical tap을 동시에 보존할 추가 synchronized capture path
- DAC playback frame과 electrical witness frame의 독립 counter/time receipt
- AB13X oscillator/rate 변경과 sample slip을 장시간 직접 관측한 raw

현재 APE 2-input을 ERR/REF에 모두 사용하면서 electrical tap을 끼워 넣어 한 microphone을
잃는 방식은 simultaneous P/S authority가 아니다. 제3 동기 ADC/recorder가 없으면 이
경로는 conditional 상태에 머문다.

### 5.4 rollback과 electrical safety

pinmux/DT 또는 배선을 바꾸기 전에 반드시 다음 순서를 지킨다.

1. 현재 DTB, `extlinux.conf`, 적용 overlay와 SHA를 보존
2. known-good boot entry와 물리 rollback 절차를 실제로 준비
3. amplifier 입력과 speaker를 물리적으로 분리
4. 무출력 상태에서 GND potential, DC bias, rail, logic level 확인
5. high-impedance divider와 DC-block capacitor 사용; amp/speaker output을 mic/line input에
   직결하지 않음
6. DMM/oscilloscope로 안전 범위를 확인한 뒤 electrical-only test
7. electrical PASS 뒤에만 새 meter와 짧은 acoustic capture를 별도 승인
8. 실패하면 즉시 무음·분리하고 같은 연결 상태에서 반복하지 않음

시스템 변경은 rollback artifact가 없으면 시작하지 않는다. 전원모드, clocks, package,
ALSA 전역 설정을 clock 문제의 우회책으로 임의 변경하지 않는다.

## 6. authority leakage 방지

다음 승격은 schema와 loader에서 모두 거부해야 한다.

- v6 diagnostic mode를 v7 q truth로 사용
- v6 failed raw에 새 estimator를 사후 fit해 canonical P/S 발행
- conditional acoustic PASS를 independent clock PASS로 변경
- timing PASS를 2/4/8 kHz attenuation PASS로 변경
- P/S identification bandwidth를 실제 ANC quiet-zone 성능으로 변경
- high-band plant residual로 q/knot/phase를 repair
- operator holdout을 support/threshold/model complexity 선택에 사용
- shared clock만 확인하고 nonlinear/P/S/holdout 없이 training admission
- legacy strict P/S, checkpoint, ONNX를 새 hardware authority에 자동 연결

발행 envelope에는 최소 다음 boolean을 서로 독립적으로 둔다.

```text
capture_transport_pass
actual_int16_lineage_pass
time_map_numerical_pass
fixed_lti_stationarity_pass
clock_authority_source
conditional_acoustic_time_map_pass
common_clock_topology_pass
hardware_frame_identity_pass
electrical_witness_pass
independent_clock_authority_pass
joint_condition_pass
absolute_delay_pass
nonlinear_gate_pass
eight_band_plant_pass
operator_holdout_pass
canonical_training_eligible
attenuation_assessed
```

상위 boolean 하나가 하위 증거를 암묵적으로 대신하지 않는다.

## 7. immutable synthetic replay 최소 목록

v7 live builder/estimator를 열기 전에 최소 다음 raw fixture를 no-replace로 봉인한다.

### Positive

1. 0 ppm fixed affine + known four-path FIR/delay
2. 기존 synthetic affine `±413.931 ppm`
3. smooth monotone drift
4. 반복되는 세 rate mode와 block 내부 여러 전환
5. transition이 역할 경계와 무관한 case
6. exact two-input simultaneous P/S + known fractional delays
7. amplitude/polarity가 선형이고 harmonic/IMD가 gate 아래인 case

### Negative

1. one-sample insert, drop, duplicated callback, silent period loss
2. path별/microphone별 다른 q
3. segment별 FIR/gain/phase/delay drift
4. common time-varying acoustic-delay byte-identical counterexample
5. periodic witness substitution과 v6 PCM splice
6. correlated P/S code로 condition `>20`
7. holdout에서만 발생하는 새 rate transition
8. holdout high-band plant mutation
9. amplitude-dependent delay, clipping, 2차/3차 harmonic, IMD cross-term
10. linear/cubic disagreement `>0.006 sample`
11. hard timing residual `>0.06755189029558946 sample`
12. holdout open 뒤 parameterization/support/threshold refit
13. raw/plan/code/repository SHA splice와 기존 target replace
14. independent hardware receipt 없이 authority boolean 변조

common time-varying acoustic-delay 반례는 estimator가 수치상 잘 맞아도 경로 A에서는
`conditional`만 발행해야 한다. 이 테스트가 authority scope를 강제한다.

## 8. 현재 단계별 상태

| 단계 | 상태 | 이유 |
|---|---|---|
| v6 raw/capture 보존 | DONE | immutable raw와 failure 존재 |
| v6 short-time forensic | DIAGNOSTIC ONLY | 여러 mode/내부 전환, authority 없음 |
| v7 time-map parameterization | BLOCKED | immutable synthetic replay와 model-selection 계약 미완성 |
| v7 simultaneous aperiodic signal | NOT SEALED | duration/amplitude/condition receipt 없음 |
| v7 exact condition `≤20` | NOT RUN | exact v7 actual-int16 PCM 미생성 |
| v7 nonlinear gate | NOT SEALED | amplitude/polarity/IMD threshold 미봉인 |
| v7 live acoustic capture | BLOCKED | 위 signal-only gate 미완성 |
| 온보드 RT5640 shared-clock 후보 | READ-ONLY DISCOVERED | codec/공통 PLL은 확인, J511 배선·동시 duplex·slip은 미확인 |
| RT5640 all-zero duplex smoke | NOT STARTED | snapshot/restore 검증기와 immutable receipt 미구현 |
| shared-clock/electrical authority | NOT STARTED | snapshot/rollback/all-zero duplex/frame receipt 없음 |
| canonical P/S | BLOCKED | independent clock와 plant gate 없음 |
| canonical pretrain/fine-tune | BLOCKED | canonical P/S 및 Elice 상태 미확인 |

2026-08-29 read-only Elice 감사에서 마지막 endpoint
`central-01.tcp.tunnel.elice.io:56230`은 TCP 터널 이후 SSH key-exchange 단계에서
`Connection closed by remote host`로 종료됐다. 인증 단계와 원격 명령 실행에 도달하지
못했으므로 GPU, 학습 프로세스, remote HEAD와 저장공간은 현재 확인되지 않았다.
따라서 “Elice 학습이 실행 중/완료”라고 주장할 수 없으며 canonical fine-tune도 현 상태에서
차단이다. 새 endpoint가 생겨도 이 문제는 v7 measurement authority gate를 낮추는 이유가
되지 않는다.

## 9. live를 열기 위한 최소 다음 행동

1. v6 forensic artifact를 diagnostic-only schema로 no-replace 봉인하고 기존 raw/failure를
   변경하지 않는다.
2. 후보 time-map classes와 model-selection rule을 문서·코드·test에서 먼저 고정한다.
3. immutable synthetic development/terminal raw suite를 생성하고 exact replay한다.
4. simultaneous two-output aperiodic signal을 만들고 actual-int16 SHA, meter power,
   support-1024 shifted condition `≤20`을 봉인한다.
5. fit-a/fit-b/validation/nonlinear/holdout/terminal 접근 mask와 no-refit receipt를 고정한다.
6. 전체 pytest 0 FAIL, clean exact commit, dry-run/no-device-open을 확인한다.
7. 별도 브랜치에서 ALSA snapshot/restore와 all-zero `hw:APE,0` playback +
   `hw:APE,1` capture 검증기를 먼저 구현하고, 소리 없이 동시 stream·frame·xrun
   **transport-smoke** receipt만 발행한다. zero 값으로 sample slip을 PASS하지 않는다.
8. 무음 PASS 뒤 J511→amplifier 실제 배선을 확인한다. 그때 처음 예상 audible time,
   작동 speaker, volume, raw 경로, 사용자 확인 사항을 다시 보고한다.
9. RT5640이 실패하면 실패 artifact를 보존하고 외부 I2S2 slave DAC 또는 독립 electrical
   witness 경로로 간다. AB13X acoustic-only 결과를 canonical로 승격하지 않는다.
10. 승인 후에도 conditional acoustic capture와 canonical hardware capture를 서로 다른
    generation/capture-id로 실행한다.

이 순서를 마치기 전에는 스피커 출력, v7 P/S 발행, canonical pretrain/fine-tune,
ONNX/TensorRT 배포로 진행하지 않는다.
