# Committed v5 live-delay offline core 계약

## 권위 경계

`deep_anc.dsp.fullband_live_delay_core`는 오디오 장치를 열지 않는 배열 전용 분석기다.
입력은 committed v5 plan, 그 plan의 actual submitted `int16` PCM, 같은 frame 수의
actual captured `<i4` ERR/REF PCM, `audio_duplex_v5` telemetry 네 개뿐이다.

별도 marker/window, callback-q, `slip_samples`는 받지 않는다. plan SHA는
`32a79b…127`, PCM SHA는 `c18416…aff`로 고정된다. payload/PCM SHA, committed builder
exact 결과, layout 순서·중복·연속성, prefix/central/suffix 길이를 검증한 뒤 central
window 여섯 개를 내부에서만 유도한다.

## Duplex telemetry의 제한된 역할

`deep_anc.audio_duplex_v5.DUPLEX_TELEMETRY_SCHEMA`가 지정하는 actual v3 schema를 exact
전체 key 집합으로 읽는다. callback sequence/start/count가 전체 capture의 exact
256-frame accounting인지, 세 timestamp 배열이 finite strict-monotonic인지,
status/xrun/error가 없는지 검사한다. `portaudio_xrun_status_witness=true`,
`hardware_sample_slip_authority=false`, host-wait watchdog 제한 문구, 정상
stop/close 결과까지 exact하게 확인한다.

frame 수만 받지 않는다. 성공 반환에 들어 있는 `actual_submitted_pcm`이 분석에 제출된
expected `<i2 [frames,2]` PCM과 dtype/shape/value 모두 일치하고, capture와 submitted
valid mask가 exact bool `[frames]` all-true여야 한다. unknown/removed key, v1 schema,
false mask, PCM 불일치는 모두 fail-closed다.

v3의 `resolved_input_device`와 `resolved_output_device`도 exact nonnegative built-in
int로 검증하고 auxiliary receipt에 보존한다. 이 숫자는 해당 PortAudio 실행에서 사용한
index 증거일 뿐 hardware identity binding 권위가 아니다. 카드/PCM identity와의 결속,
동시점 preflight 및 immutable raw publication은 상위 live raw adapter가 수행해야 한다.

그러나 frame cursor는 software accounting이므로 다음을 유지한다.

```text
hardware_slip_authority = false
timestamps_used_to_estimate_clock_q = false
slip_samples_field_expected_or_fabricated = false
```

telemetry timestamp의 비율을 clock q로 승격하지 않는다.

## Clock q

q는 committed plan의 actual pilot와 actual submitted denominator만 사용한다.
`fullband_causal_v5.estimate_common_clock_from_waveforms_v5`를 cubic과 linear 보간으로
각각 `pilot_tail_only_pre_operator_holdout` policy로 실행한다. q fit은 pilot-only lead와
fit_a/fit_b만 사용하고 validation은 pilot-only tail 하나만 사용한다. operator holdout
waveform은 clock에서 열지 않는다. 두 결과가 validation을 통과하고 두 q의 전체 capture
endpoint 차이가 0.0675519 sample 이하여야 한다. high-band 결과로 phase를 수선하거나
별도 marker를 사용하는 경로는 없다. 기존 public estimator의 기본 legacy policy는 API
호환을 위해 유지하지만 live-delay core에서는 허용하지 않는다.

## 절대·분수 지연 복구 순서

1. fit_a와 fit_b의 P/S central cyclic row를 각각 가져온다.
2. shift 없이 `max delay 4800 + compact support 1024 = 5824` tap two-input full
   causal FIR을 ERR/REF별로 joint fit한다.
3. dominant peak를 cubic local interpolation해 bulk integer와 fractional residual을
   얻는다.
4. fit_a/b peak 차이가 0.15 sample 이내인지 검사한다. holdout은 접근하지 않는다.
5. peak보다 256 sample 앞을 compact 시작으로 정한다.

```text
zeros_before = bulk_integer - 256
compact support = 1024
lead = PlantDelays(P zeros, S zeros, handoff=256).lead()
```

`PlantDelays`는 ERR peak만 사용한다. REF는 stationarity 진단 전용이다. compact 시작 전
energy 및 1024 tap 뒤 tail energy의 `1e-4` 비교는 noisy long FIR에서 편향되는
noise-sensitive 진단값일 뿐 admission gate가 아니다. representability 권위는 아래의
full→compact prediction roundtrip이다.

full 5824와 compact 1024 solve 모두 LSMR의 solution x, istop, itn, normr, normar, norma,
conda, normx가 finite여야 한다. istop은 1 또는 2, itn은 predeclared maxiter 미만이어야
하며 독립 재계산한 전체 relative residual과 normal-equation relative residual도 각각
고정 상한을 통과해야 한다. LSMR condition estimate를 exact condition이라고 부르지
않는다.

## Compact 재식별과 fractional 적용 횟수

full FIR을 단순 절단하지 않는다. fit-role에서 고정한 P/S의 서로 다른 integer zero를
입력 operator에 먼저 적용하고 1024-tap FIR을 다시 joint fit한다. fractional residual은
compact FIR shape 안에 한 번만 남는다. 별도 fractional phase/runtime stage 수는 0이다.
full-causal prediction과 compact prediction 및 양쪽 fractional peak를 roundtrip한다.
prediction 차이는 같은 fit role에서 자체 비교하지 않고 반대 fit role의
observed response에서 검사한다. 대역별 불확실성은
`max(pilot exact-null noise power, min(full observed residual, compact observed residual))`
로 사전 선언하고, full↔compact 차이 power에서 이 불확실성을 뺀 초과분만
관측 signal power 대비 2%를 통과해야 한다. 이는 임계값 완화가 아니라
20 dB 관측 noise가 만드는 예측 차이를 명시적으로 분리하는 cross-role gate다.
early 또는 long-tail 성분이 실제 shifted 1024 representation으로 옮겨지지 않으면 이
roundtrip이나 solver/대역 gate에서 실패한다.

full 5824-tap solve는 LSMR condition estimate만 기록하며 exact operator condition으로
부르지 않는다. compact admission에는 유도된 실제 P/S zeros를 각 입력에 적용한 동일
shifted 1024 operator의 periodic Gram 끝 고유값을 직접 계산한다. receipt는 zeros,
operator 정의와 SHA, plan/PCM SHA를 포함한다. unshifted condition receipt를 shifted
증거로 재사용하지 않는다.
public shifted audit는 canonical builder의 plan/layout/PCM exact equality를 자체 검증하고,
입구에서 owned copy를 만든 후 entry/exit SHA로 TOCTOU를 검사한다. 보고하는
condition은 acoustic transfer condition이 아니라 shifted finite-support periodic normal
matrix `G=A^T A`의 `κ₂(G)=λmax/λmin=κ₂(A)²` Gram condition이다. 즉
보고값을 `A`의 singular-value condition으로 해석하지 않는다. 서로 다른 4개 probe의 `A^T A`↔Gram
quadratic form을 모두 교차 검사한다.

## Fit, cross, terminal holdout

fit_a와 fit_b compact candidate는 holdout을 열기 전에 fit/cross 각 64행을 평가한다.
그 뒤 full 후보와 compact 후보를 각각 고정 가중치 `0.5/0.5`로 평균한다. 이 공식은
holdout보다 먼저 SHA로 봉인하며 selection-dependent weighting은 금지한다. 평균 full과
평균 compact 자체를 fit_a와 fit_b 입력에서 각각 roundtrip한다.

반환될 바로 그 평균 compact FIR을 다음 순서로 96행 평가한다.

```text
fit_a 32행 → fit_b 32행 → predeclared terminal holdout 32행
× primary / secondary
× ERR / REF
× physical identification 8 bands
```

각 행은 target/noise bin 수, noise 대비 20 dB 이상 target-bin density,
response-to-noise SNR, noise-conditioned relative residual, complex vector agreement를
별도로 기록한다. complex vector agreement는 coherence가 아니다. 독립 coherence를
계산하지 않았으므로 `independent_coherence_claimed=false`다.

threshold, support, peak, 후보와 평균식은 fit/cross로 먼저 고정한다. holdout waveform은
평균 FIR의 fit_a/fit_b score가 모두 통과한 뒤 처음 열며 terminal admission에만 쓴다.
clock/fit/tail 단계는 필요한 local ADC support만 owned-copy하며 holdout support를 읽지
않는다. selected q로 holdout DAC 구간을 ADC interpolation support로 수치 변환하고,
모든 preterminal owned interval과 pairwise intersection이 0임을 terminal open 전에 강제한다.
captured 전체 SHA도 terminal score 완료 후 처음 계산한다.
operator receipt는 최종 arrays별 SHA map, canonical payload SHA, captured raw SHA,
plan/submitted SHA와 telemetry, clock, timing, final roundtrip, representation/score threshold
contract SHA를 exact 결속한다. 이 SHA 결속도 immutable raw publisher나 hardware-slip
권위를 새로 만들지는 않는다.
validator는 pointer 문자열만 비교하지 않고 telemetry, clock, timing, 평균식,
최종 score, 모든 roundtrip, shifted condition, 두 threshold contract의 payload SHA를
내용에서 재계산한다. embedded score threshold exact equality와 반환 operator array
SHA도 다시 계산하여 stale receipt/component/array splice를 거부한다.

## 실패가 강제되는 경우

- plan/PCM SHA 또는 layout 순서·중복·경계 불일치
- callback 256-frame 위반, timestamp NaN/역행, status/xrun/error 존재
- clock에서 operator holdout 선소비, cubic/linear q 불일치 또는 tail validation 실패
- LSMR non-finite, istop 위반, maxiter 소진, 전체/normal-equation residual gate 실패
- fit_a/b peak 불안정 또는 peak 범위 오류
- 실제 zeros를 적용한 shifted 1024 exact Gram condition 실패
- fractional/full→compact roundtrip 실패
- 최종 고정 평균 FIR 자체의 fit_a/fit_b 또는 terminal holdout 실패
- 어느 path/mic/band에서든 density/SNR/residual/agreement 실패

## 아직 열리지 않는 것

offline 수학 gate가 통과해도 반환값은 다음과 같다.

```text
raw_publisher_bound = false
live_delay_authority_available = null
canonical_training_eligible = false
hardware_slip_authority_available = false
```

immutable raw publisher와 실제 hardware slip witness가 결속되기 전에는 strict P/S,
canonical training plant, 실제 ANC 성능 근거로 승격할 수 없다.
