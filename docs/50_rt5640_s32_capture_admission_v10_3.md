# RT5640 S32 capture admission v10.3

`deep_anc.dsp.rt5640_s32_capture_admission_v10_3`는 RT5640 S32 disarmed duplex의
**성공·부분 실패 결과를 분리하는 순수 admission boundary**다. 오디오 장치, ALSA,
sounddevice, result writer를 열지 않는다.

## [가설]

post-start pre-arm check와 S32 callback telemetry가 모두 정상이면, 이 capture는
fullband P/S와 고주파 ANC 학습에 바로 쓸 수 있다.

## [근거]

- `audio_duplex_s32_disarmed_v10_3.py`는 callback의 첫 동작을 zero-fill로 고정하고,
  `Stream.start()` 내부 callback도 `post_start_pre_arm_check` 성공 전에는 planned PCM을
  제출하지 않는다.
- 성공 telemetry에는 planned/actual `<i4` PCM, pre-arm/planned callback 배열, valid mask,
  assignment count, close/fault 상태가 있다.
- 그러나 그 값은 PortAudio application buffer에서 관측된 값이다. 실제 DAC 전압, amp
  출력, electrical tap, shared sample frame, P/S를 직접 관측한 값은 아니다.
- legacy v5 raw writer는 `<i2`/legacy schema에 고정돼 S32 result를 재사용할 수 없다.

## [확인 방법]

정상 receipt는 다음을 모두 재검산한다.

1. sealed fullband S32 plan, Q15→S32 16-bit left shift, planned/actual PCM SHA가 일치한다.
2. `post_start_pre_arm_receipt`가 stream open 뒤 APE PCM1/PCM0의 S32_LE, 48 kHz, 2 ch,
   period 256, route, J511 `HP|HS`, PCM ownership, pre/post snapshot SHA를 결속한다.
3. pre-arm callback은 zero-only이고, planned callback은 정확한 `0..N-1`/`N×256` sequence,
   zero status, complete mask, exact planned PCM을 가진다.
4. xrun/status/error/fault/termination은 모두 0 또는 empty여야 한다.
5. callback timestamps, host watchdog, stream stop/close를 검증하되 이것을 hardware sample
   identity로 해석하지 않는다.

`S32DisarmedDuplexCaptureFailure`는 raw를 버리거나 재측정하지 않고
`S32_CAPTURE_BLOCKED_PARTIAL_OR_INVALID`로 표지한다.

## [결과]

정상 반환의 status는 다음뿐이다.

```text
S32_CAPTURE_TRANSPORT_PASS_ELECTRICAL_WITNESS_UNBOUND
```

동시에 authority는 고정된다.

```text
s32_transport_capture_pass = true
hardware_sample_slip_authority = false
physical_output_authority = false
electrical_witness_bound = false
fullband_plant_identification_pass = false
canonical_training_eligible = false
deployment_eligible = false
```

## [판정]

**Confirmed — application-buffer S32 transport receipt.** 이는 S32 scale, pre-arm zero,
planned actual bytes, stream-open 뒤 hw_params/route/jack/PCM ownership 검사의 경계다.
이는 ANC attenuation이나 고주파 P/S의 증거가 아니다.

## [다음 행동]

1. 실제 RT5640 adapter는 stream open 뒤 이 module이 요구하는 post-start receipt를 생성한다.
2. 실제 output을 열기 전 외부 동기 electrical witness를 준비한다. `ERR`, `REF`,
   `NOISE_TAP`, `CANCEL_TAP`이 하나의 acquisition clock에 있거나, APE↔external recorder의
   BCLK/WS/absolute-frame bridge가 있어야 한다.
3. actual S32 callback→two tap frame identity와 native/canonical raw SHA를 no-replace로
   보존한 뒤에만 raw-first fullband P/S 분석을 연다.

따라서 현재 단계에서는 스피커를 연결해 둘 수 있지만, 이 module만으로 실제 소리를
출력하거나 Elice 학습을 시작하지 않는다.
