# RT5640 disarmed S32 transport v10.3

## [가설]

`stream.start()`가 반환된 뒤의 `on_stream_started` 확인만으로 실제 negotiated
hw_params를 검증한 다음 nonzero P/S PCM을 안전하게 제출할 수 있다.

## [근거]

PortAudio callback은 `stream.start()` 안에서 먼저 실행될 수 있다. 따라서 old planned
transport처럼 start 뒤 callback만 두면 hw_params/route/J511/occupancy 확인 전에 planned
nonzero block이 제출될 가능성이 있다.

## [확인 방법]

`capture_disarmed_planned_s32_duplex()`는 다음 순서만 허용한다.

1. 모든 callback의 첫 side effect는 output 전체 zero-fill이다.
2. `Stream.start()` 안 callback도 unarmed이면 zero만 내고 capture cursor/mask를 움직이지
   않는다.
3. start return 뒤 필수 `post_start_pre_arm_check()`가 actual hw_params/route/J511/PCM
   occupancy를 확인한다.
4. check가 성공하고 callback status가 clean일 때만 block boundary에서 arm한다.
5. arm 이후에도 status/dtype/frame/timestamp/assignment 한 항목이라도 실패하면 re-zero 후
   abort하고 partial raw mask를 보존한다.
6. close 뒤에만 `on_output_closed` notice를 부른다.

## [결과]

- implementation: `src/deep_anc/audio_duplex_s32_disarmed_v10_3.py`
- tests: `tests/test_audio_duplex_s32_disarmed_v10_3.py`
- fake backend는 실제 `start()` 내부 pre-arm callback이 zero인 것을 확인하고, 그 뒤
  planned S32 block이 byte-exact로 제출되는 경우만 PASS한다.
- pre-arm xrun/status, hw_params check failure, planned assignment mismatch, close failure는
  모두 nonzero admission/hash authority 없이 abort한다.

## [판정]

**Confirmed — backend-independent application-buffer safety primitive.** 이 code는
sounddevice/ALSA를 import하지 않으며 실제 PCM을 열지 않았다. 따라서 actual negotiated
S32, J511 cable, DAC voltage, electrical sample identity, P/S, ANC attenuation은 아직
**Inconclusive**다.

## [다음 행동]

다음 adapter는 이 primitive의 mandatory `post_start_pre_arm_check`에 다음 read-only
evidence를 실제 stream open 상태에서 결속해야 한다.

- input/output `hw_params`: S32_LE, 48 kHz, 2 ch, period 256
- APE mux route, J511 expected `HP|HS`, PCM/fuser owner, Pulse APE profile
- planned S32 SHA와 callback raw/mask
- pre/post ALSA snapshot

그 뒤에도 external synchronous electrical witness가 없으면 arm 결과는 meter transport
evidence일 뿐 canonical P/S authority가 될 수 없다. 실제 output capture가 끝난 직후
adapter는 반드시 `출력 종료 — 지금 스피커 분리`를 먼저 출력해야 한다.
