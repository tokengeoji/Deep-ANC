# v5 duplex 캡처 telemetry 계약

`deep_anc.audio_duplex_v5`는 실제 장치 정책과 raw 발행을 포함하지 않는 작은
PortAudio callback primitive다. `sounddevice`를 import하지 않고 상위 계층이 검증한
backend와 장치 번호를 주입한다.

고정 실행 계약은 48 kHz, 256 frame, low latency, 입력 2채널 exact `<i4`, 출력
2채널 exact `<i2`다. 검증과 output 제출까지 성공 commit된 callback은 sequence,
software cursor 시작점, frame 수,
PortAudio ADC/DAC/current time, status bitmask를 같은 길이 배열로 남긴다. status가
없는 callback도 bitmask 0으로 보존한다. bit는 input/output underflow/overflow,
priming output, unexpected, present를 구분한다.

`callback_start_frames`는 ALSA hardware frame counter가 아니라 소프트웨어 frame
accounting이다. 연속이라고 해서 ADC/DAC sample slip 부재가 증명되지 않는다.
PortAudio xrun status 증거와 hardware sample-slip 권위는 다르다. schema는
`portaudio_xrun_status_witness=true`, `hardware_sample_slip_authority=false`를 고정한다.
따라서 이 primitive만으로 live
plant 또는 training authority를 열 수 없다.

priming callback은 금지한다. 정상 완료는 `stop(ignore_errors=False)` 후 close하고,
실패만 `abort(ignore_errors=False)` 후 close한다. 구조와 timestamp를 모두 검증한 뒤
output과 증거 배열을 갱신하므로 실패 callback은 부분 반영되지 않는다. 실패 시에는
가능한 즉시 sink 전체를 exact zero-fill한다. sink dtype/shape가 잘못됐거나 쓰기가
실패해 무음을 확인할 수 없으면
`output_silence_not_confirmed_on_callback_failure`로 INVALID 처리한다. xrun/unexpected
status는 가능한 전체 계획을 보존한 뒤 INVALID로 반환한다.

callback abort, watchdog, stream stop/abort/close 실패에서도 `DuplexCaptureFailure`가
그 시점까지 성공 commit된 실제 제출 prefix, 캡처 raw, valid mask와 telemetry를
반환한다. 실패 callback의 hardware 출력 안전은 zero-fill 확인 범위에서만 주장한다. 계획 tail을
실제 제출 증거로 사칭하지 않는다. watchdog은 host wait 경계일 뿐 hardware deadline이나
sample-slip 증거가 아니다. 상위 live adapter는 이를 immutable
INVALID raw로 먼저 발행해야 한다. 장치 점유 검사, 저장소 audio lock, 사용자 확인,
level evidence, plan SHA 검증, no-replace/fsync publication은 이 primitive 밖에서 기존
공식 preflight와 v5 canonical writer를 사용해야 한다.
