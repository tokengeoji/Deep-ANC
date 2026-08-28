# v5 live raw publisher/loader primitive

## 범위와 권위

`deep_anc.dsp.fullband_live_raw_v5`는 오디오 장치를 열지 않는다. 상위 live adapter가
검증한 plan, capture-only authority, fresh meter, level evidence, hardware 및 input-only
preflight 결속과 `audio_duplex_v5` 결과를 받아 단일 immutable NPZ로 보존한다.

이 primitive의 `CAPTURE_PASS`는 “제출·캡처·PortAudio telemetry가 이 컨테이너 계약을
통과했다”는 뜻뿐이다. primitive가 받은 post binding은 외부 파일을 다시 읽은 권위가 아니라
caller self-attestation이다. 따라서 deterministic live adapter의 별도 post receipt file/SHA가
아직 결속되지 않은 현재 단계에서는 다음 값이 성공 raw에서도 고정된다.

```text
canonical_training_eligible = false
hardware_sample_slip_authority = false
analysis_admission_eligible = false
external_post_capture_receipt_bound = false
post_capture_binding_scope = primitive_self_attestation_not_external_receipt
```

따라서 이 raw만으로 실제 hardware sample slip 부재, plant authority 또는 canonical
training 자격을 주장할 수 없다.

## exact NPZ schema

schema는 `fullband_causal_live_raw_v5_v1`이다. NPZ에는 canonical UTF-8 JSON metadata와
다음 13개 배열만 존재한다.

```text
planned_submitted_pcm       <i2 [frames,2]
actual_submitted_pcm        <i2 [frames,2]
captured_pcm                <i4 [frames,2]
submitted_valid_mask       bool [frames]
capture_valid_mask         bool [frames]
preflight_raw_int32         <i4 [preflight_frames,2]
callback_sequence           <i8 [callbacks]
callback_start_frames       <i8 [callbacks]
callback_frame_counts       <i8 [callbacks]
input_buffer_adc_time       <f8 [callbacks]
output_buffer_dac_time      <f8 [callbacks]
callback_current_time       <f8 [callbacks]
callback_status_bitmask     <u4 [callbacks]
```

metadata는 모든 배열의 dtype·shape·bytes SHA-256을 봉인한다. NPZ member 추가/누락,
metadata key 추가/누락, compressed repack도 loader가 거부한다.

metadata의 exact `session` mapping은 고정 schema, 32자리 lowercase hex capture ID,
caller의 시작/완료 UTC, publisher가 serialization 전에 내부에서 읽은
`publisher_prepared_at_utc`와 audio-lock identity SHA를 저장한다. single immutable NPZ 내부에
durable publication 이후 시각을 다시 써 넣을 수 없으므로 이 값을 `published_at`이라고 부르거나
60초 durable-freshness 증거로 사용하지 않는다. 구조 오류는 게시 전에 거부하고, UTC 역순·0초
같이 보존 가능한 chronology 실패는 `INVALID`로 남긴다. meter 완료와 session 시작 사이는
0..600초여야 하지만 caller UTC는 실제 duration 권위가 아니다.

실제 duration 권위는 audio telemetry의 monotonic start/completed/elapsed와 그 capture에 사용한
watchdog grace다. publisher는 completed-start를 elapsed와 재계산하고, 계획 frame 수/48 kHz에서
1 ms 아래부터 nominal+실제 grace+1 ms까지인지 검사한다. loader도 같은 계산을 반복한다.

## 성공과 부분 실패

publisher의 `capture` 인자는 다음 둘 중 하나다.

1. `capture_duplex_v5()`의 정상 반환 `(captured_pcm, telemetry)`
2. 같은 함수가 던진 `DuplexCaptureFailure`

두 valid mask는 exact bool contiguous prefix이고 서로 같은 prefix를 가리켜야 한다.
prefix는 256-frame callback block 배수다. valid prefix의 actual submitted는 planned와
exact 일치해야 하며 invalid tail은 exact zero여야 한다. captured invalid tail도 exact
zero다. 이 규칙으로 계획 tail을 실제 제출 evidence로 사칭할 수 없다.

다음 중 하나라도 있으면 raw는 버리지 않고 `INVALID`로 발행한다.

- `DuplexCaptureFailure` 또는 incomplete valid prefix
- xrun을 포함한 nonzero callback status
- callback/stop/abort/close error
- telemetry incomplete 또는 normal stop 미완료
- output stop/silence 확인 불가
- `canonical_invalid_reasons`가 하나 이상
- post-capture authority/device binding 실패
- preflight PASS가 아님
- meter/session/publication chronology 또는 계획 duration/watchdog 범위 실패

`INVALID`와 partial raw는 strict loader로 진단용 재독해할 수 있지만 analysis admission은
거부된다. 현재는 `CAPTURE_PASS`도 외부 post receipt가 미결속이므로 analysis admission이 아니다.

## telemetry 결속

schema 문자열을 이 모듈에 복제하지 않는다. publish와 load 시점에
`deep_anc.audio_duplex_v5.DUPLEX_TELEMETRY_SCHEMA`와 공개 block/status 상수를 읽는다.
telemetry v3의 exact scalar key 집합과 7개 배열을 검사하고, callback sequence/start/count,
status/xrun count 및 committed frame 수를 배열과 mask에서 다시 계산한다.
성공과 실패 모두에 있는 exact nonnegative-int resolved input/output device는 hardware binding과
교차 검증한다. monotonic start/completed/elapsed/watchdog grace도 성공·실패 모두에 보존한다.

callback timestamp는 finite strict-monotonic 보조 증거다. callback frame cursor와 timestamp는
hardware sample-slip authority로 승격되지 않는다.

## 외부 evidence binding

publisher와 loader는 다음 exact binding 여섯 개를 요구한다.

- `signal_plan`: path, 전체 파일 SHA, payload SHA, PCM SHA, sealed raw path
- `live_capture_authority`: path/file/payload SHA와 plan/PCM/hardware/raw 교차 결속
- `meter`: raw/receipt/identity/followup SHA와 authority/evidence/hardware SHA
- `level_evidence`: path/file/identity SHA
- `hardware`: 고정 hardware binding schema, path/file/identity/physical-fingerprint SHA와
  resolved input/output device
- `preflight`: raw/report/identity SHA와 PASS 상태

plan envelope v5, capture authority v1, meter raw v1, evidence bootstrap-pair v2, hardware
전용 schema와 현행 preflight schema를 exact pin한다. legacy/unrelated schema는 거부한다.
authority→plan/PCM/hardware, meter→authority/evidence/hardware를 다시 비교한다. plan,
authority, hardware의 canonical path와 SHA는 committed
`fullband_live_authority_v5` 상수를 trust root로 사용하므로 caller가 함께 조작한
`expected_bindings`만으로 admission을 만들 수 없다. SHA는 exact
lowercase 문자열만 허용하며 uppercase나 숫자의 문자열 변환을 수용하지 않는다. preflight raw
SHA는 실제 NPZ 배열에서 재계산한다. 다섯 operator confirmation도 모두 exact true여야 한다.

`preflight_report`도 metadata에 inline한다. schema는
`fullband_input_preflight_report_v1`이며 PASS, preflight identity, resolved input device,
48 kHz, raw frame 수와 ERR/REF 두 channel summary를 exact key/type으로 보존한다. report의
identity/device/frame/PASS를 binding 및 raw와 대조한다. identity는 caller 임의 문자열이
아니라 preflight raw SHA, hardware identity SHA, resolved input device, 48 kHz와 frame 수의
canonical payload SHA다. 각 채널의 `rms_dbfs`, `peak`, `clip_ratio`, `stuck`, `valid`,
`unique_codes`, `raw_min`, `raw_max` 전부를
`deep_anc.audio_io.analyze_int32_input_probe(preflight_raw)`로 다시 계산해 exact 대조한다.
따라서 all-zero, 저레벨, clipped, stuck raw에 조작된 PASS report를 붙일 수 없다. canonical
compact-JSON SHA도 `binding.preflight.report_sha256`와 정확히 같아야 한다.

`post_capture_binding`은 단순 bool이 아니다. capture 뒤 다시 읽은 plan file/payload/PCM,
authority file/payload, meter raw/receipt, evidence file, hardware file/identity/fingerprint,
audio-lock identity, resolved devices 및 `raw_target_fresh=true`를 exact receipt로 보존하고
pre-binding/session과 교차 검증한다. 불일치는 `INVALID`다.

이 모듈은 경로가 가리키는 외부 파일 자체를 읽지 않는다. 상위 adapter가 fresh evidence를
검증한 뒤 exact binding을 넘기고, offline loader는 같은 authority에서 얻은
`expected_bindings`와 raw receipt의 `expected_raw_file_sha256`을 반드시 제공해야 한다.
raw 모듈의 committed authority 상수 대조는 외부 파일의 freshness 재검증을 대신하지 않는다.

## publication과 offline handoff

target은 authority가 봉인한 repository-relative raw path와 같아야 한다. 상대 target 인자는
process cwd가 아니라 명시한 `repository_root`를 기준으로 해석한다. root부터 각 parent를
`O_DIRECTORY|O_NOFOLLOW` dirfd chain으로 열고, mkdir/open도 이전 dirfd 기준으로만 수행한다.
각 component의 device/inode와 현재 lexical chain을 publish 전후에 비교한다. staging은 최종
parent dirfd 기준 `O_CREAT|O_EXCL|O_NOFOLLOW` private file이고, final hard-link도 같은 dirfd 안의
atomic no-replace 연산이다. file과 parent를 `fsync`한다. parent rename 뒤 외부 symlink를 끼우는
swap은 열린 inode 밖으로 쓰지 못하며 사후 lexical-chain 검사에서 거부된다.

final link가 성공한 뒤 staging hard-link unlink만 실패한 경우 capture 자체는 이미 immutable
이므로 거짓 실패로 반환하지 않는다. parent를 fsync하고 `publication_warnings` receipt에 남긴다.
loader 역시 같은 no-follow dirfd chain에서 regular target을 열고 fd에서만 읽으며, 읽기 전후
parent 및 target device/inode를 재검증한다.

함수 진입 시 planned/preflight/capture/telemetry 배열을 소유 C-contiguous copy로 바꿔
validation→serialization 사이 caller mutation 범위를 줄인다. metadata의 writer contract는
Python implementation/version, NumPy version, uncompressed `numpy.savez`, compact canonical JSON
규약과 `same_python_numpy_runtime_only` byte 재현 범위를 정직하게 봉인한다. 다른 runtime에서
동일 semantic NPZ bytes가 나온다고 주장하지 않는다.

offline 순서는 다음과 같다.

```text
publish_live_raw_v5
→ raw file SHA receipt 보존
→ load_live_raw_v5 (진단 포함)
→ deterministic external post receipt 발행·file/SHA 재검증(아직 이 primitive 밖)
→ 향후 external receipt-aware admission
→ 별도 v5 offline core
```

오디오 callback과 raw publication 사이에서 분석을 실행하지 않는다. 상위 live adapter는
stream close와 즉시 스피커 분리 안내를 먼저 끝내고, 그 다음 이 publisher를 호출해야 한다.
