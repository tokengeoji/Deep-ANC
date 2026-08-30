# v5 capture-only live authority 계약

## 목적과 권한 경계

`deep_anc.dsp.fullband_live_authority_v5`는 오디오 출력 직전의 파일 결속만 담당한다.
오디오 장치·네트워크에 접근하지 않고 raw 또는 실제 authority asset을 만들지 않는다.

이 primitive가 PASS해도 의미는 다음 하나뿐이다.

> 검토된 authority SHA가 가리키는 committed v5 signal plan과 현재 hardware 설정이 exact
> bytes로 존재하고, 그 plan이 미리 봉인한 raw target이 아직 비어 있다.

반환 계약은 항상 다음 상태를 유지한다.

```text
schema = fullband_causal_v5_live_capture_authority_v1
capture_only = true
plan_live_capture_enabled = false
canonical_training_eligible = false
hardware_sample_slip_authority = false
live_delay_authority = null
```

signal-only plan의 `live_capture_enabled=false`, `live_authority=null`,
`canonical_training_eligible=false`를 수정하거나 덮어쓰지 않는다. capture-only authority는
실제 P/S, clock/slip, plant, 학습 또는 ANC 성능 authority가 아니다.

## 고정 경로와 SHA

현 generation은 다음 repository-relative 경로만 받는다.

```text
plan envelope: assets/contracts/fullband_causal_v5_signal_plan.json
live authority: assets/contracts/fullband_causal_v5_live_capture_authority.json
hardware:      configs/hardware_jetson.yaml
sealed raw:    results/fullband_causal_v5/raw_capture.npz
```

tracked authority bytes는 다음 다섯 SHA에 고정된다.

```text
plan envelope file SHA-256:
  bf25f041c5c5770c01aa326e47749b4eaab9a012f9f7c69dec5cd81ae3507287
hardware file SHA-256:
  45232a45e51fd76c7b88db338b9cf4f3840a88299b4d452e259064c0ee559351
support-1024 condition receipt payload SHA-256:
  300078f714fd19e6b15eaee1bc212b196960301a1c745c256d3d46ac9295b61f
live capture authority file SHA-256:
  f090e59533fac6467f3c7c3328ebc9983deef06d8f8bc6fbf9158e4de66f8138
live capture authority payload/internal SHA-256:
  ed506255f53056724abd2fd79822e91b8879455b9cdf0c06ab942c079ae9441f
```

plan envelope는 기존 signal-only writer와 동일한 UTF-8, sorted-key, two-space indent,
trailing newline JSON bytes여야 한다. duplicate key, 의미만 같은 재포맷, extra key,
symlink parent/target을 거부한다. 호출자는 검토한 plan envelope file SHA를 별도로 전달해야
한다. condition receipt payload SHA도 별도의 external expected 인자로 전달해야 하며 코드의
pinned 값과 일치해야 한다. loader는 이 두 SHA를 현재 bytes에서 재계산한 뒤 다음도 다시
확인한다.

- envelope exact key와 schema
- `build_plan_v5()`의 exact plan 결과
- plan payload SHA `32a79b…127`
- actual submitted PCM SHA `c18416…aff`
- pinned support-1024 exact condition receipt의 exact key/field, payload SHA와 `passed=true`
- committed sealed raw 상대경로
- plan 자체의 signal-only 권한 경계

hardware도 고정 상대경로와 외부 expected file SHA를 요구한다. 따라서 plan file,
condition receipt payload, hardware file의 세 external expected SHA와 세 코드 상수가 모두
일치해야 authority를 만들 수 있다. authority 내부 SHA는
`authority_sha256` 필드를 제외한 canonical payload SHA다. 검증 시 내부 값을 그대로
신뢰하지 않고, 검토된 `expected_authority_sha256`을 별도 인자로 반드시 전달한다.

live loader는 `exact_condition_audit_v5()`를 다시 실행하지 않는다. 그 2,048×2,048
eigensolve는 Jetson에서 수 분이 걸릴 수 있고 meter/live 직전 연결 시간을 불필요하게
늘린다. heavy audit은 tracked envelope를 처음 발행하고 전체 v5 회귀를 수행할 때 이미
실행한다. live preflight는 빠른 deterministic `build_plan_v5()` exact plan과 위에 고정된
condition receipt의 canonical payload SHA, exact fields, `passed=true`만 재검증한다. 따라서
condition 숫자를 바꾸거나 receipt SHA만 다시 쓰는 변조는 plan envelope file SHA에서 먼저
거부된다.

`load_exact_saved_live_capture_authority_v5()`는 tracked live authority도 같은 방식으로
고정한다. fixed repository path의 parent/target symlink를 먼저 거부하고, duplicate key와
semantic-equal JSON reformat을 거부한다. 그 뒤 authority file SHA와
`authority_sha256`을 제외한 canonical payload SHA를 각각 external expected 및 위 pinned
상수와 비교한다. 마지막으로 in-memory `validate_live_capture_authority_v5()`를 호출하므로
plan/hardware bytes가 바뀌었거나 sealed raw target이 이미 존재하면 saved authority 자체가
정상이더라도 live preflight는 실패한다.

`build_live_capture_authority_v5()`의 결과를 canonical pretty JSON으로 직렬화한 bytes는 위
tracked authority file과 exact 동일해야 한다. authority asset은 runtime에서 다시 쓰거나
자동 갱신하지 않는다.

## raw freshness와 symlink

sealed raw target은 validation 시점에 존재하지 않아야 한다. 정상 file뿐 아니라 broken
symlink도 fresh로 세지 않는다. repository root와 plan/hardware/raw의 모든 기존 parent 및
target symlink를 거부한다. 기존 target을 덮어쓰거나 다른 generation의 v4/old broadband
path로 바꾸는 authority는 자체 SHA를 다시 계산해도 실패한다.

이 검사는 시점 검사이므로 live adapter는 stream open 직전에 같은 authority SHA와 bytes,
raw freshness를 다시 검증해야 한다. 캡처 성공·실패 뒤 실제 제출 prefix, valid mask,
PortAudio status와 partial raw를 no-replace+fsync로 먼저 발행하는 일은 별도 live adapter의
책임이다.

## 명시적으로 하지 않는 일

- speaker 출력 또는 microphone capture
- ALSA 장치 점유·physical fingerprint 확인
- level meter와 사용자 확인 결속
- callback telemetry v2 분석
- hardware sample-slip 부재 주장
- P/S delay, fractional FIR 또는 `PlantDelays.lead()` 계산
- raw/analysis/operator publication
- canonical training admission

따라서 이 primitive 단독 PASS를 근거로 소리를 내거나 학습을 시작하지 않는다. 실제 live
adapter, immutable INVALID/SUCCESS raw publisher, offline delay analyzer가 각자 독립 검토를
통과한 뒤에도 사용자에게 명령·스피커·볼륨·예상 출력 시간·저장 경로를 먼저 보고하고 명시적
승인을 받아야 한다.
