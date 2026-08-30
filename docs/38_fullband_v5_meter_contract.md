# Fullband-v5 레벨 미터·입력 preflight 계약

> [!WARNING]
> **HISTORICAL / 직접 실행 금지.** 이 문서는 이전 v5 meter 경계를 보존하는 forensic
> 기록이다. 현재 `scripts/data/set_amp_level.py`는 `--mode`를 받지 않으며, 활성 CLI는
> `--followup-mode {strict,broadband,fullband-v5}`만 허용한다. 실행 가능한 현재 계약과
> 명령은 [docs/36_fullband_v5_meter_followup.md](36_fullband_v5_meter_followup.md) 및
> `.venv/bin/python scripts/data/set_amp_level.py --help`가 단일 출처다. 이 문서의 과거
> 명령을 복사해 live 출력에 사용해서는 안 된다.

## 범위

이 계약은 실제 덕트의 committed causal v5 캡처 직전에 수행하는 두 작업만 다룬다.

1. 출력 없이 ERR/REF 입력을 총 1.5초 확인한다.
2. noise speaker(ch0)만 peak `0.003`으로 정확히 20.000초 구동해 현재 앰프 노브
   레벨을 봉인한다. cancel speaker(ch1)는 전 구간 exact zero다.

이 PASS는 레벨과 capture 입력 자격만 증명한다. P/S plant, sample-slip, 학습 또는 ANC
감쇠 성능 권한을 만들지 않는다.

## 공용 단일 출처

`deep_anc.audio_io.capture_measurement_preflight_raw()`는 입력 장치만 연다. 기본 총
1.5초 가운데 I2S 기동 트랜지언트 0.5초를 버리고, 나머지 1.0초를 소유권 있는
little-endian `<i4 [48000,2]`로 반환한다. 두 채널 모두 RMS 하한, clip 상한, raw-code
다양성을 통과해야 한다. 기존 `assert_measurement_preconditions()`도 이 함수에 위임한다.

`deep_anc.dsp.fullband_v5_meter`는 다음 소비자가 공유하는 package 계약이다.

- `scripts/data/set_amp_level.py`
- `scripts/data/measure_paths_fullband_causal_v5.py`

스크립트끼리 import하지 않는다. package 계약은 signal-plan envelope, capture-only live
authority, hardware YAML, tracked level attestation과 sealed raw target을 exact
path·schema·file SHA·payload SHA로 검증한다. PortAudio import/device query는 이 static 검증
뒤에만 허용한다.

clean exact checkout에는 2026-08-27 paired raw가 git에 들어 있지 않다. 따라서 v5 static
admission은 `measurement_level_evidence.json`의 pinned file SHA
`c76ac0d3...e73d0`와 canonical JSON, official 수치, 두 historical raw reference, physical
identity를 검증하는 좁은 attestation을 쓴다. 그 반환 범위는 반드시 다음과 같다.

- `scope=tracked_historical_attestation_for_fresh_v5_meter_only`
- `preserved_raw_revalidated=false`
- `strict_ps_authority=false`
- `plant_or_training_authority=false`
- `live_admission_eligible=false`

이것은 preserved raw 재검증을 대신하지 않는다. 기존
`load_measurement_level_evidence()`는 그대로 meter raw+receipt와 interleaved raw를 모두
요구하는 strict forensic validator이며, raw가 없는 clean checkout에서는 명시적으로
실패한다. v5 live 진입은 위 좁은 attestation과 **현재** physical fingerprint exact match,
그리고 직후 새로 얻는 20초 fullband-v5 meter를 함께 요구한다.

## 모드 경계

미터 CLI는 `strict`와 `fullband-v5` 두 모드만 허용한다. old broadband/v4 mode와 artifact를
호환 처리하지 않는다.

- `strict`: 기존 interleaved strict P/S 후속 레벨 미터
- `fullband-v5`: committed causal v5 capture 전용. 기존 paired level evidence를 요구하며
  bootstrap 우회는 금지한다.

fullband-v5는 다음 다섯 확인을 모두 요구한다.

1. speaker output 승인
2. 사용자 입회
3. 시작 전 볼륨 최소
4. ERR/REF, NS/CS 배선과 덕트 기하 확인
5. paired evidence와 같은 amplifier 설정 확인

## 실행 순서와 재검증

fullband-v5의 순서는 고정된다.

1. plan/authority/hardware/tracked attestation과 sealed target freshness 검증
2. PortAudio device resolve
3. repository audio lock 획득
4. lock 보유 상태에서 입력 전용 1.5초 preflight
5. stream open 직전 같은 파일·SHA·physical fingerprint·device를 다시 검증
6. 20.000초 meter capture
7. stream abort/close
8. 다른 저장·분석보다 먼저 스피커 즉시 분리 안내
9. capture 뒤 같은 결속을 다시 검증
10. immutable meter raw/receipt 발행
11. meter raw를 다시 읽고 후속 명령 직전 결속을 세 번째 검증

capture 뒤 결속이 바뀌면 유일한 raw는 보존하지만 status를 FAIL로 두며 후속 명령을
발행하지 않는다.

## PASS raw 결속

기존 `measurement_level_meter_raw_v1` container/receipt를 유지해 공용 PCM·레벨 검산을
재사용한다. 여기에 `fullband_v5_followup`을 추가하며 다음을 SHA 하나에 결속한다.

- exact signal plan file/payload/PCM SHA
- live capture authority file/payload SHA
- hardware file/identity/physical-fingerprint SHA
- tracked historical level attestation file/identity SHA, 좁은 scope,
  `preserved_raw_revalidated=false`
- resolved input/output PortAudio device
- 아직 존재하지 않는 sealed raw path
- 다섯 operator confirmation

consumer는 raw SHA, receipt SHA, 완료 UTC, followup SHA를 다시 묶은
`fullband_causal_v5_meter_identity_v1` SHA를 유도한다. live raw publisher는 이 값을
meter identity로 사용한다.

`require_sealed_raw_fresh=True`는 capture 전 경로다. immutable live raw 발행 뒤 external
post receipt가 meter를 다시 검증할 때만 `False`를 쓴다. 이 경우 raw 존재만 허용하고
authority/plan/hardware/evidence bytes와 SHA 검사는 그대로 수행한다. `False`인데 sealed raw가
없어도 허용하는 선택 모드는 없다. post-capture 호출은 symlink 아닌 기존 regular raw를
반드시 요구한다.

live adapter는 sounddevice를 import하기 전에
`validate_fullband_v5_meter_raw_static()`으로 raw/receipt SHA, PASS recipe, freshness,
followup contract와 embedded device index까지 먼저 검증한다. 그 다음에만 backend를 import해
실제 PortAudio index를 resolve하고, static 결과와 exact 비교한다.

## 실제 출력 전 명령

아래 명령은 실제로 noise speaker(ch0)를 nominal 20.000초, hard max 21.000초 구동한다.
먼저 사용자에게 출력 시간·speaker·볼륨 조건·artifact 경로를 보고하고 명시적 승인을
받아야 한다.

```bash
.venv/bin/python scripts/data/set_amp_level.py \
  --mode fullband-v5 \
  --confirm-speaker \
  --confirm-user-present \
  --confirm-volume-minimum \
  --confirm-routing-and-geometry \
  --confirm-same-amplifier-setting
```

PASS하면 meter가 exact `measure_paths_fullband_causal_v5.py --execute-live` 명령을 출력한다.
live adapter 구현 marker가 없으면 같은 명령을 기록하더라도 `[차단]`으로 표시하며 실행을
허용하지 않는다.

## 2026-08-29 입력 전용 현장 확인

스피커 출력 없이 PCM 무점유 뒤 1회 확인했다. 반환 raw는 `<i4 (48000,2)` 소유 배열이었다.
ERR는 `-77.41 dBFS`, 5,326 raw code, clip 0이었고 REF는 `-68.44 dBFS`, 13,189 raw code,
clip 0으로 두 채널 모두 입력 생존 gate를 통과했다. 이 raw는 저장하지 않았으므로 장기
provenance나 음향 성능 증거가 아니라, 현재 연결과 public preflight 구현의 read-only
현장 witness다.

## 실행 checkout과 recovery provenance

fullband-v5 meter는 출력 장치를 열기 전에 clean git checkout을 요구한다. meter raw에는
exact 40자리 commit, branch(또는 `DETACHED`), `repository_dirty=false`, 실행한
`scripts/data/set_amp_level.py`의 repository-relative path와 file SHA가 저장된다. live
adapter의 validator는 이 값이 현재 clean checkout과 exact 일치하지 않으면 capture를
허용하지 않는다. 생성되는 followup 명령은 현재 절대 Python interpreter와 절대 adapter
script 경로를 사용한다.

writer는 raw와 같은 inode를 가리키는 숨김 `*.v5_raw_recovery` hardlink를 성공 뒤에도
유지한다. 이는 두 번째 캡처나 복제본이 아니라 receipt 발행 중 named raw가 교체·삭제돼도
최초 bytes/SHA를 보존하는 durable evidence다. 출력된 recovery 상대경로와 SHA를 raw 및
receipt와 함께 보관하며 임의 정리하지 않는다.
