# v5 실측 capture adapter와 external post receipt

## 결론과 권한 경계

`scripts/data/measure_paths_fullband_causal_v5.py --execute-live`는 committed v5 신호를
실제 장치에 한 번 제출하고, 성공·부분 실패를 모두 immutable raw로 보존하는 capture-only
진입점이다. capture 명령은 지연 추정이나 P/S 분석을 실행하지 않는다. raw와 별도 external
post receipt가 모두 `fsync`된 뒤에만 exact offline 명령을 출력한다.

이 단계가 PASS해도 다음 값은 바뀌지 않는다.

```text
canonical_training_eligible = false
hardware_sample_slip_authority = false
```

즉 raw/receipt와 offline 수학 PASS는 실제 hardware sample slip 부재, canonical strict P/S,
학습 자격 또는 ANC 감쇠 성능을 증명하지 않는다. 별도로 사전 선언된 hardware-slip 정책과
반복 실측 authority가 생기기 전에는 해당 권한을 열지 않는다.

## 실제 출력 시간과 채널

- 입력 전용 preflight: 총 1.5초, 스피커 출력 0초
- v5 duplex nominal: 정확히 557,056 frame / 48 kHz = **11.605333초**
- host watchdog hard maximum: **13.605333초**
- 출력: ch0 소음 스피커와 ch1 상쇄 스피커
- primary/secondary near-white main slot은 시간 분리되어 있고 저레벨 pilot은 plan대로 유지

20초 fullband-v5 meter는 별도 출력 창이다. 따라서 fresh meter와 capture를 연속 실행할 때
nominal audible 합계는 31.605333초다. 두 명령 사이 앰프 노브를 바꾸지 않는다. 각 stream
close 직후 스피커/앰프를 즉시 물리적으로 분리한다.

## fail-closed 실행 순서

1. sounddevice를 import하기 전에 sealed plan/authority/hardware, pinned tracked level
   attestation과 fresh meter raw·receipt의 PASS recipe/followup, raw/post-receipt target
   freshness를 검사한다.
   historical preserved raw는 이 portable 경로에서 재검증됐다고 주장하지 않는다.
2. 같은 단계에서 PCM 무점유, capture clock, CPU idle과 actual-int16 plan SHA를 검사한다.
3. PortAudio device를 resolve하고 meter가 봉인한 input/output index와 비교한다.
4. repository audio lock을 먼저 획득한다. 이후 input-only preflight를 수행한다. 어떤 input 또는
   output open도 lock 밖에서 하지 않는다.
5. ERR/REF preflight raw와 모든 channel summary를 session raw에 결속한다. dead/stuck/railed
   microphone이면 duplex를 열지 않는다.
6. `capture_duplex_v5`가 큰 buffer를 준비한 뒤 Stream open 직전에 plan/authority/meter/
   evidence/hardware bytes, physical fingerprint, device, raw targets, audio lock, PCM/clock/CPU를
   다시 검사한다.
7. 정상 stop 또는 실패 abort/close 뒤 다른 postcheck·저장·분석보다 먼저 스피커 분리 안내를
   출력한다. close 확인이 없으면 물리 분리 경고를 출력하고 raw는 `INVALID`로만 보존한다.
8. lock을 계속 보유한 채 post bytes/fingerprint/device/lock을 다시 검사하고 success 또는
   `DuplexCaptureFailure`를 `publish_live_raw_v5`로 no-replace+fsync 발행한다.
9. raw의 실제 file SHA, metadata binding SHA와 13개 array SHA를 actual external files에 다시
   결속한 external post receipt를 no-replace+fsync 발행한 뒤 lock을 해제한다.

post 검증이 실패해도 raw를 지우거나 덮어쓰거나 즉시 재측정하지 않는다. 가능한 경우 raw SHA와
prebinding 및 exact error list를 묶은 `INVALID`, `analysis_admission_eligible=false` receipt를
발행한다. INVALID receipt 발행마저 실패하면 raw 경로/SHA와 receipt 부재를 출력하고 nonzero로
끝난다. 이 상태는 offline admission이 아니다.

## live 명령

fresh 20초 meter가 출력한 `METER_RAW`를 그대로 쓴다. 다섯 확인은 meter와 capture가 동일하다.

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal_v5.py \
  --execute-live \
  --plan-envelope assets/contracts/fullband_causal_v5_signal_plan.json \
  --live-authority assets/contracts/fullband_causal_v5_live_capture_authority.json \
  --meter-raw "$METER_RAW" \
  --level-evidence assets/measured/measurement_level_evidence.json \
  --hardware configs/hardware_jetson.yaml \
  --raw-target results/fullband_causal_v5/raw_capture.npz \
  --confirm-speaker \
  --confirm-user-present \
  --confirm-volume-minimum \
  --confirm-routing-and-geometry \
  --confirm-same-amplifier-setting
```

기존 raw 또는 post receipt, symlink target/parent, legacy/v4 meter·authority, stale meter,
device/fingerprint 변경은 모두 출력을 열기 전에 거부한다.

## offline 분석

capture PASS가 출력한 명령은 다음 필드를 포함한다.

```text
--offline-analyze
--post-receipt <raw sibling external receipt>
--expected-post-receipt-sha256 <실제 receipt file SHA>
--analysis-output results/fullband_causal_v5/analysis_<capture-id>
```

offline loader는 PortAudio를 query/open하지 않는다. receipt file/payload SHA, pinned plan와
authority, meter raw/receipt, 좁은 tracked level attestation(scope와
`preserved_raw_revalidated=false` 포함), hardware identity/fingerprint, canonical live raw bytes,
metadata binding과 모든 array SHA를 재검산한다. receipt/raw splice나 외부 file 변조가 있으면
core를 호출하지 않는다.

live 출력 전에는 clean exact checkout을 강제하며 raw session에 40자리 commit, branch 또는
`DETACHED`, `repository_dirty=false`, adapter 상대경로와 file SHA를 봉인한다. dirty tree는
PortAudio import 전에 차단된다. 생성되는 capture/offline 명령은 현재 절대 interpreter와
절대 script 경로를 사용하므로 실행 cwd가 달라도 다른 checkout의 script를 실행하지 않는다.
SIGINT/SIGTERM/SIGHUP와 post-close 예외에서도 primitive가 abort/close를 수행하고 close callback
또는 caller `finally`가 저장·분석보다 먼저 즉시 물리 분리 안내를 출력한다.

admission 뒤에는 `analyze_committed_v5_live_delay`만 호출한다. 반환된 `analysis.json`과
`operator.npz`는 sibling staging directory에서 모두 fsync한 뒤 하나의
`renameat2(RENAME_NOREPLACE)` transaction으로 공개한다. operator array SHA를 core receipt에서
다시 계산하며 결과 기반 임계 변경이나 기존 output overwrite는 없다.
