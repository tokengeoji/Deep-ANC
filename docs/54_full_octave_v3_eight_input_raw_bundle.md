# full-octave v3 8-input synchronized raw bundle

이 문서는 최종 quiet-zone physical campaign에 필요한 **소프트웨어 전용 raw bundle
경계**를 고정한다. 구현은
`deep_anc.data.full_octave_v3_physical_bundle`, 기본 설정은
`configs/full_octave_v3_physical_session_bundle.yaml`이다.

## [가설]

REF, noise/cancel 전기 tap, 서로 다른 ERR 위치 다섯 곳을 같은 48 kHz sample frame으로
수집하면, 125 Hz--8 kHz quiet-zone 평가와 P/S 분석에 필요한 원시 증거를 보존할 수 있다.

## [근거]

기존 4입력 external electrical witness 계약은 P/S 식별의 최소 역할만 고정한다.
그러나 최종 quiet-zone 주장은 `REF + NOISE_TAP + CANCEL_TAP + ERR_0..ERR_4`의 여덟
동시 입력이 필요하다. ERR 위치를 바꿔가며 순차로 수집한 결과는 이 역할을 대신할 수 없다.

## [확인 방법]

실제 음향 출력이나 장치 open 없이 다음 명령은 config와 이미 발행된 bundle만 읽는다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/data/check_full_octave_v3_physical_session_bundle.py --dry-run
```

정상적인 현재 결과는 `BLOCKED`다. future non-fixture plan/raw/sidecar가 형식상 완전해도
결과는 `BLOCKED_UNATTESTED_STRUCTURAL_RAW`이며, 이 CLI에는 success exit이 없다. 이 명령은
ALSA, sounddevice, GPU, 네트워크를 열거나 결과 파일을 만들지 않는다.

향후 live capture adapter는 출력 전에 다음 순서를 `O_EXCL` no-replace로 실제 수행해야 한다.

```text
capture_plan.json
  -> native.s32le (8ch interleaved <i4)
  -> canonical.s32le (8ch interleaved <i4)
  -> sidecar.json
```

plan과 sidecar에는 다음을 모두 포함해야 한다.

- 48 kHz, block 256, 8ch 및 정확한 channel map
  `REF`, `NOISE_TAP`, `CANCEL_TAP`, `ERR_0..ERR_4`
- BCLK/WS/absolute frame counter와 shared acquisition clock의 same-frame witness
- callback S32 SHA, xrun/drop/add 0, 연속 frame counter
- submitted source, controller, plant campaign/hardware/geometry, timing contract/
  PlantDelays lead의 각각의 SHA identity
- native/canonical raw와 plan/sidecar의 bytes SHA 및 raw-first publication declaration

checker는 raw byte 길이가 `frames × 8 × 4`인지, 모든 SHA와 predeclared target이 같은지,
plan/raw/sidecar의 약한 filesystem 순서가 맞는지도 확인한다. 이 결과의
`declared_sha_structure_valid=true`은 **선언 SHA/field 구조가 맞다**는 뜻일 뿐 physical
provenance나 canonical authority가 아니다. 사후 filesystem 상태만으로 kernel의 `O_EXCL`
이력, adapter가 실제로 열었던 장치, 실제 submitted PCM/callback telemetry, native→canonical
변환 recipe/equality를 증명할 수는 없으므로, 이 사실은 report에 명시된다.

## [결과]

현재 config의 모든 raw artifact는 `null`이다. 따라서 아래는 모두 false다.

```text
raw_bundle_structural_valid = false
declared_sha_structure_valid = false
physical_raw_provenance_attested = false
canonical_training_eligible = false
deployment_eligible = false
physical_plant_identification_pass = false
quiet_zone_performance_pass = false
```

fixture-only plan/sidecar도 `BLOCKED`이며, `fixture_only=false`인 complete-looking future
bundle도 `BLOCKED_UNATTESTED_STRUCTURAL_RAW`이다. 이때
`declared_sha_structure_valid=true`일 수 있지만 canonical 학습·배포·물리 P/S·quiet-zone
PASS가 아니다.

## [판정]

**Blocked.** 현재 장비에 8-input synchronized raw가 없으므로 static config나 legacy
2/4-channel artifact를 full-octave physical evidence로 승격할 수 없다.

## [다음 행동]

스피커를 연결하고 실제 capture를 하기 전에는 future adapter의 무음 dry-run과 전체 테스트를
먼저 통과시킨다. 실제 출력은 사용자 입회·최소 볼륨·장치 점유 확인을 거친 하나의 짧은
연결 창에서만 수행한다. 그 뒤에도 다음 독립 authority가 별도로 남아 있다.

- typed P/S operator·raw·analysis validator 및 operator/timing의 exact crosslink
- typed raw/analysis/electrical witness validator
- actual submitted PCM/callback telemetry와 native↔canonical recipe/equality
- plan nonce·device·session monotonic에 결속된 capture adapter `O_EXCL` receipt
- canonical finetune init checkpoint·experiment contract·recorded selection을 포함한
  stage-specific training schema
- immutable ON/OFF raw의 independent five-ERR quiet-zone evaluator
