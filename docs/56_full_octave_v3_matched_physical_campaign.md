# full-octave v3 physical matched OFF/DL/FxLMS campaign 경계

이 문서는 실제 ANC 성능 수치가 아니라, 미래의 OFF / Deep-ANC(DL) / FxLMS 물리 비교가
서로 다른 source·볼륨·plant·window를 섞어 비교하는 일을 막는 **읽기 전용
provenance 경계**다. 구현은 `deep_anc.data.full_octave_v3_matched_campaign`, 기본
config는 `configs/full_octave_v3_matched_campaign.yaml`이다.

## [가설]

같은 독립 source group 안에서 OFF, DL, FxLMS가 같은 source PCM bytes, level/gain,
P/S·timing·lead, geometry, window, limiter, hardware 및 8-input raw bundle을 **선언**하고
순서를 counterbalance하면, 이후 실제 capture adapter가 보존해야 할 비교 계약을 미리
고정할 수 있다. 이 문서/체커만으로 그 선언이 실제 장비에서 지켜졌음을 증명할 수는 없다.

## [근거]

`docs/27_broadband_v3_full_octave_contract.md`의 최종 v3 판정은 같은 source·SPL·P/S·
window를 요구하며, 2/4/8 kHz는 matched FxLMS보다의 paired 우위를 별도로 요구한다.
`docs/54_full_octave_v3_eight_input_raw_bundle.md`는 REF, 두 electrical tap, ERR 5개를
동시에 보존하는 raw bundle 경계만 고정한다. 두 문서 어느 하나도 순서·독립 group·
one-shot lifecycle를 검사하지 않으므로, 이 campaign receipt가 그 사이를 연결한다.
단, [54](54_full_octave_v3_eight_input_raw_bundle.md)의 complete-looking bundle도 현재는
`BLOCKED_UNATTESTED_STRUCTURAL_RAW`이며, 이 campaign은 그 report의
`declared_sha_structure_valid=true`만 재검산한다. 이를 physical capture authority로
승격하지 않는다.

## [확인 방법]

스피커·마이크를 분리한 상태에서도 아래 명령은 config와 이미 발행된 artifact만 읽는다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/data/check_full_octave_v3_matched_campaign.py --dry-run
```

이 명령은 ALSA/sounddevice/GPU/network를 열지 않고, output·ANC ON·capture·attenuation
계산·파일 쓰기를 하지 않는다. 현재 기본 null config의 정상 결과는 `BLOCKED`이다.
future non-fixture JSON/raw를 모두 채워도 현재 체커의 결과는
`BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE`이며 exit code는 0이 아니다.

향후 물리 작업은 capture 전에 immutable JSON plan을 먼저 발행하고, 각 8-input session
뒤에는 해당 session의 `comparison run receipt`를, 계획된 모든 session이 끝난 뒤에는
immutable one-shot campaign receipt를 발행한다. session receipt는 다음 SHA/value를 plan과
8-input bundle 양쪽에 다시 맞춰야 한다.

- submitted PCM/source manifest, level evidence/gain contract와 실제 gain 값
- P/S campaign, primary/secondary operator, TrainingTimingContract, PlantDelays·handoff·lead
- routing geometry, analysis window, limiter, hardware/topology
- bundle config, capture-plan evidence, sidecar evidence, native/canonical raw SHA

plan은 각 family에서 최소 네 **선언된** independent group과 3-condition cyclic
Latin-square 순서를 요구한다. 실제 canonical lineage에서 나온 group인지, source가 실제로
제출됐는지는 이 checker가 self-attested JSON만으로 증명하지 못한다.

```text
OFF → DL → FxLMS
DL → FxLMS → OFF
FxLMS → OFF → DL
```

세 condition이므로 두 condition의 ABBA를 그대로 적용하지 않고, 위 세 순서를 family마다
최소 한 번씩 쓰며 order count 차이가 1을 넘지 않게 한다. 한 group 안의 세 session만
동일 submitted PCM bytes를 공유한다. 서로 다른 independent group은 같은 family 안에서
같은 submitted PCM bytes를 재사용할 수 없다.

## [결과]

현재 config의 plan/receipt artifact는 모두 `null`이다. 따라서 checker는 다음을 모두
false로 유지한다.

```text
matched_campaign_structural_valid = false
canonical_matched_physical_pass = false
canonical_training_eligible = false
deployment_eligible = false
physical_attenuation_math_performed = false
```

future non-fixture session/campaign receipt가 SHA/field 구조상 완전해도 status는
`BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE`다. 보고서의
`declared_sha_structure_valid = true`은 선언 SHA 구조 검사 결과일 뿐, 물리 match나
`PASS`라는 성능 상태가 아니다. 다음 값은 항상 false로 유지한다.

```text
matched_campaign_structural_valid = false
physical_provenance_attested = false
canonical_matched_physical_pass = false
canonical_training_eligible = false
deployment_eligible = false
```

## [판정]

**Blocked.** 현재에는 full-octave 8-input raw session, canonical DL checkpoint, matched
FxLMS 설정 및 one-shot physical receipt가 없다. legacy P/S, 기존 checkpoint, fixture,
정적 YAML은 이 경계를 통과할 수 없다.

## [다음 행동]

실제 출력을 재개하기 전에는 fullband P/S·electrical witness·8-input capture adapter의
무음 dry-run과 전체 테스트를 먼저 통과시킨다. 사용자 입회·최소 볼륨·장치 점유 확인을
마친 한 번의 짧은 연결 창에서만 predeclared plan을 실행한다. 물리 provenance를
승격하려면 adapter가 다음을 raw-bound으로 발행·검증할 수 있어야 한다.

- O_EXCL capture event를 campaign plan SHA와 nonce, adapter build SHA, device fingerprint,
  session-monotonic sequence에 묶은 receipt
- 실제 submitted stream, level/SPL, analysis window, limiter, 8-input topology의 callback
  evidence
- raw-bound fullband P/S와 `PlantDelays.lead()` lead, canonical lineage-derived independent
  group
- native→canonical transform receipt 및 서로 다른 matched session 사이의 campaign-wide
  native/canonical raw SHA uniqueness. 단일 capture의 변환이 identity여서 native와
  canonical bytes가 같은 경우는 그 **같은 session pair에 한해** 허용한다.
- canonical finetune init checkpoint·experiment contract·recorded selection을 포함한
  stage-specific training schema와 independent raw evaluator receipt

그 뒤 capture 결과에 대해 아래를 별도로 수행해야 한다.

- immutable OFF/DL/FxLMS raw window에서 attenuation 및 paired bootstrap 계산
- P99/max/deadline/xrun/fallback을 포함한 runtime receipt 검증
- fullband causal P/S, source population/lineage, high-band coverage 검증
- one-shot physical G4와 동시 다섯 ERR 위치 quiet-zone 판정

이 checker는 위 계산·capture를 대신하지 않으며, 수치가 없으면 성능을 주장하지 않는다.
