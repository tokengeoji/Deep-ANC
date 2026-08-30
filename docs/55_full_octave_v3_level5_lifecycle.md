# 55. Full-octave v3 Level-5 unseen physical lifecycle

## 결론

Level-5는 새 녹음 파일 하나를 더 넣어 보는 테스트가 아니다. canonical model과
experiment contract를 먼저 고정한 뒤, `training∪validation∪test` source identity와
**0 교집합**인 speech/music/environment/machine source를 실제 덕트에서 한 번만 평가하는
별도 lifecycle이다. 현재 canonical practice에서 model selection은 별도 split이 아니라
`validation` manifest의 정확한 SHA alias다.

현재 기본 설정
[`configs/full_octave_v3_level5_lifecycle.yaml`](../configs/full_octave_v3_level5_lifecycle.yaml)은
의도적으로 모든 artifact가 `null`이므로 **`BLOCKED_UNATTESTED_MISSING_AUTHORITY`**다. 이 상태는 실패를 숨기지 않고,
스피커·마이크·ALSA·GPU·네트워크를 열지 않는다.

구현 authority는
[`src/deep_anc/eval/full_octave_v3_level5.py`](../src/deep_anc/eval/full_octave_v3_level5.py)다.
이 모듈은 capture adapter나 ANC evaluator가 아니다. fake checkpoint/controller, minimal
physical report, self-sealed manifest, terminal `PASS`를 포함한 **모든 self-attested 조합**은
`BLOCKED_UNATTESTED_*`다. 어떤 입력에서도 `canonical_generalization_pass=false`,
`physical_generalization_authority=false`를 고정한다.

이 문서는 [16. canonical finetune guardrails](16_canonical_finetune_guardrails.md) §2/§6/§7,
[27. full-octave v3 contract](27_broadband_v3_full_octave_contract.md) §5,
[32. population contract](32_broadband_population_contract_v3.md) §5,
[54. 8-input raw bundle](54_full_octave_v3_eight_input_raw_bundle.md)의 Level-5 후속 소비자다.
문서보다 raw artifact와 검사기가 우선한다.

## Read-only 검사

```bash
.venv/bin/python scripts/eval/check_full_octave_v3_level5_lifecycle.py \
  --config configs/full_octave_v3_level5_lifecycle.yaml --dry-run --markdown
```

이 명령은 YAML/JSON/이미 존재하는 regular file을 읽고 SHA-256을 재계산할 뿐이다.
다음을 하지 않는다.

- speaker output, playback, ANC ON/OFF, calibration, recording
- ALSA/sounddevice/PortAudio open
- GPU/Trainer/model inference/evaluation 실행
- network/subprocess 호출
- capability/token/receipt/raw/report 파일 생성 또는 수정

이 CLI는 **항상 nonzero**다. 현재 lifecycle 자체에는 independent evaluator trust root가
없으므로 success exit을 발행하지 않는다. 따라서 model lock, source manifest, 8-input
structural report, capability/consumed/terminal receipt가 형식상 완전해도
`BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN` 또는
`BLOCKED_UNATTESTED_TERMINAL_RECEIPT`로 끝난다. 이 상태는 capture 승인, capability 발급,
Level-5 성능 PASS가 아니다. 훗날 opaque verified receipt를 직접 검증하는 **별도 independent
raw evaluator CLI**만 자기 trust root 아래 success exit을 가질 수 있으며, 이 lifecycle checker가
임의 Mapping/JSON을 그 authority로 해석하는 경로는 없다.

## 고정 순서

```text
canonical model/checkpoint + experiment contract + val selection 고정
  → immutable lineage inventory/population snapshot + challenge reservation을 독립 봉인
  → raw source identity manifests (training / validation / test), base pairwise leakage=0 재검산
  → selection_raw_manifest == validation_raw_manifest의 exact SHA alias 확인
  → model-lock SHA를 가진 Level-5 challenge raw manifest와 preregistration 확인
  → physical bundle **config SHA**로 official 8-input validator 재실행 (8ch/raw/sidecar)
  → submitted PCM/controller config/P/S/timing/lead exact binding 및 네 family OFF/DL/FxLMS matched raw 확인
  → immutable challenge-input SHA
  → 별도 dirfd O_EXCL issuer가 capability.json 1회 발급
  → 별도 dirfd O_EXCL evaluator가 consumed.json과 completed.json 또는 failed.json을 mutual exclusion으로 발행
  → 독립 raw evaluator의 full-octave/spatial/runtime 재검산 및 verified receipt
```

challenge source를 model selection 전에 확보하거나, 결과를 본 뒤 controller/loss/checkpoint를
교체하면 이 lifecycle의 전제가 깨진다. 새 source를 모델 학습/validation/model selection에
재사용하는 것도 금지한다. 특히 이 checker는 **사후 self-declared challenge manifest만으로**
challenge가 결과 접근 전 예약됐는지, 전역 source inventory에서 실제 제외됐는지를 소급해
증명할 수 없다. 그것은 독립 reservation authority의 책임이다.

## Raw source manifest authority

독립 raw source manifest는 `full_octave_v3_level5_raw_source_manifest_v1` JSON이며 정확히
하나의 partition을 가진다.

```text
training / validation / test / challenge
```

모든 manifest는 네 family를 모두 포함하고, 각 record에 아래 identity를 원본 bytes와 함께
적는다.

| Identity | 검증 방식 |
|---|---|
| `source_ids` | 각 source ID가 비어 있지 않고 중복 없이 보존됨 |
| `lineage_component_id` | connected component 단위의 원본 관계 |
| `lineage_keys` | artist/album, speaker/book, machine/original-recording 등 상위 관계 |
| `native_source` | repository-relative path, size, SHA-256을 실제 bytes로 재검산 |
| `decoded_pcm` | repository-relative path, size, SHA-256을 실제 bytes로 재검산 |

`selection_raw_manifest` artifact는 별도 selection partition이 아니다. 반드시
`validation_raw_manifest`와 **exact same SHA**를 가리키는 alias여야 한다. 따라서 selection을
위장한 별도 split 또는 `validation__selection` zero-overlap 요구는 허용하지 않는다.

`challenge` manifest만 `model_lock_sha256`를 가져야 한다. training/validation/test manifest에
사후 lock SHA를 넣어 과거 identity를 재봉인하는 길은 거부한다.

검사기는 family별로 challenge와 `training∪validation∪test` union의 다음 교집합을 모두
계산하고, 하나라도 0이 아니면 fail-close한다. selection은 validation alias이므로 이 union에
이미 포함된다.

```text
source_ids
lineage_component_ids
lineage_keys
native_source_sha256
decoded_pcm_sha256
```

또한 base `training/validation/test`의 세 pair(`training__validation`, `training__test`,
`validation__test`)에 대해 같은 다섯 identity 교집합을 계산한다. 하나라도 0이 아니면 **base split leakage**로
fail-close한다. 보고서의 `base_train_val_test_pairwise_leakage`는 이 declared-byte 계산을
명시한다.

따라서 파일명을 바꾸거나 decoded PCM만 다시 만들어 source/lineage 누수를 감추는 방식은
통과하지 않는다. 동일 identity를 서로 다른 family로 표기하는 경우도 거부한다. 단, 이
계산은 현재 manifest에 선언된 identity의 구조 검사일 뿐이다. source pool 전체의 원본 관계,
누락 record, 예약 시점은 확인할 수 없으므로 immutable lineage inventory/population snapshot과
independent challenge reservation 없이는 Level-5 authority가 아니다.

## Frozen model / physical bundle binding

`full_octave_v3_level5_model_lock_v1`은 다음을 self-sealed SHA로 묶는다.

- frozen canonical checkpoint bytes
- physical controller artifact bytes (checkpoint와 다른 deployment artifact일 수 있음)
- schema-v2 experiment contract bytes 및 embedded contract SHA
- validation manifest SHA (model selection은 이 exact SHA alias)

experiment contract는 clean exact commit과 source-tree SHA까지 실제 JSON body로 다시 검증한다.
단, 이 lifecycle은 checkpoint pickle을 열거나 model inference를 하지 않는다. checkpoint 내부
experiment contract, canonical completion, verified validation selection, export provenance는
**별도 authority**가 실제 checkpoint/controller bytes에서 확인해야 하며 JSON lock의
`canonical_model_frozen=true` 선언만으로는 충분하지 않다.

`full_octave_v3_level5_physical_bundle_lock_v1`은 [54](54_full_octave_v3_eight_input_raw_bundle.md)의
`BLOCKED_UNATTESTED_STRUCTURAL_RAW` report와 plan/raw/sidecar bytes를 다시 읽는다. report에 기록된
**physical bundle config SHA**로 [54](54_full_octave_v3_eight_input_raw_bundle.md)의 official
validator도 read-only로 다시 실행한다. 다음이 정확히 맞아야 한다.

- `REF`, `NOISE_TAP`, `CANCEL_TAP`, `ERR_0..ERR_4`의 8-input role 순서
- non-fixture plan과 raw-first **declared SHA structure** report
- capture plan의 `source_manifest_sha256 == challenge manifest SHA`
- capture plan의 `controller_artifact_sha256 == frozen model controller SHA`
- official validator가 같은 config SHA, 8 channel role, canonical raw SHA, sidecar SHA를 재확인

이것은 8-input raw가 lifecycle identity에 맞는지 보장하는 **self-attested 구조 검사**다.
`declared_sha_structure_valid=true`은 adapter provenance가 아니며, report의
`physical_raw_provenance_attested`는 false여야 한다. P/S 식별 PASS,
controller 감쇠, spatial quiet zone, ANC runtime PASS로 승격하지 않는다. submitted PCM,
controller config/artifact, fullband P/S, timing contract, `PlantDelays.lead()` 결과는 각
physical session raw와 별도로 exact binding되어야 한다.

## One-shot ledger

모든 primary input bytes/path로 다음 immutable digest를 만든다.

```text
challenge_input_sha256 = SHA-256(canonical JSON(
  model lock + five raw manifests + physical bundle lock + v3 contract
))
```

허용 ledger directory는 오직 다음 하나다.

```text
results/full_octave_v3_level5_ledger/<challenge_input_sha256>/
  capability.json
  consumed.json
  completed.json | failed.json
```

future issuer/evaluator는 `dirfd` 기반 O_EXCL/no-replace와 terminal mutual exclusion으로
각각을 발행해야 한다. 이 read-only checker는 현재 파일의 경로와 SHA를 읽을 수 있을 뿐,
kernel의 실제 no-replace history·issuer 독립성·terminal mutual exclusion을 소급 증명할 수 없다.

| Artifact | 필요한 결속 |
|---|---|
| `capability.json` | challenge input, model lock, physical bundle lock, 다섯 manifest SHA, token hash |
| `consumed.json` | capability file SHA와 같은 immutable identity |
| terminal receipt | capability/consumed SHA, raw evaluation bundle/metrics/evaluator receipt file SHA, verdict |

terminal receipt가 `PASS`라고 선언해도 checker는 그것을 신뢰 성능 수치로 재해석하지 않는다.
그 경우도 `BLOCKED_UNATTESTED_TERMINAL_RECEIPT`이며 exit 0이 아니다. independent raw evaluator
receipt가 raw bytes를 직접 재계산하고 issuer history까지 검증하기 전에는 terminal JSON은
self-attestation일 뿐이다.

## Level-5 PASS를 실제로 선언하려면

별도의 raw evaluator가 최소 다음을 직접 재계산하여 terminal receipt의 해당 raw artifact를
검증해야 한다.

1. ANC OFF/Deep-ANC/FxLMS의 동일 source·SPL·P/S·window 원본
2. 125/250/500/1000/2000/4000/8000 Hz octave의 평균·worst 10%·cluster CI
3. Stage-1 150–300/300–600/600–1000/1000–1600 Hz guard
4. 2/4/8 kHz에서 matched FxLMS 대비 paired mean·worst 10%·CI 하한
5. 최소 다섯 ERR 위치의 spatial quiet-zone 결과
6. 48 kHz/256 runtime P99, deadline miss/xrun/fallback/ring/sample-slip telemetry

그 evaluator가 raw bytes, one-shot ledger, frozen identity와 모두 일치하는 PASS를 발행하기
전까지 Level-5, broadband ANC, deployment 성공을 주장하지 않는다.

## 현재 판정

현 Jetson에는 canonical full-octave checkpoint, raw source identity manifests, non-fixture
8-input physical bundle, independent lineage reservation, matched OFF/DL/FxLMS campaigns,
dirfd O_EXCL issuer proof, independent terminal raw evaluation receipt가 없으므로 이 gate는
**`BLOCKED_UNATTESTED_MISSING_AUTHORITY`**다. 기존 82 session, strict 150–1600 Hz P/S,
legacy checkpoint/ONNX 또는 fixture는 이 lifecycle을 통과시키지 못한다. 장래에 형식상 완전한
JSON artifact가 생겨도, 독립 evaluator authority가 없으면
**`BLOCKED_UNATTESTED_SELF_DECLARED_CHAIN`** 또는
**`BLOCKED_UNATTESTED_TERMINAL_RECEIPT`**로만 기록한다.
