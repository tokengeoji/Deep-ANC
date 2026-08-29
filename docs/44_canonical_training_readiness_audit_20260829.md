# Canonical full-octave 학습 readiness 현장 감사 (2026-08-29)

상태: **BLOCKED**

이 문서는 README·HANDOFF의 과거 주장 대신 현재 Jetson에 읽을 수 있는 artifact,
worktree, config, manifest, checkpoint metadata를 대조한 결과다. 이 문서의
`PASS`는 명시된 범위만 뜻하며, legacy artifact를 canonical 학습·배포 근거로 승격하지
않는다.

## [가설]

기존 strict P/S, transfer manifest, checkpoint를 사용해 canonical 125 Hz--8 kHz
Deep-ANC 학습을 바로 시작할 수 있다.

## [근거]

| 영역 | 실제 근거 | 확인 결과 |
|---|---|---|
| clean v9 기준 | `work/v9-electrical-frame-witness` @ `3f6c21a` | clean exact checkout |
| primary 작업공간 | `/home/capston/Deep_ANC`, `work/broadband-anc-v2` @ `ac645e8` | v9보다 14 commit 뒤, 사용자 소유 tracked 77개 수정·untracked 86개 존재 |
| strict P/S | `assets/measured/{primary,secondary}_path_il_strict_5dc06fdd.npz` | 48 kHz/256, P effective 1386, S effective 1245, lead 115, 150--1600 Hz 범위 |
| strict level | `assets/measured/measurement_level_evidence.json` | output=`Audio`/AB13X USB Audio, S16; RT5640/J511 S32가 아님 |
| old transfer | `data/manifests/elice_transfer_manifest.json` | schema 1, 344 files, SHA `39dc271672ac2916840a9919baaf7de5bdf078d228a68457f15096d433a76b4d`; USB Stage-1 bundle |
| old checkpoint/export | local `.pt` 26개와 legacy export | `experiment_contract_sha256` 없음, trusted band 150--600 Hz, lead 109 또는 113 |
| current v3 consumer | train YAML/trainer/eval의 v3 binding | current canonical consumer 없음 |
| Elice | 사용자가 2026-08-29 인스턴스를 삭제 | 현재 GPU·disk·job·remote manifest 없음 |

## [확인 방법]

1. P/S NPZ metadata, level evidence, runtime lead 및 output hardware identity를 직접
   비교했다.
2. checkpoint metadata에서 experiment contract, trusted band, lead, stage를 조사했다.
3. recorded holdout의 source SHA/filename과 synthetic manifest/public raw를 교집합
   대조했다.
4. v3 control-band/batch/loss 모듈이 train YAML, trainer, eval/runtime admission에서
   실제 소비되는지 코드 경로를 추적했다.
5. worktree revision/porcelain과 Elice 연결 가능 상태를 분리해서 기록했다.

## [결과]

### 실제 recorded 자료

- `recorded_regrouped.jsonl`: 82 session, 5,740초(95.67분), train/val/test=40/20/22.
- 각 session의 `session.json`, `mics.wav`, `source.wav`, `source_aligned.wav`는 존재한다.
- family별 val/test lineage component는 모두 4개 이상이고 recorded split의
  `group_id` cross-split은 0이다.
- 단, 이 양호한 recorded split은 RT5640 fullband P/S나 synthetic source lineage를
  자동으로 보증하지 않는다.

### old synthetic 자료의 실제 누수

- `recorded_holdout.json`의 speech source SHA 174개는 local LibriSpeech raw와 일치한다.
- old `speech.jsonl`에는 그 holdout 원본 8개가 그대로 있다. 예:
  `2277-149896-0016`, `2277-149896-0033`, `2277-149897-0002`,
  `2277-149897-0022`, `2428-83699-0033`, `2428-83705-0001`,
  `6345-64257-0013`, `6345-93306-0017`.
- old `esc50.jsonl`도 recorded holdout과 environment 58개, machine 24개의
  basename/lineage 교집합이 있다. raw audio 대부분은 이미 Jetson에서 제거되어
  현재는 manifest-level 증거로 보존한다.
- old `music.jsonl`은 7,877행이지만 FMA raw는 0/7,877 local paths만 존재하고,
  `esc50.jsonl`은 1,475행 중 local path 3개만 남아 있다.

따라서 old `speech.jsonl`, `music.jsonl`, `esc50.jsonl` 및 schema-1 transfer bundle은
**diagnostic-only**다. `--allow-corpus-leak` 또는 이름 변경으로 이를 canonical으로
승격하지 않는다.

### model / plant

- strict P/S는 실측이고 Stage-1 범위에서는 보존할 유효한 artifact다.
- 그러나 AB13X USB/S16 output에서 측정됐으므로 APE/RT5640/J511 S32 output의 plant가
  아니다.
- old pretrain/fine-tune/checkpoint/ONNX TensorRT artifact 모두 150--600 Hz와
  legacy lead에 묶여 있다. full-octave canonical init은 0개다.
- v3의 contract·batch·loss primitive는 존재하지만 current trainer/readiness/eval/runtime
  consumer가 그것을 canonical full-octave authority로 받지 않는다.

## [판정]

**Blocked.** 현재 Jetson에서 canonical 125 Hz--8 kHz pretrain/fine-tune을 시작하거나,
현재 모델이 그 대역에서 ANC 성능을 낸다고 주장할 artifact는 없다.

이는 recorded 82 session이 “전부 잘못됐다”는 뜻이 아니다. 그 자료는 aligned recorded
supervision 후보로 보존할 가치가 있다. 차단 사유는 새 output topology의 fullband plant,
누수 없는 synthetic corpus/manifest, full-octave training consumer 및 canonical init이
아직 하나의 exact contract로 결속되지 않았다는 점이다.

## [다음 행동]

| 순서 | gate | 완료 조건 |
|---:|---|---|
| 1 | clean fullband 학습 코드 | v3 contract를 train YAML·trainer·readiness·eval에 연결하고 clean branch 전체 test PASS |
| 2 | RT5640 electrical/frame witness | J511 `HP|HS`, safe external tap+동기 ADC 또는 검증 TRRS capture; PCM0 rail/stuck 경로는 제외 |
| 3 | 실제 덕트 fullband P/S | 80 Hz 이하--11,313.7 Hz 이상, 8 physical subband, raw-first provenance, 2/4/8 kHz consistency PASS |
| 4 | corrected transfer manifest | 새 P/S + recorded holdout + lineage receipt로 새 schema/sha 발행 |
| 5 | 새 Elice bootstrap | 새 endpoint에서 exact clean commit, `--no-update --preflight-only`, public raw six-manifest/QA PASS |
| 6 | pilot prerequisite | G0, loss pilot, strict-S gradient 0.2--0.4, 5k measured probe, exact-resume receipt, immutable ledger |
| 7 | canonical pretrain | 1--6 PASS 뒤 new 100k init |
| 8 | measured fine-tune | canonical init/readiness PASS 뒤 new 50k; val-only selection 후 one-shot test |

학습을 **시작할 최소 조건**은 1--5다. canonical 100k pretrain은 1--6, 50k fine-tune은
그 init이 실제 완료되어야 시작할 수 있다.

## 보존 규칙

다음은 삭제하거나 canonical으로 승격하지 않는다.

- 82 recorded session과 `source_aligned.wav`
- source pool v1/v2 CSV, holdout/provenance
- strict Stage-1 raw/P/S 및 legacy result
- 26개 legacy checkpoint와 ONNX/plan artifact
- schema-1 transfer와 old synthetic manifests

Jetson 여유 공간은 약 10.64 GiB 수준이므로 full public corpus를 다시 local staging하지
않는다. 새 Elice에서 untouched raw를 받아 corrected holdout을 먼저 적용하고 manifest를
재생성한다.
