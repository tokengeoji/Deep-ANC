# Jetson 로컬 broadband population-v3 availability 감사

> 감사일: 2026-08-28. 네트워크, Drive, Elice, 오디오 장치와 스피커는 사용하지 않았다.
> 기존 source/manifest/audio bytes는 읽기만 했고 수정하지 않았다.

## 1. 판정

현재 판정은 **BLOCKED**다. 로컬에 실제로 있는 source 파일의 존재·크기·SHA와 bounded
decoder probe는 재검산했지만, 이를 canonical population-v3 manifest로 승격할 수 없다.

- canonical population candidate: **0**
- physical 8구간 component-band 결손: **384** (`3 split × 4 family × 8 band × 4`)
- objective 7 octave component-band 결손: **336** (`3 × 4 × 7 × 4`)
- fullband causal P authority: **MISSING**
- decoded PCM artifact/P-applied ERR/density: **미발행·미계산**
- `POPULATION_V3_AUTHORITY`: `None`

부분 대역 기여를 전역 실패와 분리한 최신 실제 보고서는
[`results/data_audit/broadband_population_v3_availability_bandwise_v1_20260828.json`](../results/data_audit/broadband_population_v3_availability_bandwise_v1_20260828.json)이다.
기존 `broadband_population_v3_availability_20260828.json`은 이 구분을 도입하기 전의
진단 기록이므로 canonical 근거로 승격하지 않는다.

- JSON file SHA-256:
  `ca798ccaa1752a453b295946313c5aea6f77b28f2d9c8093af77e45d25160945`
- canonical payload evidence SHA-256:
  `60cda2b19c5d58abf4e0a1a6686d92e122e24ac89c6af3063df4b48e837205d5`
- control-band v3 SHA-256:
  `53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2`
- population-v3 계약 SHA-256:
  `8f0d1e3897a2ace87059cecd584ea5da3ed0ecdb01a45a5855b2475cfe6e05c1`

## 2. 실제 실행 명령

```bash
.venv/bin/python scripts/data/audit_broadband_population_v3_availability.py \
  --input public_jsonl=data/manifests/speech.jsonl \
  --input public_jsonl=data/manifests/music.jsonl \
  --input public_jsonl=data/manifests/esc50.jsonl \
  --input recorded_jsonl=data/manifests/recorded_regrouped.jsonl \
  --input source_pool_csv=data/source_pool/sources.csv \
  --input source_pool_csv=data/source_pool_v2/sources.csv \
  --scan-unreferenced speech=data/raw/speech \
  --scan-unreferenced music=data/raw/music/fma_small \
  --scan-unreferenced environment=data/raw/noise/esc50/ESC-50-master/audio \
  --scan-unreferenced machine=data/raw/machine \
  --output results/data_audit/broadband_population_v3_availability_bandwise_v1_20260828.json
```

exit code `2`는 오류 은폐가 아니라 예상된 fail-closed `BLOCKED` 판정이다. output은
`O_EXCL` no-replace로만 발행한다.

## 3. 실제 파일 availability

| 항목 | 실제 수량 |
|---|---:|
| legacy manifest 행 | 11,866 |
| unreferenced scan을 포함한 보고 행 | 12,298 |
| 실제 파일 존재 | 2,949 |
| 실제 파일 결손 | 9,349 |
| unsafe/out-of-root path | 0 |
| bounded decoder probe PASS | 2,949 |
| bounded decoder probe FAIL | 0 |
| legacy manifest가 source SHA를 선언한 행 | 0 |
| auditor가 실제 SHA를 계산한 파일 | 2,949 |

결손 9,349개는 정확히 다음 두 legacy manifest에서 발생했다.

- `music.jsonl`: **7,877/7,877 파일 결손**
- `esc50.jsonl`: **1,472/1,475 파일 결손**

로컬 raw tree를 manifest와 독립적으로 다시 훑은 결과는 다음과 같다.

- speech audio: 2,703개 = manifest 참조 2,272개 + 미참조 431개
- ESC-50 audio: 4개 = manifest 참조 3개 + 미참조 1개
- `data/raw/music/fma_small`: directory 자체가 없음
- `data/raw/machine`: directory 자체가 없음

decoder PASS는 **full decode 증거가 아니다**. libsndfile로 header와 처음/마지막 최대 1,024
frame을 실제 decode한 bounded availability probe다. 따라서 report는 모든 행에
`full_decode_verified=false`, `decoded_pcm_artifact_sha256=null`을 유지한다.

## 4. native Nyquist를 processed 48 kHz와 분리한 결과

실제 direct native 후보는 2,707개다.

- LibriSpeech FLAC 2,703개: 실제 16 kHz, native Nyquist 8 kHz
- ESC-50 WAV 4개: 실제 44.1 kHz, native Nyquist 22.05 kHz

따라서 8 kHz octave 상단 11,313.708 Hz까지 native Nyquist가 확인된 direct source는
**ESC-50 네 개뿐**이다. LibriSpeech 2,703개는 마지막 octave source로 세지 않는다.
그러나 이를 LibriSpeech 전체의 전역 실패로 취급하지 않는다. 한 source는 native
Nyquist가 실제로 덮는 physical band와 objective octave에만 기여할 수 있다. 해당 mask는
`qualification_limitations` 및 cell별 `mapping_native_components_per_*`에 보존되며,
canonical blocker와 분리된다.

recorded `source.wav` 82개와 source-pool WAV 160개는 실제 header가 48 kHz이고 decoder도
열리지만, immutable native origin과 변환 receipt가 없다. 이 242개를 native 24 kHz
증거로 세지 않는다. report에는 header Nyquist와 verified native Nyquist를 별도 필드로
저장하며 이 242개는 `native_nyquist_verified=false`다.

split까지 가진 direct full-target source는 train/environment ESC-50 세 개뿐이다. 나머지
ESC-50 한 개는 unreferenced라 split이 없다. 이것도 density나 canonical 자격이 아니라
Nyquist availability 상한일 뿐이다.

현재 mapping component의 native 대역별 availability 상한은 다음과 같다. 숫자는
canonical qualified component가 아니라, 실제 direct-native header와 mapping이 있는
component 수다.

| Split/family | physical 8구간 | objective 7 octave |
|---|---|---|
| train/speech | 36, 36, 36, 36, 36, 36, 36, 0 | 36, 36, 36, 36, 36, 36, 0 |
| val/speech | 32, 32, 32, 32, 32, 32, 32, 0 | 32, 32, 32, 32, 32, 32, 0 |
| test/speech | 35, 35, 35, 35, 35, 35, 35, 0 | 35, 35, 35, 35, 35, 35, 0 |
| train/environment | 3, 3, 3, 3, 3, 3, 3, 3 | 3, 3, 3, 3, 3, 3, 3 |
| 그 밖의 cell | 모두 0 | 모두 0 |

LibriSpeech 16 kHz 원본은 이처럼 저·중역에 계속 기여할 수 있지만 마지막 physical
구간과 8 kHz objective octave의 상단까지는 증거가 없다. 반대로 ESC-50 세 파일만으로
train/environment full-target density가 확보됐다고 볼 수 없다.

## 5. mapping 후보와 lineage 재검산

legacy v1/v2와 unmanifested source는 모두 `mapping_only=true`,
`legacy_automatic_promotion=false`다. 현재 mapping 결과는 다음과 같다.

| 상태 | 수량 | 의미 |
|---|---:|---|
| `MAPPING_CANDIDATE` | 2,357 | 파일·bounded decode·family·split·semantic lineage mapping 존재 |
| `PARTIAL_MAPPING_CANDIDATE` | 592 | 파일은 있으나 split 또는 native provenance 등이 불완전 |
| `UNAVAILABLE` | 9,349 | 실제 파일 결손 |

2,357개는 LibriSpeech 2,272개, local ESC-50 3개, recorded composite 82개다. 여기서
LibriSpeech는 native 고역이 부족하고 recorded composite는 native provenance가 없으므로
canonical candidate는 여전히 0이다.

lineage mapping은 실제 metadata bytes도 SHA로 결속해 다시 계산했다.

| Metadata | Records | SHA-256 |
|---|---:|---|
| LibriSpeech `CHAPTERS.TXT` | 5,831 | `2e9db25c250a143b031ca003180acfce84b364e31fbc79250e306303e6f25306` |
| FMA `tracks.csv` | 106,574 | `f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b` |
| ESC-50 `esc50.csv` | 2,000 | `ca660da60191a97de289983a05821c9382d852a38a2ba8428980816b68cf6246` |

reader/book, artist/album, ESC original `src_file`, source-pool group와 actual content SHA를
connected component로 묶었다. 그 결과 legacy mapping에서:

- split을 넘는 component: **515개**
- 실제 존재 파일을 포함하면서 split을 넘는 component: **38개**
- family를 넘는 component: **8개**
- 실제 존재 파일을 포함하면서 family를 넘는 component: **1개**
- split/family conflict component 전체: **517개**
- conflict component에 속한 present candidate-row reference: **2,907개**

이는 canonical leakage PASS가 아니라 **기존 split을 component 단위로 다시 만들어야 한다는
mapping conflict**다. 특히 실제 LibriSpeech reader/book component가 train/val/test를
넘는 사례가 다수다. 보고서의 `lineage_mapping_conflicts`는 component SHA, 축, split,
family, present/mapping 수와 관련 input을 모두 보존한다. metadata mapping 자체는 아직
population-v3 authority가 아니므로 `canonical_lineage_authority=false`다.

## 6. split×family availability 상한

아래 값은 `행 / 실제 decoder PASS / mapping 후보 / verified-native full-target / mapping
component` 순서다. density와 P가 없으므로 canonical qualified component는 모든 cell에서 0이다.

| Split | Family | Rows | Present | Mapping | Native full-target | Mapping components |
|---|---|---:|---:|---:|---:|---:|
| train | speech | 2,052 | 2,052 | 2,052 | 0 | 36 |
| train | music | 7,099 | 10 | 10 | 0 | 9 |
| train | environment | 1,335 | 11 | 11 | 3 | 9 |
| train | machine | 15 | 15 | 15 | 0 | 7 |
| val | speech | 118 | 118 | 118 | 0 | 32 |
| val | music | 398 | 4 | 4 | 0 | 4 |
| val | environment | 78 | 4 | 4 | 0 | 4 |
| val | machine | 8 | 8 | 8 | 0 | 4 |
| test | speech | 118 | 118 | 118 | 0 | 35 |
| test | music | 398 | 4 | 4 | 0 | 4 |
| test | environment | 80 | 6 | 6 | 0 | 4 |
| test | machine | 7 | 7 | 7 | 0 | 4 |

music/machine/recorded 숫자가 있어도 대부분 48 kHz processed composite의 availability다.
이를 native highband source나 독립 component의 canonical PASS로 읽으면 안 된다.

## 7. causal P와 발행 차단

repository JSON에서 exact
`broadband_population_causal_primary_operator_v3` payload를 검색한 결과는 **0개**다.
CLI에도 causal-P authority를 제공하지 않았으므로 `causal_p_status=MISSING`이다.

현재 strict P NPZ는 Stage-1 150--1600 Hz 자산이며 fullband population-v3 causal P
authority가 아니다. 따라서 availability auditor는 다음을 의도적으로 수행하지 않았다.

- decoded PCM 발행
- causal P 적용 ERR 발행
- physical/objective band density 계산
- population-v3 manifest 또는 training authority 발행
- untouched Level-5 challenge 편입

구조적으로 유효한 operator JSON을 `--causal-p-authority`에 주더라도 external issuer와
`POPULATION_V3_AUTHORITY`가 없으면 `STRUCTURAL_ONLY/BLOCKED`다.

## 8. 구현과 검증

구현 파일:

- `src/deep_anc/data/broadband_population_availability_v3.py`
- `scripts/data/audit_broadband_population_v3_availability.py`
- `tests/test_broadband_population_availability_v3.py`

focused 회귀 명령:

```bash
.venv/bin/python -m pytest -q \
  tests/test_control_band_contract.py \
  tests/test_broadband_population_contract_v3.py \
  tests/test_broadband_population_availability_v3.py \
  tests/test_broadband_source_contract_v2.py \
  tests/test_broadband_batch_sampler.py
```

테스트 fixture는 actual WAV/FLAC bytes, 실제 metadata mapping, missing/decoder/SHA/fs 오류,
unreferenced scan, recorded composite 분리, no-replace CLI, structural-only P와 legacy-v2
비승격을 검사한다. fixture PASS도 실제 source authority로 사용하지 않는다.

## 9. 다음 행동

1. music 원본과 native high-rate machine 원본을 새 storage/Elice 단계에서 복구하되, 복구
   전까지 현재 7,877 music path나 processed machine WAV를 있는 원본으로 세지 않는다.
2. LibriSpeech/FMA/ESC 및 source-pool 관계를 connected component 단위로 다시 split하여
   38개 present cross-split conflict부터 제거한다.
3. 새 manifest에 immutable native size/SHA, full decoder fingerprint/receipt, decoded PCM
   SHA와 native sample-rate evidence를 직접 결속한다.
4. 88.388--11313.708 Hz를 실제로 덮는 persistent causal P authority를 별도로 발급한다.
5. 그 뒤에만 actual P-applied ERR density를 재계산하고 population-v3 manifest auditor를
   실행한다.
6. population/batch coverage가 통과해도 untouched Level-5 challenge는 별도로 보존한다.
