# Recorded 82+17 세대 승격 계약

> [!CAUTION]
> 이 문서의 역사적 `highband-coverage-v1`은 **Stage-1의 600--1600 Hz 부족분**을
> 보충하는 82+17 세대다. 2/4/8 kHz 광대역-v2 데이터가 아니며, 이름에 `highband`가
> 있어도 `broadband_point_control_150_11314_v2` coverage 또는 실제 고주파 ANC
> 증거로 승격할 수 없다. 최종 광대역 데이터 계약은
> [docs/18](18_broadband_anc_guardrails.md)과
> `src/deep_anc/data/broadband_coverage_receipt.py`를 따른다.

## 1. 목적과 현재 상태

기존 `data/recorded` 82세션은 이미 holdout·원본 계보·tree SHA에 묶인 parent다.
추가 녹음을 위해 이 디렉터리나 기존 manifest를 수정하지 않는다. 추가 세션은 별도
root에 수집하고, 검증을 모두 통과한 뒤에만 parent 82 + additions 17 = 99행의 새
generation manifest를 발행한다.

현재 82세션만 존재하는 상태는 계속 유효한 transfer schema v1이다. 17세션이 실제로
녹음되기 전에는 99세션 generation을 만들었다고 간주하지 않으며 readiness도 기존
82세션 증거를 사용한다.

## 2. 권위 경로

| 역할 | 경로 | 불변식 |
|---|---|---|
| parent raw | `data/recorded/` | 기존 82세션 tree bytes 불변 |
| parent manifest | `data/manifests/recorded_regrouped.jsonl` | holdout SHA와 exact 일치 |
| parent holdout | `data/manifests/recorded_holdout.json` | lineage/tree/provenance 외부 anchor |
| addition source plan | `data/source_plans/recorded_additions/<generation-id>.csv` | exact 17행·exact header |
| addition raw | `data/recorded_additions/<generation-id>/` | 별도 root, session 17개 + progress 1개 |
| combined manifest | `data/manifests/recorded_generations/<generation-id>/recorded.jsonl` | parent 82행을 의미 변경 없이 복사 + addition 17행 |
| generation report | `data/manifests/recorded_generations/<generation-id>/generation.json` | 위 모든 현재 bytes에서 재유도 |

구현 authority는 `src/deep_anc/data/recorded_generation.py`다. transfer schema v1/v2
검증은 `src/deep_anc/data/transfer_contract.py`, generation/transfer 발행은 각각
`scripts/data/build_recorded_generation.py`와
`scripts/data/build_elice_transfer_manifest.py`가 담당한다.

## 3. source plan 계약

CSV는 `SOURCE_PLAN_FIELDS`와 열 순서까지 같아야 하며, 정확히 17행이어야 한다.
family 구성은 Stage-1 600--1600 Hz coverage 보충 목적에 맞춰 speech 5, music 4, environment 4,
machine 4로 고정한다. 모든 행은 녹음 전에 split을 지정하고, lineage component와 원본
SHA가 서로 독립적이어야 한다.

현행 generation-id는 `highband-coverage-v1`이다. canonical 구성은 environment/music
source-pool 8행, external DNS speech 5행, external ESC machine 4행이다. DNS selection
receipt가 없거나 외부 전달 receipt SHA가 없으면 source plan은 명시적으로 BLOCKED된다.

| family | path | start | split |
|---|---|---:|---|
| environment | `data/source_pool/environment/environment_008.wav` | 54.1 | test |
| environment | `data/source_pool_v2/environment/environment_012.wav` | 3.0 | test |
| environment | `data/source_pool_v2/environment/environment_004.wav` | 5.9 | test |
| environment | `data/source_pool_v2/environment/environment_017.wav` | 26.2 | val |
| music | `data/source_pool/music/music_007.wav` | 54.8 | test |
| music | `data/source_pool_v2/music/music_007.wav` | 12.8 | test |
| music | `data/source_pool_v2/music/music_012.wav` | 17.1 | val |
| music | `data/source_pool_v2/music/music_017.wav` | 20.1 | val |

CSV의 `group_id`나 `lineage_key` 문자열은 권위 정보가 아니다. validator는 기존 두
source pool 160행과 FMA `tracks.csv`, LibriSpeech `CHAPTERS.TXT`, ESC-50
`esc50.csv`에서 component를 다시 계산한다. 임의의 새 문자열로 parent 82 원본을
위장하면 실패한다.

허용 source 종류는 다음과 같다.

| `source_kind` | 용도 | 재검증 |
|---|---|---|
| `source_pool_row` | metadata DSU상 parent와 분리된 기존 pool row | family/group/component, source SHA, identity transform |
| `external_exact_composite` | source pool에 free machine component가 없을 때 쓰는 untouched ESC-50 5초 raw | ESC `src_file`, raw SHA, metadata SHA, 48 kHz Kaiser-5 resample, PCM16, 3회 반복한 15초 output SHA |
| `external_dns_speech_composite` | Elice schema-v4 speech에서 선택한 DNS component | immutable manifest/bootstrap receipt, source/raw/composite SHA, public group/reader/book alias, strict-P 네 부대역, 10.333초 PCM16 raw→15초 repeat-trim |
| `external_librispeech_file` | 향후 후보 감사용 지원 타입이며 현행 exact plan에는 0행 | raw path/SHA, CHAPTERS SHA, duration, reader↔book 전체 그래프의 transitive component |

LibriSpeech는 후보 파일의 reader/book 두 키만 직접 비교하지 않는다. CHAPTERS 전체에서
reader와 book을 bipartite graph로 묶는다. 후보 reader가 다른 book을 거쳐 active
reader와 연결되는 경우도 같은 component이므로 거부한다.

외부 ESC-50 composite 생성기는 output을 덮어쓰지 않는다.

```bash
.venv/bin/python scripts/data/build_recorded_external_composite.py \
  --generation-id <generation-id> \
  --raw-member data/raw/noise/esc50/ESC-50-master/audio/<raw.wav> \
  --family machine \
  --out-name <output.wav>
```

machine raw/split은 `1-28808-A-43.wav`→train,
`5-235507-A-44.wav`→val, `4-102871-A-42.wav`→test,
`5-222524-A-41.wav`→test로 고정한다. 실제 plan에는 도구가 출력한
path/SHA/lineage/transform 값을 그대로 넣어야 한다. 파일이 로컬에 없거나 SHA가
다르면 발행을 중단한다.

초기 external LibriSpeech 후보 네 개는 파일 단위 직접 비교에서는 free처럼 보였지만
권위 `CHAPTERS.TXT` 전체 전이 DSU를 적용하면 모두 parent82의 active component
`speech-librispeech-lineage-d697786cc484`에 연결된다. 아래 항목은 **선택 목록이 아니라
탈락 증거**이며 source plan에 넣으면 validator가 거부한다.

| 파일 | `start_seconds` | window |
|---|---:|---:|
| `2035-152373-0013.flac` | 3.00 | 15초 / train |
| `1272-128104-0004.flac` | 0.75 | 15초 / train |
| `6241-61943-0027.flac` | 0.50 | 15초 / test |
| `2412-153948-0006.flac` | 0.25 | 15초 / test |

대체 speech 다섯 행은 full source-pool 160행을 strict P로 필터링한 뒤, 15초 창 안의
결정론적 1.5초 비중첩 구간 9개(`start+0.25+1.5*k`)에서 네 부대역 energy-density를
검사해 PSD 후보로 골랐다. 임계값은 0.25이며 낮추지 않았다. 그러나 source-pool row
component만 보면 서로 달라도 각 row의 Libri reader/book identity를 권위
`CHAPTERS.TXT` 전체 그래프로 확장하면 일부 후보가
`speech-librispeech-lineage-d697786cc484`를 반복 공유하고 그 component는 parent82에도
active다. 따라서 후보 집합이 독립 group 5개를 만들지 못하며 canonical plan에서는
다섯 행 전부 rejected diagnostic evidence로만 보존한다.

Elice selector는 exact-commit bootstrap과 schema-v4 generation을 먼저 검증한다. DNS
reader/book와 보수적 numeric alias가 parent82와 겹치지 않는 public group만 남기고,
strict P 적용 뒤 네 부대역을 모두 cover하는 서로 다른 group의 global top-5를 고른다.
public manifest의 기존 split은 provenance로만 보존하고, recorded split은 선택 품질과
독립적으로 `train, train, val, test, test`를 결정론적으로 배정한다. 선택 group 전체는
이후 synthetic 모든 split에서 제외한다.

selector 발행 직전에는 bootstrap receipt의 commit과 현재 HEAD뿐 아니라 HEAD tree,
index stage/mode/blob, 실제 tracked worktree bytes, assume-unchanged/skip-worktree flag,
replace/graft, non-ignored untracked 파일을 모두 검사한다. `src/`, `scripts/`, `configs/`
아래 ignored untracked injection도 거부한다. `data/`, `results/`, `runs/`처럼 Git ignore
정책 밖의 artifact 출력은 허용하며 이 clean-source evidence 자체를 selection receipt에
봉인한다. issuer는 import 전부터 `-I -S -B` 및
`-X pycache_prefix=/dev/null/deep-anc-selector`를 강제해 `PYTHONPATH`, user-site,
`.pth`, adjacent/venv `__pycache__` 재사용을 거부한다. live interpreter/sys.path,
source/native loader origin, `__cached__`, 실제 로드된 NumPy FFT/native-extension bytes,
SoundFile이 dlopen한 `libsndfile_*.so` bytes/version, SciPy module provenance와 bootstrap
environment-freeze SHA를 scan 전·후와 publish 직전·직후에
동일하게 재검증한다. DNS coverage convolution은 SciPy 함수를 호출하지 않는 repository
NumPy power-of-two FFT 구현으로 고정한다. Jetson validator/readiness도 tracked source/index와 executable
untracked injection을 다시 계산한다. 기존 Elice checkout의 protected root에 남은
regular `__pycache__/*.pyc`는 삭제하지 않고 import 전에 repository 밖
`.deep_anc_source_cache_quarantine/`으로 no-overwrite 원자 이동한다. symlink,
special file, `.pyc` 외 member는 격리하지 않고 즉시 차단한다. manifest는 원래
경로/size/SHA/mode/source commit을 봉인하고 중단 transaction을 재실행에서 복구한다.

selector는 전체 raw를 스캔하지만 Jetson에는 선택되지 않은 DNS raw를 복제하지 않는다.
Jetson validator는 immutable manifest 전체 lineage와 선택 raw/composite/strict-P 점수를
재계산한다. full-scan ranking은 bootstrap-bound receipt와 외부 receipt 파일 SHA로
검증하며, 최종 권위 coverage는 실제 덕트 additions를 포함해 다시 생성한 recorded
subband report다. 각 후보는 scan 시 manifest content SHA/size와 대조되고 top-5는 선택 뒤
원본을 다시 열어 raw/composite를 만든다. 따라서 외부 receipt SHA와 clean-source evidence가
유지되는 동안 local에서 선택되지 않은 raw 전체를 다시 복제·재스캔하지 않는다. 외부 SHA를
잃었거나 clean-source evidence가 다르면 정적 metrics를 신뢰하지 않고 selector 전체를 exact
Elice raw에서 다시 실행한다.

## 4. 수집과 발행 순서

먼저 ESC-50 raw 4개를 exact output 이름으로 composite화한다. 아래 명령은 파일
변환만 수행하며 오디오 장치를 열지 않고, 기존 bytes가 다르면 overwrite하지 않는다.

```bash
for spec in \
  '1-28808-A-43.wav machine-28808-repeat3.wav' \
  '5-235507-A-44.wav machine-235507-repeat3.wav' \
  '4-102871-A-42.wav machine-102871-repeat3.wav' \
  '5-222524-A-41.wav machine-222524-repeat3.wav'; do
  set -- $spec
  .venv/bin/python scripts/data/build_recorded_external_composite.py \
    --generation-id highband-coverage-v1 \
    --raw-member "data/raw/noise/esc50/ESC-50-master/audio/$1" \
    --family machine --out-name "$2" || exit 1
done
```

Elice에서 먼저 selector를 실행한다. 이 명령은 저장된 파일만 읽고 오디오 장치를 열지
않는다. `--write`는 `renameat2(RENAME_NOREPLACE)`와 fsync로 receipt/raw/composite bundle을
원자 발행한다.

```bash
.venv/bin/python -I -S -B \
  -X pycache_prefix=/dev/null/deep-anc-selector \
  scripts/data/select_recorded_dns_speech.py \
  --expected-commit <40자리_SHA> \
  --bootstrap-receipt-sha256 <Elice_bootstrap_receipt_SHA256> \
  --write
```

retained cache를 명시적으로 돌려놓아야 할 때만 selector가 출력한 absolute
transaction path를 사용한다. 복구는 기존 cache를 overwrite하지 않으며,
같은 transaction을 두 번 복구하면 실패한다.

```bash
.venv/bin/python -I -S -B \
  -X pycache_prefix=/dev/null/deep-anc-selector \
  scripts/data/select_recorded_dns_speech.py \
  --expected-commit <40자리_SHA> \
  --restore-source-cache-quarantine </absolute/transaction/path>
```

출력한 selection receipt 파일 SHA를 별도 채널로 Jetson에 전달한다. exact 17행 plan은
그 외부 anchor를 필수 인자로 받아 receipt와 plan의 동시 재봉인을 막는다.
기존 bundle 검증도 commit/bootstrap/selection receipt 외부 anchor 세 개를 모두
다시 제공해야 한다.

```bash
.venv/bin/python -I -S -B \
  -X pycache_prefix=/dev/null/deep-anc-selector \
  scripts/data/select_recorded_dns_speech.py \
  --expected-commit <40자리_SHA> \
  --bootstrap-receipt-sha256 <Elice_bootstrap_receipt_SHA256> \
  --receipt-sha256 <Elice_selection_receipt_SHA256> \
  --verify-existing
```

no-replace publish 후 source/runtime/receipt 검증이 실패하면 방금 게시한
writer-owned inode만 `.publish-failures/`에 원자 격리하고 failure receipt를 남긴다.
canonical 경로는 비워져 원인 수정 후 새 no-replace 실행이 가능하며, 경쟁에서
게시권을 얻지 못한 process는 다른 process의 bundle을 격리할 수 없다.

```bash
.venv/bin/python scripts/data/build_recorded_additions_plan.py \
  --generation-id highband-coverage-v1 \
  --dns-selection-receipt-sha256 <Elice_selection_receipt_SHA256> \
  --check-only
```

receipt/bundle이 아직 없으면 위 명령의 nonzero `BLOCKED`가 정상이다. check-only가 0으로
끝난 뒤 같은 외부 SHA로 `--write`, `--verify-existing`을 각각 실행한다. 발행된 실제
source plan은 다음 무음 dry-run으로 검사한다. 이 명령은 파일을 만들거나 오디오 장치를
열지 않는다.

```bash
.venv/bin/python scripts/data/record_session_batch.py \
  --sources data/source_plans/recorded_additions/highband-coverage-v1.csv \
  --out-root data/recorded_additions/highband-coverage-v1 \
  --canonical-additions-generation highband-coverage-v1 \
  --dry-run
```

17행이 모두 15초라면 audible 합계는 255초(4분 15초)다. 실제 연결 시간은 dry-run이
출력한 input-only preflight·settle 포함 상한을 따른다. 실제 실행은 사용자 입회,
볼륨 최소, 배선/덕트 geometry 확인, 오디오 장치 무점유 확인 뒤 별도 승인된 연결
창에서만 한다. 실패 세션은 자동 재시도하지 않고 failure evidence를 먼저 분석한다.

각 성공 session은 다음 exact 집합이어야 한다.

- `mics.wav`: 48 kHz, 2ch, PCM_32, 계획된 frame 수
- `source.wav`: 48 kHz, mono, FLOAT, 계획된 frame 수
- `source_aligned.wav`: 48 kHz, mono, FLOAT, 계획된 frame 수
- `session.json`: source-list path/SHA/row, source SHA, start, family, group,
  lineage, preassigned split, 세 WAV의 size/SHA

canonical resume은 `session.json` 존재만으로 완료 처리하지 않는다. exact plan과
`source.wav`를 source bytes/start/amplitude에서 재생성해 수치 일치시키고, 세 WAV
format/content/SHA와 `batch_progress.csv` PASS 행의 `seconds`까지 source plan과 모두
맞을 때만 완료로 센다.
symlink out-root나 불완전 session은 소리를 내기 전에 실패한다.

canonical 수집 자식은 직접 addition root에 발행하지 않는다. 세션별 staging root에서
record_duct 내부 검증과 batch QA/exact artifact 검증을 모두 통과한 뒤에만
`renameat2(RENAME_NOREPLACE)`로 addition root에 원자 발행한다. QA 실패 raw는
`results/recording_failures/record_duct/batch_qa/<generation-id>/`에 no-replace 보존하고
즉시 nonzero로 끝내며 다음 소리를 자동 재생하지 않는다. 과거 실패 세션이 canonical
root에 남은 경우 자동 삭제하지 않고 명시적으로 quarantine하거나 새 generation-id를
사용해야 한다.

17세션과 `batch_progress.csv`가 exact 일치한 뒤 generation을 no-replace 발행한다.
`--allow-missing-source-files` 같은 publish 우회는 제공하지 않는다.

```bash
.venv/bin/python scripts/data/build_recorded_generation.py \
  --generation-id highband-coverage-v1 \
  --expected-holdout-sha256 <recorded_holdout_sha256>
```

그 다음 Jetson에서 source raw가 아직 존재할 때 transfer schema v2를 발행한다.
schema v2는 parent manifest, generation report, source plan, combined manifest, parent와
addition raw exact 집합을 모두 포함한다. Elice의 validator는 외부 전달된 transfer SHA와
bootstrap receipt를 anchor로 삼고, config의 `data.recorded_generation` 경로와
`data.recorded_generation_sha256`도 exact 비교한다.

schema v2 bootstrap은 검증된 generation report의 path/SHA를
`prepare_noise_pool.py --recorded-generation ...
--expected-recorded-generation-sha256 ...`로 전달한다. prepare는 source plan의 17행에서
source/output SHA, external raw-member SHA, raw lineage와 authority component를 재유도해
`manifest_generation.json.recorded_generation_exclusion`에 결속한다. public 6종은
basename뿐 아니라 content SHA와 lineage component 단위로 제외하며, readiness의
`corpus_disjoint`가 transfer가 검증한 같은 generation인지와 최종 6종 교집합 0을 다시
계산한다. 따라서 parent82 holdout 밖 external raw를 sidecar에서 생략하거나 다른
generation의 exclusion으로 바꾸면 학습 전에 실패한다.

## 5. 실패 조건과 lifecycle

다음 중 하나라도 발생하면 generation/transfer/readiness는 FAIL이다.

- parent 82 파일·manifest·holdout·provenance의 1 byte 변경
- addition 17 중 누락·교체·중복 source row 또는 session
- source plan header/count/family composition 변경
- preassigned split, start, source SHA, session metadata 불일치
- pool metadata DSU 또는 external metadata component가 parent active component와 겹침
- external transform output/authority metadata SHA 불일치
- session WAV format/frame/channel 또는 session artifact SHA 불일치
- combined manifest에서 parent 의미 변경 또는 99행 exact 집합 불일치
- generation report/transfer/config SHA trust-chain 불일치
- schema v2 generation exclusion sidecar 누락 또는 public 6종과 source/raw SHA·lineage 교집합

새 녹음 세대를 만들 때 기존 generation 파일을 덮어쓰지 않는다. 새 generation-id,
새 source plan, 새 addition root, 새 report 경로를 사용하고 config/transfer receipt를 새
SHA로 함께 전환한다. 구형 82-only schema v1과 과거 generation은 진단 증거로 보존한다.
