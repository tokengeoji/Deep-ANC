# Recorded Stage-1 82+19 세대 승격 계약

> [!CAUTION]
> 이 문서의 역사적 `highband-coverage-v1`은 **Stage-1의 600--1600 Hz 부족분**을
> 보충하는 82+17 세대다. 2/4/8 kHz 광대역-v2 데이터가 아니며, 이름에 `highband`가
> 있어도 `broadband_point_control_150_11314_v2` coverage 또는 실제 고주파 ANC
> 증거로 승격할 수 없다. 최종 광대역 데이터 계약은
> [docs/18](18_broadband_anc_guardrails.md)과
> `src/deep_anc/data/broadband_coverage_receipt.py`를 따른다.
>
> `stage1-coverage-v2`는 fixed `0.06`과 environment_006 `25.75 s`를 사용했던
> **immutable stale 세대**다. source별 physical cap 계약과 맞지 않으므로 기존
> receipt/plan/raw는 진단 증거로만 보존하고 덮어쓰거나 현행 수집에 재사용하지 않는다.
> 현행 수집 세대는 새 no-replace ID `stage1-coverage-v3-gain012`다. v1 첫 environment 원본의 두 실제
> capture는 하드웨어/xrun/clip 문제가 없었지만 source→ERR 150--600 Hz coherence²가
> `0.872375`, `0.836006`으로 canonical `0.90`에 미달했다. 임계값을 낮추거나 같은
> church-bells 원본을 반복하지 않고, 안정적인 DEMAND DKITCHEN immutable 원본으로
> 교체한다. v1 raw와 progress는 diagnostic evidence로 보존하고 v2가 resume하지 않는다.

## 1. 목적과 현재 상태

기존 `data/recorded` 82세션은 이미 holdout·원본 계보·tree SHA에 묶인 parent다.
추가 녹음을 위해 이 디렉터리나 기존 manifest를 수정하지 않는다. 추가 세션은 별도
root에 수집하고, 검증을 모두 통과한 뒤에만 parent 82 + additions 19 = 101행의 새
generation manifest를 발행한다.

현재 82세션만 존재하는 schema v1은 기존 bytes를 재검증하는 forensic/readiness 계약으로만
유효하다. 현행 `stage1-coverage-v3-gain012` 신규 학습 입력으로는 사용할 수 없다. 19세션이 실제로
녹음되기 전에는 101세션 generation을 만들었다고 간주하지 않으며 readiness도 기존
82세션 증거를 사용한다.

## 2. 권위 경로

| 역할 | 경로 | 불변식 |
|---|---|---|
| parent raw | `data/recorded/` | 기존 82세션 tree bytes 불변 |
| parent manifest | `data/manifests/recorded_regrouped.jsonl` | holdout SHA와 exact 일치 |
| parent holdout | `data/manifests/recorded_holdout.json` | lineage/tree/provenance 외부 anchor |
| addition source plan | `data/source_plans/recorded_additions/<generation-id>.csv` | exact 19행·exact header |
| addition raw | `data/recorded_additions/<generation-id>/` | 별도 root, session 19개 + progress 1개 |
| combined manifest | `data/manifests/recorded_generations/<generation-id>/recorded.jsonl` | parent 82행을 의미 변경 없이 복사 + addition 19행 |
| generation report | `data/manifests/recorded_generations/<generation-id>/generation.json` | 위 모든 현재 bytes에서 재유도 |

구현 authority는 `src/deep_anc/data/recorded_generation.py`다. transfer schema v1/v2
검증은 `src/deep_anc/data/transfer_contract.py`, generation/transfer 발행은 각각
`scripts/data/build_recorded_generation.py`와
`scripts/data/build_elice_transfer_manifest.py`가 담당한다.

## 3. source plan 계약

CSV는 `SOURCE_PLAN_FIELDS`와 열 순서까지 같아야 하며, 정확히 19행이어야 한다.
family 구성은 Stage-1 coverage 보충과 current-plant train anchor를 함께 만족하도록
speech 5, music 5, environment 5,
machine 4로 고정한다. 모든 행은 녹음 전에 split을 지정하고, lineage component와 원본
SHA가 서로 독립적이어야 한다.

현행 generation-id는 `stage1-coverage-v3-gain012`다. canonical 구성은 environment/music
source-pool 9행, immutable DEMAND environment 1행, external DNS speech 5행, external
ESC machine 4행이다. DNS와 DEMAND selection receipt가 없거나 두 외부 전달 receipt
SHA 중 하나라도 없으면 source plan은 명시적으로 BLOCKED된다. bounded physical
gain-linearity PASS receipt와 외부 SHA도 필수이며, 그 receipt가 허용한 최대값(현재
`0.012` 이하)에서 exact 19행 모두가 feasible일 때만 plan을 발행한다.

| family | path | start | split |
|---|---|---:|---|
| environment | `data/source_pool/environment/environment_006.wav` | 42.0 | train |
| environment | `data/source_pool_v2/environment/environment_014.wav` | 30.0 | val |
| environment | `data/source_pool/environment/environment_003.wav` | 44.5 | test |
| environment | `data/source_pool/environment/environment_008.wav` | 53.25 | test |
| environment | immutable origin `.../origin-environment-demand-dkitchen-ch01-f7e2a2868219.wav` 185.6--200.6 s → exact peak-normalized composite `.../environment-demand-dkitchen-ch01-f7e2a2868219-185600ms-peaknorm.wav` | composite 0.0 | test |
| music | `data/source_pool/music/music_007.wav` | 54.8 | test |
| music | `data/source_pool_v2/music/music_007.wav` | 43.0 | test |
| music | `data/source_pool_v2/music/music_012.wav` | 17.1 | val |
| music | `data/source_pool_v2/music/music_017.wav` | 20.1 | val |
| music | `data/source_pool_v2/music/music_008.wav` | 31.5 | train |

CSV의 `group_id`나 `lineage_key` 문자열은 권위 정보가 아니다. validator는 기존 두
source pool 160행과 FMA `tracks.csv`, LibriSpeech `CHAPTERS.TXT`, ESC-50
`esc50.csv`에서 component를 다시 계산한다. 임의의 새 문자열로 parent 82 원본을
위장하면 실패한다.

허용 source 종류는 다음과 같다.

| `source_kind` | 용도 | 재검증 |
|---|---|---|
| `source_pool_row` | metadata DSU상 parent와 분리된 기존 pool row | family/group/component, source SHA, identity transform |
| `external_demand_environment_file` | 안정적인 지속성 환경음으로 실패한 v1 church-bells 행을 대체 | exclusion 전 96행 DEMAND manifest, exact clean commit/bootstrap/freeze/holdout, DKITCHEN 16채널 public group, full 300초 원본 SHA, origin 185.6--200.6 s→15초 peak-normalized composite bytes 재유도, strict-P 네 부대역, rendered level/SNR |
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

DEMAND DKITCHEN의 public manifest split은 `train`이지만 recorded split은 coverage가
부족한 `test`다. 이는 split 문자열을 바꿔 누수를 숨기는 것이 아니다. 선택 당시의 exclusion
전 96행 manifest와 `manifest_generation.json`, bootstrap receipt, environment freeze,
parent82 holdout, full 300초 source와 실제 재생할 15초 composite를 별도 no-replace bundle에
복사한다. full origin→composite transform은
`mono_48000_pcm16_window_peak_normalize_720000/v1`로 고정하고, validator가 origin
185.6--200.6초 PCM에서 composite bytes를 다시 만들어 exact 일치를 검사한다.
source path는 일반 basename `ch01.wav`를 쓰지 않고 content-addressed 고유 basename을 쓴다. 이후 public
producer가 DKITCHEN public group의 16채널만 제거해 DEMAND 96→80행, 나머지 5개 환경은
보존하고 교집합 0임을 확인하기 전에는 pilot/pretrain도 fail-closed한다.

원본 300초 WAV를 그대로 `NoiseProgram(file)`에 넣으면 150.652초의 full-file
global peak가 amplitude 정규화 기준이 되어, 선택한 창의 실제 재생은 RMS
`-63.85 dBFS`, 150--1600 Hz `-78.39 dBFS`까지 내려간다. 이는 strict
capture gate를 통과시킬 실효 excitation이 아니다. composite를 amplitude `0.06`으로
렌더한 canonical 수치는 peak `0.059999999` (`-24.437 dBFS`), RMS
`-43.490 dBFS`, 150--1600 Hz Hann band RMS `-56.665 dBFS`다. 공식 level-meter
playback `-56.487 dBFS`보다 `0.177 dB` 낮고, meter 하한 `-52.1 dBFS`와
보수적 ERR quiet ceiling `-64.0 dBFS`를 결합한 predicted signal-to-quiet은
`11.723 dB`로 coherence² `0.90`의 이론적 SNR `9.542 dB`를 넘는다. 이 수치와
안전 peak/RMS 범위를 receipt에 결속하며, 이 게이트는 물리 level meter를 대체하지
않는다. DEMAND bundle은 receipt+bootstrap+freeze+generation+manifest+holdout+full
origin+composite의 exact 8파일이다.

두 selector는 격리된 `-I -S -B` import path에서 exact commit, bootstrap/freeze,
pre-exclusion manifest generation을 검증하고 시작·종료 clean-source evidence가 같을 때만
발행한다. 그중 DNS selector는 발행 직전 bootstrap receipt의 commit과 현재 HEAD뿐 아니라 HEAD tree,
index stage/mode/blob, 실제 tracked worktree bytes, assume-unchanged/skip-worktree flag,
replace/graft, non-ignored untracked 파일을 모두 검사한다. `src/`, `scripts/`, `configs/`
아래 ignored untracked injection도 거부한다. `data/`, `results/`, `runs/`처럼 Git ignore
정책 밖의 artifact 출력은 허용하며 이 clean-source evidence 자체를 selection receipt에
봉인한다. DNS issuer는 import 전부터 `-I -S -B` 및
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
special file, `.pyc` 외 member는 격리하지 않고 즉시 차단한다. DNS manifest는 원래
경로/size/SHA/mode/source commit을 봉인하고 중단 transaction을 재실행에서 복구한다.

DEMAND selector는 현재 DNS의 pre-import cache quarantine과 NumPy/SoundFile/SciPy native
loader inventory 전체를 복제하지 않는다. 대신 canonical isolated sys.path, exact clean
commit, bootstrap/freeze, immutable 96행 manifest·generation·parent82·strict P·full source를
발행과 소비 시점에 재검산한다. 따라서 DNS 전용 runtime provenance를 DEMAND receipt가
제공한다고 주장하지 않는다.

DNS selector는 전체 raw를 스캔하지만 Jetson에는 선택되지 않은 DNS raw를 복제하지 않는다.
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
    --generation-id stage1-coverage-v3-gain012 \
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

같은 clean exact checkout에서 DEMAND DKITCHEN bundle도 exclusion 전
`canonical_v4` generation에 결속해 발행한다. 이 selector는 full 300초 source와
96행 parent manifest를 immutable bundle에 복사할 뿐 오디오 장치를 열지 않는다.
`<manifest_generation_SHA256>`는 발행 직전
`data/manifests/canonical_v4/manifest_generation.json`의 외부 전달 SHA다.

```bash
.venv/bin/python -I -S -B \
  -X pycache_prefix=/dev/null/deep-anc-selector \
  scripts/data/select_recorded_demand_environment.py \
  --expected-commit <40자리_SHA> \
  --bootstrap-receipt-sha256 <Elice_bootstrap_receipt_SHA256> \
  --expected-manifest-generation-sha256 <manifest_generation_SHA256> \
  --write
```

DEMAND bundle의 receipt SHA도 stdout와 다른 전달 채널에서 고정한다. Jetson 전송 뒤에는
그 외부 SHA를 다시 제공하지 않으면 기존 bundle을 검증할 수 없다.

```bash
.venv/bin/python -I -S -B \
  -X pycache_prefix=/dev/null/deep-anc-selector \
  scripts/data/select_recorded_demand_environment.py \
  --expected-commit <40자리_SHA> \
  --bootstrap-receipt-sha256 <Elice_bootstrap_receipt_SHA256> \
  --expected-manifest-generation-sha256 <manifest_generation_SHA256> \
  --expected-receipt-sha256 <Elice_DEMAND_receipt_SHA256> \
  --verify-existing
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

출력한 selection receipt 파일 SHA를 별도 채널로 Jetson에 전달한다. exact 19행 plan은
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
  --generation-id stage1-coverage-v3-gain012 \
  --dns-selection-receipt-sha256 <Elice_selection_receipt_SHA256> \
  --demand-selection-receipt-sha256 <Elice_DEMAND_receipt_SHA256> \
  --gain-linearity-receipt results/recording_gain_linearity/<capture>/receipt.json \
  --gain-linearity-receipt-sha256 <Jetson_gain_linearity_receipt_SHA256> \
  --check-only
```

receipt/bundle이 아직 없으면 위 명령의 nonzero `BLOCKED`가 정상이다. check-only가 0으로
끝난 뒤 같은 외부 SHA로 `--write`, `--verify-existing`을 각각 실행한다. 발행된 실제
source plan은 다음 무음 dry-run으로 검사한다. 이 명령은 파일을 만들거나 오디오 장치를
열지 않는다.

```bash
.venv/bin/python scripts/data/record_session_batch.py \
  --sources data/source_plans/recorded_additions/stage1-coverage-v3-gain012.csv \
  --out-root data/recorded_additions/stage1-coverage-v3-gain012 \
  --canonical-additions-generation stage1-coverage-v3-gain012 \
  --amplitude 0.06 \
  --dry-run
```

위 `--amplitude 0.06`은 v1 CLI 호환을 위한 **unused legacy sentinel**이다. canonical
v2 child 명령의 실제 출력은 PASS source-gain plan의 행별 값만 사용하고 모두
`0.012` 이하임을 parent와 child가 다시 검사한다. sentinel을 실제 출력 레벨로
해석하거나 plan 없이 직접 canonical root에 녹음할 수 없다.

이 pre-campaign dry-run은 source/lineage/plan/기존 session을 전부 읽되 오디오 장치와 fresh
campaign만 열지 않는다. 실제 실행은 아래처럼 방금 발행한 campaign의 외부 SHA를 반드시 함께
지정해야 하며, dry-run 성공만으로 live admission이 열리지 않는다.

20초 official meter가 PASS하고 출력 stream이 닫힌 직후 그 raw와 canonical sibling receipt를
다음처럼 묶는다. 첫 명령은 무출력 check-only이며, 같은 bytes가 PASS한 뒤에만 `--write`를
붙여 no-replace 발행한다. stdout의 `receipt_path`와 `receipt_sha256`을 live 명령에 그대로 쓴다.

```bash
.venv/bin/python scripts/data/issue_recording_level_campaign.py \
  --meter-raw results/calibration_interleaved/level_bootstrap/<capture>/meter_raw.npz \
  --meter-receipt results/calibration_interleaved/level_bootstrap/<capture>/meter_raw.receipt.json

.venv/bin/python scripts/data/issue_recording_level_campaign.py \
  --meter-raw results/calibration_interleaved/level_bootstrap/<capture>/meter_raw.npz \
  --meter-receipt results/calibration_interleaved/level_bootstrap/<capture>/meter_raw.receipt.json \
  --write
```

19행이 모두 15초라면 audible 합계는 285초(4분 45초)다. 실제 연결 시간은 dry-run이
출력한 input-only preflight·settle 포함 상한을 따른다. 실제 실행은 사용자 입회,
볼륨 최소, 배선/덕트 geometry 확인, 오디오 장치 무점유 확인 뒤 별도 승인된 연결
창에서만 한다. 실패 세션은 자동 재시도하지 않고 failure evidence를 먼저 분석한다.
`0.06`은 과거 82세션과 legacy additions 계획의 file playback 기준값이지, 현재
canonical additions의 source-independent live authority가 아니다. 2026-08-31 strict-P
재계산에서 source별 예측 peak가 약 22.82 dB 범위로 달라 fixed `0.06` batch를 차단했다.
canonical live는 source별 gain plan과 REF 상한 및 다중레벨 선형성 receipt를 결속해야 한다.
schema-v1 plan은 ERR-only 무출력 계산이므로 `canonical_live_eligible=false`이며,
아래 실행은 PASS v2 physical authority가 없으면 audio open 전에 실패해야 한다.
어떤 digital amplitude도 같은 physical SPL이나 과거 82세션과 같은 amplifier gain을
자동으로 증명하지 않는다.
canonical batch/session/generation은 fresh meter raw·receipt를 recording-level campaign
path/SHA로 결속한다. 각 session 시작 시 meter 완료 후 최대 600초 이내인지, 같은 hardware
config/fingerprint와 amplifier setting 확인이 유지되는지 다시 검사한다. 시간이 만료되면 임계값을
늘리지 않고 새 meter/campaign을 발행해 resume한다. 이 결속은 아날로그 노브를 자동 판독하는
것은 아니므로 다른 amplifier setting을 같은 물리 레벨이라고 주장할 수는 없다.
0.15는 안전 상한일 뿐 canonical 수집 레벨이 아니다.

신규 세션은 publish 전에 다음 일곱 조건을 하나의 공용 capture-gate 계약으로 모두
통과해야 한다. 저역 코히런스 하나만으로 성공 처리하지 않는다.

1. `coh²(source_aligned→ERR, 150--600 Hz) >= 0.90`
2. `coh²(source_aligned→ERR, 600--1600 Hz) >= 0.60`
3. `coh²(REF→ERR, 150--600 Hz) >= 0.60`
4. source→REF raw valid-window ratio `>= 0.90`
5. source_aligned→ERR valid-window ratio `>= 0.77`
6. 잔여 지연 robust standard deviation `<= 3.41254 samples`
7. 잔여 지연 p95−p5 `<= 48 samples`

실패 조건은 durable `failure.json`에 실제 측정값과 함께 저장되고 active additions에는
발행되지 않는다. batch resume과 최종 101세션 generation도 저장된 `TimelineReport`에서
같은 공용 계약을 다시 계산하여, 수집 시점과 최종 승격 시점의 판정식이 갈라지지 않게 한다.

아래 명령은 과거 fixed-gain 형식을 보존한 예시다. 현 canonical 실행에는 추가로
`--source-gain-plan`과 외부 SHA가 필요하며, schema-v1 plan을 주어도 live는 열리지 않는다.

```bash
.venv/bin/python scripts/data/record_session_batch.py \
  --sources data/source_plans/recorded_additions/stage1-coverage-v3-gain012.csv \
  --out-root data/recorded_additions/stage1-coverage-v3-gain012 \
  --canonical-additions-generation stage1-coverage-v3-gain012 \
  --amplitude 0.06 \
  --recording-level-campaign results/recording_level_campaigns/<campaign-id>/campaign.json \
  --recording-level-campaign-sha256 <외부_SHA256> \
  --source-gain-plan results/recording_source_gains/<plan-id>.json \
  --source-gain-plan-sha256 <외부_SHA256> \
  --confirm-same-amplifier-setting \
  --confirm-user-present \
  --confirm-volume-minimum \
  --confirm-routing-and-geometry
```

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

19세션과 `batch_progress.csv`가 exact 일치한 뒤 generation을 no-replace 발행한다.
`--allow-missing-source-files` 같은 publish 우회는 제공하지 않는다.

```bash
.venv/bin/python scripts/data/build_recorded_generation.py \
  --generation-id stage1-coverage-v3-gain012 \
  --expected-holdout-sha256 <recorded_holdout_sha256>
```

generation 뒤에는 clean exact commit에서 old82 train split로 발행한
`data/manifests/recorded_level_calibration/<commit>.json`도 필요하다. 이 receipt는 WAV를
수정하지 않고 historical ERR만 current strict-P 단위로 보정하며, val/test는 fit이 아니라
사전 고정 품질 gate 진단에만 쓴다.

그 다음 Jetson에서 source raw가 아직 존재할 때 transfer schema v2를 발행한다.
schema v2는 parent manifest, generation report, source plan, combined manifest, parent와
addition raw, recording-level campaign/meter, historical level calibration receipt exact 집합을
모두 포함한다. Elice의 validator는 외부 전달된 transfer SHA와
bootstrap receipt를 anchor로 삼고, config의 `data.recorded_generation` 경로와
`data.recorded_generation_sha256`도 exact 비교한다.

schema v2 bootstrap은 검증된 generation report의 path/SHA를
`prepare_noise_pool.py --recorded-generation ...
--expected-recorded-generation-sha256 ...`로 전달한다. prepare는 source plan의 19행에서
source/output SHA, external raw-member SHA, raw lineage와 authority component를 재유도해
`manifest_generation.json.recorded_generation_exclusion`에 결속한다. public 6종은
basename뿐 아니라 content SHA와 lineage component 단위로 제외하며, readiness의
`corpus_disjoint`가 transfer가 검증한 같은 generation인지와 최종 6종 교집합 0을 다시
계산한다. 따라서 parent82 holdout 밖 external raw를 sidecar에서 생략하거나 다른
generation의 exclusion으로 바꾸면 학습 전에 실패한다.

기존 canonical transfer가 82세션 schema v1이면 파일을 지우거나 덮어쓰지 않는다.
검증된 기존 SHA를 명시하여 content-addressed history를 먼저 만들고, 완성된 101세션
schema v2만 원자 교체한다. 전체 인자는 `docs/05_training_elice.md`의 canonical 명령을
byte 그대로 사용한다. 그 완전한 명령에서 `--recorded-generation`은
`data/manifests/recorded_generations/stage1-coverage-v3-gain012/generation.json`,
`--rotate-existing-transfer-sha256`은 검증 직전 canonical schema v1의 실제 SHA-256이어야 한다.
설명용 placeholder를 bash에 복사해 실행하지 않는다.

builder는 기존 v1 전체와 임시 v2 전체를 각각 검증한 뒤 `os.replace`한다. 어느 검증이든
실패하면 canonical v1은 그대로 남고 v2를 학습 입력으로 주장할 수 없다.

## 5. 실패 조건과 lifecycle

다음 중 하나라도 발생하면 generation/transfer/readiness는 FAIL이다.

- parent 82 파일·manifest·holdout·provenance의 1 byte 변경
- addition 19 중 누락·교체·중복 source row 또는 session
- source plan header/count/family composition 변경
- preassigned split, start, source SHA, session metadata 불일치
- pool metadata DSU 또는 external metadata component가 parent active component와 겹침
- external transform output/authority metadata SHA 불일치
- session WAV format/frame/channel 또는 session artifact SHA 불일치
- combined manifest에서 parent 의미 변경 또는 101행 exact 집합 불일치
- generation report/transfer/config SHA trust-chain 불일치
- schema v2 generation exclusion sidecar 누락 또는 public 6종과 source/raw SHA·lineage 교집합

새 녹음 세대를 만들 때 기존 generation 파일을 덮어쓰지 않는다. 새 generation-id,
새 source plan, 새 addition root, 새 report 경로를 사용하고 config/transfer receipt를 새
SHA로 함께 전환한다. 구형 82-only schema v1과 과거 generation은 진단 증거로 보존한다.
