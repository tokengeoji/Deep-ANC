# Elice 외부 데이터 staging 규칙

갱신일: 2026-08-31

## 1. 상태와 권위 경계

Jetson의 로컬 여유 공간이 작아도 public corpus는 Elice에서 직접 내려받을 수 있다. Kaggle이나
Google Drive는 **임시 공급원 또는 cache**일 뿐이며, 그곳에 파일이 있다는 사실만으로 canonical
raw나 학습 준비 완료가 되지 않는다. 이 문서는 특정 Elice 인스턴스가 살아 있거나 GPU job이
실행 중이라고 가정하지 않는다.

새 인스턴스의 **유일한 권위 진입점**은 `scripts/elice/bootstrap_all.sh`다. full bootstrap에는
다음 네 항목이 모두 필요하다.

- `--expected-commit <40자리_SHA>`
- `--expected-holdout-sha256 <64자리_SHA256>`
- `--expected-transfer-manifest-sha256 <64자리_SHA256>`
- `--no-update`

폐기된 `scripts/elice/bootstrap.sh`는 `--train`을 포함한 모든 실행에서 `exit 2`로 끝나는
fail-closed stub이다. Drive 복원 뒤에도 이 경로를 환경 준비나 학습 진입점으로 사용하지 않는다.

## 2. 과거 Drive snapshot의 용도

`results/elice_snapshots/predelete_917aa25a0315247f/BACKUP_README.md`가 설명하는 snapshot은
commit `917aa25a0315247f398b111323ec09e49a64d19f` 당시의 **forensic backup/cache**다. 전체
repository, `.git`, data, results와 과거 receipt를 함께 포함하므로 새 exact checkout 위에 통째로
복원하면 안 된다. 특히 다음 항목을 현재 세대의 증거로 복원·승격하지 않는다.

- snapshot의 `.git`, tracked working tree와 `.venv`
- `data/manifests/canonical_v4/`와 `data/manifests/elice_bootstrap_receipt.json`
- schema v1 `data/manifests/elice_transfer_manifest.json`
- 과거 readiness, selector, pilot, run과 checkpoint
- 환경 fingerprint가 다른 decoder audit

과거 snapshot에서 재사용할 수 있는 범위는 검증된 **raw-only cache인 `data/raw/`**뿐이다.
이와 별도로 아래 §3.1의 official public archive cache는 고정 10개 archive의 전송 가속용으로만
사용할 수 있다. recorded
101세션, RIR, strict P/S, holdout/provenance와 transfer schema v2는 현재 Jetson exact commit이
발행한 transfer manifest의 상대경로와 SHA대로 별도 전송한다. 과거 snapshot의 recorded 82세션과
schema v1을 additions 19세션이 포함된 현재 generation처럼 취급하지 않는다.

## 3. Drive 검증 receipt와 삭제 gate

업로드 로그의 `100%`, remote 디렉터리 존재, metadata 일부 비교는 각각 완료 receipt가 아니다.
Drive cache를 쓰기 전에 별도 final **Drive 검증 receipt**에 다음을 봉인한다.

1. remote 이름과 content-addressed 경로
2. 생성 시각, 원본 exact commit과 데이터 generation
3. 일반 파일 count, total bytes, 정렬된 path/size/content-SHA inventory SHA
4. monolithic archive라면 archive size, SHA-256과 provider hash
5. `rclone check`의 differences 0, errors 0과 검증 시각
6. 원격에서 다시 읽은 metadata/inventory bytes의 SHA 일치

predelete archive를 사용하는 경우 BACKUP_README의 size `41,299,005,440 bytes`, SHA-256
`a743fe4a4761b6d743171c94b6366d74fa199bb1b0361585ed27547fa627b994`, provider MD5
`010aea6e045d64662b288cd0185928c7`을 모두 확인한다. 이 세 값의 일치도 archive 내부의 raw가
현재 transfer generation과 맞는다는 뜻은 아니므로, raw 복원 뒤 bootstrap decoder audit과
lineage/holdout 검증을 다시 수행한다.

Elice 인스턴스 삭제와 Drive remote snapshot 삭제는 서로 다른 결정이다. 인스턴스는 위 archive
검증과 metadata readback, 필요한 고유 결과의 별도 보존이 모두 확인된 뒤에만 삭제 후보가 된다.
Drive remote는 후속 snapshot의 전체 cold-restore가 독립 검증되고 사용자가 정확한 삭제 대상을
다시 승인하기 전까지 보존한다. 이 문서의 절차에는 archive 또는 remote 삭제 작업이 없다.

### 3.1 official public archive cache 계약

`scripts/elice/public_archive_cache.py`는 임의 URL·dataset ID·target을 받지 않는다. 정확히
DNS Fullband 3개, DEMAND 6개, MIMII fan 1개만 고정 순서로 다룬다. FMA-small/FMA metadata와
ESC-50은 현재 Drive의 extracted restore 또는 기존 bootstrap 경로를 그대로 사용하며 이 cache에
넣지 않는다. LibriSpeech는 현행 public raw 37,761개 계약의 extra이므로 다운로드·복원 대상이
아니다.

| ID | archive bytes | 공급자 검증 |
|---|---:|---|
| `dns_noise_000` | 5,364,611,964 | Content-Length + strong ETag `0x8D9B5BADC55F6BB` |
| `dns_noise_001` | 5,357,916,291 | Content-Length + strong ETag `0x8D9B5B0DB55DD9B` |
| `dns_speech_000` | 4,664,045,287 | Content-Length + strong ETag `0x8D9B5BA782095C9` |
| `demand_dkitchen` | 336,992,458 | provider MD5 `b4d38241fbd50d8a17f8742ca6870c10` |
| `demand_dwashing` | 306,101,499 | provider MD5 `ecf765f12b8d3ada7ef0ec664b8f8d73` |
| `demand_ooffice` | 277,643,831 | provider MD5 `6f87edf8a6b03f17b3f693af1754aab4` |
| `demand_ohallway` | 252,905,617 | provider MD5 `cb9227a75d2c1342de0b6548da4bbb1b` |
| `demand_tmetro` | 367,513,573 | provider MD5 `00d895020233f94348aafce7140d671f` |
| `demand_tcar` | 373,520,251 | provider MD5 `8550f03e8356d8054ae845e8b4b6c773` |
| `mimii_fan` | 928,511,244 | provider MD5 `a1a9b488934a82426bacc933d87aacde` |

DNS 공급자는 cryptographic archive checksum을 공개하지 않는다. 따라서 ETag를 MD5/SHA로
오인하지 않고 manifest에 `provider_checksum_kind=none`, 고정 ETag, 계산한 content SHA-256을
각각 분리한다. 모든 archive는 size/provider validator 뒤 bzip2 또는 ZIP CRC, traversal,
absolute path, symlink/hardlink/device, duplicate/case-collision, 고정 prefix와 WAV count/bytes를
검사한다. DNS noise aggregate는 WAV 16,000개/15,360,708,826 bytes, speech는
8,065개/8,000,834,860 bytes, DEMAND는 96개/2,764,841,088 bytes, MIMII는
3,600개/1,152,158,400 bytes여야 한다.

clean exact checkout에서 caller가 만든 저장소 밖 staging root와 rclone remote root를 명시한다.
다음 명령은 이 문서를 작성하면서 실행하지 않았으며, 실제 실행 시에도 OAuth/service-account
정보를 인자로 넘기지 않는다. 인증은 기존 rclone config에서만 읽는다.

```bash
test -d /dev/shm
.venv/bin/python -I -B scripts/elice/public_archive_cache.py publish \
  --staging-root /dev/shm \
  --remote-root "gdrive:DeepANC/public_archive_cache/$EXPECTED_COMMIT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --rclone "$(command -v rclone)"
```

publisher는 exact commit에 tracked된 `pget.py`로 archive를 하나씩만 받고 검증한다. 성공한
archive도 manifest-last publish가 끝날 때까지 지우지 않으므로 `/dev/shm` peak는 10개 archive
합계 18,229,762,015 bytes + 고정 512 MiB headroom이다. publisher는 이 전체 공간을 첫
provider 요청/remote write 전에 검사한다. 실패한 `.part`, 완성 archive와 이미
올라간 content-addressed remote object는 덮어쓰거나 삭제하지 않는다. Drive 명령은
`copyto --immutable`, `transfers=1`,
`checkers=1`, `tpslimit=2`, `drive-pacer-min-sleep=500ms`로 고정하고, 각 archive마다
`rclone check --download`와 `rclone cat` SHA-256 readback을 수행한다.

publisher의 exact-source gate는 `git status`만 보지 않는다. replace ref, graft,
assume-unchanged/skip-worktree, HEAD/index/tree/mode와 실제 tracked blob 전체를 기존
`exact_clean_source_evidence()`로 재검증한다. 그 checker 자체, cache entry script와 tracked
`pget.py`는 실행 전에 expected commit의 blob과 직접 비교하며, manifest에는 entry script SHA와
`publisher_pget_sha256`을 함께 봉인한다. bootstrap Git은 PATH/상속 `GIT_*`를 쓰지 않고 regular
non-symlink `/usr/bin/git`과 명시 git-dir/work-tree만 사용한다. 검증 뒤 경로를 다시 여는 대신
expected-commit의 `source_trust.py`/`pget.py` blob bytes 자체를 compile/exec하며, CLI는
`python -I -B`를 강제한다. hidden pget 변조·path swap·다른 downloader path는 허용하지 않는다.

remote 상대경로는
`archives/v1/<id>/bytes_<N>/sha256_<SHA>/<official-name>`이다. 10개가 모두 통과한 뒤에만
`manifests/v1/sha256_<SHA>/archive_cache_manifest.json`을 마지막 immutable object로 발행한다.
stdout의 manifest 상대경로와 SHA-256을 서로 다른 신뢰 채널로 Elice에 전달한다. mutable
`latest` pointer나 remote delete/sync는 없다.

## 4. 권장 cold-start 흐름

### 4.1 exact checkout

SHA 세 개는 Drive snapshot 내부가 아니라 신뢰한 Jetson/GitHub 채널에서 전달한다.

```bash
git clone https://github.com/tokengeoji/Deep-ANC.git Deep_ANC
cd Deep_ANC
git checkout --detach "$EXPECTED_COMMIT"
```

### 4.2 raw-only restore 또는 public 재다운로드 선택

검증 receipt가 있는 별도 raw tree를 cache로 쓸 때만 repository 밖 incoming에 먼저 받는다.
아래 `<verified-raw-root>`는 final receipt가 결속한 정확한 경로여야 한다.

```bash
INCOMING=/mnt/data/deep_anc_incoming/raw
test ! -e "$INCOMING"
mkdir -p "$(dirname "$INCOMING")"
rclone copy "gdrive:DeepANC/<verified-raw-root>" "$INCOMING" \
  --immutable --checkers 8 --transfers 8
rclone check "$INCOMING" "gdrive:DeepANC/<verified-raw-root>" \
  --one-way --download

test ! -e data/raw
mkdir -p data
mv "$INCOMING" data/raw
```

`mv` 전에는 receipt의 count/bytes/inventory SHA와 로컬 incoming을 대조해야 한다. manifest,
receipt, results, runs는 incoming에서 복사하지 않는다. monolithic archive밖에 없거나 raw-only
inventory receipt가 없으면 archive 전체를 `/home/elicer`에 풀지 말고 이 경로를 건너뛴다. 이 경우
아래 full bootstrap이 공식 public source에서 다시 받게 하는 것이 권장 경로다.

### 4.2.1 content-addressed archive cache restore / held-fd consume

archive cache는 remote를 repository에 직접 mount하지 않는다. repository 밖 read-only mount 또는
별도 incoming tree로 정확한 content-addressed layout을 노출하고, manifest SHA를 외부에서
명시한다. 예를 들어 Elice에서 read-only mount를 쓰면 다음과 같다.

```bash
mkdir -p /mnt/deep_anc_archive_cache
rclone mount \
  "gdrive:DeepANC/public_archive_cache/$EXPECTED_COMMIT" \
  /mnt/deep_anc_archive_cache \
  --read-only --vfs-cache-mode off --daemon \
  --transfers 1 --checkers 1 --tpslimit 2 --drive-pacer-min-sleep 500ms

ARCHIVE_CACHE_ROOT=/mnt/deep_anc_archive_cache
ARCHIVE_CACHE_MANIFEST="$ARCHIVE_CACHE_ROOT/$ARCHIVE_CACHE_MANIFEST_RELATIVE"
```

mount 대신 local incoming을 쓰면 `rclone copy --immutable`로 저장소 밖에 받고
`rclone check --one-way --download`까지 마친다. 어느 방식도 cache source에 sidecar나 marker를
쓰지 않는다. 먼저 archive만 no-replace 복원하고 종료하려면 다음 명령을 사용한다.

```bash
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --archive-cache-root "$ARCHIVE_CACHE_ROOT" \
  --archive-cache-manifest "$ARCHIVE_CACHE_MANIFEST" \
  --expected-archive-cache-manifest-sha256 "$ARCHIVE_CACHE_MANIFEST_SHA256" \
  --archive-cache-only \
  --no-update
```

`--archive-cache-only`는 manifest external anchor와 10개 archive의 size/checksum/SHA/member
inventory를 다시 계산하고, 고정 canonical archive filename에 exclusive staging copy+fsync+hardlink
no-replace로 발행한 뒤 끝난다. 이미 exact한 target은 받아들이지만 다른 bytes, symlink 또는
non-regular target은 바꾸지 않고 실패한다. 복사 중 실패한 staging은 forensic evidence로
보존한다. extractor, 37,761 raw audit, decoder audit, six manifest, bootstrap receipt는 모두
발행하지 않으므로 이 성공은 raw/training authority가 아니다.

restore 성공은
`data/raw/noise/.archive_cache_origins/archive_cache_origin.<manifest-sha>.<commit>.json`도
no-replace로 발행한다. 이 receipt는 10개 target의 SHA/size, manifest SHA, exact commit,
entry/pget SHA를 결속하지만 `cache_origin_only_not_official_raw_or_training_authority`일 뿐이다.
다음 full bootstrap에서도 반드시 같은 cache 세 인자와 외부 manifest SHA를 다시 제시해야 한다.
receipt 파일만 남기거나 복사한 것은 external anchor를 대체하지 않는다.

full bootstrap은 restored pathname을 shell `tar`/`unzip`에 넘기지 않는다. 별도 `consume` subprocess가
external SHA로 검증한 manifest를 메모리에 고정하고, cache root를 dirfd로 순회해 10개 archive
inode를 `O_NOFOLLOW`로 모두 연다. 각 descriptor는 size/provider checksum/SHA/member inventory 검증,
decompression/no-replace raw publish, 마지막 SHA/metadata readback과 cache-origin receipt 발행이 끝날
때까지 유지한다. 따라서 검증 뒤 archive pathname rename/replacement는 추출 입력이 될 수 없고,
same-inode 변경은 최종 gate에서 실패한다. raw target도 모든 ancestor fd/identity를 유지해 중간
symlink 교체를 거부하고 existing target은 decompressed bytes가 exact할 때만 재개한다.

manifest는 archive member의 path/size뿐 아니라 각 decompressed member content SHA와 최종 raw
상대경로 projection SHA도 봉인한다. `consume`은 약 27,761개 cache-origin WAV의 descriptor를
RLIMIT_NOFILE 범위 안에서 모두 열어 둔 채 content SHA, inode, link count, mtime/ctime과 named path를
inventory→origin→completion receipt 전후로 반복 대조한다. 첫 raw보다 먼저 immutable intent를 쓰며,
완료 시 per-member `{archive_id,path,size,sha256}` inventory와 completion을 no-replace로 결속한다.
중단되어 raw count만 완성됐거나 marker directory 생성 직후 종료된 경우에도 cache 세 external
anchor가 없는 plain bootstrap은 이를 official raw로 승격하지 않는다.

shell은 consume 직후, decoder audit 직전, decoder audit/prepare 뒤와 최종 receipt 직전에 current
raw를 durable per-member inventory에 다시 결속한다. decoder report의 `relative_path`,
`content_size`, `content_sha256` projection도 external archive manifest에서 유도된 projection과
exact해야 한다. `dns_fullband`/`speech`/`demand`/`machine/fan` 네 cache-owned root의 nofollow
WAV exact-set도 durable inventory와 같아야 하므로 extra WAV를 decoder에 넣어 재봉인하는 방식은
거부된다. 최종
bootstrap receipt는 **schema v3**이며 cache를 쓰지 않았으면 `archive_cache_consumption=null`, 썼으면
manifest/completion/member-inventory/decoder file·semantic·projection SHA를 명시한다. schema v2 receipt는
forensic 호환 입력으로만 읽을 수 있고 새 bootstrap이 발행하지 않는다.

canonical campaign과 direct trainer/ledger issuer는 schema-v3 receipt 내부 문자열만 신뢰하지
않는다. 저장소 밖 external manifest path와 별도 SHA-256 anchor를 resolved data config까지 전달하고,
manifest bytes·publisher source·archive origin·archive별 output-content projection을 재검증한다. 이
external manifest SHA는 experiment contract와 canonical prerequisite에도 남는다. campaign CLI는
project import shadow를 막기 위해 항상 `.venv/bin/python -I -B`로 실행한다.

### 4.3 현재 transfer schema v2 배치

Jetson에서 발행한 `data/manifests/elice_transfer_manifest.json`과 manifest가 열거한 파일을
상대경로 그대로 `rsync`/`scp` 또는 content-addressed Drive bundle로 전송한다. manifest
`schema_version`은 2, selected recorded manifest는 combined 101세션이어야 한다. raw cache보다
나중에 배치하여 transfer가 결속한 FMA/lineage metadata가 최종 bytes가 되게 한다. 전송 도중에는
full bootstrap이나 `train.py`를 동시에 실행하지 않는다. 과거 Drive snapshot에서 schema v1
manifest나 bootstrap receipt를 가져와 빈자리를 채우지 않는다.

### 4.4 read-only preflight

schema v2 bundle이 배치되어 canonical holdout가 존재할 때 처음 bootstrap 명령을 실행한다.

```bash
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update --preflight-only
```

`--preflight-only`는 exact code와 canonical holdout bundle만 확인한다. 전달한 transfer SHA의
semantic bytes, GPU, storage, environment 또는 public raw를 확인했다는 뜻이 아니다.

### 4.5 full bootstrap

현재 schema v2 transfer 파일 배치 뒤 cache를 쓰려면 세 external anchor를 그대로 포함한다.

```bash
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --archive-cache-root "$ARCHIVE_CACHE_ROOT" \
  --archive-cache-manifest "$ARCHIVE_CACHE_MANIFEST" \
  --expected-archive-cache-manifest-sha256 "$ARCHIVE_CACHE_MANIFEST_SHA256" \
  --raw-hash-workers 8 \
  --no-update
```

full bootstrap은 full A100 80GB, storage, exact environment, transfer bundle, public raw count/SHA,
decoder audit, six manifests, QA와 pytest를 검사하고 bootstrap receipt를 발행한다. raw cache가
있어도 검증을 생략하지 않는다. bootstrap은 학습을 자동 시작하지 않으므로 receipt 이후 readiness,
G0, gradient, pilot/probe, A100 resume-equivalence smoke와 campaign ledger를
`docs/05_training_elice.md` 순서대로 통과해야 한다.

파일시스템 total 127.875 GiB 계약은 cache 여부와 무관하다. 미완성 raw의 최초 실행에서 official
download는 repository archive/staging peak를 포함해 available 96 GiB를, held-fd cache consume은
외부 archive를 repository에 재복사하지 않으므로 available 72 GiB를 요구한다.
이 shell 수치와 별도로 cache CLI가 각 canonical archive/raw target의 실제 parent dirfd에
`fstatvfs`를 실행한다. 따라서 `data/raw`가 repository와 다른 mount이면 그 mount별 missing bytes,
sequential staging peak와 512 MiB reserve가 첫 archive/raw publish 전에 충족되어야 한다.

archive cache 세 인자는 all-or-nothing이다. full bootstrap은 transfer/environment/early pytest 뒤
`public_archive_cache.py consume`으로 고정 10개를 held-fd 추출한다. 이후 DNS shell bzip2/tar와
DEMAND/MIMII ZIP pathname extractor는 cache mode에서 호출하지 않는다. cache object 하나라도
missing/손상/manifest 불일치이거나 held source/target identity가 바뀌면 즉시 종료하며 해당 corpus를
official network에서 다시 받는 fallback은 0회다. 반면 FMA/ESC는 이 cache의 대상이 아니므로 기존
official/extracted restore 경로를 따른다.

consume은 성공해도 raw/bootstrap authority가 아니다. no-replace raw tree를 만든 뒤 기존 exact
16,000 DNS noise + 8,065 speech + 96 DEMAND + 3,600 MIMII count, 전체 decoder fingerprint/SHA audit,
six manifests, QA, pytest와 bootstrap receipt를 그대로 실행한다. 실패 중간에 발행된 exact member와
forensic staging은 자동 overwrite/delete하지 않는다. origin receipt는 completion보다 먼저 발행될 수
있어 crash 뒤 남을 수 있지만 cache transport provenance일 뿐 raw/training authority가 아니다. 같은
external anchors로 재실행하면 기존 raw member가 archive에서 다시 읽은 bytes와
exact할 때만 이어간다.

consume/restore는 실제 target parent의 filesystem별로 byte와 inode 여유를 모두 첫 publish 전에
검사한다. raw/archive output뿐 아니라 intent/inventory/origin/completion parent mount도 포함한다.
missing final bytes/inodes, 한 번에 하나인 staging peak, 512 MiB 및 receipt/forensic 64 inode
reserve 중 하나라도 부족하면 intent/raw/archive를 발행하지 않는다. `data/raw` 별도 mount도 이 검사를
우회하지 않는다.

완료된 cache-origin raw에서 `--cache-preflight-only`를 다시 쓸 때는 cache root/manifest/external
manifest SHA 세 인자를 함께 유지한다. decoder reuse verifier의 전수 대조가 끝난 뒤 단순 projection
비교까지 raw pathname이 바뀌는 창을 없애기 위해, archive verifier가 cache 대상 raw descriptor를
전부 다시 열어 보유한 채 current content와 durable member inventory 및 decoder report를 한 경계에서
재결속한다. 따라서 cache-origin raw는 의도적으로 두 번 hash한다. `--decoder-projection-only`는 이
TOCTOU를 닫을 수 없어 거부하며, preflight도 receipt/raw authority를 발행하지 않는다.

반대로 cache 세 인자가 없는 plain bootstrap은 시작 시점에 위 DNS/DEMAND/MIMII working archive
10개 중 하나라도 이미 있으면 CRC/bzip2가 정상이어도 즉시 거부한다. 이는 cache-only bytes를
official-origin archive로 세탁하는 경로를 막기 위한 의도된 restart 경계다. official downloader가
같은 실행에서 새로 만든 완성 archive만 즉시 extractor로 넘길 수 있다. 중단 뒤 남은 `.part`는
pget의 held validator와 archive별 0700 deterministic private staging으로 같은 output에서 재개할
수 있지만, 완성 archive는 자동 재사용·삭제하지 않는다. final은 no-replace publish하며 fixed target
parent의 intermediate symlink도 setup/network 전에 거부한다. matching cache anchor를 제시하거나
bytes를 별도로 보존·격리한 뒤 official download를 새로 시작한다.

## 5. 외부 공급원 공통 규칙

1. `train.py`나 bootstrap이 실행 중이면 staging·manifest 교체를 하지 않는다.
2. dataset ID, 버전/업로드 시각, 라이선스, official archive hash를 별도 ledger에 기록한다.
3. repository 밖 incoming에 먼저 받고 archive와 추출 파일의 count/path/content SHA를 확인한다.
4. 현재 corpus와 byte가 다르면 기존 manifest를 고치는 대신 새 generation으로 격리한다.
5. canonical holdout, FMA/DNS/DEMAND lineage와 recorded-generation exclusion을 적용해 six
   manifests를 새로 만들고 noise QA, recorded QA, pytest와 readiness를 다시 실행한다.
6. 모든 gate가 통과한 뒤에만 새 transfer/bootstrap receipt를 발행한다.

Kaggle mirror는 official archive와 hash가 일치할 때만 raw cache 후보가 된다. 토큰은 Elice의
비밀 저장소에서 주입하며 저장소, shell history, log에 넣지 않는다.

```bash
INCOMING=/mnt/data/deep_anc_incoming/kaggle_<version>
test ! -e "$INCOMING"
mkdir -p "$INCOMING"
umask 077
kaggle datasets download -d <owner>/<slug> -p "$INCOMING" --unzip
sha256sum "$INCOMING"/*
```

Google Drive OAuth token과 service-account JSON도 개인키와 같은 비밀정보다. raw가 MP3이면
file hash뿐 아니라 bootstrap의 실제 SoundFile/libsndfile/libmpg123 환경에서 full decoder audit을
다시 통과해야 한다. Jetson으로 public corpus를 되받아 복제하거나 Git에 raw·token을 올리지 않는다.
