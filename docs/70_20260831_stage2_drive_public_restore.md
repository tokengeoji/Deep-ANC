# Stage-2 public 데이터 Drive read-only 감사와 Elice partial restore

> 기준일: 2026-08-31
>
> 판정: partial restore input `PASS`, public scratch-pretrain `BLOCKED`

이 문서는 Google Drive의 실제 객체 목록과 SHA metadata를 read-only로 재감사한 결과다.
Elice에는 접속하지 않았고 Drive write, raw download, 오디오, GPU 사용은 0회다.

## 1. 가설 — 기존 Drive만 복원하면 Stage-2 scratch pretrain을 바로 열 수 있다

### [가설]

기존 `DeepANC` Drive backup에 public raw와 official fixed archive가 모두 있어 새
Elice에서 official download 없이 바로 Stage-2 scratch pretrain을 열 수 있다고 가정했다.

### [근거]

2026-08-27 Jetson snapshot에는 FMA, LibriSpeech, ESC-50가 업로드됐고, 과거 Elice
snapshot과 bootstrap decoder audit도 Drive에 남아 있다. 문서가 아닌 현재 remote object를
다시 확인할 필요가 있었다.

### [확인 방법]

`rclone cat`으로 snapshot SHA manifest를 읽고 `rclone lsjson --recursive --files-only
--hash`로 현재 `data/` 13,428개 객체의 path/size/SHA-256을 전수 열거했다. manifest와
actual object path set 및 개별 SHA를 비교했다. 별도로
`DeepANC/public_archive_cache`를 read-only 열거해 fixed DNS3+DEMAND6+MIMII1 archive와
manifest 존재 여부를 검사했다.

### [결과]

전체 actual audit:

- `results/data_audit/stage2_drive_public_restore_20260831.json`
- file SHA-256:
  `d37591577f447c2cc063ef360ff6eea3519ac9d9e2f6d92e910c8de561424119`
- internal evidence SHA-256:
  `f1ca93f2dcd024051201b5d331feffc76a8aea9182457f25e1ff09dfb825b37a`

Git에 둘 수 있는 작은 external anchor:

- `configs/stage2_drive_public_restore_anchor_20260831.json`
- file SHA-256:
  `9380183c5abe2049a939f6738899fac2141f81b32a4d0c8b150380c8d2b38f9a`
- internal evidence SHA-256:
  `1bd1e7264b5f48206584e449116c072d90933b66f611d73775bca26d7d517360`

Snapshot 전체는 다음 external anchor와 exact했다.

| 항목 | 실제 값 | 판정 |
|---|---:|---|
| manifest SHA-256 | `1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2` | exact |
| manifest/actual file | 13,428 | exact set |
| actual bytes | 17,439,445,191 | exact |
| path/size/SHA projection | `7ff53f692532d57c6c1d2acbb737ed72003b294b703750119eaef3ebc78c1888` | exact |
| 개별 content SHA metadata mismatch | 0 | PASS |

빠르게 partial restore할 수 있는 public cohort는 다음 12,819개,
9,480,223,737 bytes다.

| Cohort | 전체 객체/bytes | 핵심 audio | projection SHA-256 |
|---|---:|---:|---|
| FMA music | 8,003 / 8,235,886,703 | MP3 8,000 / 7,975,016,002 | `7c104d5ff9c2fbd22b27e75d867f637e31e2a2b63e64473d1509041dd5fe7319` |
| LibriSpeech | 2,805 / 360,289,905 | FLAC 2,703 / 359,034,309 | `859bcbeaa8f3ace90fee7b823ef8a2caefd25c2009ef7116ff809c8df52a8cb7` |
| ESC-50 | 2,011 / 884,047,129 | WAV 2,000 / 882,088,000 | `5b9e606c64974e5ef0c4d834ed98338eee279ab6f7019b38199dd1b98f9cfb83` |

반면 `DeepANC/public_archive_cache`는 현재 `directory not found`다. 따라서 총
18,229,762,015 bytes인 DNS3+DEMAND6+MIMII1 fixed archive 10개와 manifest는 **0개**다.
2026-08-30 Elice split snapshot도 10 part 중 1개가 없으므로 대체 archive가 아니다.

### [판정]

가설은 **Contradicted**다. 기존 Drive는 약 9.48 GB partial restore에는 exact하지만,
Stage-2 public scratch-pretrain 전체 admission에는 부족하다. 현재 별도 status는 다음과 같다.

- partial restore input: `PASS_INPUT_ELIGIBLE_NOT_TRAINING_READY`
- official fixed archive cache: `BLOCKED_ARCHIVE_CACHE_ABSENT`
- public scratch-pretrain: `BLOCKED`
- recorded fine-tune 모집단: 이 restore와 독립이며 필요조건이 아니다

### [다음 행동]

새 Elice에서는 FMA/LibriSpeech/ESC-50 9.48 GB를 Drive에서 local SSD로 복원한다.
DNS/DEMAND/MIMII는 clean exact commit으로 cache를 먼저 발행하지 못했다면 official source에서
한 번 받아야 한다. 두 경로를 합친 뒤 decoder, lineage, five-octave source-density와 Stage-2
manifest bundle을 새로 발행해야 public scratch-pretrain이 열린다.

## 2. 새 Elice exact restore 명령

아래는 새 exact checkout에서 실행한다. `$EXPECTED_COMMIT`은 최종 push된 40자리 SHA여야
하며 이 문서의 과거 HEAD를 복사하지 않는다. Drive mount를 training random-I/O 경로로
쓰지 않고 Elice local SSD로 복원한다.

```bash
git clone https://github.com/tokengeoji/Deep-ANC.git Deep_ANC
cd Deep_ANC
git checkout --detach "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=no)"

SNAPSHOT_ROOT='gdrive:DeepANC/jetson_data_backup_20260827'
INCOMING='/mnt/data/deep_anc_stage2_partial_1dd9fef8'
test ! -e "$INCOMING"
mkdir -p "$INCOMING/data/raw/noise"

rclone copyto \
  "$SNAPSHOT_ROOT/data_backup_manifest.sha256" \
  "$INCOMING/data_backup_manifest.sha256" \
  --immutable --transfers 1 --checkers 1

rclone copy "$SNAPSHOT_ROOT/data/raw/music" \
  "$INCOMING/data/raw/music" --immutable --transfers 8 --checkers 8
rclone copy "$SNAPSHOT_ROOT/data/raw/speech" \
  "$INCOMING/data/raw/speech" --immutable --transfers 8 --checkers 8
rclone copy "$SNAPSHOT_ROOT/data/raw/noise/esc50" \
  "$INCOMING/data/raw/noise/esc50" --immutable --transfers 8 --checkers 8

rclone check "$SNAPSHOT_ROOT/data/raw/music" \
  "$INCOMING/data/raw/music" --download
rclone check "$SNAPSHOT_ROOT/data/raw/speech" \
  "$INCOMING/data/raw/speech" --download
rclone check "$SNAPSHOT_ROOT/data/raw/noise/esc50" \
  "$INCOMING/data/raw/noise/esc50" --download

.venv/bin/python scripts/data/audit_stage2_drive_pretrain_restore.py \
  verify-local \
  --anchor configs/stage2_drive_public_restore_anchor_20260831.json \
  --restore-root "$INCOMING" \
  --snapshot-manifest "$INCOMING/data_backup_manifest.sha256" \
  --output "$INCOMING/stage2_partial_restore_receipt.json"
```

`verify-local`은 12,819개 파일을 전부 다시 SHA-256으로 읽고, missing/extra, symlink,
size/content/projection mismatch를 하나라도 발견하면 exit 2로 중단한다. PASS receipt도
`PASS_PARTIAL_RESTORE_ONLY`이며 training-ready를 주장하지 않는다.

검증된 incoming을 exact target에 놓을 때 기존 target이 하나라도 있으면 병합하지 않고
멈춘다.

```bash
test ! -e data/raw/music
test ! -e data/raw/speech
test ! -e data/raw/noise/esc50
mkdir -p data/raw/noise
mv "$INCOMING/data/raw/music" data/raw/music
mv "$INCOMING/data/raw/speech" data/raw/speech
mv "$INCOMING/data/raw/noise/esc50" data/raw/noise/esc50
```

그 뒤 DNS/DEMAND/MIMII cache external manifest가 실제 발행됐다면
`docs/14_elice_external_data_staging.md`의 held-fd consume 경로를 사용한다. 현재는 그
manifest SHA가 없으므로 placeholder를 채우거나 cache가 있다고 가정하지 않는다. cache가
계속 없다면 `bootstrap_all.sh` full bootstrap의 official download 경로를 사용한다.

## 3. 시간·비용 병목

- Drive metadata 13,428개 actual audit는 현재 shared client에서 약 1--2분이 걸렸다.
- 9.48 GB restore 시간은 Elice--Drive 실제 throughput을 아직 측정하지 않아 숫자를 만들지
  않는다. rclone 출력의 actual rate/ETA를 receipt와 별도 log에 보존한다.
- 가장 큰 미해결 전송은 Drive에 없는 fixed archive 18.23 GB다. 이를 Elice에서 official
  download하면 그 시간이 cold-start 병목이다.
- Drive의 rclone shared Google client ID는 2026년 중 종료 경고가 실제 출력됐다. 사용자
  소유 OAuth client로 전환하기 전에는 Drive를 유일한 복구 경로로 두지 않는다.
- restore 후에도 decoder 전수 검증과 Stage-2 lineage/frequency manifest 생성이 필요하다.
  데이터가 존재한다는 이유로 이 검사를 생략하지 않는다.

## 4. 구현과 회귀 경계

- remote/local verifier:
  `src/deep_anc/data/stage2_drive_pretrain_restore.py`
- CLI: `scripts/data/audit_stage2_drive_pretrain_restore.py`
- negative tests: `tests/test_stage2_drive_pretrain_restore.py`

remote audit 서브명령은 오직 `rclone cat`과 `rclone lsjson`을 호출한다. `copy`, `sync`,
`delete`, mount 또는 Elice 접속을 하지 않는다. local verifier는 Drive에 접속하지 않는다.
