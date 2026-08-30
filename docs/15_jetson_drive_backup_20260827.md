# Jetson 데이터 Google Drive 백업 기록

최초 시작: 2026-08-27 (KST)<br>
상태: **누락 343개만 순차 업로드 재개 — 검증 완료한 개별 public 원본만 부분 정리**

## 백업 대상과 무결성 기준

백업 원본은 저장소의 `data/` 디렉터리이며, 업로드 전에 생성한 고정 목록을
기준으로 한다.

| 항목 | 값 |
|---|---:|
| 파일 수 | 13,428 |
| 총 바이트 | 17,441,317,063 |
| 목록 파일 | `/tmp/deep_anc_data_backup_20260827.sha256` |
| 목록 SHA-256 | `1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2` |
| Drive 경로 | `gdrive:DeepANC/jetson_data_backup_20260827/data` |
| 원격 목록 복사본 | `data_backup_manifest.sha256` |

원격 목록 파일의 SHA-256은 위 값과 일치하는 것을 먼저 확인했다. 원본 파일은
업로드 중 변경하지 않는다.

## 실행 상태

처음 실행한 전체 `data` 복사는 하위 디렉터리 전용 복사와 겹치고 Google Drive
요청률 제한을 일으킬 수 있어 중단했다. 하위 경로를 병렬로 재개한 시도에서도
`RATE_LIMIT_EXCEEDED`가 관찰되어 해당 작업을 중단하고, 이미 올라간 객체를
그대로 둔 채 `setsid`로 분리한 저요청률 순차 복사를 재개했다. 현재 순차 작업은
다음 경로를 정확한 동일 목적지에 차례로 복사한다.

```text
data/raw/music
data/raw/noise
data/raw/speech
data/recorded
data/recorded_broken
data/source_pool
data/source_pool_v2
data/manifests
data/rir_bank
```

현재 작업은 `--fast-list`, `--checkers 2`, `--transfers 2`, `--tpslimit 4`,
`--drive-pacer-min-sleep 250ms`를 사용한다. 명령은 `rclone copy`이며 `sync`,
`move`, 원본 삭제 옵션을 사용하지 않는다. 따라서
현재는 원본과 이미 올라간 원격 파일 모두 보존된다. 업로드가 끝난 뒤에만 다음을
실행한다.

1. 네 경로별 원격 파일 수·바이트를 로컬 고정 목록과 대조한다.
2. `rclone check`로 내용 해시/크기를 확인한다.
3. 원격 중복 경로가 없는지 목록을 검사한다.
4. manifest SHA를 다시 대조하고 백업 receipt를 Drive에 기록한다.
5. 검증된 경로만 휴지통으로 이동하거나 정확한 목록을 삭제한다.

### 2026-08-28 부분 검증·정리 receipt

순차 작업과 다른 source를 삭제하지 않도록, 삭제 전에는 해당 source의 `rclone copy`
종료, local/remote 파일 수·바이트 대조, `rclone check --one-way` 0 differences, 실행
프로세스 미점유를 각각 확인했다.

| Local path | Drive 검증 | 삭제 결과 | 보존한 항목 |
|---|---|---|---|
| `data/raw/music/fma_small` | 8,002 files, 7,975,472,258 bytes, fixed manifest SHA 일치, `rclone check` 0 differences | 삭제 완료 | `fma_metadata/tracks.csv` |
| `data/raw/noise/esc50/ESC-50-master/audio` | 2,000 files, 882,088,000 bytes, `rclone check --one-way`: 0 differences / 2,000 matching | 삭제 완료 | `meta/esc50.csv` 및 ESC-50 repository metadata |

`data/raw/speech`는 이 기록 시점에도 전송 중이므로 표의 검증·삭제 대상이 아니다.
그 외 `recorded`, `recorded_broken`, `source_pool`, `source_pool_v2`도 각 경로의
전송·검증이 끝나기 전까지 보존한다. 검증된 Drive 사본에서 복구할 수 있지만, Jetson의
로컬 삭제 자체는 되돌릴 수 없으므로 다음 후보도 같은 순서로만 처리한다.

### 2026-08-28 16:19 KST 전체 목록 재감사·누락 전송 재개

업로드가 끝났다는 화면 상태를 그대로 믿지 않고 Drive를 다시 열어 고정 목록과 대조했다.
Drive에 보존된 `data_backup_manifest.sha256`은 13,428행이고 파일 SHA-256은
`1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2`로 최초 목록과
일치했다. 그러나 실제 `data/` 원격 inventory는 **13,085개 / 12,789,037,252바이트**뿐이었다.
고정 목록과 원격 상대경로의 집합 차이는 다음과 같다.

| 구분 | 파일 수 | 로컬 바이트 | 상태 |
|---|---:|---:|---|
| `recorded/` | 263 | 아래 합계에 포함 | Drive에 누락 |
| `recorded_broken/` | 39 | 아래 합계에 포함 | Drive에 누락 |
| `source_pool/` | 16 | 아래 합계에 포함 | Drive에 누락 |
| `source_pool_v2/` | 25 | 아래 합계에 포함 | Drive에 누락 |
| **합계** | **343** | **4,650,407,939** | 원격 extra 0 |

343개 로컬 파일은 업로드 전에 모두 고정 목록의 SHA-256과 다시 일치함을 확인했다. 그 뒤
삭제 옵션이 없는 `rclone copy data .../data --files-from <missing-list>`로 이 343개만
저요청률(`checkers=2`, `transfers=2`, `tpslimit=4`) 재개했다. 전송 중에는 source를 삭제하지
않는다. 프로세스 종료 뒤 다음 세 조건이 모두 PASS하기 전에는 백업 완료로 바꾸지 않는다.

1. 원격 inventory가 정확히 13,428개 / 17,441,317,063바이트일 것
2. 원격 extra와 missing이 각각 0일 것
3. remote download hash 또는 동일 강도의 `rclone check`가 13,428개 전부 일치할 것

현재 rclone shared Google Drive client ID의 2026년 폐기 예고와 요청률 변동도 관찰됐다.
전송 속도 때문에 검증 단계를 줄이거나 전송 프로세스를 중복 실행하지 않는다.

## 삭제·보존 정책

검증 전에는 `data/`를 삭제하지 않는다. 검증이 끝나도 계약·실행에 필요한 다음
항목은 Jetson에 남긴다.

```text
data/manifests/
data/rir_bank/
assets/
runs/
results/
```

용량을 회수할 때 삭제 대상은 업로드가 확인된 `data/raw/`, `data/recorded/`,
`data/recorded_broken/`, `data/source_pool/`, `data/source_pool_v2/`로 한정한다.
개인키, rclone 설정·토큰, `.venv`는 Drive나 Git에 올리지 않는다.

## 복구 방법

복구가 필요하면 receipt의 고정 목록과 SHA를 먼저 확인한 뒤, 필요한 하위 경로에
대해 다음처럼 `rclone copy`를 사용한다.

```bash
/home/capston/.local/bin/rclone copy \
  gdrive:DeepANC/jetson_data_backup_20260827/data/<subdir> \
  data/<subdir>
```

복구 후에는 manifest와 `rclone check`를 다시 실행한다. 이 문서는 백업 완료를
주장하지 않으며, receipt가 추가될 때 완료 시각·검증 결과·삭제 결과를 갱신한다.
