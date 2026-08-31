# Deep-ANC 노트북 작업 지시서 — Stage-2 2 kHz

이 문서는 일반 노트북 Agent가 **Jetson의 오디오 장비를 건드리지 않고** public 데이터 준비를
병렬 처리하기 위한 실행 계약이다. 목표는 125 Hz–2 kHz 단일 ERR mic ANC 학습 입력을 빠르게
준비하는 것이다. 2 kHz 실기 감쇠 합격선은 최소 `+3 dB`이고, 1.6 kHz도 near-zero를 허용하지
않는다. 이 문서의 작업 자체는 성능 달성을 주장하지 않는다.

## 0. 절대 규칙

1. 이 문서가 포함된 최신 `origin/dev`를 새 디렉터리에 clone하고 **detached exact commit**으로
   실행한다. 기존 clone, `main`, 축약 SHA, dirty checkout은 사용하지 않는다.
2. public raw와 archive는 Git에 commit하지 않는다. 토큰, OAuth JSON, rclone config, SSH key,
   `.pem`도 commit/status receipt에 넣지 않는다.
3. Drive에는 `copyto --immutable`만 사용한다. `sync`, `delete`, `purge`, overwrite를 금지한다.
4. 기존 Jetson snapshot과 Drive archive를 수정·삭제하지 않는다. 다운로드 실패물도 자동 삭제하지
   말고 저장소 밖 forensic staging에 둔다.
5. 노트북에서는 스피커/마이크/P/S/ANC ON을 실행하지 않는다. P/S와 physical latency는 Jetson 전용이다.
6. threshold를 낮추거나 missing corpus를 synthetic으로 대체해 PASS시키지 않는다.
7. `notebook_todo.md`를 작업 중 수정하지 않는다. 진행 상태는 아래 append-only Drive status로만
   발행한다.

## 1. 교환 경로

- 기본 remote: `gdrive:DeepANC/notebook_exchange/stage2_2khz_v1`
- 상태: `status/<UTC>_<commit12>_<phase>_<payload16>.json`
- 작은 증거: `receipts/sha256_<SHA>/<basename>`
- 고정 public archive cache: `gdrive:DeepANC/public_archive_cache/<EXPECTED_COMMIT>`

상태는 receipt 업로드가 끝난 뒤 마지막에 발행된다. mutable `latest.json`은 만들지 않는다. Jetson
Agent는 같은 exact commit을 지정해 status를 read-only로 집계하므로 서로 다른 commit의 PASS가
섞이지 않는다. 모든 phase가 PASS여도 결과 이름은 `ADVISORY_COMPLETE`이며
`canonical_pretrain_ready=false`다. Git에 review된 authority, held source bytes, strict P/S 및 A100
smoke를 notebook status로 대체하지 않는다.

## 2. 새 clone과 사전검사

```bash
git clone https://github.com/tokengeoji/Deep-ANC.git Deep_ANC_notebook
cd Deep_ANC_notebook
git fetch --prune origin
EXPECTED_COMMIT="$(git rev-parse origin/dev)"
git checkout --detach "$EXPECTED_COMMIT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -f notebook_todo.md
test -f configs/stage2_2khz_campaign.yaml
test -f scripts/data/notebook_exchange.py
command -v rclone
rclone listremotes
```

필요 조건:

- Linux 또는 동등한 POSIX 환경
- Python 3.10+
- 저장소 밖 SSD 가용공간 최소 32 GiB(archive cache 발행만), partial raw까지 복원하면 최소 48 GiB
- rclone의 Google Drive remote가 이미 인증되어 있을 것. 인증 파일 내용을 출력하거나 receipt에
  넣지 않는다.
- remote 이름이 `gdrive`가 아니면 아래 모든 명령에
  `--remote-root "<remote>:DeepANC/notebook_exchange/stage2_2khz_v1"`를 명시한다.

venv는 저장소 지침대로 만든다. Jetson 전용 NVIDIA wheel을 노트북에 설치하지 않는다. 데이터 전용
명령에 필요한 최소 의존성 설치가 현재 스크립트로 불가능하면 `BLOCKED` receipt만 발행하고 임의
package version으로 official PASS를 만들지 않는다.

사전검사 상태 발행:

모든 `--artifact KIND=PATH` 파일은 UTF-8 JSON object이며 phase별 exact schema와 의미를 모두
통과해야 한다. schema/status 문자열만 있는 빈 receipt는 거부된다. `PASS` status는 phase별 필수
kind의 **exact 집합**이 아니면 발행되지 않고, Jetson readback은 Drive의 실제 bytes·size·SHA뿐
아니라 receipt 의미와 artifact 사이 cross-SHA를 다시 확인한다.

```bash
WORK_ROOT="/mnt/ssd/deep_anc_stage2_${EXPECTED_COMMIT:0:12}"
mkdir -p "$WORK_ROOT/receipts"

.venv/bin/python scripts/data/notebook_exchange.py audit-checkout \
  --expected-commit "$EXPECTED_COMMIT" \
  --work-root "$WORK_ROOT" \
  --out "$WORK_ROOT/receipts/notebook_preflight_receipt.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase preflight --state PASS \
  --message "clean detached checkout, rclone, external SSD preflight PASS" \
  --artifact checkout_audit="$WORK_ROOT/receipts/notebook_preflight_receipt.json"
```

## 3. 작업 A — Drive partial snapshot 검증·복원

먼저 remote metadata를 다시 감사한다. 새 output 이름을 사용하며 기존 결과를 덮어쓰지 않는다.

```bash
.venv/bin/python scripts/data/audit_stage2_drive_pretrain_restore.py audit-remote \
  --output "$WORK_ROOT/receipts/drive_remote_audit.json" \
  --anchor-output "$WORK_ROOT/receipts/drive_restore_anchor.json"
```

정상 기대값은 FMA/LibriSpeech/ESC-50 `12,819 files / 9,480,223,737 bytes`의 partial restore
input PASS다. 이것은 전체 pretrain PASS가 아니다.

`docs/70_20260831_stage2_drive_public_restore.md` §2의 명령으로 저장소 밖 incoming에 세 cohort를
`--immutable` 복원하고 `rclone check --download`를 수행한다. 이어서 다음 verifier를 실행한다.

```bash
.venv/bin/python scripts/data/audit_stage2_drive_pretrain_restore.py verify-local \
  --anchor "$WORK_ROOT/receipts/drive_restore_anchor.json" \
  --restore-root "$WORK_ROOT/partial_restore" \
  --snapshot-manifest "$WORK_ROOT/partial_restore/data_backup_manifest.sha256" \
  --output "$WORK_ROOT/receipts/partial_restore_receipt.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase drive_partial_restore --state PASS \
  --message "FMA/LibriSpeech/ESC partial restore exact verification PASS" \
  --artifact restore_receipt="$WORK_ROOT/receipts/partial_restore_receipt.json"
```

## 4. 작업 B — DNS/DEMAND/MIMII fixed archive cache 발행

현재 Drive에는 다음 고정 archive 10개가 없다.

- DNS noise 2 + speech 1
- DEMAND 6
- MIMII fan 1
- 합계 `18,229,762,015 bytes`

저장소 밖 staging은 archive 합계와 512 MiB 여유를 동시에 제공해야 한다. 다음 official publisher만
사용한다. URL을 바꾸거나 Kaggle mirror를 섞지 않는다.

```bash
ARCHIVE_STAGE="$WORK_ROOT/archive_cache_staging"
mkdir -p "$ARCHIVE_STAGE"

.venv/bin/python -I -B scripts/elice/public_archive_cache.py publish \
  --staging-root "$ARCHIVE_STAGE" \
  --remote-root "gdrive:DeepANC/public_archive_cache/$EXPECTED_COMMIT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --rclone "$(command -v rclone)" \
  | tee "$WORK_ROOT/receipts/public_archive_cache_publish.log"
```

publisher는 각 archive와 manifest의 `rclone check --download` 및 `rclone cat` SHA readback을
끝낸 뒤에만 JSON을 반환한다. 그 JSON의 `staging_manifest`를 exact tracked publisher와 다시 대조해
작은 advisory receipt를 만든다. shell log만 직접 receipt로 사용하지 않는다.

```bash
ARCHIVE_MANIFEST="$(.venv/bin/python -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["staging_manifest"])' \
  "$WORK_ROOT/receipts/public_archive_cache_publish.log")"

.venv/bin/python scripts/data/notebook_exchange.py audit-archive-cache \
  --expected-commit "$EXPECTED_COMMIT" \
  --manifest "$ARCHIVE_MANIFEST" \
  --out "$WORK_ROOT/receipts/archive_cache_readback_advisory.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase public_archive_cache --state PASS \
  --message "official fixed archive cache 10/10 immutable publish and readback PASS" \
  --artifact archive_cache_receipt="$WORK_ROOT/receipts/archive_cache_readback_advisory.json"
```

위 `ARCHIVE_MANIFEST`는 publisher stdout의 관측값에서만 읽는다. 경로나 SHA를 추측하지 않는다.

## 5. 작업 C — decoder·lineage·주파수 coverage

전체 DNS/DEMAND/MIMII raw 추출과 six canonical manifest 발행은 Elice full bootstrap의 held-fd
consumer가 최종 권위다. 노트북은 partial cohort에 대해 선행 계산할 수 있지만 이를 full canonical
PASS로 승격하지 않는다.

반드시 확인할 항목:

1. 실제 audio 전수 decode, sample rate/channel/duration/content SHA
2. FMA artist+album, speech speaker+book, ESC original clip의 connected component
3. recorded 82세션 및 추가 녹음 source와 public train/val/test의 transitive component 교집합 0
4. 125/250/500/1000/2000 Hz octave source-density
5. 1.6 kHz sentinel `[1425.437949, 1795.939277]` source-density와 family/component 하한
6. legacy public manifest 자동 승격 금지

현재 legacy manifest 감사는 exact basename overlap뿐 아니라 transitive component overlap도 검출한다.
새 canonical manifest는 exclusion 뒤 **새 generation**으로 만들어야 한다.

lineage receipt는 반드시 `stage2_2khz_public_lineage_receipt_v2`이어야 하며,
`data/manifests/recorded_holdout.json`의 actual file SHA, `clip_lineage.clips_sha256`, clip 수,
`transitive_basename_content_sha256_lineage_keys_v1` 알고리즘을 결속한다. manifest가
제공하는 path basename/content SHA/lineage key 이외의 신원은 추정해 만들지
않는다. Jetson/Elice production admission이 holdout bytes와 manifest rows에서 교집합 0을
독립 재계산하므로, 수기 zero scalar만 있는 v1 receipt는 완료 증거가 아니다.

decoder `PASS`는 partial cohort가 아니라 **37,761개 전수 audit**만 허용한다. 현재 고정
partition은 `36,868 accept + 893 reject`이며, reject는 삭제하거나 억지로 PASS시키지 않고
forensic inventory에 보존한다. downstream manifest에는 accept projection만 들어간다. production decoder
audit 뒤 아래 projection을 먼저 만든다.

```bash
.venv/bin/python scripts/data/notebook_exchange.py audit-full-decoder \
  --expected-commit "$EXPECTED_COMMIT" \
  --decoder-audit "$WORK_ROOT/receipts/decoder_audit_full.json" \
  --out "$WORK_ROOT/receipts/decoder_audit_full_advisory.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase decoder_qa --state PASS \
  --message "full six-corpus 37,761 = 36,868 accept + 893 reject audit advisory PASS" \
  --artifact decoder_qa_receipt="$WORK_ROOT/receipts/decoder_audit_full_advisory.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase lineage_manifest --state PASS \
  --message "new-generation component exclusion receipt PASS" \
  --artifact manifest_bundle="$WORK_ROOT/receipts/six_manifest_bundle.json" \
  --artifact lineage_receipt="$WORK_ROOT/receipts/lineage_manifest_receipt.json"

.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase frequency_coverage --state PASS \
  --message "125-2000Hz public source-density audit PASS" \
  --artifact manifest_bundle="$WORK_ROOT/receipts/six_manifest_bundle.json" \
  --artifact frequency_coverage_receipt="$WORK_ROOT/receipts/stage2_frequency_coverage.json"
```

coverage receipt는 `stage2_2khz_public_frequency_coverage_v2`여야 한다. 5 octave와 1.6 kHz
sentinel의 각 qualified source를 `{dataset_index, component_id, path, content_sha256}`로 담으며,
exchange가 이를 함께 발행한 manifest actual row와 exact 비교한다. 부분 cohort만 검사했다면
artifact 없이 message에 `PARTIAL_ONLY`를 명시한 `IN_PROGRESS`만 발행한다. full six-corpus receipt
없이 `bundle_publish PASS`를 내면 안 된다.

## 6. 완료 bundle

완료 bundle은 raw 자체가 아니라 다음 작은 증거 파일의 content SHA를 묶는다.

- archive cache manifest
- partial restore receipt
- full decoder audit
- six canonical manifest bundle receipt
- public-recorded lineage exclusion receipt
- five-octave source-density receipt

이 bundle을 Drive의 새 content-addressed 경로에 immutable 발행하고 readback한 뒤 마지막 상태를 낸다.

```bash
.venv/bin/python scripts/data/notebook_exchange.py publish \
  --expected-commit "$EXPECTED_COMMIT" \
  --phase bundle_publish --state PASS \
  --message "Stage2 notebook public-data bundle immutable publish PASS" \
  --artifact manifest_bundle="$WORK_ROOT/receipts/six_manifest_bundle.json" \
  --artifact lineage_receipt="$WORK_ROOT/receipts/lineage_manifest_receipt.json" \
  --artifact frequency_coverage_receipt="$WORK_ROOT/receipts/stage2_frequency_coverage.json" \
  --artifact transfer_bootstrap_receipt="$WORK_ROOT/receipts/stage2_transfer_bootstrap_receipt.json"
```

`stage2_transfer_bootstrap_receipt.json`은 production schema
`stage2_2khz_transfer_bootstrap_receipt_v1`이어야 한다. outer status의
`artifact_bundle_sha256`가 manifest/lineage/coverage/transfer 네 file SHA를 함께 묶고, 각 receipt의
`manifest_bundle_sha256`와 `source_inventory_commit_sha`가 동일한지 재검산한다.

## 7. Jetson Agent가 상태를 읽는 명령

Jetson에서는 Drive를 쓰지 않고 다음 한 명령으로 같은 commit의 최신 phase만 집계한다.

```bash
EXPECTED_COMMIT="$(git rev-parse origin/dev)"
.venv/bin/python scripts/data/notebook_exchange.py read \
  --expected-commit "$EXPECTED_COMMIT"
```

모든 advisory 단계가 PASS일 때만 exit 0을 요구하려면:

```bash
.venv/bin/python scripts/data/notebook_exchange.py read \
  --expected-commit "$EXPECTED_COMMIT" --require-complete
```

이 exit 0은 `ADVISORY_COMPLETE`일 뿐 canonical 사전학습 READY가 아니다. 새 상태는 append-only이므로
이 명령을 다시 실행하면 자동으로 최신 상태를 읽는다. Codex/Agent에게는
“`notebook_todo.md`를 읽고 실행한 뒤 각 phase를 Drive receipt로 발행하라”라고 지시하면 된다.
Jetson 측 Agent에게는 “notebook exchange read 명령으로 상태를 재검증하라”라고 지시한다.

## 8. 노트북에서 하지 않는 작업

- 새 P/S 측정, mic recording, ANC OFF/ON, latency/xrun 판정
- 100k 사전학습 또는 50k 파인튜닝을 일반 CPU에서 시작
- legacy checkpoint/ONNX를 canonical로 승격
- 2 kHz `+3 dB`를 데이터 coverage만으로 달성했다고 주장
- 4/8 kHz single-point 결과를 quiet zone으로 주장
- Drive/GitHub의 기존 객체 삭제 또는 branch history rewrite
