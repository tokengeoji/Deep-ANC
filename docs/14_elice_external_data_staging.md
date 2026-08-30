# Elice 외부 데이터 staging 규칙

갱신일: 2026-08-30

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

새 Elice에서 재사용할 수 있는 범위는 검증된 **raw-only cache인 `data/raw/`**뿐이다. recorded
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

현재 schema v2 transfer 파일 배치와 선택적 raw cache 검증을 마친 뒤 같은 네 anchor로 실행한다.

```bash
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --raw-hash-workers 8 \
  --no-update
```

full bootstrap은 full A100 80GB, storage, exact environment, transfer bundle, public raw count/SHA,
decoder audit, six manifests, QA와 pytest를 검사하고 bootstrap receipt를 발행한다. raw cache가
있어도 검증을 생략하지 않는다. bootstrap은 학습을 자동 시작하지 않으므로 receipt 이후 readiness,
G0, gradient, pilot/probe, A100 resume-equivalence smoke와 campaign ledger를
`docs/05_training_elice.md` 순서대로 통과해야 한다.

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
