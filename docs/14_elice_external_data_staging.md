# Elice 외부 데이터 staging 규칙

작성일: 2026-08-27

## 결론

Jetson의 로컬 여유 공간이 작아도 public corpus는 Elice에서 직접 내려받을 수 있다.
Kaggle 또는 Google Drive는 **임시 공급원(staging)** 으로 사용할 수 있지만, 미러 파일을
그대로 canonical raw로 인정하지 않는다. 현재 Elice에는 `data/raw` 약 34 GiB와 약 80 GiB의
가용 공간이 있고, DNS·speech·ESC-50·FMA·DEMAND·MIMII의 6종 raw와 manifest가 이미
존재한다. 따라서 현재 실행 중인 loss pilot 동안에는 재다운로드하거나 `data/raw`를 교체하지
않는다.

## 절대 순서

1. `train.py`와 bootstrap이 실행 중인지 확인한다. 실행 중이면 staging·manifest 교체를
   하지 않는다.
2. Kaggle/Drive 파일의 dataset ID, 버전/업로드 시각, 라이선스, 제공자가 제시한 SHA-256을
   별도 ledger에 기록한다.
3. `/tmp` 또는 Elice 외부의 임시 디렉터리에 먼저 다운로드한다. `data/raw`에 바로
   덮어쓰지 않는다.
4. archive SHA와 추출된 각 raw의 개수·경로·content SHA를 확인한다. 현재 canonical
   manifest/receipt와 byte가 다르면 새 corpus 세대로 간주한다.
5. canonical holdout·FMA lineage·recorded exclusion을 적용해 6종 manifest를 새로 만들고
   noise QA, recorded QA, pytest, readiness를 다시 실행한다.
6. 모든 gate가 통과한 뒤에만 transfer/bootstrap receipt를 새 SHA로 승격한다. 검증 전
   archive는 지우지 않고, 승격 후 중복 archive만 정리한다.

## Kaggle 예시

Elice에 이미 설치·검증된 downloader가 있을 때만 아래 형식을 사용한다. `<owner>/<slug>`와
`<sha256>`는 실제 dataset 버전의 값으로 치환하며, 토큰은 저장소나 명령행에 넣지 않는다.

```bash
INCOMING=/tmp/deep_anc_incoming/kaggle_<version>
mkdir -p "$INCOMING"
umask 077
# KAGGLE_CONFIG_DIR 또는 KAGGLE_API_TOKEN은 Elice 비밀 저장소에서 주입한다.
kaggle datasets download -d <owner>/<slug> -p "$INCOMING" --unzip
sha256sum "$INCOMING"/*
```

Kaggle mirror의 파일이 공식 archive와 다르면 기존 manifest를 수정해 맞추지 않는다.
공식 원본 URL/버전을 사용하거나 새 provenance 세대로 격리한다.

## Google Drive 예시

공개 파일이면 `gdown`을, 권한이 필요한 파일이면 Elice에 등록된 `rclone` remote를 사용한다.
OAuth token·service-account JSON은 개인키와 동일하게 비밀정보로 취급하고 작업 종료 후
Elice에서 제거한다.

```bash
INCOMING=/tmp/deep_anc_incoming/drive_<version>
mkdir -p "$INCOMING"
gdown --id <drive-file-id> -O "$INCOMING/archive.bin"
# 또는: rclone copy "<remote>:<path>" "$INCOMING" --immutable
sha256sum "$INCOMING/archive.bin"
```

Drive 링크에서 내려받은 파일이 재압축·재인코딩되었거나 SHA가 없으면 canonical 학습에
사용하지 않는다. 특히 MP3는 decoder warning 여부까지 원본 파일별로 audit해야 한다.

## 현재 실행에 대한 처리

현재 Elice loss pilot은 20k 후보를 순차 실행 중이다. pilot 완료 전에는 위 downloader를
실행하지 않는다. 완료 후에도 반복되는 `libmpg123` 경고의 원본을 먼저 분류하고, 문제가
확인되면 해당 raw/manifest를 새 SHA로 재생성한 뒤에야 probe·canonical pretrain을 연다.
Jetson으로 public corpus를 되받아 복제하거나 Git에 raw·토큰을 올리는 경로는 사용하지 않는다.
