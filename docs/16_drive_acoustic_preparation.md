# 16. Drive 중심 acoustic 데이터·bootstrap 준비

> 비상업 학업 실험, acoustic REF, ERR 한 점, 저역과 800–1600 Hz 및 음성·음악이 목표다.
> 최신 사용자 지시: **PC Docker에 임시 다운로드 → 검증 → Drive 업로드 확인 → 해당 PC 원본 삭제**.
> 기존 Drive 백업은 보존한다. 코드·검증 기록과 실제 원본 확보·학습·실기 성능은 별개다.

## 1. 완료 범위를 구분하는 기준

| 상태 | 의미 | 의미하지 않는 것 |
|---|---|---|
| PC bootstrap 완료 | Docker 의존성 검사·합성 RIR 생성 | 음원 전수 확보, GPU/Jetson 준비 완료 |
| Drive 목록 관측 | 선택한 폴더의 이름·ID·크기 확인 | 원본 내용 hash/PCM, 백업 복원 가능 |
| 원본 전송 receipt | 공식 archive checksum 또는 고정 Git 객체 출처·로컬 archive hash, Drive ID/이름/부모/크기 확인 | 원격 내용 hash, 압축 해제된 PCM/학습 분할 검증 |
| `prepare_acoustic_corpus`의 `data_ready` | 명시 stage된 목록·PCM·그룹 분할 검사 | 공식 archive 전체 확보, 실기 감쇠 |
| strict 학습 QA | 해당 자료·RIR·로더의 정합성 | 학습 완료, 목표대역의 S 신뢰 확보 |
| CPU 2-step fixture 테스트 | 모델·손실·checkpoint 코드 연결 | 실제 음성·음악 학습이나 유용한 모델 |

현재 체크아웃의 strict 그룹 manifest 전수 QA와 독립 acoustic 모델 학습은 미완료다.
따라서 **Jetson 실측만 남았다고 보고하지 않는다**. 실제 완료 목록은 HANDOFF와 Drive receipt를 따른다.

2026-09-15 실제 확보·아카이브 순회는 아래와 같다. 총 **13개, 18,599,035,802 byte**, 음원
**47,558개**다. 이 표는 Drive 전송 완료표나 strict 학습 그룹 검증 결과가 아니다.

| 공개 원본 | archive 수 / 96MiB 조각 수 | 음원 수치 QA |
|---|---:|---|
| LibriSpeech train100/dev/test | 3 / 72 | 33,862 / 33,862 통과 |
| FMA small + metadata | 2 / 81 | 7,993 / 8,000 통과, 7 실패; 별도 metadata 12개 |
| ESC-50 고정 Git | 1 / 7 | 2,000 / 2,000 통과 |
| DEMAND 6개 환경 | 6 / 22 | 96 / 96 통과 |
| MIMII DG fan | 1 / 10 | 3,600 / 3,600 통과 |

FMA 실패 ID는 `098565`, `098567`, `098569`, `099134`, `107535`, `108925`, `133297`이다.
디코딩 오류 또는 완전 무음으로 실패했으며 원본에서 임의 제거하지 않았다. 그 외 soft decoder
경고도 관측되어, 7개 실패만 제외하면 나머지의 디코딩 품질이 보장된다는 뜻은 아니다.
원본 archive checksum은 ESC를 제외한 12개에서 공식 배포 값과 일치했다. ESC는 아래의
고정 Git 객체 출처 검증을 사용한다. 조각 결합/Drive 업로드 상태는 개별 transfer receipt를 따른다.

## 2. Drive 최적화와 원본 정책

기존 `DeepANC`의 snapshot·bootstrap cache·과거 manifest를 확인했다. 선택한 과거 QA에는
DEMAND 96, DNS 15,553, ESC 1,006, machine 3,600, music 6,308, speech 7,969 파일이
기록되어 있지만 일부는 표본 검사이고 현재 원본 전수 검증이 아니다. 일부 음원 leaf는 연결 도구에
목록이 보이지 않았다. 과거 41,299,005,440 byte 아카이브는 10개 중 9개 part만 관측됐으며
`part-0005-of-0010`은 미확인이다. 이를 실제 삭제/분실 확정으로 표현하거나 나머지를 지우지 않는다.
과거 recorded bootstrap의 요청 subband 전체 통과 플래그도 false였다.

새 준비물은 기존 루트 아래 `acoustic_preparation_20260915`로 모으고 원본 archive는
그 아래 `public_archives`로 구분한다. 기존 폴더를 이동·삭제하거나 공유 범위를 변경하지 않는다.
개인 Drive ID/전송 inventory는 ignored `results/`와 Drive에만 두고 공개 Git에는 넣지 않는다.

- 다운로드 전 기존 원본·archive의 이름/출처를 검색한다. 같은 이름만으로 내용 동일을 판정하지 않는다.
- 공식 배포 archive checksum, 추가 SHA-256·MD5·byte 수·라이선스를 보존한다.
- 임시 폴더는 이번 작업만을 위한 새 경로다. 업로드 실패·미확인은 삭제하지 않는다.
- 업로드 응답과 후속 Drive 파일 ID·이름·부모·크기를 대조한 뒤 해당 PC archive만 삭제한다.
  원격 checksum을 도구가 제공하지 않으면 **원격 hash 미검증**으로 기록한다.
- 연결 업로드 도구는 파일당 100MiB 상한이다. 큰 archive는 96MiB씩 분할하고 조각마다
  번호/전체 수/크기/SHA-256/Drive ID를 복원 목록에 기록한다. 모든 조각 확인 전 완료로 표시하지 않는다.
  조각을 개별 ZIP/TAR로 풀지 않고 순서대로 결합한 전체 checksum을 먼저 검사해야 한다.
- 빈 폴더·중복처럼 보여도 기존 백업은 자동 삭제하지 않는다. 이번 최적화는 중복 방지·정리·추적성이다.
- 원본 음원/개인 인증은 Git에 올리지 않는다. 출처·저작자·트랙별 라이선스 표시는 보존한다.

## 3. 이 PC에서 가능한 bootstrap

아래 명령은 원본 다운로드나 학습을 하지 않는다. 매번 **새 출력 경로**를 사용한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/bootstrap_acoustic.py \
  --out results/acoustic_preparation/NEW_RUN

# 선택한 Drive 관측 JSON이 있을 때만 추가한다.
# --drive-inventory results/drive_preparation/20260915_01/inventory.json
```

`pip check`, duct 설정 검증, seed 20260915의 합성 `p_ref/p_err/f_fb` 300×8192 @48 kHz와
`preparation.json`을 생성한다. S 자산·신뢰대역·지연은 수정하지 않는다.
`pc_bootstrap_complete=true`, `dataset_training_ready=false`, `jetson_ready=false`를 분리한다.
RIR은 덕트 기하 시뮬레이션이며 실측 절대 gain/실제 F/장치 비선형성이 아니다.
전체 코드 회귀·필터 뱅크·데이터 QA는 아래 별도 명령이다.

## 4. 원본 확보 후보와 라이선스

공식 checksum과 URL은 `configs/public_archive_sources.json`에 있다. 이 목록 자체는
다운로드 완료 기록이 아니다. 기존 Drive 자료를 확인하고 필요한 항목만 확보한다.

| 계열 | 준비 대상·출처 | 분할·주의 사항 |
|---|---|---|
| 음성 | [LibriSpeech](https://www.openslr.org/12/) clean train100/dev/test, CC BY 4.0 | speaker 단위, 공식 subset 유지. 16 kHz이므로 8 kHz 초과 정보 없음 |
| 음악 | [FMA small 및 metadata](https://github.com/mdeff/fma) | artist 단위. metadata CC BY 4.0, 음원은 `track.license`별 조건 |
| 환경음 | [ESC-50](https://github.com/karolpiczak/ESC-50) 기존 원본/공식 CSV | `src_file` 단위. ESC-50 CC BY-NC 3.0, ESC-10 예외 별도 |
| 실제 환경 잡음 | [DEMAND 48 kHz](https://zenodo.org/records/1227121), CC BY-SA 3.0 | 동시 녹음 16채널을 같은 recording/group으로 유지 |
| 기계음 | [MIMII DG fan](https://zenodo.org/records/6529888), CC BY-NC-SA 4.0 | section/기계 개체/원녹음 관계를 검토한 목록 필요 |

비상업 연구 승인은 저작자 표시·개별 라이선스 조건을 생략하는 허가가 아니다.
기존 DNS 원본의 출처 그룹 매핑을 모르는 상태에서 파일명만으로 새 split을 만들지 않는다.
새 strict 기본 설정은 DNS를 필수 계열에서 제외하되 기존 DNS/구형 설정은 변경하지 않는다.
MIMII의 domain/공식 train-test 디렉터리를 우리의 ANC train/val/test로 무조건 복사하지 않는다.

PC 임시 다운로드 명령은 아래와 같다. `NEW_STAGING`은 이번 작업의 새 임시 하위 폴더다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/stage_public_archive.py \
  --catalog configs/public_archive_sources.json --source librispeech_test_clean \
  --out results/NEW_STAGING/librispeech_test_clean --confirm-temporary-local-staging
```

공식 HTTPS/redirect 허용 목록, byte 수·배포 checksum을 검사한 뒤 `receipt.json`을 만든다.
실패한 부분 파일에는 완료 receipt가 없으며 자동 재시도·삭제·업로드하지 않는다.
원본 GET·다운로드와 소리를 재생하는 작업은 다르며 오디오 장치를 열지 않는다.

큰 archive의 분할과 **복원 파일을 새로 만들지 않는** 결합 검증은 아래와 같다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/audit_public_archive.py \
  --archive results/NEW_STAGING/librispeech_test_clean/test-clean.tar.gz \
  --staging-receipt results/NEW_STAGING/librispeech_test_clean/receipt.json \
  --out results/NEW_STAGING/librispeech_test_clean/qa

bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/split_staged_archive.py \
  --staging-receipt results/NEW_STAGING/librispeech_test_clean/receipt.json \
  --out results/NEW_STAGING/librispeech_test_clean/parts

bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/split_staged_archive.py \
  --verify-manifest results/NEW_STAGING/librispeech_test_clean/parts/manifest.json
```

ZIP/TAR.GZ의 항목 이름·링크·중복·종결·크기를 검사하고 음원은 전수 PCM을 읽는다.
음원 entry의 인코딩된 바이트 상한은 기본 64MiB이며 초과도 실패로 기록한다.
전체 프로세스 메모리 상한이 아니며 PCM 블록·FFT·목록 메모리가 추가된다. 비오디오 metadata는
1MiB 청크로 해시만 계산한다. 전체 archive QA는 공식metadata의 의미/학습 split QA가 아니다.
`source_archive_content_qa_passed`와 `data_ready=false`를 분리하고 실패도 inventory에 보존한다.
PCM QA는 현재 libsndfile/mpg123가 반환한 PCM의 frame 수·유한값·대역 파워 정합 검사다.
예외로 전파되지 않은 디코더 stderr 경고는 자동 판정에 반영하지 않으므로 원래 음원에 대한
PCM 충실도나 경고 없는 디코딩을 보장하지 않는다. 실제 FMA small 검사에서
`dequantization failed`, `resync` 등 경고를 관측했다. 해당 트랙 식별·독립 디코더 검증 전에는
경고 영향을 미확인으로 남긴다. 공식 archive checksum 일치도 내부 MP3 무손상 보장은 아니다.
복원 검증은 **manifest.json과 그 조각만 있는 폴더**에서 실행하며 누락·추가 파일·hash 불일치를 거부한다.
전송 receipt와 QA는 parts 폴더 밖에 둔다. Drive에서 조각을 받은 뒤에도 같은 검사를 먼저 실행한다.

검증한 조각을 실제 압축 원본 파일로 복원하려면 다음 명령을 쓴다. 출력은 parts 폴더 밖의
**존재하지 않는 파일**이어야 한다. 예시 이름은 실제 manifest의 archive 이름에 맞춘다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/split_staged_archive.py \
  --restore-manifest results/NEW_RESTORE/parts/manifest.json \
  --archive-out results/NEW_RESTORE/restored/test-clean.tar.gz
```

사전 전수 검증, 결합 중 part/전체 checksum, fsync 후 출력 재읽기 hash·파일 상태 검사를 모두
통과해야 성공 JSON을 출력한다. 기존 파일은 덮어쓰지 않는다. 생성 후 실패한 출력은 보존되며,
완전한 크기로 보여도 성공 JSON이 없으면 사용하지 않는다. 다음 시도에는 다른 새 경로를 지정한다.
이 명령은 Drive 다운로드·삭제·압축 해제를 하지 않는다. 복원 후 PCM/그룹 QA는 여전히 별도다.

공식 ESC-50은 공개 archive checksum 대신 공식 Git의 고정 commit을 사용한다.
`package_esc50_git.py`는 이미 받은 독립 원본 repo를 검사·포장하며 다운로드하지 않는다.
`source_checksum_verified=false`를 유지하고 `official_git_commit_and_objects` 검증과
로컬 생성 archive checksum을 구분한다. Git 출처 선언도 서명된 원격 증명은 아니다.

선택적 `stage_public_archive_parallel.py`는 최대 4개 HTTPS Range와 공식 전체 checksum을
검사하는 도구다. 현재 실전 시도에서는 Libri 연결 실패가 있었고 FMA는 기존 전송과 중복되어
중단했다. 속도 개선/완주를 주장하지 않는다. 실패 부분 파일은 정식 원본 전송 확인 전 보존한다.

인증된 별도 환경용 `stream_public_to_drive.py`도 준비돼 있다. 기본은 계획만 출력하고,
실행은 운영자가 승인된 원격 환경으로 표시했을 때만 가능하다. 현재 PC는 연결 Drive의
파일 업로드를 사용한다. Google 인증을 플러그인에서 추출하지 않는다. 스트리밍 도구의
공식 checksum/청크 ACK/최종 Drive MD5 검사는 mock 회귀만 수행했으며 실서비스 인증 전송은 미검증이다.
중단된 resumable session을 자동 재개·삭제하지 않는다.

## 5. 공개 PCM 검사와 strict 로더

Drive 원본을 사용하는 **명시적으로 승인된 staging 작업**에서 다음 트리를 만든다.
원본 없이 smoke fixture를 만든 것을 실제 corpus로 등록하지 않는다.

```text
RAW_ROOT/
  esc50/   공식 esc50.csv와 해당 오디오
  music/   공식 tracks.csv와 fma_small 오디오
  speech/  SPEAKERS.TXT, train-clean-100/, dev-clean/, test-clean/
  demand/  source_index.csv, source_index_meta.json, 원본 오디오
  machine/ source_index.csv, source_index_meta.json, 원본 오디오
```

DEMAND/machine CSV 필수 열은 `path,group_id,official_split,license,source_recording_id`다.
`path`는 계열 루트 기준 상대 경로, 같은 동시녹음·원녹음은 같은 group이다.
`official_split`은 검토된 `train/val/test` 또는 공백(그룹 분할 도구에 위임)이다.
메타 JSON은 `source_family`, `expected_inventory_complete:true`, `inventory_origin`,
`grouping_basis`가 필요하다. 이 true는 staging 작성자의 책임 있는 선언이지 외부 완전성 인증이 아니다.
실제 archive 목록과 대조하지 않고 자동으로 true를 채우지 않는다.

```bash
# RAW_ROOT에는 실제 승인된 staging 경로를 넣는다. 이 CLI는 다운로드하지 않는다.
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/prepare_acoustic_corpus.py \
  --raw-root RAW_ROOT --out data/acoustic/manifests

# 기존 결과 재사용은 모든 원본 PCM·metadata·산출물을 다시 검사하며 수정하지 않는다.
bash scripts/docker/dev.sh exec .venv/bin/python scripts/data/prepare_acoustic_corpus.py \
  --raw-root RAW_ROOT --out data/acoustic/manifests --reuse

bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_training_data.py \
  --config configs/train_acoustic_prepared.yaml \
  --set data.rir_bank=results/acoustic_preparation/NEW_RUN/duct_rirs_v1.npz --json
```

PCM 전체 frame·finite·hash·near-full-scale·대역 파워를 읽는다. 오류가 있으면 inventory와 QA만
저장하고 계열 manifest는 발급하지 않는다. 동일 byte hash 중복은 같은 split 안에서도 모두
거부하며 좋은 것 하나를 임의 선택하지 않는다. 재인코딩/부분 복제 검출은 별도 미구현이다.
LibriSpeech는 발견된 transcript와 FLAC를 대조한다. transcript와 해당 chapter가 **함께**
없으면 외부 완전 목록 없이 검출할 수 없으며 `official_archive_completeness_verified=false`다.

strict 로더는 hash/원본 header/모든 split/group/manifest와 inventory 동등성/RIR 변형을 다시
검증한다. 누락 시 synthetic fallback하지 않는다. 원본을 Drive로 옮겨 PC에서 삭제한 뒤에는
로컬 `data_ready`가 다시 false가 되는 것이 정상이다. Drive 영구 보관과 로컬 학습 staging을 혼동하지 않는다.
학습 checkpoint에는 실제 loader 검사 시점의 `prepared_data_snapshot`이 남는다.

새 acoustic 소스 비율은 기존 분포에서 DNS를 제외해 정규화한 연구 출발점이다. 최적 비율이 아니다.
합성원은 기존 저역 분포와 800–1600 Hz 분포를 0.5 확률로 혼합하며 legacy 생성기는 불변이다.
PCM inventory의 목표 파워는 **[800,1600]**, ERR/loader smoke 평가는 **[800,1600)**다.
둘 다 FFT 진단이지 SNR·반복 일관성·측정 S 신뢰대역 판정이 아니다.

`configs/train_acoustic_prepared.yaml`은 tiny/batch1/FP32/최대2step/no-resume의 연결 검사 설정이다.
실행 시 실제 corpus가 없으면 실패한다. 기존 측정 S의 trusted loss 대역을 유지하므로 이 설정을
대규모로 돌리는 것만으로 800–1600 Hz 학습 문제가 해결되지 않는다. 새 S 검증 후 별도 연구 설정이 필요하다.

## 6. 논문 구조를 확장한 합성 필터 뱅크

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/prepare_filter_bank.py \
  --out results/filter_bank/NEW_RUN
```

`causal_toy`와 `long_delay_stress`마다 5소스×2개 독립 train seed의 후보를 따로 준비한다.
이전 docs/15 단일 후보의 toy→긴 지연 전이는 이 도구에서는 하지 않는다. 과거 REF의
PSD/RMS와 후보 특징 거리만으로 선택하며 ERR/정답/미래 블록은 선택에 쓰지 않는다.
학습된 CNN은 아니고 방향 분류도 아니다. bank JSON/NPZ/source hash·context를 검증하며
live callback에는 여전히 연결하지 않았다.

validation/test seed를 train과 분리하고, 미학습 주파수·레벨 2종·linear/tanh·1차경로 gain 변화를
zero/cold/fixed/selected 4군으로 비교한다. 기본 실행은 **20후보, 896 run**이다.
고정 후보는 평가 결과로 고르지 않고 사전 지정 mixed 후보를 쓴다. 모든 실패·무신호·증폭을
JSON/CSV에 남긴다. 집계 중앙값만으로 최악 소스의 실패를 덮지 않는다.

초기 연구 설정 mu=0.25의 train tone clipping을 확인해 기본 비교율을 0.05로 낮췄다.
모든 학습·적응 대조군에 동일하게 적용했고 성공 seed만 선택하지 않았다. 설정은 보고서에 남는다.
긴 지연에서 selected가 cold보다 나쁜 사례가 있어 **기본 FxNLMS보다 항상 우수하다고 주장하지 않는다**.
실제 음악/음성·실측 비선형성·클록 drift는 이 시험에 없다. tanh는 임의 스트레스일 뿐이다.

실제 CPU 기본 실행의 **linear/test/steady/800–1600 Hz** 비교는 다음과 같다.
각 군 28행 중 유효 24행·무신호 4행이며 아래 중앙값은 유효 행만의 값이다.

| 합성 조건 | cold 중앙값 | fixed 중앙값 | selected 중앙값 | selected 증폭 행 / 최악 |
|---|---:|---:|---:|---|
| causal toy | 8.39 dB | 10.03 dB | 14.98 dB | 0/24, 8.58 dB |
| 긴 지연 | 6.37 dB | −2.04 dB | 2.80 dB | 6/24, −0.74 dB |

양수는 감소, 음수는 증폭이다. 긴 지연 cold에도 미약한 음수 2행이 있고 fixed는 18행 증폭했다.
이 선택 규칙은 주파수·위상 불일치와 REF만으로 관측 불가능한 경로 변화에 취약한 연구 기준선이다.
전체 결과·tanh·초기/변경 후 구간을 함께 보고한다. 위 표는 실제 ERR 감쇠 결과가 아니다.

## 7. 남은 단계

- 실제 원본의 확보/Drive 전송 기록과 압축 해제 목록·PCM 전수 QA를 연결한다.
- DEMAND는 [공식 배포 설명](https://zenodo.org/records/1227121)에 따라 환경 디렉터리의
  16개 채널을 하나의 원녹음 그룹으로 묶는다. 같은 녹음의 16/48 kHz 배포본과 잘라낸 구간도
  같은 그룹을 유지한다. 이는 환경 간 녹음 세션·부분 중복 부재까지 인증하는 것은 아니다.
- MIMII DG fan의 section은 장치 ID가 아니라 혼합 기계음·공장 배경음·SNR의 도메인 변화다.
  [저자 논문 §2](https://arxiv.org/html/2205.13879)와
  [공식 CSV 규약](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)을
  실제 attributes CSV 3개와 대조했다. 각각 데이터 1,200행, `file_name,d1p,d1v` 3열이며
  3,600개 파일 목록과 CSV hash는 기존 archive QA inventory와 일치했다. 원 팬 녹음 및
  혼합 배경음의 부모 녹음·시간 구간 ID는 없으므로 section 간 부모 녹음 독립성은 여전히 미확인이다.
- 현 strict 기본 계약은 5계열 각각 train/val/test 존재를 요구한다. MIMII에 임의 group ID를
  발급해 READY 검사를 우회하지 않는다. 학습 보조 전용 사용 또는 보류에 대한 사용자 결정은
  대기 중이며 아직 채택하지 않았다. 출처 그룹·평가 정책 확정 후 실제 음성·음악도 독립 평가한다.
- 학습된 선택기/계수 생성기, 온라인 S/F 식별, F 보상, 오디오 스레드 통합은 후속 구현이다.
- Jetson 이미지·CUDA/TensorRT·장치/경로/THD·실제 지연·ERR OFF–ON–OFF는 [docs/14](14_pc_jetson_workplan.md)를 따른다.

약 2.9 ms의 REF 기하선행 추정과 기록된 약 35.9 ms의 제어음 지연 차이는 그대로다. 필터를 사전 측정해도
불규칙한 광대역의 미래를 알 수 없다는 제약은 없어지지 않는다. 새 코드·데이터 준비를 실제 감쇠 달성과 분리한다.
