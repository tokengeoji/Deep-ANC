# 2026-08-31 학습·백업·bootstrap 권위 상태

이 문서는 Elice 크레딧 종료 직후 실제 Jetson, GitHub remote, Google Drive를 다시 읽어
확인한 상태다. 문서의 존재를 학습 완료나 성능 증거로 해석하지 않는다.

> [!CAUTION]
> 이 문서는 2026-08-31의 historical snapshot이다. 2026-09-01 actual USB output-master
> split-clock failure 뒤에는 여기의 Stage-1 19 additions/101-session·schema-v1/v2 진행선을
> 현행 학습·녹음 authority로 사용하지 않는다. 현행 순서는 same-card RT5640/J511 S32 actual
> P/S → Stage-2 47개 독립 component source plan/QA → transfer-v3 → exact Elice bootstrap이며,
> `HANDOFF.md` 최상단과 `docs/75`가 우선한다.

## 1. 학습 판정

- canonical surrogate pretrain 100k: **미시작**
- canonical measured fine-tune 50k: **미시작**
- canonical checkpoint/G4/현장 ON-OFF raw: **없음**
- 로컬 `runs/`의 2026-08-02~05 checkpoint는 legacy/diagnostic-only이며 init, resume,
  모델 선택 또는 현재 성능 근거로 재사용하지 않는다.
- 종료된 Elice 인스턴스에서 완료된 것은 exact checkout, 전체 pytest, public manifest,
  decoder-audit 재검증, DNS/DEMAND source 선택까지다. 최신 선택 bundle은 인스턴스 종료
  전에 Jetson으로 전송되지 않았으므로 다음 인스턴스에서 재발행한다.

## 2. GitHub

2026-08-31 readback 기준:

| ref | SHA | 판정 |
|---|---|---|
| 마지막 clean local `dev` | `65ebd73df51a686865a2eb7e7532d1b1bea2a78c` | clean 기준선 |
| `origin/dev` | `65ebd73df51a686865a2eb7e7532d1b1bea2a78c` | 기준선과 exact |
| local/`origin/main` | `67572b14436f65c917f3cd4bb18e64b121898fa1` | 배포 기준선 유지 |

실제 remote URL은 `git@github.com:tokengeoji/Deep-ANC.git`이다. 환경 문서의
`Roka-jsj/Deep-ANC` URL과 양쪽을 read-only `ls-remote`한 결과 `main`/`dev` SHA가 각각
exact하게 같았다. remote에는 `main`, `dev` 두 branch만 있고 tracked private-key/rclone
config/`.env` 파일과 private-key header는 `0`이었다.

Git에는 코드·계약·문서만 보존한다. raw, selector bundle, decoder audit, checkpoint는 Git
백업으로 간주하지 않는다.

이 문서를 포함한 gainprobe006/archive-cache 변경은 전체 검증 뒤 새 clean `dev` commit으로
push한다. 따라서 위 SHA를 새 물리 plan이나 archive manifest의 expected commit으로 사용하지
않고, 실행 직전에 `git rev-parse HEAD`와 `origin/dev`가 exact한지 다시 읽는다.

## 3. Google Drive

### 3.1 완료된 Jetson 고정 데이터 snapshot

`DeepANC/jetson_data_backup_20260827/data`를 Drive에서 다시 전수 열거한 결과:

- 파일: `13,428`
- 파일 bytes 합계: `17,439,445,191`
- `data_backup_manifest.sha256`: `13,428`행
- manifest SHA-256:
  `1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2`
- 현재 `elice_transfer_manifest.json`의 `data/` 입력 `337/337`은 위 manifest에 모두
  존재하고 SHA-256 mismatch가 `0`이다.

따라서 2026-08-27에 고정한 data snapshot과 현재 82-session transfer의 data 부분은
완전하다. 이후 새로 생긴 파일 전체를 포함한다는 뜻은 아니다.

### 3.2 bootstrap decoder-audit cache

새 no-replace 폴더 `DeepANC/bootstrap_cache_b074`에 다음 완료 audit을 별도 업로드했다.

- Drive 파일: `decoder_audit_917aa25a.json`
- bytes: `90,424,631`
- 파일 SHA-256:
  `ed3b379827f1a38b170f3ab394a689e8164a0251d8f5ff84642f61765a15c9e8`
- local/Drive MD5 readback:
  `f9da2fed203abf3343a2fc9eda1ad924`
- 내부 `audit_sha256`:
  `0b39019a347f3a555d73c45b455e47b62f51a2faf52138c664eb85ddc1e56975`
- 후보/accept/reject: `37,761 / 36,868 / 893`

이 audit은 decoder policy/fingerprint와 raw 37,761개의 path/size/SHA가 새 인스턴스에서도
모두 같을 때만 `verify_decoder_audit_reuse.py`가 승인한다. Drive에 있다는 이유만으로
재사용하지 않으며, 검증 실패 시 자동 fallback도 하지 않는다.

같은 folder의 `transfer_authority_b074/`에는 current transfer manifest, holdout, strict P/S
NPZ와 raw/analysis, meter, provenance report 총 `11`개를 상대경로 그대로 보존했다.
`rclone check --one-way` readback은 `0 differences / 11 matching`이었고, decoder audit을
포함한 folder 전체는 `12`개 파일, `102,458,703`바이트다.

### 3.3 완료되지 않은 Elice snapshot

`DeepANC/elice_snapshot_20260830/.../archives/parts`는 `10`개 중 `9`개만 존재한다.
`deep_anc_project_predelete_917aa25a.tar.part-0005-of-0010`이 없으므로 전체 tar 복원본으로
사용하지 않는다. 최신 `b074...` DNS/DEMAND selector bundle도 이 snapshot에 없다.

### 3.4 2026-08-28 Elice 결과 snapshot의 정확한 범위

`DeepANC/elice_snapshot_20260828/predelete_49dd6c7/manifests_results_runs.tar`는
Drive 객체를 로컬 디스크에 복제하지 않고 끝까지 스트리밍해 tar 구조와 SHA를 다시
검사했다.

- bytes: `479,846,400`
- SHA-256: `bfa3d7de91e747049eeebde99d844b87c4973bbfa1985a63e9c49cdc10542589`
- tar member: `275`개 (`data` 23, `results` 156, `runs` 96)
- 포함 범위: 당시 manifest, bootstrap/evaluation 결과, diagnostic run
- 제외 범위: public raw corpus와 Elice venv

따라서 이 tar는 과거 결과의 forensic 복구본으로는 완전하지만 새 Elice의 public raw
cache가 아니다. 2026-08-30 split tar도 한 part가 없으므로, 현재 Drive 어디에도 Elice
public raw 37,761개를 완전하게 복원할 수 있는 단일 backup은 없다.

### 3.5 2026-08-31 gain probe 실패 forensic backup

첫 bounded v2 capture는 `.003/.006/.009`까지만 완료됐고 `.009` REF peak
`0.4177616`에서 다음 `.012` 출력 전에 안전 중단됐다. raw는
`results/recording_gain_linearity_v2/ab0d2402f6c96232/raw_measurement.npz`
(SHA-256 `e2616dfbabfeb18afd79a1d0c00c4bb191c9bf93b1224975a736bf8b924eee1e`), FAIL receipt는
`results/data_audit/recording_gain_linearity_v2_receipt.json`
(SHA-256 `d888948d46fa8748fd8d9f15f155acbb5c529b00865f54f6773e19920972608e`)다.
Drive `DeepANC/jetson_measurements_20260831/forensic_fail/recording_gain_linearity_v2/`
아래 plan/receipt/metadata/raw/manifest 5개는 `rclone check --download`에서
0 differences/5 matching이었다. 이 backup은 실패 원인 분석용이며 source-gain authority가
아니다.

### 3.6 public archive transport cache 구현 상태

DNS noise 2 + DNS speech 1 + DEMAND 6 + MIMII fan 1의 fixed archive 10개,
총 `18,229,762,015`바이트를 Jetson `/dev/shm`에 하나씩만 staging해 Drive로 immutable
발행하는 도구를 추가했다. publisher는 exact Git tree와 entry/pget blob, provider checksum,
archive traversal/CRC/bzip2, member inventory를 검증하고 각 remote 객체를 download-readback한
뒤 manifest를 마지막에 발행한다. restore는 external manifest SHA와 exact commit을 요구하며,
cache origin receipt 없는 기존 archive를 일반 download 결과로 세탁하지 못한다.

현재 actual archive download와 Drive publish는 **0회**다. 이 cache는 transport acceleration일
뿐 extracted raw, decoder audit, selector 또는 training readiness authority가 아니다.

## 4. 다음 Elice에서 비싼 전수 decode를 먼저 반복하지 않는 순서

Elice를 켜기 전에는 현재 Jetson만으로 가능한 코드·물리 gain/linearity probe·schema-v1
transfer·GitHub/Drive backup을 먼저 닫는다. 다만 19개 추가 세션의 DNS/DEMAND source와 최종
source-gain plan은 public raw를 가진 Elice의 schema-v1 selector가 있어야 확정되므로, 이를
Elice 이전 완료 항목으로 잘못 세지 않는다. 인스턴스를 준비 작업 대기용으로 유지하지 않고,
첫 full bootstrap 직후 selector를 발행한다. Jetson의 짧은 19세션 수집·QA와 schema-v2 이관 뒤
같은 cache의 두 번째 full bootstrap을 통과하면 즉시 G0/pilot/smoke와 승인된 학습으로
이어간다. Jetson에는 public raw 전체를 새로 staging하지 않는다.

Drive에는 현재 Jetson authority와 decoder audit이 있지만 Elice public raw 37,761개의
완전한 복원본은 아직 없다. clean commit 뒤 위 fixed archive cache를 Jetson에서 먼저
발행·readback하면 새 Elice는 official source를 다시 받는 대신 그 archive를 local SSD에
no-replace 복원할 수 있다. cache 발행이 완료되지 않으면 첫 학습 인스턴스가 official source에서
한 번 받아야 한다. 어느 경우에도 Drive mount를 학습 dataset의 직접 random-I/O 경로로 쓰지
않고 Elice local SSD를 작업 cache로 사용한다.

1. GitHub `dev`의 신뢰한 전체 40자리 SHA를 clean detached checkout한다.
2. Jetson의 schema-v1 82-session transfer와 Drive decoder audit을 새 Elice로
   스트리밍하고, code/holdout/transfer/GPU/storage의 cheap gate를 먼저 통과한다.
3. archive cache가 발행됐다면 publisher manifest file SHA와 exact commit을 외부 anchor로
   전달해 `--archive-cache-only` restore를 먼저 수행한다. 이 단계는 raw/bootstrap receipt를
   0개만 발행해야 한다. 이어 같은 anchor를 붙인 **full bootstrap**이 archive를 해제하고
   audit fingerprint와 raw 37,761개 SHA를 전수 대조한다. cache가 없다면 full bootstrap이
   official URL에서 한 번 받는다.

   ```text
   --reuse-decoder-audit
   --expected-decoder-audit-sha256 \
     0b39019a347f3a555d73c45b455e47b62f51a2faf52138c664eb85ddc1e56975
   --expected-decoder-audit-file-sha256 \
     ed3b379827f1a38b170f3ab394a689e8164a0251d8f5ff84642f61765a15c9e8
   --raw-hash-workers 8
   ```

4. schema-v1 bootstrap receipt에서 같은 commit의 DNS/DEMAND selector를 발행하고 즉시
   Jetson과 Drive에 각각 전송해 독립 verifier를 통과시킨다.
5. Jetson에서 새 source plan과 source-gain authority로 19세션을 수집·QA한 뒤 101-session
   generation과 schema-v2 transfer를 같은 Elice cache로 보낸다.
6. schema-v2 SHA로 full bootstrap을 재실행한다. 이미 있는 public raw/venv를 재사용하되
   exact 검증을 반복하고, readiness가 init만 FAIL인 상태일 때 campaign 진입점으로 즉시
   G0/pilot/probe/smoke/100k/50k를 진행한다.

`--cache-preflight-only`는 public raw와 exact venv가 이미 존재하는 재개 인스턴스에서만
사용한다. fresh 인스턴스에서 이를 먼저 호출해 raw missing으로 실패시키지 않는다. schema-v1
receipt는 selector 발행까지만 쓰며 canonical 학습 입력으로 승격하지 않는다.

2026-08-31 Jetson에서 official download endpoint를 실제 HEAD와 1-byte Range로 확인했다.
DNS 세 archive는 각각 `5,364,611,964`/`5,357,916,291`/`4,664,045,287`바이트이며 strong
ETag와 HTTP 206을 반환했다. FMA-small/metadata도 `7,679,594,875`/`358,412,441`바이트,
strong ETag와 HTTP 206을 반환했고, Zenodo MIMII fan도 `928,511,244`바이트와 HTTP 206을
반환했다. 이는 현재 URL 접근성과 range 호환 증거이지 다음 인스턴스의 다운로드 시간을
보장하는 SLA는 아니다.

Jetson의 현 `gdrive:` remote는 2026년 중 종료 예정인 rclone shared Google Drive
client ID 경고를 실제로 출력한다. 현재 readback은 성공했지만 향후 bootstrap의 유일한
복원 경로로 간주하지 않는다. 사용자 소유 OAuth client로 전환하기 전까지 GitHub의 코드,
Drive 객체 SHA/manifest, official public source URL을 서로 독립적인 세 경로로 유지한다.

정상 bootstrap 시간을 아직 숫자로 보장하지 않는다. 과거 장시간의 직접 원인이었던
manifest-entry별 전체 index 재구축 O(N^2) 결함은 process-local index 1회 구축으로 수정됐지만,
fresh download와 fresh full decode는 여전히 비싼 작업이다. 따라서 cache preflight, 조기
focused test, 진행 상태/종료 receipt를 통과하지 않은 실행을 장시간 방치하지 않는다.

## 5. 이번 통합의 검증 결과와 남은 물리 경계

이 문서를 포함한 `dev` tip에서 변경 Python compile, Elice shell syntax, static
pytest/SHA reference audit, 관련 통합 focused 묶음과 전체 pytest를 실행했다. 최종 전체
pytest는 **0 FAIL**이고 선택적 실기 test 3개만 skip됐다. private key는 `*.pem` ignore에
머물며 tracked 비밀정보 pattern은 0이다.

새 Elice용 코드 경계는 다음과 같다.

- `bootstrap_all.sh`는 expected commit/holdout/transfer SHA, clean tree, A100 80GB와
  storage를 조기에 검사하고 range-resume downloader의 strong ETag와 block hash를 검증한다.
- schema-v1 selector bundle과 schema-v2 training bundle을 한 실행으로 오인하지 않는다.
  `push_transfer_bundle.py`가 Jetson의 exact authority만 전달하고, current generation
  `stage1-coverage-v4-gainprobe006`는 물리 gain receipt와 19/19 source feasibility가 없으면
  발행되지 않는다.
- `run_canonical_campaign.py`는 완료 상태 문구가 아니라 raw G0/gradient/pilot/probe/
  completion/evaluation artifact를 다시 검증해 다음 단계 하나만 실행한다. 경계 결과의
  두 번째 seed도 별도 fixed prerequisite와 같은-seed 100k init을 요구한다.

따라서 다음 Elice가 생기기 전에 남은 핵심은 Jetson의 bounded physical gain probe와 archive
cache 실발행이다. 현행 probe는 `.003/.004/.005` fit + 독립 `.006` holdout이며 nominal
active `15.615667초`, exact int16 nonzero `13.821375초`, output-open `26초`, 실제 monotonic
software hard deadline `37초`다. 같은 plant를 지난 pilot response 5개로 clock trajectory를
검사하고, low-SNR distortion은 `INCONCLUSIVE/distortion_certified=false`로 남긴다. PASS도
tested range ADC peak safety만 인증한다. 아직 actual v3 raw/PASS receipt는 없다.

이 raw와 PASS receipt 및 archive cache manifest를 Drive에서 SHA readback하기 전에는 Elice
selector를 canonical additions plan으로 승격하지 않는다. 이 PASS 뒤에도 DNS/DEMAND selector와
19-session 수집이 남으므로 “파인튜닝 준비 완료” 또는 “학습 완료”라고 쓰지 않는다.
