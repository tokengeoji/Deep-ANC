# HANDOFF — 파인튜닝 준비 복구 상태

> “이어서 진행해줘”를 받으면 이 파일과 `AGENTS.md`를 먼저 읽는다.
> 최종 갱신: 2026-08-27. 작업 브랜치: `fix/finetune-readiness-repair`.

## 0. 현재 결론

파인튜닝은 아직 시작하지 않는다. 이번 브랜치는 준비 계약과 계보를 복구하는 중이며,
다음 증거가 모두 생긴 뒤에만 canonical 학습을 연다.

1. (완료) 새 strict P/S 캡처와 level evidence
2. (로컬 완료) 해당 P/S·82세션·계보 자료를 결속한 Elice transfer manifest
3. Elice에서 재생성한 public corpus manifest 6종과 전체 QA
4. 선택된 계약으로 처음부터 완료한 tiny 100k canonical init checkpoint

과거 `pretrain_*_corrected`, `finetune_tiny`, legacy P/S는 삭제하지 않지만 모두
diagnostic-only다. init, resume, 모델 선택, 성능 주장의 근거로 사용하지 않는다.

라이브 측정은 전체 테스트, 무음 dry-run, 장치 점유·CPU gate와 사용자 입회 뒤 실행했다.
측정 종료 직후 오디오 스트림은 닫혔고 스피커 분리 안내를 출력했다.

2026-08-27 확인 결과:

- strict P/S 실제 덕트 측정은 xrun/clip 0과 ERR/REF 입력 무클리핑으로 통과했다.
  P `primary_path_il_strict_5dc06fdd.npz`는 bulk 1642/effective 1386샘플,
  S `secondary_path_il_strict_5dc06fdd.npz`는 bulk 1501/effective 1245샘플이며,
  handoff 256에서 `PlantDelays.lead()`가 lead 115샘플을 유도한다.
  150–1600Hz consistency는 P 0.9998, S 0.9997, kept repeats 19/64,
  clock drift 중앙 2.48샘플/주기, fractional joint-LS·cubic crosscheck·compact round-trip이 PASS다.
- paired level evidence `assets/measured/measurement_level_evidence.json`도 PASS다
  (meter -48.278 dBFS, interleaved ERR -48.246 dBFS,
  SHA-256 `c76ac0d3c52c20fadd761d1ed0c85e27e3599328f60ca0d164535594336e73d0`).
- strict raw/analysis와 P/S를 묶은 로컬 transfer manifest를 생성했다
  (SHA-256 `39dc271672ac2916840a9919baaf7de5bdf078d228a68457f15096d433a76b4d`,
  344 files/82 sessions).
- Elice `~/Deep_ANC`에 transfer bundle을 전송하고 exact checkout/holdout/strict P·S
  SHA를 원격에서 대조했다. 이후 full `NVIDIA A100 80GB PCIe`(world-size 1)를 확인하고
  public raw 6종을 untouched 상태로 다운로드·manifest 재생성·QA했다. bootstrap 전체
  pytest는 0 FAIL이며 receipt SHA는
  `f56c3d1042211112627380f74315d5949f05bcf274bdcf3fefc588ea3d3caa7e`다.
- 현재 Elice에서 seed `20260803`의 승인 loss grid 4개(각 20k surrogate pilot)를
  순차 실행 중이다. 2026-08-27 마지막 확인 시 `alpha=0.7, lambda_frame=0.5`,
  `alpha=0.7, lambda_frame=0.2`, `alpha=1.0, lambda_frame=0.5`는 20k를 완료했고,
  앞의 두 후보는 val CVaR가 각각 +0.02/+0.00 dB, alpha 1.0/frame 0.5는
  +0.20 dB(최악 +3.07 dB)로 불안정했다. 네 번째 `alpha=1.0, lambda_frame=0.2`는
  약 6.1k/20k에서 진행 중이었다. 모든 후보 checkpoint는 init으로 승격하지 않는다.
  현재 Elice checkout은 실행 중단을 피하기 위해 `2d19f140…`에 남겨 두었으며, pilot·
  decoder warning 점검·gradient·G0·measured probe·A100 resume 증거를 ledger로 묶은
  뒤 canonical 기준선 `c21fe1a`로 맞춘다.
- canonical `recorded_regrouped.jsonl` 전수 QA는 82/82 세션·95.67분·오류/경고 0으로 통과했다.
  불변 `session.json`의 원본 pool group과 재그룹화 manifest의 lineage group을 직접 비교하던
  QA 결함을 수정했으며, 회귀 테스트와 전체 pytest도 0 FAIL이다.
- QA 수정 커밋 `cef615ec40b18e26c1fe3e7fa53a09c715cb7a67`, strict 자산 승격 커밋
  `4c55386`, Elice 이관 상태 문서 커밋 `86c5c45`, 데이터/손실 안정화 커밋
  `2d19f14`, 예산 근거·진행 문서 커밋 `bd2a0cf`/`43563be`가 모두 원격 브랜치에
  push 완료됐다. 현재 준비 기준선은 clean 상태로 push된
  `fix/finetune-readiness-repair`의 HEAD이며, 실행에 사용할 exact SHA는
  `git rev-parse HEAD`로 확인한다. 브랜치별 범위는 `docs/08_dev_workflow.md §7`에 고정했다.
- Elice receipt가 생긴 뒤 `check_finetune.py`의 외부 입력 차단은 해소됐지만, canonical
  init checkpoint·campaign ledger가 아직 없어 readiness는 의도적으로 15/15가 아니다.
- 2026-08-27 고주파 진단 캡처는 공식 자산과 분리해 수행했지만 유효한 clock witness를
  만들지 못했다. `results/experimental_high_band/20260827_fullband/20260827_203328_1b24d0c2/`
  의 immutable raw에서 ERR/REF 공통 clock 유효 주기가 0개(최소 8, score≥0.995)로 판정되어
  `Invalid experiment`이다. xrun/clip은 0이지만 P/S NPZ를 만들지 않았고 2/4/8kHz 성능
  숫자로 사용하지 않는다. 고주파 재측정은 자극을 좁은 대역으로 재설계한 뒤 별도 dry-run과
  사용자 입회 절차를 통과해야 한다.
- 처음 듣는 소리 검증을 Level 1–5로 고정한 OOD 게이트를 `docs/07_evaluation_protocol.md
  §3.1`에 기록했고 기준선 commit `98df0b0`으로 push했다. 현재 Level 5(모델 선택 후 실제
  덕트 새 녹음) raw/session artifact는 아직 없으므로 현장 OOD 일반화는 `Not yet
  demonstrated`이다. 이 challenge는 학습·val 선택·test에 재사용하지 않는다.
- 현재 기준선 `c21fe1a`에서 전체 pytest는 **0 FAIL**(경고는 로컬에 없는 downstream public
  manifest를 진단 fixture가 알리는 것), `bash -n`과 `git diff --check`도 통과했다. Elice의
  canonical 실행 전에는 detached checkout `2d19f140a66e3d0264694e8f2e2941bce4fbd3bc`를
  기준선 전체 SHA로 다시 맞춘다.

## 1. 구현된 계약

### 시간축

- `TrainingTimingContract`가 strict P/S NPZ, compact FIR peak, 256-sample handoff,
  `PlantDelays.lead()`를 구분해 합성·실측 총 선행량을 유도한다.
- config, 합성/실측 dataset, readiness, 평가가 같은 계약을 소비한다. P/S delay나 lead를
  YAML과 문서에 수동 숫자로 복사하지 않는다.
- `recorded_lead_mode: timeline`은 합성 총 선행량에서 세션별 정렬 잔여 지연을 빼서
  실측 branch lead를 만든다. 공식 1차 실행의 lead jitter와 session mixing은 0이다.

### 체크포인트와 재현성

- 체크포인트는 commit, model/stage, loss, optimizer/schedule, P/S·RIR·manifest,
  sampler/augmentation, seed와 selection metric을 포함한 `experiment_contract_sha256`를
  저장한다.
- 자동 resume은 없다. `--resume`은 같은 전체 계약의 `last.pt`에만 허용한다.
  `init_ckpt`는 완료된 canonical surrogate-pretrain의 weight-only 전이다.
- 공식 경로는 `runs/<stage>_<contract-sha>_<seed>` 형식의 신규 디렉터리이며,
  기존 결과를 덮어쓰지 않는다.
- 공식 경로는 A100 한 장(`required_world_size: 1`)이다. global sample index 기반 sampler와
  RNG 상태를 사용하며, 중단/재개 등가를 smoke에서 확인해야 한다.

### 측정

- `MeasurementLevelContract`가 meter와 strict P/S의 probe peak `0.003`을 공유한다.
- meter raw와 strict raw는 실제 제출 int16/수신 PCM, 장치·clock·recipe, SHA receipt를
  보존한다. strict 분석은 level target 대응을 raw에서 오프라인 재검증한다.
- 출력 stream이 닫힌 직후 저장·분석보다 먼저
  `[스피커 출력 종료 — 지금 스피커/앰프를 분리하세요]`를 출력한다.
- `--confirm-user-present`, `--confirm-volume-minimum`, routing/geometry와 same-amplifier
  확인이 빠지면 장치를 열기 전에 실패한다. PCM 점유, CPU idle, clock gate도 선행한다.

### 데이터와 Elice

- v1/v2 historical builder를 재현해 160개 source WAV 불변성과 CSV prefix를 검증하고,
  누락 `sources.csv`만 복구한다. `identify_pool_clips.py`는 진단용이지 권위 자료가 아니다.
- FMA artist/album, speech speaker/book, 원본 clip 공유 관계의 transitive component를
  절대 나누지 않는 `recorded_regrouped.jsonl`을 만든다.
- active 82세션만으로 canonical holdout을 만들고, synthetic manifest는 raw audio content
  SHA와 holdout SHA를 결속한다.
- content-addressed provenance report와 transfer manifest가 recorded 전체 파일, RIR,
  strict raw/analysis/P/S, regrouped manifest, FMA tracks, holdout, 두 CSV를 한 번에 결속한다.
- Elice bootstrap은 전체 40자리 commit, holdout SHA, transfer manifest SHA, `--no-update`를
  요구한다. dirty/숨김 index/graft/replace/symlink/byte mismatch는 다운로드 전에 실패한다.

## 2. 지금부터의 로컬 실행 순서

### 2.1 코드·계보 게이트

에이전트 작업이 모두 합쳐진 뒤 아래를 순서대로 실행하고 결과를 이 문서에 기록한다.

```bash
.venv/bin/python scripts/data/repair_source_pool_provenance.py \
  --repair-csv --write-active-holdout --write-regrouped-manifest --jobs 4

.venv/bin/python -m pytest -q
bash -n scripts/elice/bootstrap_all.sh scripts/elice/setup_env.sh
git diff --check
```

로컬에는 untouched public raw 6종이 없으므로 provenance 명령에
`--require-downstream-gates`를 붙이지 않는다. downstream synthetic gate가 BLOCKED인 것은
정상이며 Elice raw를 확보한 뒤에만 연다.

오디오 없는 공식 dry-run:

```bash
.venv/bin/python scripts/data/set_amp_level.py --self-test
.venv/bin/python scripts/data/measure_paths_interleaved.py \
  --dry-run --bootstrap-level-evidence \
  --primary-out assets/measured/_dry_run_primary_do_not_create.npz \
  --secondary-out assets/measured/_dry_run_secondary_do_not_create.npz \
  --diagnostics-root results/_dry_run_measurement_do_not_create
```

dry-run 뒤 위 세 경로가 생성되지 않았는지 확인한다. 그 다음 비밀정보·대용량·ignored data가
staging에 없는지 검사하고 한국어 메시지로 commit/push한다. 커밋 메시지에 AI 표기를 넣지 않는다.

### 2.2 strict P/S 라이브 측정

커밋·push 뒤에도 자동으로 소리를 내지 않는다. 먼저 read-only로 다음을 확인한다.

```bash
fuser -v /dev/snd/*
for f in /proc/asound/card*/pcm*/sub*/status; do printf '%s: ' "$f"; cat "$f"; done
```

다른 프로세스가 점유하면 중단한다. 두 마이크 입력, ERR/REF와 noise/cancel speaker 배선,
사용자 입회, 볼륨 최소, 즉시 분리 준비를 다시 확인한 뒤 다음 두 출력 창만 실행한다.

```bash
# 1) input-only 1.5초 후 nominal 20.0초, hard max 21.0초 출력
.venv/bin/python scripts/data/set_amp_level.py --bootstrap-level-evidence \
  --confirm-speaker --confirm-user-present --confirm-volume-minimum

# 출력 종료 즉시 분리. 노브를 바꾸지 않고, 10분 안에 다음 명령 직전에만 재연결한다.
# meter가 출력한 METER_RAW와 strict 명령 전체를 그대로 사용한다.
METER_RAW=results/calibration_interleaved/level_bootstrap/<session>/meter_raw.npz
.venv/bin/python scripts/data/measure_paths_interleaved.py \
  --bootstrap-level-evidence --meter-raw "$METER_RAW" \
  --confirm-same-amplifier-setting --confirm-user-present \
  --confirm-volume-minimum --confirm-routing-and-geometry \
  --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \
  --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
```

strict stream은 input-only preflight 3초 뒤 nominal 12.5초, hard max 13.5초다. nominal audible
합계는 32.5초다. 각 출력 close 직후 즉시 분리한다. 실패해도 재측정하지 않고 immutable raw를
먼저 분석한다. 기존 legacy NPZ는 덮어쓰지 않는다.

합격 조건은 48 kHz/256/low, 지정 channel/operator, xrun/clip 0, observed PCM과 raw/analysis
SHA, clock-q witness, fractional joint-LS+cubic crosscheck, compact round-trip,
150–1600 Hz 모든 부대역 consistency ≥0.9406, kept repeats ≥8, 안정적인 P−S 상대 지연이다.
임계값은 낮추지 않는다.

## 3. 계보와 Elice 이관

strict P/S가 합격한 뒤 canonical transfer manifest를 만든다. 실제 capture 파일을 모두
`--strict-raw`/`--strict-analysis`로 열거한다.

```bash
EXPECTED_HOLDOUT_SHA256=$(sha256sum data/manifests/recorded_holdout.json | awk '{print $1}')
.venv/bin/python scripts/data/build_elice_transfer_manifest.py \
  --rir-bank data/rir_bank/duct_rirs_v1.npz \
  --strict-raw results/<capture>/raw_measurement.npz \
  --strict-raw results/<capture>/metadata.json \
  --strict-analysis results/<capture>/analysis_results.npz \
  --strict-analysis results/<capture>/analysis_metadata.json \
  --primary-npz assets/measured/primary_path_il_strict_<capture-id>.npz \
  --secondary-npz assets/measured/secondary_path_il_strict_<capture-id>.npz \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256"

EXPECTED_TRANSFER_MANIFEST_SHA256=$(sha256sum \
  data/manifests/elice_transfer_manifest.json | awk '{print $1}')
```

manifest에 열거된 상대경로를 rsync/scp로 그대로 스트리밍한다. 로컬 tar 복제본을 만들지 않는다.
Elice에서는 exact detached checkout을 사용한다.

```bash
EXPECTED_COMMIT=<신뢰한-전체-40자리-SHA>
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update --preflight-only

# A100 80GB 1장·가용 128GiB 확인 뒤 full bootstrap
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update
```

bootstrap은 torch `2.5.1+cu121`, CUDA 12.1 계약, A100, 저장공간, public raw 수량과 FMA
metadata를 검증하고 manifest 6종을 untouched raw에서 재생성한다. noise/recorded QA,
전체 pytest와 readiness까지 통과한 정상 사전학습 출발 상태는 init 하나만 FAIL인 14/15다.

## 4. 공식 학습 순서

1. family→lineage component→session 균등 sampler와 공통 gain/polarity/EQ, input-only mic
   noise를 사용한다. session mixing과 lead jitter는 0이다.
2. strict S로 `lambda_dnh` gradient 비중 0.2–0.4를 확인한다.
3. 고정 batch G0에서 trusted NMSE < −6 dB와 lead metadata를 확인한다.
4. seed `20260803`, `alpha∈{0.7,1.0} × lambda_frame∈{0.5,0.2}`의 20k surrogate +
   5k measured probe를 recorded val로만 비교한다. 0.2 dB 이내 동률/불안정이면 alpha 0.85를
   추가하고, 계속 동률이면 alpha 0.7을 택한다. pilot checkpoint는 init 자격이 없다.
5. 선택 계약의 tiny를 새 run에서 100k 처음부터 사전학습한다. 200–500 step smoke에서 VRAM,
   처리량/ETA, 중단·재개 등가를 먼저 확인한다.
6. canonical init 지정 뒤 readiness 15/15를 확인하고 open-loop, recorded 70% + synthetic 30%,
   bf16 forward + FP32 loss, 50k fine-tune을 실행한다.
7. checkpoint 선택은 recorded val만 사용한다. 선택을 고정한 뒤 test를 정확히 한 번 연다.
   경계 0.3 dB 이내 또는 INCONCLUSIVE일 때만 seed `20260903`의 100k+50k를 한 번 더 한다.

공식 test G4는 trusted 150–1600 Hz 평균/모든 family 평균/최악 10%/family cluster-bootstrap
95% CI 상단이 모두 0 dB 미만, fullband 평균 ≤0 dB, 대역 밖 octave 최악 10% 증폭 <1 dB이며
판정이 PASS여야 한다.

## 5. G4 이후

G4 PASS 뒤에만 완전 미사용 natural-crest source로 speech/music/environment/machine 각 1세션
(1차 약 4분 40초 audible)을 녹음한다. 네 계열이 모두 개선되면 계열당 3개 독립 그룹을 더해
총 16세션으로 확장한다(누적 최대 약 18분 40초 audible). challenge는 학습에 쓰지 않는다.

G4와 crest challenge를 모두 통과하기 전에는 closed-loop, ONNX export/배포, 실제 ANC ON
평가로 진행하지 않는다.

## 6. 아직 남은 실행 항목

- 진행 중인 4개 loss pilot 완료 및 recorded-val 기준 winner 선택
- winner의 5k measured probe, 실제 A100 bf16 중단→resume 수치등가 smoke, G0·gradient
  ledger 작성 및 SHA 결속
- canonical tiny 100k surrogate-pretrain init checkpoint 생성 후 readiness 15/15 확인
- canonical measured 50k fine-tune, 고정 checkpoint의 단 한 번 G4 평가
- G4 PASS 뒤 natural-crest challenge 녹음·평가

이 항목들은 코드로 우회하거나 legacy artifact로 대체하지 않는다.

## 7. 외부 폴더 감사 및 저장소 정리(2026-08-27)

- `/home/capston/DeepANC_CRN_n_codex/duct_cnn_anc`는 읽기 전용으로 전수 감사했다.
  논문형 3,232-parameter 모델과 Primary 진단 raw는 있었지만 checkpoint/학습/ONNX가
  없고, Secondary는 전기적 speaker-input이 없는 proxy와 입력 프레임 불일치로
  `Invalid experiment`이다. 현행 P/S·lead·학습 계약은 변경하지 않는다.
- 외부 감사 상세와 재사용/차단 목록은 `docs/11_external_duct_cnn_audit.md`에 기록했다.
- 명백한 임시 readiness snapshot 두 디렉터리, 종료된 PID의 stale audio lock, 저장소
  Python/test cache는 휴지통으로 이동했다. 데이터·raw·RIR·checkpoint·legacy 결과는
  보존했다. 삭제/보존 목록과 SHA는 `docs/13_repository_cleanup_20260827.md`에 있다.
- 현재 branch HEAD는 외부 감사 기록과 후속 정리 변경을 포함한 최신 commit이다. Elice의
  진행 중 pilot은 중단하지 않고 종료 뒤 이 branch의 exact commit으로 동기화한다.
- 2026-08-27 22:03 KST read-only poll에서 Elice pilot parent PID 58467과 네 번째
  `alpha=1.0, lambda_frame=0.2` worker가 살아 있었고 로그는 약 9.3k/20k까지 진행됐다.
  A100 80GB는 GPU 56%, VRAM 6.6/81.9 GiB, Elice 디스크는 80 GiB 여유였다. 세 완료
  후보와 진행 후보 모두 `libmpg123`의 MP3 dequantization/illegal-header 경고를
  남기고 있다(완료 로그 각 159건, 진행 로그 76건 이상). 경고가 발생한 원본을 decoder
  audit로 분류하기 전에는 어떤 pilot도 canonical init이나 최종 성능 근거로 승격하지
  않는다. 실행 프로세스는 중단하지 않았다.
- 2026-08-27 22:25 KST 재확인에서 같은 worker가 약 13.8k/20k, 3.4 step/s로 진행 중이었다.
  단순 속도 외삽상 네 번째 pilot 자체는 약 30–35분 뒤 종료 예상이지만, 이는 decoder
  audit·winner 선택·measured probe·resume smoke를 포함하지 않은 추정이다. Elice에는
  `data/raw` 34 GiB, manifest 31 MiB, `runs` 110 MiB가 있고 전체 디스크 여유는 약
  80 GiB다. 현재 데이터는 이미 Elice에 있으므로 Jetson 용량 부족을 해소하기 위해
  같은 corpus를 다시 외부 저장소에서 중복 다운로드할 필요가 없다.
