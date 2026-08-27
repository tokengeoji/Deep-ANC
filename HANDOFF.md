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
- canonical `recorded_regrouped.jsonl` 전수 QA는 82/82 세션·95.67분·오류/경고 0으로 통과했다.
  불변 `session.json`의 원본 pool group과 재그룹화 manifest의 lineage group을 직접 비교하던
  QA 결함을 수정했으며, 회귀 테스트와 전체 pytest도 0 FAIL이다.
- 수정 커밋 `cef615ec40b18e26c1fe3e7fa53a09c715cb7a67`은 원격 브랜치에 push 완료했다.
- `check_finetune.py`는 외부 Elice `bootstrap_receipt`가 없어 의도적으로 중단된다. strict P/S와
  로컬 계보는 통과했지만, Elice corpus/receipt 없이는 readiness를 통과시킬 수 없다.

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

## 6. 아직 외부에서 필요한 것

- 현재 접근 가능한 Elice A100 80GB 인스턴스와 128GiB 저장공간
- strict P/S 라이브 측정 시 사용자의 현장 입회와 배선/노브 확인
- public raw corpus 다운로드(로컬 디스크에는 staging하지 않음)
- canonical 100k+50k 학습 시간

이 항목들은 코드로 우회하거나 legacy artifact로 대체하지 않는다.
