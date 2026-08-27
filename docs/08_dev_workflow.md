# 08. 개발 워크플로와 프로젝트 정책

## 1. 저장소와 시스템 정책

| 정책 | 내용 |
|---|---|
| 읽기 전용 | `~/anc_project`, `~/DeepANC_CRN_n_codex`는 복사/참고만 가능하다. import나 cache 생성도 금지한다. |
| Jetson 변경 | 기본은 user-space만 사용한다. pinmux/device-tree는 되돌릴 방법을 먼저 마련한 경우에만 사용자 허용 범위에서 변경할 수 있다. 전원모드·jetson_clocks·apt·limits는 건드리지 않는다. |
| 오디오 | 사용자 입회, 볼륨 최소, ANC OFF로만 시작한다. 다른 저장소가 장치를 점유하면 측정하지 않는다. |
| 대용량 | `data/`, `runs/`, `*.pt`, `*.onnx`, `*.plan`은 Git에 넣지 않는다. transfer manifest로 외부 전송한다. |
| 비밀정보 | 키·토큰·개인키·환경변수 값을 commit/push하지 않는다. |
| 커밋 | 메시지는 한국어로 작성하고 AI attribution을 넣지 않는다. |

legacy FxLMS 실행은 정상 종료 때도 원본 디렉터리에 가중치를 쓸 수 있다. 재현이 꼭 필요하면
읽기 전용 원본에서 실행하지 말고, 복사한 코드와 저장소의 ignored 결과 경로를 사용한다.

## 2. Jetson ↔ GitHub ↔ Elice

```text
Jetson: 코드·계보·strict P/S 검증 → clean exact commit/push
                              ↓
GitHub: 신뢰한 40자리 commit SHA
                              ↓
Elice: detached checkout + transfer SHA 검증 → public raw/QA → 학습
```

recorded 4.11 GiB와 측정 raw는 Git이나 로컬 tar로 복제하지 않는다. content-addressed transfer
manifest에 열거한 상대경로를 rsync/scp로 스트리밍하고, Elice에서 모든 byte SHA와 aggregate를
다시 계산한다. 자세한 절차는 [docs/05](05_training_elice.md)다.

## 3. 재현성 계약

- `TrainingTimingContract`가 strict P/S NPZ, compact FIR peak, 256-sample handoff와
  `PlantDelays.lead()`를 구분한다. delay/lead를 config나 문서에 수동 숫자로 복사하지 않는다.
- `experiment_contract_sha256`는 clean commit, model/stage, 150–1600 Hz loss,
  optimizer/schedule, P/S·RIR·synthetic/recorded manifest, sampler/augmentation과 seed를 묶는다.
- 자동 resume은 없다. 같은 계약의 명시적 `--resume`만 허용하며 init은 완료된 canonical
  pretrain checkpoint의 weight-only 전이로 분리한다.
- 공식 실행은 A100 한 장이다. global sample index 기반 sampler/RNG와 checkpoint stochastic
  state로 중단/재개 수치 등가를 smoke에서 확인한다.
- test capability는 val selection과 campaign ledger에서 발급되며 캠페인 전체에서 정확히
  한 번만 소비한다.

## 4. 코드 품질 게이트

commit 전 최소 게이트:

```bash
.venv/bin/python -m pytest -q
bash -n scripts/elice/bootstrap_all.sh scripts/elice/setup_env.sh
git diff --check
```

테스트가 강제하는 핵심 불변식:

1. 모델 인과성과 streaming/offline 수치 등가
2. `e = d + S·y` 극성과 FP32 plant/loss
3. P/S bulk delay, compact FIR peak, handoff와 lead의 단일 timing contract
4. 256 배수 segment와 plant 적용 뒤 closed-loop warmup 절단
5. lineage component·holdout·raw content SHA 기준 무누수
6. checkpoint 전체 계약, explicit resume, completion receipt와 run no-overwrite
7. strict measurement raw/analysis provenance, xrun/clip/clock/consistency gate
8. val-only selection과 one-shot test capability

오디오 없는 측정 게이트도 commit 전에 실행한다.

```bash
.venv/bin/python scripts/data/set_amp_level.py --self-test
.venv/bin/python scripts/data/measure_paths_interleaved.py \
  --dry-run --bootstrap-level-evidence \
  --primary-out assets/measured/_dry_run_primary_do_not_create.npz \
  --secondary-out assets/measured/_dry_run_secondary_do_not_create.npz \
  --diagnostics-root results/_dry_run_measurement_do_not_create
```

## 5. 파인튜닝 준비 체크리스트

### 5.1 코드와 데이터 계보

- [ ] 전체 pytest 0 FAIL, clean exact commit, push
- [ ] v1/v2 historical builder 재현과 160 WAV exact/≤1 LSB 불변성
- [ ] active 82세션 canonical holdout
- [ ] shared clip/artist/album/speaker/book transitive component를 나누지 않은 regrouped split
- [ ] family별 val/test 독립 component 각각 ≥4
- [ ] synthetic train/val/test와 recorded holdout의 원본·계보 교집합 0

기존 82세션은 계보 복구 뒤 사용한다. 이전 문서의 “전량 붕괴/재녹음” 판정은 폐기됐다.
학습 전 93분 장시간 재녹음은 하지 않는다.

### 5.2 strict P/S

- [ ] 전체 테스트와 무음 dry-run
- [ ] `/dev/snd` 미점유, CPU idle, 두 입력과 ERR/REF 확인
- [ ] 사용자 입회·볼륨 최소·noise/cancel routing/geometry 확인
- [ ] meter 20초와 strict 12.5초를 각각 출력 close 직후 즉시 분리
- [ ] immutable raw를 먼저 보존하고 분석; 실패 시 즉시 반복 금지
- [ ] 48 kHz/256/low, xrun/clip 0, observed PCM/SHA/clock-q, fractional joint-LS,
  cubic crosscheck, compact round-trip, 모든 150–1600 Hz 부대역 consistency ≥0.9406,
  repeats ≥8, 안정적인 P−S 상대 지연

### 5.3 Elice

- [ ] transfer manifest에 recorded 전체, RIR, strict raw/analysis/P/S, regrouped, FMA tracks,
  holdout, 두 CSV와 provenance report 결속
- [ ] A100 80GB 한 장과 가용 128 GiB
- [ ] torch 2.5.1+cu121/CUDA 12.1 environment receipt
- [ ] untouched public raw 6종을 복수 full-decode/seek decoder audit으로 전수 검증하고,
  audit-bound `data/manifests/canonical_v4` content-hashed manifest 재생성
- [ ] noise QA, recorded QA, 전체 pytest
- [ ] canonical init만 FAIL인 readiness 14/15

### 5.4 학습과 평가

- [ ] strict S 기준 `lambda_dnh` gradient 비중 0.2–0.4
- [ ] G0 trusted NMSE < −6 dB
- [ ] frame-metric-only 2개 alpha 20k+5k 후보와 필요 시 alpha 0.85를 val-only로 선택
- [ ] A100 200–500 smoke와 중단/재개 등가 receipt
- [ ] 선택 tiny 계약의 새 100k canonical pretrain
- [ ] readiness 15/15 뒤 open-loop 50k fine-tune
- [ ] val selection 고정 뒤 campaign one-shot test G4 PASS
- [ ] G4 뒤 natural-crest challenge PASS

G4와 crest challenge 전에는 closed-loop, ONNX export, 실제 ANC ON 평가를 하지 않는다.

## 6. 주요 진입점

| 작업 | 진입점 |
|---|---|
| 계보 복구 | `scripts/data/repair_source_pool_provenance.py` |
| transfer 결속 | `scripts/data/build_elice_transfer_manifest.py` |
| Elice 준비 | `scripts/elice/bootstrap_all.sh` |
| strict P/S | `scripts/data/set_amp_level.py` → `measure_paths_interleaved.py` |
| readiness | `scripts/train/check_finetune.py` |
| canonical pretrain | `scripts/train/train.py --config configs/train_pretrain_tiny.yaml` |
| fine-tune | `scripts/train/run_finetune_pipeline.py --config configs/train_finetune.yaml` |
| recorded 평가 | `scripts/eval/evaluate_recorded.py` |

실행 상태와 정확한 다음 명령은 [HANDOFF.md](../HANDOFF.md)를 단일 인수인계 문서로 사용한다.

## 7. 브랜치 경계와 병합 규칙 (2026-08-27)

현재 브랜치는 내용이 섞인 임시 작업선이 아니라, 서로 의존하는 준비 계약을 순서대로 쌓은
검증 기준선이다. 과거 커밋을 재작성하거나 강제 reset하지 않고 아래 네임스페이스로 이후
변경의 목적을 분리한다.

| 브랜치 | 허용 범위 | 금지 범위 |
|---|---|---|
| `main` | 공개 릴리스 기준선과 검토된 병합만 | 직접 실측·학습·실험 커밋 |
| `fix/finetune-readiness-repair` | timing/계보/strict P/S/readiness/bootstrap 계약의 기준선 유지 | 고주파 가설, canonical 학습 결과, legacy 결과를 기준선으로 승격 |
| `work/canonical-training` | 선택된 loss 계약의 tiny 100k pretrain과 50k measured fine-tune 코드·설정 | pilot을 init으로 재사용, 고주파 P/S·모드 실험 |
| `work/high-frequency-validation` | 1.6 kHz 밖의 P/S·덕트 모드·기하 확인과 별도 고주파 목적함수 실험 | 공식 150–1600 Hz 계약과 strict 자산 덮어쓰기 |
| `archive/*` | 역사적 분석·재작성 전 ref 보존과 read-only 비교 | 실행, 재학습, 성능 근거 승격 |

두 `work/*` 브랜치는 2026-08-27 기준선에서 의도적으로 같은 commit으로 시작한다. 이후
실험은 해당 브랜치에서만 커밋하고, 결과·체크포인트·raw는 Git에 넣지 않고 Elice 실행
receipt/transfer manifest로 결속한다. 한 실험의 변경을 다른 작업선에 cherry-pick할 때는
전체 계약 SHA, pytest, provenance와 음향 평가 artifact를 함께 재검증한다.

Elice 학습은 항상 `--expected-commit`으로 기록된 detached checkout에서 실행한다. 학습 중인
checkout의 branch 이동·reset·force-push는 하지 않는다. 기준선으로 병합하기 전에는 다음을
모두 통과해야 한다.

```bash
.venv/bin/python -m pytest -q
bash -n scripts/elice/bootstrap_all.sh scripts/elice/setup_env.sh
git diff --check
```
