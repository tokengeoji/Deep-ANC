# 05. Elice Cloud 학습 가이드

현행 1차 대상은 **digital-reference tiny open-loop** 하나다. Elice bootstrap은 exact code,
canonical 데이터 계약과 환경만 준비하며 학습을 자동 시작하지 않는다. 과거 base/tiny 병렬
실행과 `pretrain_*_corrected`는 150–600 Hz/legacy plant 진단 기록이라 init·resume·성능
근거로 사용하지 않는다.

## 1. 필요한 인스턴스

| 항목 | 최소 계약 |
|---|---|
| GPU | A100 80GB 1장 |
| 저장공간 | 가용 128 GiB 이상 |
| 학습 환경 | torch `2.5.1+cu121`, CUDA runtime 12.1 |
| 공식 world size | 1 |

bootstrap은 실제 GPU 이름/메모리, 가용 디스크, Python/torch/CUDA 환경 receipt를 검증한다.
200–500 step smoke에서 VRAM, 처리량과 ETA를 기록하기 전에는 장기 인스턴스를 방치하지 않는다.

## 2. Jetson에서 먼저 고정할 입력

Elice로 가기 전에 Jetson에서 다음을 완료한다.

1. 전체 pytest와 무음 측정 dry-run
2. historical source CSV 복구와 160 WAV 불변성 검증
3. active 82세션 holdout과 lineage-component regrouped split
4. 사용자 입회 strict P/S 측정과 raw/analysis/NPZ 보존
5. clean exact commit과 push

strict P/S가 없거나 legacy P/S만 있으면 transfer manifest를 만들지 않는다. 준비된 뒤 다음처럼
canonical transfer manifest를 만든다. 실제 capture가 만든 파일을 모두 열거해야 한다.

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

manifest가 결속하는 범위:

- `data/recorded/`의 82세션 전체 파일 byte와 aggregate SHA
- `data/rir_bank/duct_rirs_v1.npz`
- strict P/S raw, metadata, analysis, primary/secondary NPZ
- `data/manifests/recorded_regrouped.jsonl`
- `data/raw/music/fma_metadata/tracks.csv`
- `data/manifests/recorded_holdout.json`
- v1/v2 `sources.csv`
- holdout가 가리키는 content-addressed provenance report

이 파일들을 manifest의 상대경로 그대로 `rsync`/`scp`로 스트리밍한다. 4.11 GiB recorded
tree의 로컬 tar 복제본을 만들지 않는다. 전송 뒤에는 manifest validator가 각 경로, 크기,
content SHA와 recorded aggregate를 다시 계산한다.

## 3. exact checkout bootstrap

GitHub에서 받은 전체 40자리 commit SHA와 두 data SHA를 서로 다른 신뢰 채널로 Elice에
전달한다. branch name이나 축약 SHA를 쓰지 않는다.

```bash
git clone https://github.com/Roka-jsj/Deep-ANC.git
cd Deep-ANC
git checkout --detach "$EXPECTED_COMMIT"

# 먼저 다운로드와 venv 생성 없이 코드+bundle만 검증
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update --preflight-only

# full environment/data bootstrap. 학습은 시작하지 않음
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update
```

`--no-update`는 필수다. bootstrap은 pull하지 않는다. 다음은 모두 즉시 실패한다.

- HEAD/expected commit 불일치 또는 dirty tree
- replace/graft, assume-unchanged/skip-worktree, tracked blob byte 불일치
- holdout/provenance/CSV/transfer SHA 또는 schema 불일치
- symlink·비정규 파일·검증 도중 파일 변경
- recorded 82세션 aggregate, strict P/S, RIR, lineage metadata 누락
- A100 80GB/가용 128GiB/torch·CUDA 환경 계약 불일치

긴 다운로드의 직전과 직후에도 code와 transfer bundle을 다시 검증한다. bootstrap 잠금은 동시
데이터 준비를 차단하며, 실행 중인 `train.py`가 있으면 manifest를 건드리지 않는다.

## 4. public raw와 manifest

public raw는 Elice에서 직접 받는다. Jetson의 3–4 GiB 여유 공간에 staging하지 않는다.

| corpus | 기대 항목 수 | 용도 |
|---|---:|---|
| DNS noise | 16,000 | 광대역 생활 소음 |
| DNS speech | 8,065 | 음성 |
| ESC-50 | 2,000 | 환경음 |
| FMA-small | 약 8,000 | 음악 |
| DEMAND | 96 | 실환경 |
| MIMII fan | 3,600 | 기계음 |

FMA metadata는 archive SHA와 `tracks.csv` content SHA, 106,574행 및 FMA-small ID 대응을
검증한다. bootstrap은 corrected holdout를 먼저 확인한 뒤 untouched raw에서 다음 manifest
6종을 새로 만든다.

```text
dns_fullband, speech, music, demand, machine, esc50
```

manifest schema는 각 raw audio content SHA, lineage component와 holdout SHA를 결속한다.
여기에 더해 canonical schema v4는 `data/raw` 전체를 65,536·262,144-frame full sequential
decode와 고정 seek grid로 읽은 `decoder_audit.json`을 transaction 안에서 함께 복사한다.
audit은 현재 Python/SoundFile/libsndfile/libmpg123 fingerprint, raw inventory SHA/size, stderr
warning·decode error·nonfinite·peak>2·RMS≤1e-8의 판정을 보존한다. reject raw는 원본을
수정·삭제하지 않고 새 v4 manifest에서만 제외한다. audit 당시와 다른 decoder 환경, raw
추가/변경, reject와 같은 content SHA, audit 없는 v3 manifest는 모두 canonical 학습을 막는다.
`NoisePool`도 v4 경계에서는 이후 decode 이상을 다른 파일로 재시도하지 않고 즉시 실패한다.
recorded/holdout와 synthetic train·val·test의 원본 계보 교집합은 모두 0이어야 한다.

bootstrap은 `results/provenance/decoder_audit.json`을 먼저 만들고
`data/manifests/canonical_v4/`에 새 세대를 발행한 뒤, 이 경로만 대상으로 noise QA를 실행한다.
기존 `data/manifests/` v3 세대는 forensic/diagnostic 자료로 보존하며 canonical 설정이 읽지
않는다. bootstrap 종료 후 noise QA, recorded QA, 전체 pytest와 readiness를 다시 실행한다.
strict P/S·transfer/public corpus가 모두 정상이고 canonical init만 없는 상태가
**14/15 PASS**다. 그보다 적으면 학습을 시작하지 않는다.

## 5. 사전학습 계약 선택

공식 sampler는 family→lineage component→session을 각 단계에서 균등 추첨한다. 공통
gain/polarity/EQ와 input-only mic noise만 켜고, 1차 실행의 session mixing과 lead jitter는 0이다.

1. strict S로 `lambda_dnh`의 gradient 비중이 주 NMSE의 0.2–0.4인지 측정한다.
2. 고정 batch G0에서 trusted NMSE < −6 dB와 timing metadata를 확인한다.
3. seed `20260803`으로 frame-metric-only 상태에서 다음 2개를 20k surrogate-pretrain
   + 5k measured probe한다. signed frame-CVaR는 λ=0.5와 0.2가 모두 fixed-batch
   control에서 영출력 붕괴를 재현해 후보에서 제외한다.

```text
alpha ∈ {0.7, 1.0}, lambda_frame = 0.0
```

test는 열지 않고 recorded val만 사용한다. 차이가 0.2 dB 이내거나 alpha 1.0이 불안정하면
alpha 0.85(`lambda_frame=0`)를 추가한다. 계속 동률이면 단순한 alpha 0.7을 선택한다.
frame은 170 ms metric으로 계속 기록해 후보 비교·원인 분석에 사용한다. 고정 local pass
threshold는 아직 증거가 없으므로 성능 주장에는 쓰지 않으며, signed frame gradient를
되살리려면 one-sided/item-wise v2 guard와 별도 control evidence가 필요하다.
pilot/probe checkpoint는
`init_eligible=false`라 canonical init이 될 수 없다.

선택 계약을 campaign prerequisite ledger로 고정한 뒤 tiny 100k를 처음부터 실행한다.
`configs/train_pretrain_tiny.yaml`은 canonical A100 한 장 전용이다. 실행 디렉터리는 계약 SHA와
seed에서 파생하며 기존 경로를 덮어쓰지 않는다.

```bash
.venv/bin/python scripts/train/train.py --config configs/train_pretrain_tiny.yaml
```

자동 resume은 없다. 동일한 `experiment_contract_sha256`와 global sample index/RNG 상태를 가진
같은 실행의 `last.pt`만 명시적으로 재개한다.

```bash
.venv/bin/python scripts/train/train.py \
  --config configs/train_pretrain_tiny.yaml \
  --resume runs/<canonical-contract-seed>/ckpt/last.pt
```

smoke 중단/재개와 uninterrupted 결과가 수치 등가이고, completion receipt가 100k `last.pt`,
recorded-val `best.pt`, best metric/step 관계를 결속해야 init 자격이 생긴다.

## 6. 공식 파인튜닝과 test

canonical 100k `best.pt`를 `init_ckpt`로 명시하고 readiness **15/15 PASS**를 먼저 확인한다.

```bash
INIT_CKPT=runs/<canonical-pretrain>/ckpt/best.pt
.venv/bin/python scripts/train/check_finetune.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured \
  --set init_ckpt="$INIT_CKPT"

.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured \
  --set init_ckpt="$INIT_CKPT"
```

공식 설정은 tiny/open-loop, recorded 70% + synthetic 30%, bf16 forward + FP32 loss, 50k다.
checkpoint 선택은 recorded val만 사용한다. 선택을 원자 고정한 뒤 capability ledger가 허용한
test를 캠페인 전체에서 정확히 한 번 연다.

첫 seed의 val 지표가 경계 0.3 dB 이내이거나 INCONCLUSIVE이면 seed `20260903`으로 선택 계약의
100k+50k를 한 번 더 실행한다. clear PASS는 1시드로 종료하고 clear FAIL은 두 번째 seed를
소모하지 않고 원인을 수정한다. 2시드가 필요하면 val G4를 통과하면서 모든 gate까지의 최소
여유가 큰 모델만 최종 선택할 수 있다.

G4 PASS와 별도 natural-crest challenge 전에는 closed-loop, ONNX export, 실제 ANC ON 평가로
진행하지 않는다.

## 7. 장애 대응 원칙

- OOM: smoke 계약을 새로 기록하고 batch/sample schedule을 함께 재검토한다.
- val이 0 dB에 고정: 기다리지 말고 timing, gradient, manifest와 plant fingerprint를 점검한다.
- decode 오류: 항목을 조용히 건너뛰지 않는다. manifest에 결속된 raw 문제를 먼저 해결한다.
- SSH 종료: PID만 보고 재시작하지 않는다. run lock, completion/evaluation ledger와 checkpoint
  계약을 먼저 확인한다.
- 인스턴스가 없거나 접속 실패: 외부 차단 상태로 기록한다. legacy 결과로 대체하지 않는다.
