# 05. Elice Cloud 학습 가이드

현행 1차 대상은 **digital-reference tiny open-loop** 하나다. Elice bootstrap은 exact code,
canonical 데이터 계약과 환경만 준비하며 학습을 자동 시작하지 않는다. 과거 base/tiny 병렬
실행과 `pretrain_*_corrected`는 150–600 Hz/legacy plant 진단 기록이라 init·resume·성능
근거로 사용하지 않는다.

## 0. 유일한 운영 진입점

새 Elice 인스턴스의 **유일한 권위 진입점**은
`scripts/elice/bootstrap_all.sh`다. full bootstrap은 신뢰 채널에서 따로 전달받은 다음 네
항목을 모두 명시해야 하며, branch 이름·축약 SHA·원격 working tree의 값으로 대신하지 않는다.

- `--expected-commit <40자리_SHA>`
- `--expected-holdout-sha256 <64자리_SHA256>`
- `--expected-transfer-manifest-sha256 <64자리_SHA256>`
- `--no-update`

`scripts/elice/bootstrap.sh`는 호환용 wrapper가 아니라 **폐기된 legacy 진입점**이다. 인자 없음,
`--train`, 그 밖의 어떤 인자로 실행해도 항상 `exit 2`로 끝나는 fail-closed stub이며 환경 준비나
학습을 시작하지 않는다. 파일을 복구하거나 과거 commit의 내용을 복사해 되살리지 않는다.

`bootstrap_all.sh --preflight-only`는 exact checkout과 canonical holdout bundle까지만
read-only 검증한다. 이 성공은 transfer manifest, A100/스토리지, venv, public raw, decoder
audit 또는 readiness 성공을 뜻하지 않는다. full bootstrap만 transfer 외부 SHA anchor와
하드웨어를 먼저 검사한 뒤 환경·데이터를 준비하며, full bootstrap도 학습은 시작하지 않는다.

## 1. 필요한 인스턴스

| 항목 | 최소 계약 |
|---|---|
| GPU | 이름이 `NVIDIA A100`으로 시작하고 usable VRAM이 79 GiB 이상인 80GB 1장(MIG partition 불가) |
| 저장공간 | 파일시스템 총 127.875 GiB 이상; 미완성 corpus 최초 bootstrap은 가용 96 GiB 이상 |
| 학습 환경 | torch `2.5.1+cu121`, CUDA runtime 12.1 |
| 공식 world size | 1 |

bootstrap은 실제 GPU 이름/메모리, 가용 디스크, Python/torch/CUDA 환경 receipt를 검증한다.
기존 `.venv`를 재사용할 때도 `.setup-complete`만 신뢰하지 않는다. editable 설치가 현재
checkout을 가리키는지 확인한 뒤 `environment-freeze.txt`를 원자적으로 다시 만들고,
그 안의 유일한 Deep-ANC VCS requirement가 `--expected-commit` 전체 40자리와 exact할 때만
transfer 검증·public download·최종 bootstrap receipt 발행으로 진행한다. freeze 파일 SHA를
새로 봉인하는 것만으로 과거 source SHA를 현재 commit으로 승격할 수 없다.
200–500 step smoke에서 VRAM, 처리량과 ETA를 기록하기 전에는 장기 인스턴스를 방치하지 않는다.

## 2. Jetson에서 먼저 고정할 입력

Elice로 가기 전에 Jetson에서 다음을 완료한다.

1. 전체 pytest와 무음 측정 dry-run
2. historical source CSV 복구와 160 WAV 불변성 검증
3. parent 82세션 holdout과 lineage-component regrouped split 보존
4. 사용자 입회 strict P/S 측정과 raw/analysis/NPZ 보존
5. 모든 tracked 코드·config·문서 변경을 review/commit/push하고 clean exact commit 고정
6. 그 exact commit에서 DNS/DEMAND selector와 source plan을 새로 발행하고 추가 19세션 QA PASS,
   combined 101세션 generation/manifest no-replace 발행
7. old82 train-only strict-P level calibration receipt를 발행하고 같은 commit/generation으로
   transfer schema v2 발행

5번 뒤에는 tracked 파일을 한 byte도 바꾸지 않는다. commit이 달라지면 이전 DNS/DEMAND selector,
source plan, recorded generation과 transfer manifest는 새 commit의 증거가 될 수 없으므로 selector부터
다시 발행한다. data artifact의 SHA를 새 commit 문자열로 바꾸는 재봉인은 금지한다.

strict P/S가 없거나 legacy P/S만 있으면 transfer manifest를 만들지 않는다. parent 82만 담은
schema v1도 새 학습의 입력이 아니다. 추가 19세션을 봉인한 generation을 먼저 검증한 뒤 다음처럼
**schema v2** transfer manifest를 만든다. 실제 capture가 만든 파일을 모두 열거해야 한다.

```bash
EXPECTED_HOLDOUT_SHA256=$(sha256sum data/manifests/recorded_holdout.json | awk '{print $1}')
RECORDED_GENERATION=data/manifests/recorded_generations/<generation-id>/generation.json
EXACT_COMMIT=$(git rev-parse HEAD)
RECORDED_LEVEL_CALIBRATION="data/manifests/recorded_level_calibration/${EXACT_COMMIT}.json"

.venv/bin/python scripts/data/build_recorded_level_calibration.py \
  --out "$RECORDED_LEVEL_CALIBRATION"
```

calibration builder는 clean tree와 전체 40자리 HEAD를 다시 검사하고 output을 canonical
directory 아래 no-replace로만 발행한다. stdout의 SHA-256을 별도 anchor로 보존한다. 기존 파일을
삭제하거나 덮어써 같은 경로를 새 분석으로 재사용하지 않는다.

builder는 출력 경로를 `data/manifests/elice_transfer_manifest.json`으로 고정하고 기본 동작에서는
다른 bytes를 자동 overwrite하지 않는다. 기존 schema v1을 schema v2로 바꿀 때만 Jetson의 clean
`dev`에서 기존 SHA를 외부 anchor로 명시한다. builder는 기존 schema v1 전체를 먼저 검증하고
`data/manifests/elice_transfer_history/elice_transfer_manifest.<old-sha>.json`에 content-addressed
hard-link를 보존한 뒤, 새 schema v2 전체 검증이 끝나야 canonical 경로를 원자 교체한다.

```bash
CURRENT=data/manifests/elice_transfer_manifest.json
OLD_TRANSFER_SHA256=$(sha256sum "$CURRENT" | awk '{print $1}')

.venv/bin/python scripts/data/build_elice_transfer_manifest.py \
  --rir-bank data/rir_bank/duct_rirs_v1.npz \
  --strict-raw results/<capture>/raw_measurement.npz \
  --strict-raw results/<capture>/metadata.json \
  --strict-analysis results/<capture>/analysis_results.npz \
  --strict-analysis results/<capture>/analysis_metadata.json \
  --primary-npz assets/measured/primary_path_il_strict_<capture-id>.npz \
  --secondary-npz assets/measured/secondary_path_il_strict_<capture-id>.npz \
  --recorded-generation "$RECORDED_GENERATION" \
  --recorded-level-calibration "$RECORDED_LEVEL_CALIBRATION" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --rotate-existing-transfer-sha256 "$OLD_TRANSFER_SHA256"

test -f "$CURRENT"
test -f "data/manifests/elice_transfer_history/elice_transfer_manifest.${OLD_TRANSFER_SHA256}.json"
test "$(sha256sum \
  "data/manifests/elice_transfer_history/elice_transfer_manifest.${OLD_TRANSFER_SHA256}.json" \
  | awk '{print $1}')" = "$OLD_TRANSFER_SHA256"
test "$(.venv/bin/python -I -B - "$CURRENT" <<'PY'
import json
import sys
with open(sys.argv[1], "rb") as handle:
    print(json.load(handle).get("schema_version"))
PY
)" = 2

EXPECTED_TRANSFER_MANIFEST_SHA256=$(sha256sum "$CURRENT" | awk '{print $1}')
```

위에서 고정한 exact commit을 유지한다. data tree의 generation/schema-v2 manifest와 보존한
schema-v1 archive는 Git 외부 immutable artifact로 함께
백업하되, Elice에는 canonical v2 경로와 그 manifest가 열거한 파일만 전송한다. 회전 anchor가
다르거나 old/new 전체 검증이 실패하면 builder가 canonical v1을 보존하고 실패해야 한다. history를
지우거나 schema v1 bytes를 수정해 v2처럼 보이게 하거나 Elice의 schema v1 receipt를 재봉인하지
않는다. 기존 canonical 파일이 없는 최초 schema-v2 발행에는 회전 옵션을 지정하지 않는다.

manifest가 결속하는 범위:

- parent `data/recorded/` 82세션과 별도 additions 19세션의 전체 파일 byte 및 combined aggregate SHA
- 101세션 combined manifest, source plan과 recorded generation report
- fresh recording-level campaign/meter raw/receipt와 old82 strict-P level calibration receipt
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
git clone https://github.com/tokengeoji/Deep-ANC.git
cd Deep-ANC
git checkout --detach "$EXPECTED_COMMIT"

# 먼저 다운로드와 venv 생성 없이 코드+holdout bundle만 검증.
# transfer/GPU/storage 검증은 아직 아니다.
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
  --raw-hash-workers 8 \
  --no-update
```

`--no-update`는 필수다. bootstrap은 pull하지 않는다. 다음은 모두 즉시 실패한다.

- HEAD/expected commit 불일치 또는 dirty tree
- replace/graft, assume-unchanged/skip-worktree, tracked blob byte 불일치
- holdout/provenance/CSV/transfer SHA 또는 schema 불일치
- symlink·비정규 파일·검증 도중 파일 변경
- recorded generation 101세션 aggregate, strict P/S, RIR, lineage/level metadata 누락
- full A100 80GB, 파일시스템 total/available budget, torch·CUDA 환경 계약 불일치

`--preflight-only`가 transfer SHA 인자를 파싱하더라도 그 단계에서는 transfer bytes나 하드웨어를
검증하지 않는다. 반드시 같은 네 anchor로 full bootstrap을 끝내고
`data/manifests/elice_bootstrap_receipt.json`을 얻어야 다음 단계로 갈 수 있다.

### 3.0.1 Elice 실패 복구와 재발 방지

Elice working tree만 고치는 hotfix는 금지한다. 원격에서 발견한 결함은 Jetson의 이
저장소에서 코드·negative regression fixture·문서를 함께 수정하고 GitHub에 push한 뒤,
Elice가 새 40자리 SHA를 detached checkout하는 방식으로만 복구한다. 이전 checkout에서
만든 freeze/bootstrap/DNS selection receipt를 새 commit의 증거로 다시 봉인하지 않는다.

긴 data scan보다 먼저 다음 pure-stdlib 검사를 로컬과 Elice에서 모두 실행한다.

```bash
python3 -I -B scripts/ci/check_static_contract_references.py --repo-root .
```

이 검사는 운영 코드가 참조하는 `tests/...::test_*` node의 파일·함수 존재 여부를 AST로
확인한다. 또한 실행 commit을 코드에 40자리 literal로 고정하는 것을 금지한다. 예외는
source-pool v1/v2를 byte-exact하게 재현하기 위한 네 개의 명시적
`historical_builder_commit` 위치뿐이다. 역사적 builder SHA, SHA-256 digest, docs와
diagnostic artifact를 현재 HEAD로 바꾸는 검사가 아니다. `tests/conftest.py`도 같은 검사를
pytest collection 전에 실행하므로, stale test 이름이 전체 회귀 중간에서 늦게 발견되어서는
안 된다.

bootstrap 단계와 실패 시 행동은 다음처럼 고정한다.

| 순서 | 단계 | 실패 시 행동 |
|---:|---|---|
| 0 | exact HEAD/clean tree + static contract reference | venv·raw·GPU를 열기 전에 종료 |
| 1 | holdout/transfer 외부 SHA preflight | bundle을 수정하지 않고 로그 보존 |
| 2 | venv 재사용 또는 생성 + environment freeze 재발행 | stale freeze를 재봉인하지 않고 종료 |
| 3 | public raw count/SHA + decoder audit | 원본과 transaction을 보존하고 자동 재다운로드 금지 |
| 4 | manifest transaction + lineage/QA | published old manifest를 current로 승격하지 않음 |
| 5 | 전체 pytest | 첫 실패 node와 전체 log를 보존하고 학습·selector 금지 |
| 6 | bootstrap receipt + readiness | receipt가 없으면 selector·G0·학습 금지 |

실패 뒤 재개할 때는 기존 raw/venv를 삭제하는 것으로 문제를 숨기지 않는다. 새 exact commit의
preflight가 PASS한 경우에만 verified count와 외부 decoder SHA를 사용해 cache를 재사용한다.
각 실행 log에는 commit을 파일명으로 넣고, `HANDOFF.md`에 실패 단계·node·log 경로·재사용한
bytes·다음 명령을 기록한다. bootstrap receipt가 발행된 뒤에만 같은 SHA로 DNS selector를
발행하고, 같은 pre-exclusion public generation에서 DEMAND DKITCHEN selector도 발행한다.
두 receipt SHA를 각각 stdout와 별도 전달 채널로 Jetson source plan의 외부 anchor로
전달한다. selector의 exact 명령과 immutable bundle 구성은 `docs/17_recorded_generation.md`
§4를 따른다. 둘 중 하나만 발행된 source plan은 canonical 19행이 아니다.

긴 다운로드의 직전과 직후에도 code와 transfer bundle을 다시 검증한다. bootstrap 잠금은 동시
데이터 준비를 차단하며, 실행 중인 `train.py`가 있으면 manifest를 건드리지 않는다.

`--raw-hash-workers 8`은 A100 80GB/16 vCore 인스턴스에서 **이미 완료된 decoder audit의
원본 SHA-256·size 재대조와 manifest transaction postcondition**만 병렬화한다. PCM decoder
audit 자체, 입력 순서, manifest bytes, 첫 실패 보고 순서는 바꾸지 않는다. 이 값은 검증을
생략하는 옵션이 아니며 1~32만 허용한다. vCPU·스토리지 병목이 확인되지 않은 환경은 기본값
`1`을 유지한다.

### 3.1 125 Hz–8 kHz full-octave 확장 gate

위 기본 bootstrap은 기존 Stage-1 corpus를 준비하는 경로다. 이를 125/250/500/1000/2000/
4000/8000 Hz canonical 학습으로 해석해서는 안 된다. 특히 16 kHz MIMII를 업샘플해 8 kHz
octave의 native machine source로 세는 것을 금지한다.

full-octave를 요청할 때는 실제 BSD35k `fx-m` 등 **native 22,628 Hz 이상** machine source의
다음 evidence file과 외부에서 확인한 file SHA를 명시해야 한다.

- official archive size/MD5 및 selected ZIP member ↔ extracted WAV bytes
- official metadata에서 다시 계산한 selection·uploader lineage
- complete decoder audit
- deterministic native PSD와 split×band 독립 uploader coverage

```bash
# evidence 자체는 먼저 read-only로 검증한다.
.venv/bin/python scripts/data/audit_bsd35k_highrate_machine.py verify \
  --evidence results/provenance/bsd35k_fx_m_highrate_source.json \
  --expected-file-sha256 "$EXPECTED_BSD35K_EVIDENCE_SHA256"

# evidence와 full-octave admission이 모두 PASS일 때만 bootstrap을 시도한다.
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --full-octave \
  --full-octave-highrate-machine-evidence \
    results/provenance/bsd35k_fx_m_highrate_source.json \
  --expected-full-octave-highrate-machine-evidence-sha256 \
    "$EXPECTED_BSD35K_EVIDENCE_SHA256" \
  --no-update
```

evidence가 없거나 full-octave P/S·population·family batch·DNH·raw-bound execution
admission 중 하나가 BLOCKED이면 bootstrap은 **public raw download, manifest 발행, 학습
시작 전** 종료한다. 이 상태에서 `--preflight-only`로 high-rate evidence를 우회할 수 없다.

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
않는다. bootstrap 종료 후 noise QA, recorded QA, strict 부대역 coverage, 전체 pytest와
readiness를 다시 실행한다. 현재 82세션은 cross-public speech lineage, recorded strict
부대역 coverage, canonical init 세 blocker가 남아 **가능한 최대치가 14/17 PASS**이지만, 새 exact
bootstrap receipt가 나오기 전에는 현재 점수를 확정하지 않는다. 먼저 numeric alias가 겹치는
speech lineage를 독립 원본으로 복구해 **15/17 PASS**를 확인한다. 다음으로 부족 family×대역을
추가 녹음해 coverage PASS인 **16/17 PASS**를 확인한다. 이 순서를 모두
마친 뒤에만 G0를 시작하며, 그보다 적거나 coverage가 FAIL이면 학습을 시작하지 않는다.

단, 비용이 발생하는 새 GPU의 kernel/VRAM/처리량/중단·재개 경로만 조기에 확인하는
`a100_pretrain_smoke`는 15/17 상태에서 200--500 step으로 병렬 실행할 수 있다. 이 예외는
`init_eligible=false`인 diagnostic-only이며 G0, loss 후보 점수, checkpoint 전이, test 또는
readiness PASS로 사용할 수 없다. DKITCHEN selection receipt가 이미 존재한다면 public
DEMAND 16행 exclusion과 새 manifest sidecar가 먼저 검증돼야 smoke도 열린다. selector 발행
전 82세션 상태에서 시작한 smoke는 500 step을 넘기지 않고, 101세션 canonical 학습의 데이터
또는 성능 근거로 승격하지 않는다.

## 5. 사전학습 계약 선택

공식 sampler는 family→lineage component→session을 각 단계에서 균등 추첨한다. 공통
gain/polarity/EQ와 input-only mic noise만 켜고, 1차 실행의 session mixing과 lead jitter는 0이다.

1. alpha별 현재 `lambda_dnh`로 고정 batch G0를 **처음부터** 실행해 trusted NMSE
   `< -6 dB`와 timing metadata를 확인한다.
2. 합격 G0의 같은 checkpoint/fixed batch와 strict S, Trainer의 settle 절단,
   150–1600 Hz를 사용해
   `‖lambda_dnh·∂L_dnh/∂y‖ / ‖∂L_nmse/∂y‖`를 다시 계산한다. 이는 model
   parameter-gradient가 아니라 **model output waveform y-gradient**이며, 현재 cfg의
   실제 share가 0.2–0.4일 때만 PASS다. 범위 안이면 현재 λ를 유지한다.
3. G0가 실패했거나 share가 범위 밖이면 sealed receipt의 추천 λ는 다음 실행을 위한
   정보일 뿐이다. 추천값으로 새 contract를 만들고 weight 전이 없이 G0부터 다시 실행한다.
   실패 checkpoint/추천 receipt는 pilot·init 자격을 열 수 없다.
4. seed `20260803`으로 frame-metric-only 상태에서 다음 2개를 20k surrogate-pretrain
   + 5k measured probe한다. signed frame-CVaR는 λ=0.5와 0.2가 모두 fixed-batch
   control에서 영출력 붕괴를 재현해 후보에서 제외한다.

```text
candidate identity = (alpha, lambda_frame, lambda_dnh)
alpha ∈ {0.7, 1.0}, lambda_frame = 0.0,
lambda_dnh = 해당 alpha가 G0+pre-pilot share gate를 통과한 값
```

한 `lambda_dnh`를 모든 alpha에 강제하지 않는다. 각 후보의 exact identity는
G0→pre-pilot gradient→20k→5k 전 구간에 결속된다. 선택된 후보는 20k best에서 모든
후보 G0가 공유한 fixed-batch SHA 중 **winner G0의 authoritative artifact path/SHA
자체**로 share를 한 번 더 계산해 출력 분포 drift도 통과해야 하며, post-pilot batch를
복제하거나 새로 골라서는 안 된다. 그 winner identity 전체가
smoke→100k→50k에 그대로 이어진다.

alpha별 G0와 pre-pilot receipt의 기본 흐름은 다음과 같다. 경로에는 alpha와 현재 λ를
반드시 포함해 no-replace evidence가 서로 충돌하지 않게 한다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
BOOT=$(sha256sum data/manifests/elice_bootstrap_receipt.json | awk '{print $1}')
ALPHA=0.7
LAMBDA_DNH=0.00075  # 시작값일 뿐 승인값이 아님
G0_DIR="results/training_prerequisites/evidence/g0_a${ALPHA}_dnh${LAMBDA_DNH}"

.venv/bin/python scripts/bench/diagnose_training_overfit.py \
  --config configs/train_pretrain_tiny.yaml \
  --loss-alpha "$ALPHA" \
  --loss-lambda-dnh "$LAMBDA_DNH" \
  --bootstrap-receipt-sha256 "$BOOT" \
  --evidence-dir "$G0_DIR"

# 위 명령이 PASS(exit 0)한 G0에만 공식 pre-pilot gate를 발행한다.
.venv/bin/python scripts/train/measure_gradient_budget.py \
  --g0-receipt "$G0_DIR/receipt.json" \
  --out-dir "results/training_prerequisites/evidence/prepilot_a${ALPHA}_dnh${LAMBDA_DNH}"
```

G0가 exit 2이면 같은 디렉터리에는 campaign-eligible G0가 아니라 failed-diagnostic kind가
봉인된다. 추천이 필요할 때만 아래 명령을 사용한다. 이 명령도 의도적으로 exit 2이며,
출력된 추천값으로 **새 경로의 fresh G0**를 실행해야 한다.

```bash
.venv/bin/python scripts/train/measure_gradient_budget.py \
  --failed-g0-receipt "$G0_DIR/receipt.json" \
  --out-dir "results/training_prerequisites/evidence/failed_gradient_a${ALPHA}_dnh${LAMBDA_DNH}"
```

5k measured-probe로 winner를 고른 뒤 selected-20k drift 검사는 새 validation batch를 만들지
않는다. 반드시 winner의 위 G0 receipt를 다시 넘겨 그 receipt가 가리키는 exact batch
path/SHA를 사용한다.

```bash
.venv/bin/python scripts/train/measure_gradient_budget.py \
  --checkpoint runs/<winner-20k>/ckpt/best.pt \
  --authoritative-g0-receipt results/training_prerequisites/evidence/<winner-g0>/receipt.json \
  --out-dir results/training_prerequisites/evidence/gradient_selected20k
```

test는 열지 않고 각 20k pilot best.pt를 init으로 한 5k measured 70:30
probe의 recorded val만 최종 선택에 사용한다. 두 measured-probe 점수 차이가
0.2 dB 이내이면
alpha 0.85(`lambda_frame=0`)를 추가한다. alpha 1.0의 non-finite/실행 실패는 현재
"불안정"이라는 수기 표기로 0.85를 자동 승인하지 않는다. pre-forward model·batch·RNG를
재실행해 검증하는 immutable failure receipt가 준비되기 전까지는 canonical을 차단하고 raw를
보존한다. 계속 동률이면 단순한 alpha 0.7을 선택한다.
frame은 170 ms metric으로 계속 기록해 후보 비교·원인 분석에 사용한다. 고정 local pass
threshold는 아직 증거가 없으므로 성능 주장에는 쓰지 않으며, signed frame gradient를
되살리려면 one-sided/item-wise v2 guard와 별도 control evidence가 필요하다.
pilot/probe checkpoint는
`init_eligible=false`라 canonical init이 될 수 없다.

선택 계약을 campaign prerequisite ledger로 고정한 뒤 tiny 100k를 처음부터 실행한다.
`configs/train_pretrain_tiny.yaml`은 canonical A100 한 장 전용이다. 실행 디렉터리는 계약 SHA와
seed에서 파생하며 기존 경로를 덮어쓰지 않는다.

아래 값은 모두 같은 Elice exact checkout에서 읽고, `ALPHA`와 `LAMBDA_DNH`는 raw
measured-probe winner identity로 유도한 값만 쓴다. bare `train.py --config ...`는
의도적으로 fail-closed한다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
BOOT=$(sha256sum data/manifests/elice_bootstrap_receipt.json | awk '{print $1}')
LEDGER=$(sha256sum results/training_prerequisites/canonical_pretrain.json | awk '{print $1}')
ALPHA=0.7  # raw measured-probe winner(0.7/0.85/1.0) 값으로 교체
LAMBDA_DNH=<winner의 alpha별 approved 값>

.venv/bin/python scripts/train/train.py \
  --config configs/train_pretrain_tiny.yaml \
  --set data.bootstrap_receipt_sha256="$BOOT" \
  --set campaign_prerequisite_sha256="$LEDGER" \
  --set loss.nmse_cvar_alpha="$ALPHA" \
  --set loss.lambda_dnh="$LAMBDA_DNH"
```

`scripts/elice/run_pretrain.sh`, `run_parallel_models.sh`, `queue_gpu0.yaml`,
`queue_gpu1.yaml`은 historical legacy diagnostic 전용이며 canonical 명령으로 사용할 수 없다.
실행하려면 별도의 legacy acknowledgement가 필요하고, canonical evidence·init·성능 근거로
승격되지 않는다.

자동 resume은 없다. 동일한 `experiment_contract_sha256`와 global sample index/RNG 상태를 가진
같은 실행의 `last.pt`만 명시적으로 재개한다.

```bash
.venv/bin/python scripts/train/train.py \
  --config configs/train_pretrain_tiny.yaml \
  --set data.bootstrap_receipt_sha256="$BOOT" \
  --set campaign_prerequisite_sha256="$LEDGER" \
  --set loss.nmse_cvar_alpha="$ALPHA" \
  --set loss.lambda_dnh="$LAMBDA_DNH" \
  --resume runs/<canonical-contract-seed>/ckpt/last.pt
```

smoke 중단/재개와 uninterrupted 결과가 수치 등가이고, completion receipt가 100k `last.pt`,
recorded-val `best.pt`, best metric/step 관계를 결속해야 init 자격이 생긴다.

## 6. 공식 파인튜닝과 test

canonical 100k `best.pt`를 `init_ckpt`로 명시하고 readiness **17/17 PASS**를 먼저 확인한다.

```bash
INIT_CKPT=runs/<canonical-pretrain>/ckpt/best.pt
# canonical pretrain을 만든 같은 Elice generation과 loss winner를 그대로 쓴다.
# 이 두 값이 없으면 canonical fine-tune config stamp가 fail-closed한다.
BOOT=$(sha256sum data/manifests/elice_bootstrap_receipt.json | awk '{print $1}')
ALPHA=0.7  # canonical ledger의 raw measured-probe winner(0.7/0.85/1.0)로 교체
LAMBDA_DNH=<같은 winner의 approved 값>
.venv/bin/python scripts/train/check_finetune.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured \
  --set init_ckpt="$INIT_CKPT" \
  --set data.bootstrap_receipt_sha256="$BOOT" \
  --set loss.nmse_cvar_alpha="$ALPHA" \
  --set loss.lambda_dnh="$LAMBDA_DNH"

.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured \
  --set init_ckpt="$INIT_CKPT" \
  --set data.bootstrap_receipt_sha256="$BOOT" \
  --set loss.nmse_cvar_alpha="$ALPHA" \
  --set loss.lambda_dnh="$LAMBDA_DNH"
```

`ALPHA`와 `LAMBDA_DNH`는 `canonical_pretrain.json`이 검증해 선택한 identity와 같아야
한다. YAML의 기본값을 자동으로 믿으면 winner가 다른 alpha/λ일 때 init checkpoint의
`loss_selection_sha256`와 달라져 fine-tune은 의도적으로 시작 전에 중단된다.

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
