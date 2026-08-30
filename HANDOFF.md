# HANDOFF — 파인튜닝 준비 복구 상태

> “이어서 진행해줘”를 받으면 이 파일과 `AGENTS.md`를 먼저 읽는다.
> 최종 갱신: 2026-08-30. 현행 통합 개발 브랜치: `dev`.

## 0. 통합 상태와 절대 판정 (2026-08-29)

이 파일 아래의 누적 기록에는 과거 작업선 이름과 당시 시점의 상태가 남아 있다. 그것을
현재 authoritative HEAD나 학습 가능 판정으로 읽으면 안 된다. 현재 개발선은 `dev`이며,
`fix/dev`의 광대역 준비 변경과 V10--V14의 지연·동기 witness·인과 P/S 경계 보강을 모두
그 ancestry에 통합 중이다. `main`은 canonical 학습과 현장 평가가 합격하기 전까지 배포
기준선으로 유지한다.

현재 실제 장비에서 강하게 확인된 사실은 다음과 같다.

1. RT5640 드라이버와 APE 카드는 존재하지만, 2026-08-29 무음 3회 probe에서
   `CVB-RT Jack-state`는 모두 `None`이었다. 이는 **J511 HP/HS 감지 실패**이며 USB DAC
   출력 경로의 재생 가능/불가능을 뜻하지는 않는다. 어느 출력 경로도 현장 검증되지 않았고,
   speaker가 분리된 동안에는 출력·P/S·ANC ON 실험을 하지 않는다. 다시 연결한 뒤에도
   장치 점유 확인→무음 preflight→사용자 입회/최소 볼륨의 짧은 검증 창 순서를 지킨다.
2. 2026-08-29 read-only ALSA inventory에서 `card2` AB13X USB Audio의
   `/proc/asound/card2/stream0`은 **48 kHz/S16, playback 2채널, capture 1채널**만
   제공했다. APE의 ADMAIF 20개 열거는 8개 동기 물리 ADC 입력을 뜻하지 않는다. 모든 PCM은
   `closed`였고 PulseAudio는 control node만 열고 있었다. 따라서 현 연결만으로는
   `REF + NOISE_TAP + CANCEL_TAP + ERR_0..ERR_4` 8-input same-frame quiet-zone raw를
   만들 수 없다. full-octave 다점 물리 판정에는 verified 8채널 동기 ADC/전기 tap acquisition
   또는 동등한 hardware-frame bridge가 별도 blocker다.
3. 현재 strict P/S는 150--1600 Hz **Stage-1에만 authoritative**하다. 2/4/8 kHz 광대역
   식별·학습·배포 authority가 아니며, 기존 checkpoint·ONNX·고역 결과도 canonical
   성능 근거로 승격하지 않는다.
4. 광대역 canonical 학습을 여는 최소 순서는 **동기화된 다채널 electrical witness 확보 →
   fail-closed P/S raw 분석 합격 → lineage-clean public/recorded manifest 합격 →
   surrogate pretrain → recorded fine-tune → one-shot physical G4**다. 어느 하나라도 없는
   상태에서 GPU 학습을 시작하거나 성능 수치를 주장하지 않는다.

### 0.1 2026-08-29 짧은 출력 경로 진단 (ANC/P/S authority 아님)

사용자 입회에서 `record_duct.py`의 300 Hz/3초/ANC OFF 진단을 두 번만
실행했고, 각 출력 stream 종료 직후 분리 안내를 냈다. 첫 minimum 조건 raw는 신호가
잡음 바닥에 묻혔고, 사용자 승인으로 gain을 최저 위치에서 한 단계 조정한 두 번째 raw에서는 300 Hz
source→ERR coherence² `0.741641`, source→REF `0.730343`, ERR↔REF `0.992065`가
관측됐다. 즉, 그 짧은 조건의 noise speaker→ERR/REF 물리 경로는 진단상 살아 있다.

두 capture 모두 canonical collection plan이 없는 `unbound_diagnostic`이며,
단일 tone은 150–700 Hz timeline witness를 제공하지 못해 `timeline_gate`가 정상적으로
거부했다(`valid_window_ratio=0`). raw를 수정·승격·재측정하지 않고 failure artifact로
보존한다. **이는 P/S, lead, ANC 감쇠, 고역, 모델 또는 quiet-zone 증거가 아니다.**
파일 경로·SHA·독립 재계산·다음 안전 절차는
[`docs/57_20260829_output_route_diagnostic.md`](docs/57_20260829_output_route_diagnostic.md)에
기록했다. 다음 물리 출력은 strict/broadband 계획의 무음 dry-run과 별도 출력 창 보고 뒤에만
실행한다.

### 0.2 2026-08-29 현재 gain 공식 meter PASS (strict P/S는 물리 재연결 대기)

current USB AB13X/APE Stage-1 경로에서 1.5초 무출력 input preflight 뒤 ch0만 20초
출력하고 ch1을 exact silence로 둔 fresh meter가 PASS했다. 마지막 구간 중앙값은
`-48.197019 dBFS`로 공식 target `-50.1 ± 2 dBFS` 안이며, xrun/queue drop/예기치
callback/중단은 모두 0이고 stream close도 확인됐다.

- raw:
  `results/calibration_interleaved/level_bootstrap/20260829_215459_1a6a12bb/meter_raw.npz`
  SHA-256 `ed6fddae136f468f7b44539874e35b996a71db1968760f5eb1495970e15f7028`
- receipt:
  `results/calibration_interleaved/level_bootstrap/20260829_215459_1a6a12bb/meter_raw.receipt.json`
  SHA-256 `71e62bd75931fe3bba901c85556a466a4146123d7de805124a3ef23ac885c334`
- `validate_bootstrap_meter_raw(..., require_fresh=True)`와 현재 strict P/S
  meter-bound dry-run은 모두 PASS했다. dry-run은 output/raw/session 파일을 만들지 않았다.

이것은 **현재 노브의 Stage-1 측정 레벨 증거**일 뿐 새 P/S·lead·ANC·고역·quiet-zone
evidence는 아니다. 출력 종료 직후 분리 안내를 냈으므로, 같은 gain의 12.5초 strict P/S는
실제 재연결이 확인될 때만 fresh window 안에서 한 번 실행한다. 재연결이 확인되지 않거나
freshness가 만료되면 raw를 재사용하거나 임계값을 완화하지 않고, 다음 안전 창에 새 meter부터
시작한다.

### 0.3 2026-08-29 스피커 분리 상태의 소프트웨어 경계 복구

이번 작업에서는 오디오 장치를 열지 않았다. 다음은 실제 파일을 읽어 재확인했거나,
fixture-only가 아닌 artifact 없이는 **BLOCKED**를 유지하도록 코드로 고정한 항목이다.

- 82 recorded 세션은 현행 QA 기준 82/82, 95.67분, lineage component 교집합 0의
  Stage-1 자료다. 그러나 실제 재감사
  `results/audits/broadband_prerequisite_20260829_postrepair.json`에서
  2.828--5.657/5.657--11.314 kHz joint independent group은 0이므로, full-octave
  학습 자료로 승격하지 않는다.
- `configs/full_octave_v3_execution.yaml`과
  `scripts/train/check_full_octave_v3_execution.py`는 raw/analysis/witness/P-S
  operator/population/sampler/DNH/non-fixture binding/training YAML/nonce receipt를
  SHA로 교차 결속한다. 선언 SHA 구조가 모두 맞아도
  `BLOCKED_UNATTESTED_EXECUTION_PROVENANCE`이며 성공 exit이나 Trainer·GPU·run directory를
  만들지 않는다. typed P/S/raw/analysis/witness와 stage별 init 계약이 생기기 전에는
  generic Stage-1 Trainer로 대체 실행하지 않는다.
- `configs/full_octave_v3_physical_session_bundle.yaml`과
  `scripts/data/check_full_octave_v3_physical_session_bundle.py`는 최종 quiet-zone
  주장에 필요한 `REF + NOISE_TAP + CANCEL_TAP + ERR_0..ERR_4`의 8-input,
  48 kHz/256/S32, BCLK/WS/absolute-frame witness, raw-first no-replace bundle만
  선언 구조로 읽는다. 현재 null config는 audio/ALSA/GPU/network를 열지 않은
  `BLOCKED`이며, non-fixture bytes가 맞아도
  `BLOCKED_UNATTESTED_STRUCTURAL_RAW`와 nonzero다. 이 정보는 ANC/P/S/배포 authority가
  아니며, capture adapter receipt·submitted PCM telemetry·native→canonical 변환 증거가
  추가로 필요하다.
- `configs/full_octave_v3_matched_campaign.yaml`의 OFF/DL/FxLMS Latin-square 비교와
  `configs/full_octave_v3_level5_lifecycle.yaml`의 Level-5 미사용 source lifecycle은
  선언된 SHA/순서가 맞아도 각각 `BLOCKED_UNATTESTED_PHYSICAL_PROVENANCE`,
  `BLOCKED_UNATTESTED_*`로 유지한다. 현재 checker들은 self-attested JSON·checkpoint·raw
  또는 terminal `PASS`에 성공 exit/학습/배포 authority를 주지 않는다. 실제 capture adapter
  O_EXCL provenance, full-octave P/S·lead, native lineage inventory, 완료 checkpoint·selection,
  독립 raw evaluator가 별도 authority로 생겨야 한다.
- 2026-08-29 무음 실행 감사에서 v5 `--execute-live`가 false authority에도 backend로
  진입할 수 있던 안전 결함을 수정했다. 이제 tracked
  `fullband_causal_v5_live_capture_authority.json`의
  `plan_live_capture_enabled=false`이면 CLI와 내부 `_execute_live()`가 모두
  audio primitive/`sounddevice` import 전에 exit 2로 fail-closed한다. 회귀는 false
  authority에서 execute/backend import가 각각 0회임을 검사한다. 이 변경은 live 권한을
  여는 것이 아니라, 현재 `BLOCKED` 경계를 실제 실행에도 강제하는 것이다.
- historical high-band raw
  `results/experimental_high_band/20260827_fullband/20260827_203328_1b24d0c2/`
  (raw SHA `46acda579a4ba7069844cc6824fcf4e475edc750d1b43a80cfe40a2e9ffe1ec7`)는
  `design_band_hz=[60,8000]`이다. strict 재분석 승격은 immutable raw recipe가
  현행 `[60,1650]` 및 required/consistency `[150,1600]`과 exact할 때만 허용하도록
  보강했다. 따라서 sidecar marker 유무와 무관하게 이 diagnostic raw는 official P/S가
  될 수 없다.

이 구현들은 물리 측정 전 **오인 경로를 닫는** 소프트웨어 준비다. fullband raw P/S·8-input
acquisition adapter·high-rate public corpus·고역 recorded population·raw-bound trainer loader와
독립 physical evaluator가 없다는 실제 blocker를 PASS로 바꾸지 않는다.

### 0.4 2026-08-29 현재 장비 2/4/8 kHz 결합 진단 (P/S·ANC authority 아님)

현재 meter의 고정 gain에서 48 kHz/256/low, peak `0.003`으로 각 2/4/8 kHz에
input-only 2초→NS(ch0) 2초 tone→CS(ch1) 2초 tone을 한 번씩만 실행했다. 모든 raw는
callback xrun/clip 0, 양채널 zero flush/stream close PASS였다. 2/4 kHz의 네
NS/CS→ERR/REF 경로와 8 kHz의 CS→ERR·양 REF 경로는 단일-tone coupling detector를
통과했다. 8 kHz NS→ERR은 +23.30 dB 상승이 있으나 절대 tone이 `-101.775 dBFS`라
보수적 `-100 dBFS` floor에서 `UNRESOLVED`다. 재생을 반복하거나 임계값을 낮추지 않는다.

artifact SHA, 독립 raw projection, 정확한 출력 시간 및 authority 한계는
[`docs/58_20260829_highband_coupling_diagnostic.md`](docs/58_20260829_highband_coupling_diagnostic.md)에
기록했다. 결론은 현재 speaker가 2/4/8 kHz를 전혀 내지 못한다는 주장이 반증됐다는
것뿐이다. USB DAC↔APE 독립 시간축과 2-input 공간 관측 한계 때문에 이 raw는
P/S·lead·학습·ANC 감쇠·FxLMS 비교·quiet-zone 근거가 아니다.

### 0.5 2026-08-29 RT5640/J511 common-clock 출력 후보 (아직 사용 불가)

read-only ALSA/DT 감사에서 `APE PCM0 → ADMAIF1 → I2S1 → RT5640 → J511` route와
APE PCM1/I2S2 ERR/REF route가 실제로 노출됐고, I2S1--6은 APE `PLL_A` 공유 후보임을
확인했다. 따라서 USB AB13X의 adaptive/asynchronous endpoint보다 timing 구조상 유리할
가능성은 있다. 그러나 현 J511 state는 세 번 모두 `None`, PCM은 closed, electrical/
acoustic output witness와 hardware-frame identity는 모두 없다. 이 route는 즉시
high-band P/S·학습·ANC 권한이 아니며 8-input quiet-zone acquisition의 대체도 아니다.

근거·정확한 config SHA·무음 다음 gate는
[`docs/59_20260829_rt5640_common_clock_route_audit.md`](docs/59_20260829_rt5640_common_clock_route_audit.md)에
기록했다. J511 cable이 실제로 연결될 때만 `HP`/`HS` 세 번 일치부터 다시 확인한다.

### 0.6 2026-08-29 현재 Stage-1 학습 admission 재확인

현재 `dev`에서 `check_finetune.py --config configs/train_finetune.yaml --set
data.digital_primary_path_mode=measured`를 실제 실행하면 canonical bootstrap receipt와
외부 `bootstrap_receipt_sha256`가 없다는 설정 admission에서 exit 2로 멈춘다. 결과
directory도 만들지 않았으므로, 이 fail-closed 결과를 readiness PASS나 학습 실행으로
오인하지 않는다.

82세션의 64-segment coverage 진단에는 12 family×split×subband 부족 행이 있으며, 하한은
독립 신규 session/group 17개다. local source/lineage로 확정된 environment/music 8개와
ESC-50 machine 4개는 보존됐지만, train 2·val 1·test 2의 DNS speech 선택은 새 Elice
canonical_v4 bootstrap receipt와 full public manifest에서만 exact SHA로 발행할 수 있다.
따라서 현재 12행 임시 CSV나 임의 speech source로 녹음을 시작하지 않는다.

다음 학습 전 무음 순서는 새 A100 80GB Elice exact checkout/bootstrap → DNS selection
receipt → no-replace 17행 plan/dry-run → 짧은 Stage-1 additions 수집 → coverage 재감사다.
그 뒤에만 G0/pilot/probe/smoke/100k pretrain/50k fine-tune을 연다.

### 0.7 2026-08-29 개발선 통합

clean linked worktree 7개와 stale registration을 Git으로 해제했고, `dev`에 완전히
흡수된 작업 branch와 안전하지 않은 구형 high-frequency USB experiment를 제거하는
정리 근거를 [`docs/60_20260829_branch_consolidation.md`](docs/60_20260829_branch_consolidation.md)에
기록했다. 이 정리는 raw/model/data를 삭제하거나 `main`에 미검증 결과를 병합하지 않는다.
최종 상태는 `main`(배포 기준선)과 `dev`(통합 개발선)만 유지한다.

### 0.8 2026-08-29 실제 acquisition witness readiness 재감사

현재 Jetson ALSA 장치와 full-octave fail-closed checker를 다시 대조했다. AB13X는
playback 2채널/capture mono 1채널의 asynchronous USB endpoint이고, APE PCM1의 ERR/REF와
같은 hardware frame을 증명하지 못한다. J511/RT5640은 output 후보일 뿐 최근 plug state가
세 번 `None`이고 4-input electrical witness 또는 8-input quiet-zone acquisition을 만들지
못한다. 따라서 current device set만으로 125 Hz--8 kHz canonical P/S·학습·배포를 여는 것은
**BLOCKED**다.

무출력 static checker는 `static_gate_pass=true`와 동시에
`electrical_witness_pass=false`, `canonical_training_eligible=false`를 반환했고, 8-input
physical bundle checker도 raw/plan/sidecar 부재로 정상적으로 exit 1 `BLOCKED`였다. 이는
하드웨어 통과가 아니라 우회가 막혔다는 증거다. 최신 `dev` 전체 pytest는 0 FAIL이며 local
canonical_v4 부재 RuntimeWarning 두 건만 남았다.

문서 commit 뒤 실제 Jetson PCM inventory도 다시 읽었다. 모든 stream은 `closed`이고
AB13X는 계속 2ch adaptive playback/mono asynchronous capture이며, J511 checker의 세 표본도
`None`이었다. 최신 `check_finetune.py`는 `data.bootstrap_receipt`와 외부 receipt SHA가
없어 exit 2로 멈췄고 run directory를 만들지 않았다. 즉, 현재 학습 미시작은 GPU 유휴를
방치한 것이 아니라 canonical admission이 실제로 닫힌 결과다.

정확한 실제 inventory, 4/8-input 최소 조건, safety tap 조건과 Stage-1의 별도 17세션
순서는 [`docs/61_20260829_acquisition_witness_readiness.md`](docs/61_20260829_acquisition_witness_readiness.md)에
기록했다. 현재 소프트웨어로 가능한 다음 단계는 새 Elice A100 exact bootstrap → DNS
selection receipt → 17행 no-replace plan/dry-run이며, final high-band는 동기 acquisition
topology 확정 전까지 녹음·학습으로 우회하지 않는다.

현재 `elice_transfer_manifest.json`은 344파일·4,689,042,188 bytes의 82세션 schema v1
bundle이다. 현 `dev`의 docs-only 변경은 bundle bytes를 무효화하지 않으므로 새 Elice의
canonical_v4/DNS selection에는 쓸 수 있다. 그러나 이 bundle을 canonical 학습 입력으로
재사용하지 않는다. DNS receipt를 받아 17세션을 수집한 뒤에는 99세션 schema v2 transfer를
no-replace 재발행·검증 전송하고 새 receipt로 다시 결속해야 한다.

### 0.9 2026-08-30 새 Elice bootstrap의 실제 pre-venv 경계 복구

새 Elice clone에서 82세션 transfer manifest SHA와 recorded tree byte count는 실제로
관측됐지만, 파일별 SHA/full semantic 검증은 아직 통과 전이었다. 종전
`bootstrap_all.sh`는 venv를 만들기 전에 system `python3`로 full transfer validator를
import했다. 이 import graph가 `soundfile`뿐 아니라 NumPy/Pydantic 경로까지 끌어와 fresh
환경에서 `ModuleNotFoundError`로 실패했다. 이것은 data 부족이나 GPU 문제가 아니라
**bootstrap 순서 결함**이며, 실패를 무시하고 다운로드·학습으로 진행하지 않는다.

복구된 경계는 다음과 같다.

1. system Python은 canonical holdout과 transfer manifest의 regular-file/no-symlink,
   외부 SHA-256 anchor만 확인한다.
2. exact A100 CUDA venv를 만든다.
3. public raw 다운로드, manifest 생성, QA, DNS selection, 학습보다 **먼저** 그 venv에서
   기존 full transfer validator를 실행한다. 따라서 모든 transferred file SHA, strict P/S,
   generation/DNS receipt, recorded lineage 의미 검증은 약화되지 않는다.

새 exact commit을 push한 뒤 remote checkout을 그 commit으로 바꾸고, 먼저 `--preflight-only`,
그 다음 일반 Stage-1 bootstrap을 tmux log로 한 번 실행한다. full validator가 실패하면 raw와
log를 보존한 채 원인을 분석하며 자동 재실행하지 않는다. 이 절은 high-band/quiet-zone
hardware blocker를 해제하지 않으며, Stage-1 82세션 bootstrap과 추가 17세션 수집 전에는
pretrain/fine-tune을 시작하지 않는다.

### 0.10 2026-08-30 Elice bootstrap pytest 중단 및 재개 경계

`560d44b1b4c7b2a47db4d273054bd20402d889d2`로 같은 Elice 인스턴스에서 Stage-1
bootstrap을 실행했다. A100 80GB, exact torch `2.5.1+cu121`, public raw 다운로드/decoder
감사(`candidate=37761`, `accept=36868`)와 transfer full semantic 검증은 통과했지만,
마지막 pytest에서 5개 실패로 exit 1했다. 따라서 bootstrap receipt·readiness·checkpoint·학습
디렉터리는 생성되지 않았고, 기존 venv/raw는 보존돼 있다. 원본 로그는
`/home/elicer/deep_anc_logs/stage1_bootstrap_560d44b.log`에 남아 있다.

실패는 다음처럼 분리했다.

1. Torch 2.5.1에서 causal P/S 독립 oracle의 FP32 누적 순서가 기본 `allclose`보다 작은
   오차를 냈다. adapter 내부 composition은 `torch.equal`로 유지하고 독립 oracle 비교에
   기존 crop 검증과 같은 `atol=2e-7, rtol=2e-6`을 적용했다.
2. `fullband_v5_meter` portable static loader가 오디오 없는 Elice의 `/proc/asound`를
   읽던 결함을 수정했다. 기본 static 검증은 tracked attestation의 physical snapshot만
   사용하며, 실제 Jetson live 경계는 현재 ALSA fingerprint를 명시적으로 수집해 두 번째
   결속 호출을 한다. live hardware gate를 제거한 것이 아니다.
3. `set_amp_level.py`는 fullband-v5의 다섯 operator confirmation을 raw/evidence preflight
   전에 검사하도록 순서를 고쳤다.
4. tracked `measurement_level_evidence.json`이 참조하는 historical meter raw/receipt가
   transfer bundle에 없던 계보 결함을 고쳤다. `transfer_contract`에
   `level_meter_raw/level_meter_receipt` role과 pointer/SHA/receipt exact 검증을 추가했고,
   builder가 evidence에서 두 파일을 자동 발견한다. 새 로컬 manifest는 346파일,
   SHA-256 `7881262a574b8ad793e43879fcab0bcb297831213496195ae0877b9807ccf261`이다.

로컬 변경 실패군 pytest는 통과했고 `git diff --check`/`bash -n`도 통과했다. 전체 pytest는
Jetson의 장시간 scipy 회귀 도중 중단했으므로 새 exact commit의 Elice 전체 pytest가 최종
게이트다. 다음 순서는 새 커밋 push → 기존 Elice checkout fast-forward/clean 확인 → 새
346파일 manifest와 두 meter 파일 전송 → 동일 venv에서 bootstrap 재개다. pytest와 canonical
recorded coverage가 통과하기 전에는 pretrain/fine-tune을 시작하지 않는다.

### 0.11 2026-08-30 추가 녹음 첫 행 fail-closed와 file playback 계약 수정

Elice bootstrap/readiness 15/17 뒤 DNS speech 5개를 exact selector로 선택했고,
`highband-coverage-v1` 17행 plan의 dry-run이 PASS했다. 사용자 승인 아래 첫 environment
세션만 출력했으며, stream 종료 직후 분리 안내를 내고 batch QA가 실패하자 나머지 16개는
자동 재생하지 않았다. 실패 session은
`results/recording_failures/record_duct/batch_qa/highband-coverage-v1/`에 no-replace 보존했고,
active addition session은 0개다.

실패는 장비·gain 문제가 아니었다. 실제 amplitude/peak는 `0.15`, xrun/clip 0이고 timeline
coherence도 150--600 Hz `0.086→0.906`으로 합격했다. 원인은 file producer가 1초 settle
무음 중에도 cursor를 48,000샘플 진행시켜 계획 54.1초가 아니라 55.1초부터 출력한 반면,
generation validator는 planned start에서 fade 없이 재유도한 코드 계약 불일치다. 실패
`source.wav`는 `1초 advance + 양끝 0.1초 fade`를 적용하면 720,000샘플 bit-exact지만,
그 파형은 plan window가 아니므로 재봉인·승격하지 않는다.

복구 규칙은 다음과 같다.

1. 공용 file renderer가 planned start에서 exact keep frame만 생성하고 0.1초 fade를 적용한다.
2. duplex settle은 exact zero prefix이며 file cursor를 소비하지 않는다.
3. producer와 generation validator는 같은 renderer를 사용하고 one-second shift/fade 누락을
   회귀 테스트로 거부한다.
4. 기존 `qa_failed` progress와 raw는 삭제하지 않는다. 수정 commit의 새 Elice bootstrap/DNS
   receipt/plan을 다시 결속하고 무음 dry-run을 통과한 뒤 같은 generation의 row 2를 새
   explicit attempt로 녹음한다.
5. 수정·검증 전에 소리를 반복하지 않고, 재실행도 새 출력 창 보고와 승인 뒤에만 한다.

### 0.12 2026-08-30 amplitude·capture gate·Elice freeze 후속 복구

수정된 exact file renderer로 첫 행을 다시 확인했을 때 settle 48,000 frame 뒤의 15초
source가 현재 계획에서 렌더한 bytes와 max error 0으로 일치했다. 그러나 당시 CLI amplitude가
`0.15`였으므로 기존 82세션의 단일 레벨 `0.06` 계약과 달랐고, 다음 측정값으로 capture gate가
실패했다.

- 실패 raw: `results/recording_failures/record_duct/20260830_120721_461980_timeline_gate_9a157941/`
- `failure.json` SHA-256:
  `246281a690c201c05ab187f8b29152b86acfa533c79ecad3218be602720dbee8`
- `mics.wav` SHA-256:
  `fffa87bfd40542d27ea99ff3a3209c7f4648610e7e484092a173a3fadf98ccad`
- 저역/고역/REF 코히런스: `0.859238 / 0.594673 / 0.857512`
- raw valid-window ratio `0.915254`, 잔여 robust std `2.49684`, p95−p5 `11.8999`
- xrun/clip 0, active additions session 0

따라서 이 raw는 수정·승격하지 않고 실패 증거로 보존한다. canonical additions amplitude는
기존 82세션과 같은 exact `0.06`으로 코드·batch·최종 generation에 모두 강제했다. 신규
capture는 저역/고역/REF 코히런스, raw/aligned valid-window ratio, 잔여 지연 robust std와
p95−p5의 일곱 조건을 공용 `RecordedCaptureGateContract`로 publish 전에 판정한다. durable
`failure.json`은 symlink component를 거부하고 같은 file descriptor에서 snapshot/SHA를
읽은 경우에만 batch progress 근거로 채택한다. resume과 최종 generation도 같은 계약을
재계산한다.

기존 82세션은 raw WAV 전체 reissue QA에서 `82/82 PASS`, 오류·경고 0, 95.67분, 61 group을
재확인했다. 결과는
`results/data_audit/recorded_qa_reissue/20260830_post_capture_gate/recorded_qa.json`
(SHA-256 `484599c489c9c6d4daaf2e4cece1f327cfb44af777e5c02e077a6957d7f04db9`)에
보존했다. 이는 82세션이 Stage-1 자료로 유효하다는 증거이며 2/4/8 kHz full-octave authority로
승격하는 증거는 아니다.

Elice의 기존 venv는 현재 checkout을 import했지만 저장된 `environment-freeze.txt`가 과거
commit을 가리키고 있었다. bootstrap/setup은 venv를 재사용하더라도 freeze를 원자 재생성하고,
유일한 Deep-ANC editable VCS requirement의 전체 40자리 SHA가 `--expected-commit`과 exact할
때만 진행하도록 고쳤다. transfer/DNS receipt 소비측도 stale source commit을 새 외부 SHA로
재봉인하는 우회를 거부한다. A100 80GB와 public raw/decoder audit는 보존하며 36 GB를 다시
받지 않는다.

현재 실제 다음 순서는 **전체 pytest 0 FAIL → clean exact dev commit/push → 같은 commit의
Elice bootstrap/DNS selector 재결속 → 0.06 첫 15초 세션 1개 → 즉시 일곱 gate QA → PASS일
때만 나머지 16개**다. dirty tree를 우회한 녹음, 0.15 재시도, 실패 세션 자동 반복은 금지한다.
99세션 transfer/bootstrap 뒤 readiness가 init만 FAIL인 상태가 되기 전에는 GPU pilot을
시작하지 않는다.

### 0.13 파인튜닝 준비 완료 후 Git history 정리 예약

사용자 지시에 따라 준비 과정의 미세한 수정 commit을 최종 history에 그대로 누적하지 않는다.
다만 bootstrap/DNS selection/recording/transfer receipt가 exact commit SHA에 결속되는 동안에는
rebase로 그 근거를 무효화하지 않는다. 현재 `dev` 고유 이력은 155개 commit과 3개 merge라
직접 interactive rebase의 누락 위험이 크다. 따라서 **현재 코드·테스트 준비가 clean 상태로
고정된 뒤, 성공 canonical 17세션과 checkpoint를 만들기 직전** 다음 절차로 한 번만 정리한다.
이 시점이 사용자 지시의 “파인튜닝 준비 코드 완료 후”이며, 이미 녹음·pilot을 만든 뒤 SHA를
다시 바꾸는 비용을 피하는 마지막 안전 창이다.

1. 현재 `dev` tip을 remote backup ref로 보존하고 local/remote SHA를 기록한다.
2. `main`은 변경하지 않는다. 별도 review worktree에서 최종 tree를 다음 7개 중요 단계로
   재구성한다: timing/measurement 불변식, strict P/S·계보·Stage-1 readiness, exact-env·
   checkpoint·학습 계약, full-octave source/loss/batch 계약, Jetson clock/witness/acquisition,
   physical G4 fail-closed gate, 최종 Elice/recording 통합 복구.
3. private key, token, raw corpus, run artifact가 commit에 들어오지 않았는지 전 history를 검사한다.
4. old/new tree SHA가 byte-exact인지 먼저 확인하고, 정리된 새 `dev`에서 전체 pytest,
   `git diff --check`, bootstrap/selector/transfer/readiness의 exact-SHA 검증을 다시 수행한다.
   rebase 전 receipt를 새 SHA의 증거로 재사용하지 않는다.
5. old/new merge-base와 backup ref를 출력해 검토한 뒤 `--force-with-lease`로만 `dev`를 갱신한다.
6. canonical fine-tune과 현장 평가가 합격하기 전에는 `main`에 병합하지 않는다.

즉 history 정리는 예약돼 있지만, 현재 유효 데이터 수집을 앞두고 SHA를 계속 바꾸는 식으로
실행하지 않는다.

### 0.14 Elice stale reference를 비용 발생 전에 차단하는 정적 게이트

7개 milestone history 후보에서 전체 pytest를 Jetson과 Elice에 병렬 실행했을 때 양쪽 모두
`tests/test_gate_registry.py::test_every_declared_gate_has_a_failing_fixture` 한 건만 실패했다.
원인은 `recording_timeline_fail_closed` registry가 이미 개명된
`test_cli_refuses_to_loosen_the_gates`를 계속 가리킨 것이었다. 모델·데이터·오디오 결함은
아니었지만, 기존 메타 테스트가 전체 회귀 중간에 있어서 Jetson 약 19분과 Elice bootstrap
scan을 낭비했다. Elice log는
`/home/elicer/deep_anc_logs/bootstrap_full_a7d75d050e55256d0e587cdaddabf7e14d919eab.log`에
보존했고 bootstrap receipt·selector·학습은 발행하지 않았다.

옛 node 한 줄만 바꾸고 끝내지 않고 다음 조기 경계를 추가했다.

1. `scripts/ci/check_static_contract_references.py`는 프로젝트 의존성을 import하지 않는
   pure-stdlib 검사다. 현재 운영 Python source 149개에서 static pytest node 172개와 test
   file 23개를 AST로 검증한다.
2. 파일/함수 개명·누락, malformed/parameter node, duplicate test definition, symlink target을
   fail-closed한다. 여러 gate가 같은 fixture를 공유하는 것은 정상으로 허용한다.
3. `src/scripts/configs`의 새 40자리 Git SHA literal은 대소문자 모두 거부한다. source-pool
   v1/v2 byte-exact 재현을 위한 지정 파일 네 위치의 historical builder SHA만 exact
   allowlist이며, 64자리 SHA-256과 docs/runs/results의 역사 증거는 current commit으로
   오인하지 않는다.
4. `tests/conftest.py`가 pytest collection 전에 같은 API를 호출한다. 따라서 stale node는
   전체 회귀 중간이 아니라 시작 시점에 종료된다.
5. `bootstrap_all.sh`는 exact checkout을 확인한 직후 이 checker를 `python3 -I -B`로
   실행한다. holdout/transfer/hardware/venv/raw/manifest/pytest보다 앞이며, stale fixture에서
   뒤 단계와 `.venv` 생성이 모두 0회임을 shell regression이 강제한다.
6. Elice-only hotfix를 금지하고, 원격 실패는 local 코드·negative fixture·문서·GitHub exact
   commit 한 세트로만 복구하도록 `AGENTS.md`와 `docs/05_training_elice.md`에 고정했다.

focused 결과는 static checker 18/18, Elice script 52/52, renamed registry 9개 fixture PASS다.
현재 Elice의 raw `36,403,604,715` bytes와 venv `5,639,428,687` bytes는 보존됐고 관련
bootstrap/selector/train process는 0이다. 다음 순서는 최종 milestone tip의 정적 검사·전체
pytest 0 FAIL → `dev` force-with-lease → Elice exact preflight/full bootstrap → 새 DNS selector
receipt → Jetson 무음 dry-run → amplitude `0.06` 첫 15초 세션이다.

### 0.15 2026-08-30 병목 완화 경계와 현행 82+19 수집 계약

이 절은 위 0.6--0.14에 남은 `17 addition / 99 combined` 실행 수치를 대체한다. 역사적
`highband-coverage-v1` 17행과 그 실패 raw는 그대로 보존하지만 현행
`stage1-coverage-v2` 입력으로 재사용하지 않는다.

Git history 정리는 이미 완료됐다. 실제 `main..dev`는 7개 milestone commit과 main 통합 merge
1개이고 local/remote branch도 `main`, `dev`만 존재하며 worktree는 이 디렉터리 하나다. 따라서
위 0.13의 “155개를 앞으로 재구성” 설명은 과거 예약 기록이며 다시 rebase/force-push하지
않는다. 이 통합 복구는 별도의 의미 있는 한 commit으로 추가한 뒤 exact SHA를 새 artifact의
단일 출처로 사용한다.

사용자 지시에 따라 기술적으로 본질적이지 않은 병목은 다음처럼 완화했다. 최종 2/4/8 kHz
다채널 hardware authority가 없어도 150--1600 Hz Stage-1의 처리량 smoke와 정식 준비는
진행할 수 있다. coverage가 아직 완성되기 전의 A100 200--500 step 실행도
`init_eligible=false`인 finite/VRAM/ETA/resume 진단으로만 허용한다. 반면 lead·극성·인과성,
deadline/xrun/clip, P/S와 raw SHA, 동기·coherence, split lineage 누수, trusted 대역 악화와
대역 밖 고주파 증폭은 결과에 맞춰 완화하지 않는다. 전체 표는
`docs/16_canonical_finetune_guardrails.md` §1.1이 권위다.

실제 old82 `source_aligned→ERR`와 current strict P를 Welch/H1으로 전수 대조한 결과,
2026-08-04 cohort는 train 중앙값 `-25.441966 dB`, 2026-08-06 cohort는
`-20.333239 dB`로 같은 digital reference 단위가 아니었다. 이 차이를 무시한 70:30 학습은
성능이 아니라 서로 다른 plant gain을 동시에 가르치므로 차단한다. 새
`recorded_primary_level_calibration_v1`은 WAV와 source/REF를 바꾸지 않고 historical ERR만
train-only cohort scalar로 current strict-P 단위에 맞춘다. val/test 중앙 잔차 최대
`0.384963 dB`, 전체 session 최악 잔차 `1.679043 dB`, train complex agreement
`0.986107`, scalar+delay 뒤 relative error `0.166113`, 보정 뒤 ERR peak 약 `0.669`로
사전 고정 gate를 통과했다. 실제 receipt는 dirty tree에서 발행하지 않으며 새 clean exact
commit에 결속한다.

또한 네 family 모두에서 `historical_calibrated`와 `current_strict`를 train에 노출하지 않으면
plant-domain 차이를 검증할 수 없다는 새 병목을 발견했다. 종전 17행에는 current music/train과
environment/train이 0개였으므로, lineage가 독립인
`environment/environment_006.wav@25.75s`와
`source_pool_v2/music/music_008.wav@31.5s`를 각각 train 1개씩 추가했다. 현행 exact 구성은
speech 5, music 5, environment 5, machine 4의 **19세션**, parent 포함 **101세션**이다.
family→plant-domain→component→session sampler는 각 family에서 current/historical을 정확히
50:50으로 노출하고 worker/resume global index에도 결정적이다. 서로 다른 plant-domain session
mix는 허용하지 않는다.

신규 19세션은 amplitude `0.06`, 48 kHz, 15초이고 audible 합계는 **285초(4분 45초)**다.
canonical live는 fresh `measurement_level_meter_raw_v1`과 그 receipt를 no-replace
`recording_level_campaign_v1`으로 묶은 뒤에만 열린다. 각 session은 campaign path/SHA,
hardware identity/fingerprint, 같은 amplifier setting 확인, 실제 callback에 줄 float32 source
sample SHA와 peak/RMS/trusted RMS를 저장한다. meter 완료 후 600초가 지나면 임계값을 늘리지
않고 중단해 새 meter/campaign으로 resume한다. campaign 없는 canonical live는 audio import/open
전에 실패하지만, pre-campaign `--dry-run`은 source/lineage/plan/기존 artifact를 무출력으로
끝까지 검사할 수 있다. session 시작은 meter 완료 뒤이면서 campaign 발행 시각 이후여야 하므로
과거 raw를 사후 campaign으로 승격할 수 없다.

Red-team에서는 level calibration receipt의 164개 `source_aligned/mics` 참조를 가짜 SHA/size로
재봉인해도 `verify_bound_audio=False` 경로가 통과하던 결함을 재현했다. schema-v2 transfer
validator는 이제 추가 WAV 재읽기 없이 이미 검증한 `role=recorded` entry map과 164개
path/size/SHA를 exact 대조하고, builder/validator 모두 receipt source commit과 현재 checkout
HEAD의 exact 일치를 요구한다. stale audio ref와 stale commit negative fixture가 각각 이 경계를
강제한다.

학습 전 남은 실제 순서는 다음과 같다.

1. 통합 변경의 focused/전체 pytest, static reference, shell syntax, secret, diff 검사를 0 FAIL로
   끝내고 `dev` clean exact commit을 push한다.
2. 같은 Elice A100의 검증된 raw/venv cache를 새 exact commit으로 fast-forward한 뒤
   `bootstrap_all.sh --no-update`와 DNS/DEMAND selector를 재발행한다.
3. Jetson에서 exact 19행 plan의 pre-campaign dry-run과 audio occupancy를 확인한다.
4. fresh meter 20초 뒤 19×15초를 campaign freshness 안에서 수집하고 각 session의 일곱 capture
   gate를 즉시 판정한다. 실패 행은 자동 재생·임계 완화 없이 raw-first 분석한다.
5. 101세션 generation, old82 level-calibration receipt, transfer schema v2를 발행·전송하고
   Elice readiness를 init만 FAIL인 16/17까지 올린다.
6. G0/loss pilot/measured probe와 200--500 step A100 smoke를 통과한 뒤에만 selected 100k
   pretrain과 50k fine-tune을 실행한다.

현재 canonical checkpoint와 실제 OFF/ON raw는 아직 없으므로 감쇠 dB나 broadband 성공을
주장하지 않는다. Stage-1 성공 뒤에도 2/4/8 kHz와 8 kHz octave 상단 11.314 kHz 최종 목표는
동기 다채널 P/S·matched FxLMS·Level-5 actual G4가 별도로 필요하다.

### 0.16 2026-08-30 Elice DEMAND selector build-id 단일 출처 복구

exact commit `474d6175814fad9170e098960929903461da3814`의 Elice bootstrap은 전체 pytest
0 FAIL과 receipt SHA-256
`999db4c21bd8a1648ddda375b980b32f98f0e4683b3396dcc4809d4b07036e45`로 끝났다. 같은
checkout의 DNS selector도 14파일 bundle을 발행하고 `--verify-existing`을 통과했지만,
DEMAND selector는 8파일을 no-replace 발행한 뒤 self-validation에서 정확히 차단됐다.

원인은 데이터나 threshold가 아니라 manifest generation build-id 직렬화의 코드 불일치다.
공식 생성기와 `manifest_contract`는 `sort_keys=True`, `indent=2`, trailing newline UTF-8
bytes의 SHA-256 `2dc54c3ad06ad6cbd512ef6757b378e8963af97ad0e6ed5b7b6357c5cdede1ec`를
사용하지만, DEMAND immutable verifier만 compact JSON digest
`1eb47c877321e79e91802039932b556d5cd2315724f3de1e873142d60f73091e`를 다시 계산했다.
따라서 그 DEMAND receipt SHA-256
`2593bf61f0d836dea032e9e1d9e672843f03d2654eff3719406ac88a1c167b98`는 **INVALID**이며
selection authority로 사용하지 않는다. 정상 DNS bundle도 source commit이 바뀌면 stale이므로
새 commit의 plan에 섞지 않는다.

복구는 `manifest_contract.manifest_generation_build_id()`를 단일 출처로 두고 manifest
생성기·공식 validator·DEMAND immutable verifier가 모두 같은 함수를 사용하게 한다. 회귀는
공식 pretty digest를 수용하고, 구 compact digest로 generation/ref/receipt를 모두 다시
봉인해도 거부하는지 검사한다. Elice 전용 hotfix, threshold 완화, invalid bundle 삭제·덮어쓰기는
하지 않는다. 수정 commit의 전체 pytest와 clean exact bootstrap 뒤 DNS/DEMAND를 모두 새로
발행·검증한 경우에만 Jetson으로 이관한다.

### 0.17 2026-08-30 Stage-1 수집 첫 행 fail-closed와 사용자 요청 중단 지점

사용자 요청에 따라 **추가 출력·녹음·학습·Drive 업로드를 모두 중단**했다. Jetson의 모든
PCM status는 `closed`이고 PulseAudio는 control node만 열고 있다. `record_duct`, batch
recorder, realtime, calibration, trainer, `rclone` 프로세스는 없다. Elice도 exact detached
HEAD `b9138e395505ad5507547738a8d1e8a2c3c384e5`이며 bootstrap/selector/trainer/GPU process가
0임을 read-only SSH로 확인했다. 따라서 중단 중에는 스피커·앰프·마이크를 분리해도 된다.

중단 전 완료한 소프트웨어·Elice 경계는 다음과 같다.

- `b9138e395505ad5507547738a8d1e8a2c3c384e5`에서 focused 78개와 전체 pytest가 0 FAIL,
  `git diff --check`가 PASS했다. Elice full bootstrap도 같은 clean exact source에서 0 FAIL로
  끝났고 receipt SHA-256은
  `d5a668957c63e7c66aa25e319f83f42c92a1651723043517efdc3e9f9e28b54b`이다.
- 같은 commit의 DNS 14-file/25,505,331-byte selector와 DEMAND
  8-file/45,355,946-byte selector를 Elice에서 각각 write+verify한 뒤 별도 임시 staging으로
  Jetson에 전송했다. DNS receipt SHA-256은
  `5c2bc946300bfffb924dad4a78dba0c4f6ab7d1a8678e58b631ffeddfa3b0db0`, DEMAND receipt
  SHA-256은 `bcf3395b823b4d15e3fd02e3fa317d86bfac6f68a8e3ce7026dac38df62d9aaf`이며 로컬
  immutable verifier도 PASS했다. 구 474 DNS와 invalid DEMAND는 삭제·덮어쓰기 없이
  Elice의 `.superseded` forensic 경로로 분리했다.
- old-82 level calibration receipt
  `data/manifests/recorded_level_calibration/b9138e395505ad5507547738a8d1e8a2c3c384e5.json`
  (SHA-256 `77c3690ee4ba06d082bdf521ce89337a1e74b8a1e3a2904097cf8f4bd7794bd2`)은
  82/82 `source_aligned.wav`/`mics.wav` path·size·SHA 결속을 검증했다.
- 19행 plan `data/source_plans/recorded_additions/stage1-coverage-v2.csv`는 check-only,
  no-replace write, verify를 모두 통과했다. SHA-256은
  `f1b3d63fa1e455bac723a7a323aede0b602486c955bcb945281dc129cc7bc574`다. batch dry-run은
  파일 변경과 audio open 0으로 PASS했고, 공식 출력은 19×15초=285초, output-open 304초,
  분석을 제외한 연결 상한 388.5초, amplitude exact `0.06`, 자동 retry 없음으로 계산됐다.

실제 수집 직전 기존 level evidence를 사용하는 fresh 20초 meter가 PASS했다. raw는
`results/calibration_interleaved/level_bootstrap/20260830_201645_fe5f40a9/meter_raw.npz`
(SHA-256 `490cf6a85c0eaadf7ad9674dc946f66d7dbf8820173ad757078d43b7c05ed0db`), receipt는 같은
경로의 `meter_raw.receipt.json`(SHA-256
`de1ab8488806fc7a0c86a9f085013d87893a0ac6bcda1b341d256c4c8a283285`)이다. 중앙값
`-48.192936 dBFS`는 목표 `-50.1 ± 2 dBFS` 안이었다. 이 meter에 결속한 campaign
`recording-level-3427c7690fbc8ece8fa593487f149580b9aa372aa8b82caecfd1f6c3201fc35c`도 발행·검증했다.
중단으로 600초 freshness가 만료됐으므로 **재개 때 이 campaign을 재사용하지 않는다.**

19행 실제 batch는 첫 행만 실행한 뒤 정상적으로 fail-closed했다. 나머지 18개 source는
재생하지 않았고 자동 retry도 없었다. 첫 행은 environment `water-drops`,
`data/source_pool/environment/environment_006.wav`, start `25.75초`, amplitude `0.06`,
15초였다. active session을 발행하지 않았으므로 현재 추가 수집은 **0/19**이고 82세션
generation 상태는 바뀌지 않았다.

- immutable failure root:
  `results/recording_failures/record_duct/20260830_201833_757643_timeline_gate_e61aad3e/`
- `failure.json` SHA-256:
  `0a902a0dc5ec06baf2092e35a8f06cf3c64628803b551a5bbcc824c26fbef9bd`
- `mics_raw.wav` SHA-256:
  `ab989f15ed2990353a067dc14843aeb6da341c46d537548e3f2ac359d914cfb5`
- `source_raw.wav` SHA-256:
  `886c1ecd3495807e1ca4e15cae5f87a670f2da82f823753d7b9418dc06cebc2f`
- batch progress는
  `data/recorded_additions/stage1-coverage-v2/batch_progress.csv`의 단 한 행
  `record_failed/timeline_gate`뿐이다.

실패값은 source-aligned→ERR coherence²가 150–600 Hz `0.465841`(<`0.9`),
600–1600 Hz `0.433336`(<`0.6`), source→REF raw valid-window ratio `0.737288`(<`0.9`),
aligned valid-window ratio `0.762712`(<`0.77`)이었다. 반면 REF↔ERR 저역 coherence²
`0.807587`, residual robust std `1.944` samples, p95−p5 `8.927` samples는 통과했고,
768,000 frames/3,000 callbacks가 모두 exact 256 frames였다. raw의 active mic peak도
ERR/REF 약 `0.620/0.728`로 경로 무출력은 아니다. 첫 15초 water-drop window에 긴
저활성 구간이 있다는 점은 확인했지만, 이것만으로 source-window 문제라고 확정하지 않는다.
**xrun/clip, 시간축 구현, source 선택의 기여를 immutable raw로 분리하기 전에는 gate를
낮추거나 재녹음하지 않는다.**

재개 순서는 다음과 같다.

1. 스피커 없이 위 failure raw를 오프라인 재검산해 source-window/capture-timing/code 중
   원인을 분리한다.
2. 1행 source/window 교체가 필요하면 lineage-clean plan을 새 exact commit으로 다시 발행한다.
   코드·plan이 그대로여도 이 중단 기록 commit 때문에 HEAD가 `b9138e3`에서 바뀌므로,
   current-HEAD 실행 전에 Elice bootstrap/selector/plan의 exact-source binding을 새로 만든다.
3. 모든 PCM 점유를 다시 확인한 뒤 기존 evidence mode로 fresh 20초 meter와 새 campaign을
   발행한다. `--bootstrap-level-evidence`를 다시 주지 않는다.
4. 원인이 해결됐다는 무출력 증거가 있을 때만 실패 행부터 한 번 실행한다. 통과한 뒤에만
   나머지 18행을 계속하고, 임계값 완화·자동 retry는 하지 않는다.
5. 19/19 뒤 101세션 generation/transfer/readiness 16/17을 만들고서야 G0/pilot/probe/
   resume smoke와 canonical 100k pretrain→50k fine-tune으로 진행한다. 현재 학습은 시작하지
   않았고 checkpoint·감쇠 dB도 없다.

Elice 삭제 전 Drive snapshot은 10개 archive part 중 6개만 성공하고 part 0/1/3/5의
rate-limit retry가 남은 상태에서 함께 중단했다. full size/MD5 검증은 아직 완료되지 않았으며
원본은 삭제하지 않았다. 재개용 로컬 기록은
`results/elice_snapshots/predelete_917aa25a0315247f/BACKUP_README.md`와
`retry_failed_parts_remote.sh`다. 이 백업은 학습 admission의 대체가 아니다.

V10--V14의 구현·검증 경계는 `docs/42_rt5640_j511_connection_gate.md`,
`docs/45_s32_capture_admission.md`부터 `docs/51_causal_ps_prefix_adapter.md`까지를 우선
참조한다. 아래 내용은 보존된 역사·진단 기록이다.

### 0.18 2026-08-30 source preflight 복구와 녹음 재승인

사용자는 녹음을 다시 허용했다. 그러나 새 19행 plan과 무음 dry-run이 통과하기 전에는
소리를 내지 않는다. 실제 출력 순서는 새 exact Elice bootstrap/selector → Jetson bundle
검증 → 새 19행 plan → batch dry-run → 장치 점유 확인 → fresh 20초 meter/campaign →
19×15초 수집이다. 예상 audible은 meter 20초 + additions 최대 285초다. 자동 retry와
임계값 완화는 계속 금지한다.

0.17의 첫 failure raw를 오프라인 독립 재계산한 결론은 다음과 같다.

- `source_raw.wav`의 1초 settle은 exact zero이고 뒤 15초는 plan 원본 SHA/start
  `25.75`/amplitude `0.06`/fade에서 재렌더한 배열과 bit-exact(`max error=0`)였다.
  renderer와 plan binding 오류는 반증됐다.
- timeline estimator가 실제로 읽는 source span은 `12,000 + 2×600 = 13,200` frame,
  hop은 `3,000`, RMS 하한은 `2e-4`다. 첫 환경 source는 이 필요조건의 최대 비율이
  `200/236=0.847458`라 capture gate `0.90`을 물리 경로와 무관하게 통과할 수 없었다.
- 기존 19행을 같은 exact renderer로 전수 검사하면 environment 1행과 DNS speech 5행이
  각각 `0.847458`, `0.889831/0.813559/0.822034/0.847458/0.682203`이었다. 즉
  **6/19는 출력 전에 거절됐어야 했고**, 기존 dry-run의 PASS는 software contract 누락이었다.
- raw lag ptp/std와 정렬 후 residual은 성공 82세션 범위였다. USB clock/timeline estimator가
  주원인이라는 가설은 반증됐다. 다만 종료부 약 1.25초 ERR/REF 동시 바닥 하강의 물리 층은
  전기 DAC witness가 없어 inconclusive로 보존한다.

복구 중인 공용 `recording_source_preflight/v1`은 exact rendered 720,000-frame source에
timeline eligible ratio `>=0.95`, 공식 150–1600 Hz absolute level, quiet ceiling
`-64.0 dBFS`와 coherence² `0.90`에 필요한 predicted SNR을 동시에 강제한다. DNS selector,
source-plan builder, batch dry-run이 같은 evidence를 재계산하고 selected DNS composite
bytes도 다시 검증한다. strict-P 상대 density만 높은 간헐/저레벨 speech는 선택할 수 없다.

새 source-pool 9행의 무출력 exact 검증은 모두 PASS했다. 환경 4행은 서로 다른 authority
component이며 다음으로 고정했다.

| split | source | start | timeline ratio | trusted 150–1600 Hz |
|---|---|---:|---:|---:|
| train | `environment_006.wav` | 42.00 s | 0.991525 | -39.31 dBFS |
| val | `source_pool_v2/environment_014.wav` | 20.50 s | 0.983051 | -40.38 dBFS |
| test | `environment_003.wav` | 24.50 s | 1.000000 | -52.49 dBFS |
| test | `environment_008.wav` | 53.25 s | 1.000000 | -35.32 dBFS |

music 5행은 timeline ratio가 모두 `1.0`, trusted level은 `-40.52`~`-33.86 dBFS`다.
선택된 source-pool 9행은 parent82와 disjoint한 9개 component/158개 authority token이며
상호 교집합 0이다. 관련 focused source/DNS/generation/batch 테스트는 현재 작업 tree에서
PASS했다. 전체 pytest와 exact commit/push, Elice 새 receipt는 아직 완료 전이므로 이 절만으로
recording/training admission을 열지 않는다.

Elice는 삭제하면 안 된다. 현재 clean detached `13aaad649661cda320151d2ca02046a7d0181631`,
A100 80GB idle, 가용 디스크 38,798,897,152B이며 public raw 36,403,604,715B와 manifest 6종의
참조 missing 0이 보존돼 있다. 새 commit을 받은 뒤 full bootstrap과 DNS/DEMAND selector를
즉시 다시 발행해야 한다. 기존 receipt/freeze는 `b9138e3` 결속이라 stale이다.

Elice pre-delete Drive archive는 전체 원본 41,299,005,440B, SHA-256
`a743fe4a4761b6d743171c94b6366d74fa199bb1b0361585ed27547fa627b994`를 재계산했다.
Drive에는 9/10 part, 37,004,038,144B만 있고 이 9개는 size/SHA-256/MD5가 원본과 같다.
누락 part 5는 4,294,967,296B, SHA-256
`c083fc64e2e941c13795a940ccc92b8fd39889c8b91363613699713806c3da7c`다. 공유 Drive API
quota 때문에 세 번 최종 확정에 실패했으므로 backup 완료·Elice 삭제 가능으로 판정하지 않는다.
Jetson의 2026-08-27 고정 snapshot은 Drive 경로 집합 13,428/13,428, missing/extra 0으로
재확인했다. 로컬에 남은 고정 객체 3,429개는 rclone MD5 check 0 differences였고, 이미 로컬에서
정리된 FMA 8,002개와 ESC-50 2,000개의 원격 파일 수/bytes도 당시 개별 PASS receipt와 exact했다.
Drive 파일 bytes 17,439,445,191과 과거 `du -sb` 17,441,317,063의 1,871,872-byte 차이는
디렉터리 metadata 집계 차이이며 파일 누락이 아니다. 따라서 그 고정 snapshot 자체는 완전하다.
다만 이후 Jetson `data/`에 생긴 신규 87개/216,813,288B와 새 results/runs/assets는 이 snapshot
범위 밖이다. 별도 no-replace snapshot의 원격 readback 전에는 그 신규 로컬 자료를 삭제하지 않는다.

### 0.19 2026-08-30 source-guard 이후 첫 15초 현장 진단

사용자의 녹음 승인 뒤 먼저 장치 점유와 모든 PCM `closed`를 확인하고, 새 source preflight를
통과한 `environment_008.wav@53.25s`, amplitude `0.06`, 15초를 **canonical plan에 결속하지
않은 진단 1회**로 실행했다. stream 종료 직후 스피커 분리 안내를 냈고, capture gate 실패 뒤
추가 출력은 하지 않았다.

- failure root:
  `results/diagnostic_source_guard_live/failures/20260830_215127_639247_timeline_gate_22881ff7/`
- `failure.json` SHA-256:
  `77dbbaf83c9cd1061852a1509856b4646bc60ca4624c3f4aed41ab608c960ddb`
- `mics_raw.wav` SHA-256:
  `3ee2aa6f8855deb9041cd4c26b437398c6b1903a69de66ae909c6757a4d2d949`
- `source_raw.wav` SHA-256:
  `90fc25891972dc272438d068a9c76b560793953381b3e81fc034f003292c0106`

settle 뒤 source는 exact 720,000 frame이고 전체 15초 동안 끊기지 않았다. peak는 source
`0.0570`, ERR `0.5846`, REF `0.7137`, ADC clip 0이었다. source-only preflight가 제거하려던
무음/저레벨 window 문제는 이 capture에서 반증됐다. 그러나 현행 alignment 뒤에도
source→ERR coherence²가 150--600 Hz `0.598340`, 600--1600 Hz `0.337707`, raw valid-window
ratio `0.741525`, 잔여 robust std `3.840601 samples`라 공식 gate를 통과하지 못했다.
source→REF도 두 대역 `0.666874/0.493331`, REF→ERR은 `0.833722/0.650410`이었다.

같은 source의 2026-08-30 앞선 amplitude `0.06` 진단 두 건은 source→REF가 저역
`0.938/0.904`, 고역 `0.877/0.874`였고, 0.85초 offset을 제외한 digital source interior는
이번 capture와 bit-exact했다. 여러 window/quality/band/refine 폭의 오프라인 재정렬 중 최선도
source→ERR `0.708/0.419`에 그쳐 acceptance를 회복하지 못했다. 따라서 이번 실패는 source
선택이나 단순 renderer 오류가 아니라 **재연결 뒤 물리 경로 또는 독립 USB DAC--ADC 시간축의
비정상 변동**일 가능성이 높다. raw를 승격하거나 gate를 낮추지 않는다. 다음 출력은 이 raw의
오프라인 forensic이 끝나고, 원인을 구분하는 단 한 번의 bounded probe가 정해진 뒤에만 한다.

## I. 보존된 광대역 준비 기록

파인튜닝은 아직 시작하지 않는다. 이번 브랜치는 준비 계약과 계보를 복구하는 중이며,
다음 증거가 모두 생긴 뒤에만 canonical 학습을 연다.

1. (완료) 새 strict P/S 캡처와 level evidence
2. (로컬 완료) 해당 P/S·82세션·계보 자료를 결속한 Elice transfer manifest
3. Elice에서 재생성한 public corpus manifest 6종과 전체 QA
4. 선택된 계약으로 처음부터 완료한 tiny 100k canonical init checkpoint

과거 `pretrain_*_corrected`, `finetune_tiny`, legacy P/S는 삭제하지 않지만 모두
diagnostic-only다. init, resume, 모델 선택, 성능 주장의 근거로 사용하지 않는다.

### 2026-08-28 최신 실행 상태 (아래 누적 기록보다 우선)

현행 pushed HEAD는 `work/broadband-anc-v2`의
`ec27098839a94a21d24044e1d1f881435ccabd47`이다. 최근 독립 격리 커밋은 다음과 같다.

- `6f113f4`: 광대역 source population의 실제 로컬 가용성을 8 physical/7 objective
  대역별로 감사. 12,298 후보 중 2,949개만 로컬에서 bounded decode됐고 canonical
  후보는 0이다. missing 9,349개와 lineage/split 교차는 Elice raw 재구축 전 blocker다.
- `2f9e7ec`: 11.605333초/557,056-frame v5 persistently-exciting 측정 신호와
  fixture-only raw/offline publisher를 봉인. plan SHA는
  `32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127`, PCM SHA는
  `c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff`다.
  live authority는 여전히 `None`이며 실제 출력은 하지 않았다.
- `ec27098`: 125/250/500/1000/2000/4000/8000 Hz 동일가중 baseline과 additive
  worst guard를 가진 diagnostic v3 loss primitive. relative NMSE floor -80 dB와
  CPU/CUDA exact-cancellation finite-gradient 회귀를 포함하지만 trainer admission에는
  연결하지 않았다.

이 절의 v5가 현행이다. 아래 누적 기록에 남은 **49.627초 multi-panel v5** 설명과
`measure_paths_broadband_interleaved.py` followup은 폐기된 설계 기록이며 현재 live 명령이
아니다. 이를 실행하거나 현행 plan으로 승격하지 않는다.

Jetson 무출력 실제 확인에서는 ERR/REF 입력이 각각 약 -64.89/-68.77 dBFS로 살아 있었고,
`/proc/asound/.../status`의 모든 PCM은 `closed`였다. PulseAudio는 control node만 열고
PCM을 점유하지 않았다. 이 값은 연결·liveness 증거이지 P/S나 ANC 성능 증거가 아니다.

현재 최우선 blocker는 두 개다.

1. v5 exact PCM에서 waveform pilot로 공통 q를 구하고, central cyclic fit_a/fit_b에서
   absolute/fractional P/S를 복원해 서로 다른 integer zeros와 compact FIR을 만드는
   raw-bound live analyzer
2. 모든 callback의 exact status/timestamp/actual submitted prefix를 보존하고
   priming·xrun·partial failure·normal drain을 fail-closed로 처리하는 live capture adapter

두 구현의 합성 fixture만 통과해서는 소리를 내지 않는다. exact plan/raw SHA 결속,
독립 red-team, 전체 무음 dry-run 뒤에만 사용자에게 meter 20초 + v5 11.605초의 명령,
스피커, 볼륨, 저장 경로를 먼저 보고하고 명시적 승인을 받는다. 현재 Elice 인스턴스와
canonical pretrain/fine-tune 실행은 없으며, 이 두 물리 blocker와 public corpus authority가
닫히기 전에는 새 GPU 인스턴스를 열지 않는다.

### 2026-08-28 광대역-v2 작업선 즉시 상태

현행 작업은 `work/broadband-anc-v2`에서만 진행한다. Stage-1 readiness 기준선과 과거
high-frequency diagnostic branch는 움직이지 않는다. 현재 물리 성능 수치를 새로 주장할
canonical checkpoint/raw ON-OFF session은 없다.

| 축 | 현재 검증 결과 | 권위 판정 |
|---|---|---|
| Tiny 표현력 | 100 Hz--11.314 kHz delay-only deterministic structural G0 통과 | 구조 진단 PASS, 실제 덕트 성능 아님 |
| runtime telemetry | callback timestamp/frame/completion, engine-step budget, fallback, xrun/ring/watchdog를 분리하고 무음 회귀 100개 통과 | 실제 session 없음; software 단독 authority 최대 INCONCLUSIVE |
| source-v2 | compressed full-decode 계보, 3개 이상 component union, actual Q15 source/P-applied ERR 9×7 재계산 구현. 계약 SHA `0bb458d68ec2a00466a96a35bb1da3bc3ed13d5085b782aefb6cb4b849d58e8a` | 실제 source와 causal P가 없어 0/48 BLOCKED |
| continuous causal P/S | actual-int16 continuous pilot + 독립 PE fit/holdout + 공통 q 설계를 검증 중 | live raw/7대역 stationarity·SNR 증거 전 BLOCKED |
| 광대역 학습 | causal P·n, S·y의 prefix/state 및 artifact SHA admission을 별도 schema로 연결 중 | 실제 causal authority 전 시작 금지 |

`PortAudio` callback 이전 capture period 유실은 Python timestamp만으로 증명하지 못한다.
다만 이를 영구 하드웨어 blocker로 단정하지 않고, highband 결과와 독립인 continuous pilot,
구간별 q/change-point 및 physical REF witness를 runtime receipt에 추가 검증한다. 구조 fixture나
문서 문구로 이 판정을 PASS로 바꾸지 않는다.

### 2026-08-28 최종 광대역 목표 전환 — 계약 배경과 누적 증거

사용자는 최종 목표를 150–1600 Hz에 제한하지 않고, 저역을 유지하면서 2/4/8 kHz에서
Deep-ANC가 matched FxLMS보다 우수함을 실제 덕트에서 증명하는 것으로 확정했다. 8 kHz는
octave 중심이므로 v2 point-control 식별·데이터 상단은 11,313.708 Hz다. source family는
speech/music/environment/machine 네 계열과 Level-5 unseen을 모두 유지한다.

새 코드 authority는 `src/deep_anc/dsp/control_band_contract.py`, 운영 규범은
`docs/18_broadband_anc_guardrails.md`다. 기존 150–1600 strict 계약과 자산은 Stage-1로
보존하지만 최종 광대역 성공·init·배포 증거로 승격하지 않는다.

실제 82세션을 `source_aligned.wav→mics.wav ERR ch0`, 각 5–65초, Welch/coherence
8192/4096으로 전수 재계산한 diagnostic report는
`results/data_audit/broadband_prerequisite_diagnostic_20260828.json`이다. file SHA-256은
`c9bbad10c7b2f0fde658394023e7da7c7ed8757d7bfcb3950a8a25d085f0d4f9`, control-band
contract SHA는 `eebdba2f82ee9e85cc117a612aa3ef592f43d6ebbd19f16a1207469fae103864`다.
결과는 BLOCKED다.

- 150–300/300–600/600–1000/1000–1600 Hz의 coherence median은
  0.968/0.952/0.904/0.827이다.
- 1600–2828/2828–5657/5657–11314 Hz는 0.592/0.220/0.018이다.
- 같은 고역 세 구간의 ERR target-d density PASS는 3.7%/0%/0%, coherence+density
  독립 group은 전체 기준 3/0/0이다. family×split 하한 4를 충족하지 못한다.
- current strict P/S는 same-capture/lead=115 증거는 유효하지만 excitation upper가
  P 1648/S 1640 Hz라 광대역 역할은 BLOCKED다.

2026-08-27 fullband diagnostic raw
`results/experimental_high_band/20260827_fullband/20260827_203328_1b24d0c2/raw_measurement.npz`
는 2/4/8 kHz 에너지는 전달했으나 저장 계약에서 clock valid repeat가 0/64다. 임계를
낮추거나 사후 clock band를 바꿔 official P/S로 승격하지 않는다.

v4 multi-panel plan은 diagnostic-only로 강등했다. drive별 16 Hz grid의 3000-sample
bulk-delay alias, strict drift의 panel 간 약 181-sample 누적, 결과 기반 ±16-sample stitch를
공식 timing 증거로 사용할 수 없기 때문이다. 기존
`results/data_audit/broadband_measurement_signal_plan_live_authority_v4_20260828.json`과 세 SHA는
forensic 기록으로 보존하지만 live에서 거부한다.

대체 v5 signal-only plan은 P/S 각각 0.25초 비주기 marker+0.125초 guard, 4초 warmup,
panel당 63회, panel 사이 10-period 저역 anchor를 사용하며 block padding 포함 49.627초다.
각 period의 **actual submitted int16 pilot spectrum**을 분모로 사용하고, 같은 bin의 반대
DAC channel null을 actual DFT에서 absolute `≤1e-8`, main 대비 ratio `≤1e-12`로 강제한다.
ERR/REF×P/S는 하나의 global ADC→DAC map에 동의해야 하며 11.314 kHz의 20 dB-grade
residual/trajectory budget은 0.0675518903 sample, highband 결과 기반 phase repair는 exact 0이다. PortAudio time_info는
monotonic/slip witness로만 저장한다. 제한된 measured tone은 finite causal history operator가
아니므로 `measured_band_training_eligible=false`이고 상태는
`blocked_until_fullband_persistently_exciting_causal_history`이며,
1024-tap compact FIR은 항상 diagnostic-only다.

**v5 live authority는 현재 `None`으로 잠겨 있고 실제 오디오는 0회다.** root의 exact
file/payload/PCM SHA 검토 전에는 signal-only dry-run만 가능하다. 다음 순서는 v5 dry-run과
offline publisher 최종 회귀 → 별도 persistently-exciting fullband causal P/S 식별 설계 검토 →
사용자 승인 뒤 meter+측정 한 연결 창 → raw 분석 → THD/IMD → sub-sample alignment를 가진
recorded v2 → loss/G4 v2다.
광대역 직전 meter는 `set_amp_level.py --followup-mode broadband`에 exact v5 plan과 새
`results/` raw session을 함께 넘긴다. 이 모드는 plan recipe/PCM과 no-replace target을
sounddevice import 전에 검증하고, PASS fresh meter raw와 다섯 확인이 포함된
`measure_paths_broadband_interleaved.py --execute-live` 명령만 출력한다. 기본 strict/
bootstrap 후속 안내는 기존 복구 절차와 byte/의미 호환을 유지한다.
광대역 meter raw에는 v5 authority의 exact 상대경로/file SHA/canonical payload SHA/PCM SHA,
실제 hardware 상대경로/SHA, `verified_existing` level evidence 상대경로/SHA와 예정 raw target을
`broadband_meter_followup_v1`로 봉인한다. meter는 stream open 직전·capture 종료 직후·명령
출력 직전에 이 bytes/target을 다시 검증하고, live도 같은 metadata와 invocation을 exact
교차검증한다. 따라서 semantic-equal byte 변조, 다른 evidence, 기본값으로 되돌아간 hardware,
meter 도중 raw target 선점은 출력 전에 실패한다.
기존 dense fullband 자극을 그대로 반복하지 않는다. NS/CS는 같은 USB DAC clock이므로 먼저
sparse pilot+fractional q/no-slip을 검증하고, 실패할 때만 rollback 가능한 I2S DAC 공통-clock
경로를 연다.

THD/IMD도 signal-only 48.000초 계획까지 생성했다.
`results/data_audit/broadband_nonlinearity_signal_plan_strict_v2_20260828.json`의 file SHA는
`2d40ee05a1587f470a0de67eec35e765a58c8929cd07b89037be62aeb515e590`, PCM SHA는
`59e8845f4ea898afbacba8afd030bec13f0005b571b78f20ac8329e695615d14`다. raw publisher와
THD/IMD 분석 gate가 생기기 전에는 live 출력이 exit 2로 닫힌다.

recorded v2 campaign receipt의 fail-closed 구조는
`src/deep_anc/data/broadband_coverage_receipt.py`에 추가했다. native Nyquist, source/mics SHA,
sub-sample alignment+clock witness, segment metric 재집계, 동일 source/lineage 독립-group 위장,
split×family×7 subband group 하한을 모두 검사한다. 현재 82세션 진단 BLOCKED를 PASS로
바꾸는 도구가 아니며 새 broadband v2 recording이 생긴 뒤에만 receipt를 발행한다.

Synthetic public corpus의 native-Nyquist/lineage 1차 게이트도
`src/deep_anc/data/synthetic_broadband_coverage.py`와
`scripts/data/audit_synthetic_broadband_native.py`에 추가했다. split×family×7 subband마다
독립 group 4개를 요구하며, built-in synthetic generator나 16 kHz 원본의 업샘플을
8 kHz octave 증거로 세지 않는다. 현재 Jetson에는 `data/manifests/canonical_v4`가 없어
BLOCKED이며, 기존 MIMII fan(16 kHz)은 native Nyquist 8 kHz라 마지막
5657--11314 Hz machine cell을 원천적으로 채울 수 없다. Elice 새 세대에는 native
high-rate machine 원본과 source spectral-density receipt가 추가로 필요하다.

광대역 성능 판정의 pure raw-segment gate는
`src/deep_anc/eval/broadband_point_control.py`에 추가했다. 같은 OFF/DL/FxLMS window만
받아 family×7 subband의 target-d coverage, 양의 Deep-ANC 평균/worst10/cluster-CI를
검사하고, 고역 세 구간은 DL−FxLMS 평균/worst10/paired cluster-CI 하단까지 0 dB 초과를
요구한다. 저역 평균이 고역 실패를, 고역 평균이 저역 실패를 숨기는 fixture와 다점 ERR를
단일 point로 평균내지 못하게 하는 fixture가 있다. 아직 test-once/provenance/runtime
receipt writer와 연결하지 않았으므로 acoustic metric-only다.

광대역 Jetson runtime gate는 `src/deep_anc/eval/broadband_runtime.py`에 분리했다.
최소 30초 관측, P99 `<3ms`, max `<5.333ms`, 네 위치의 lead
(plant/checkpoint/deployment/runtime) exact 일치와 miss/xrun/ring drop·add·backlog/
fallback/watchdog/sample slip 0을 모두 요구한다. 이는 아직 실제 canonical broadband
ONNX/runtime log가 없으므로 schema와 fixture만 PASS이며 현 모델의 고역 실시간 성능
증거가 아니다.

### 2026-08-28 현재 운영 상태와 강제 가드레일

단계별 PASS/FAIL/BLOCKED, 저·고역·네 source family·latency·one-shot G4·배포 중단 조건은
[`docs/16_canonical_finetune_guardrails.md`](docs/16_canonical_finetune_guardrails.md)가
권위 인덱스다. `src/deep_anc/dsp/invariants.py`의 절대목표 family도
speech/music/environment/machine 네 계열로 정합화했고, environment는 실제 canonical pool인
DEMAND+ESC-50 비중으로 검사한다. 이 임계·family를 결과에 맞춰 낮추지 않는다.

Elice 인스턴스는 광대역 P/S·데이터 blocker 때문에 당장 공식 학습을 시작할 수 없고 GPU가
유휴였으므로 **삭제 가능으로 확정했다**. 삭제 전 고유 `manifests/results/runs`를 Google Drive
`DeepANC/elice_snapshot_20260828/predelete_49dd6c7/manifests_results_runs.tar`에 직접 스트리밍했고,
크기 479,846,400 bytes와 MD5 `b4d999c70ba4e1fc745490a15faaae23`를 Drive 객체에서 다시
검증했다(SHA-256 `bfa3d7de91e747049eeebde99d844b87c4973bbfa1985a63e9c49cdc10542589`).
약 36.4GiB public raw는 재다운로드 가능한 corpus라 이 archive에서 제외했다. 따라서 아래
과거 Elice receipt/경로는 forensic 기록일 뿐 현재 살아 있는 인스턴스나 학습 프로세스로
가정하지 않는다. 광대역 local blocker가 해결된 뒤 A100 80GB 인스턴스를 새 exact commit으로
재구축한다.

2026-08-28 이번 read-only SSH 재확인에서도 기존 tunnel
`central-01.tcp.tunnel.elice.io:56230`은 key exchange 전에 원격이 연결을 닫았다. 따라서
그 endpoint에는 실행 중 학습/GPU가 있다고 주장하지 않으며 반복 접속도 하지 않는다. 새
인스턴스가 필요해지는 시점은 광대역 P/S·recorded/synthetic blocker와 exact source commit이
모두 고정된 뒤다.

기존 Elice bootstrap receipt는 이전 commit에 결속됐고 최신 v13 시도도 runtime plant test
1건 실패로 receipt를 발행하지 못했으므로 재사용 금지다. 49dd6c7에서 `/tmp`에만 둔
production-batch-96 current-loss G0는 500 step trusted NMSE **-7.239dB**, y_rms 0.03786,
1.52 step/s로 진단 기준 `< -6dB`를 통과했다. 그러나
`role=diagnostic_overfit`, `init_eligible=false`, deterministic backend false, receipt 없음,
secondary-surrogate이므로 공식 G0나 실제 덕트 감쇠 수치가 아니다. 같은 batch SHA의
NMSE-only/no-DNH 대조를 끝내고 로그를 diagnostic-only로 보존한다.

같은 old exact checkout에서 alpha 1.0의 DNH 붕괴 경계를 좁히는 1,000-step 진단도
끝냈다. 모두 `deterministic_algorithms=false`, `role=diagnostic_overfit`, init/G4 부적격이며
실제 덕트 감쇠 dB가 아니다. measured-P fixed batch에서는 lambda_dnh 0.0001875,
0.000375, 0.0005625, 0.00075가 각각 trusted NMSE -16.229, -16.075, -16.642,
-16.329dB로 영출력 붕괴를 피했다. secondary-surrogate는 0.00075에서 약 0dB로 붕괴했지만
0.0005625에서는 -13.404dB였고, RIR-surrogate 0.00075는 -12.861dB였다. 마지막 RIR 진단은
로그상 lead=0이므로 strict lead=115 실험과 섞지 않는다. 이 결과는 붕괴가 lambda 하나만의
함수가 아니라 plant 표현과 결합된다는 반증이며, 후보별 schema-v7 strict-S gradient gate를
생략할 근거가 아니다.

Jetson 보존 로그 SHA-256은 0.0001875 measured
`bd74ed88ebd910b36254f0db907ff16fe2671c01c4a362b7fd420335132d4b44`, 0.000375 measured
`343e7acec7a2c12f81744c933bb3942fce223fc6392e9162e9d67e3eba0951c8`, 0.0005625 measured
`1b737f0a83a739101fc07c8ee75286325c8fdcfa150655b7d6d2fabfa50c9896`, 0.0005625
secondary-surrogate `7d2943109f7b0f65bd92ca830e47fbdc8fc421a8acfba6029ef2c6ed964e755f`, 0.00075 measured
`dbe63d597427a921de19ec1bb0e89da4db2749f62b6e8b50a9a0c88225c149ae`, 0.00075
RIR-surrogate `c8b658ec2916ab8701b7cdc601d8e92799b5789b62d4ed9aae4cd1dcd127e06d`다. 원본은
`results/diagnostic_gpu_keepalive/49dd6c7bcfa99fa7565fa3130779f5b1e6388476/`에 보존한다.

공식 G0 evidence 경로는 실제 실행의 PyTorch/cuDNN/CUBLAS 결정론 상태를 시작·종료 시점과
`environment.json` SHA로 결속하도록 복구했다. 무증거 diagnostic 경로는 이 evidence를
발행할 수 없다.

추가로 실제 원격 canonical_v4와 recorded holdout를 대조해 보수적 numeric identity 기준
`dns_book` 340/5946/8201, `dns_reader` 422/652 교집합을 발견했다. 기존
`namespace_disjoint_no_official_crosswalk` 정책은 이를 놓치므로 과거 15/17 기록은
authoritative하지 않다. 공식 crosswalk를 사실로 단정하지 않되 false-negative를 막는
cross-corpus numeric alias를 양쪽에 적용하고 public manifest를 재생성하기 전까지
`corpus_disjoint`는 BLOCKED로 본다. 알려진 init·coverage·speech-lineage 세 blocker를
적용한 현재 최대치는 **14/17**이다.

마이크는 다시 연결됐지만 역사적 `highband-coverage-v1` 17세션은 **Stage-1
600--1600 Hz 보충용**이지 2/4/8 kHz 광대역-v2 데이터가 아니다. 이 17개 plan의 exact
source/start/split/lineage, 전체 무음 dry-run, PCM 점유 재확인, family별 예상 출력 시간이
고정되기 전에는 Stage-1 추가 녹음도 열지 않는다. 최종 광대역-v2는 별도의 sub-sample
alignment와 7대역 ERR target-d receipt를 가진 새 generation으로 수집한다. 기존 82세션과
strict P/S raw는 그대로 보존한다.

### 2026-08-28 canonical G4 raw 계약과 다음 Elice 실행

공식 recorded val/test가 summary scalar만으로 PASS되는 경로를 닫았다. 현행 계약은
다음을 모두 raw `metrics.npz`에서 다시 계산한다.

- global trusted/fullband/gap per-segment 배열
- speech/music/environment/machine의 session·lineage group별 평균, 최악 10%, cluster CI
- 125/250/500/1000/**1600**/2000/4000/8000Hz의 8-center octave 감쇠
- 150–300/300–600/600–1000/**1000–1600Hz** strict 부대역의 target(`d`=ERR)
  energy-density coverage, family별 평균·최악 10%·독립 group CI
- selected manifest의 모든 val/test session과 raw segment session의 전단사, 그리고 각
  session의 family/group exact binding
- checkpoint/experiment contract/selection/test capability/consumed marker SHA의 one-shot
  ledger 결속

대역 밖 do-no-harm은 trusted 내부 실패를 중복 집계하지 않고 125/2k/4k/8kHz만 본다.
trusted 내부는 네 strict 부대역 게이트가 별도로 막는다. stored dtype·threshold·center·boolean도
cast하지 않고 schema대로 검사한다. manifest session을 좋은 결과만 골라 생략하거나 running
ledger와 metrics를 함께 변조해도 completion marker를 만들 수 없는 회귀 테스트가 있다.

canonical data 설정은 declared public source manifest 누락을 synthetic fallback으로 바꾸지
않고 즉시 실패한다. loss pilot은 정확히 20k, measured probe는 정확히 5k이며 pilot/probe/smoke/
pretrain/finetune 역할마다 데이터 분포·증강·P/S·sampler·loss 계약을 고정한다. 전체 pytest는
이 변경 worktree에서 0 FAIL로 통과했다. Elice로 넘길 exact source는 이 섹션을 포함한 branch
HEAD를 push한 뒤 40자리 SHA로 다시 확인한다.

현재 `configs/duct.yaml`의 strict P1386/S1245/lead115를 직접 다시 푼 인과 FIR 설계 상한은
150–1600Hz 전대역 26.10dB, 최악 옥타브 17.28dB(125Hz)다. 이는 **모델의 실측 감쇠가 아니라
플랜트의 이론적 상한**이다. readiness 선언 2.15dB는 strict 캡처가 독립 반복되지 않은 상태의
보수적 prior로 유지하며 optimizer 성능을 제한하지 않는다. legacy P/S의 4.83/2.16dB를
current strict 값으로 읽던 테스트는 제거하고 `configs/duct.yaml` 경로에서 재계산한다.

Elice read-only preflight 결과 원격 `~/Deep_ANC`는 clean detached `49dd6c7...`, A100 80GB
PCIe는 0MiB/0%로 유휴, 디스크 여유는 84,320,649,216 bytes(약 78.53GiB)였다. 학습·bootstrap
프로세스는 없다. decoder audit file/internal SHA는 각각
`7fb16a9b27e5115c458a61c4138173a35a198fec1d30c0437e42facee71fbec2` /
`ceac538487ffe1414d433e3a83fdee11a0d17c204427cf8e7fed92bb73c2940f`로 canonical_v4와
일치한다. 기존 bootstrap receipt는 old source에 결속됐으므로 재사용하지 않는다.

현장 schema v2 coverage 감사는 manifest
`ad5978f2ecd0dd2e9c1b3c8b286b0bf9868054b4071c66525a8b061ce3410575`, timing
`1d6723bbfbad1371fab9d38e827c59789eba35a98fae67e478c44e1fdb0061db`에 결속됐고
**FAIL**이다. 8개 균등 표본은 70초 세션의 고역 구간을 놓쳐 불필요한 재녹음을 유발하므로,
canonical population과 공식 recorded evaluator를 `max_segments_per_session=64`로 함께
고정했다. 현 세션 길이에서는 양끝 0.25초를 제외한 사실상 모든 비중첩 segment다.
64-segment 결과의 부족 행은 train 2개, val 5개, test 5개다. 1000–1600Hz는
train machine=3/speech=2, val environment=3/machine=3/music=2/speech=3,
test environment=1/machine=2/music=2/speech=2 독립 그룹이다. 600–1000Hz는
val speech=3과 test machine=2만 부족하다. 임계값 0.25 또는 그룹 하한 4를 낮추지 않는다.

Jetson의 64-segment 실제 보고서는
`results/data_audit/recorded_subband_coverage_fullscan_20260828.json`, coverage contract SHA
`db305cf183caa7816a5d8abcbb44bfae4206873778cef218f973adb4607e3e55`, file SHA
`00d705890799024cc00be12f8d39ef99bec5fa71a925005dcd9427f1f7da1f94`, semantic evidence SHA
`cb80b78a84869628355bab3b553cd1b120ee1c99b4e6fa649efac551c680922f`다. 이는 부족 행을
확인한 로컬 schema v2 진단 증거이며, 현행 readiness의 canonical 증거로는 재사용하지 않는다.
현행 schema v3는 여기에 `segment_seconds=1.5`와 hop 정렬 후 실제 segment sample 수까지
exact 결속한다. Elice bootstrap은 같은 manifest bytes/timing/threshold와 Elice 실제 경로에서
schema v3 contract SHA를 유도하고, 그 이름의 canonical report를 no-replace로 발행해 receipt에
결속해야 한다.
과거 8-segment `98e9c4…` 보고서는 삭제하지 않되 canonical readiness 근거로
재사용하지 않는다.

report lifecycle은 고정 파일 덮어쓰기가 아니다.
`results/data_audit/recorded_subband_coverage/<coverage-contract-sha256>.json`에 manifest/timing/
threshold 세대별로 no-replace 발행한다. 새 녹음으로 manifest가 바뀌면 새 이름을 만들고 옛
FAIL report는 그대로 보존한다. Elice bootstrap receipt schema v2가 report path/file SHA와
manifest/timing/contract SHA를 외부 `bootstrap_receipt_sha256` 아래 결속하므로, ignored report를
수정하고 자체 digest를 다시 봉인해도 readiness가 거부한다.

공식 recorded val/test도 동일 모집단을 독립 재검산한다. max 64, segment 1.5초, edge
0.25초뿐 아니라 checkpoint `model.hop`/timing/PlantSettle, 기본 feedback와 warmup,
manifest sample rate, 각 session의 immutable `session.json` 정렬 지연, 모든 결정론적 start
exact set을 metrics와 대조한다. 이 값의 CLI override는 diagnostic 전용이며 selection,
campaign capability, test completion에는 사용할 수 없다.

따라서 다음 순서는 새 HEAD push → Elice exact clean checkout → audit-reuse full bootstrap →
public/recorded speech lineage를 독립 원본으로 복구한 뒤 **15/17 readiness**
(의도된 FAIL: init + recorded subband coverage) 확인 → 부족 family×대역의
독립 원본만 짧게 추가 녹음 → coverage PASS인 **16/17 readiness** →
G0 500 step → alpha 0.7/1.0의 각 20k pilot+5k measured probe
(필요하면 0.85 chain 추가) → measured-probe winner →
selected-loss A100 exact-resume smoke → tiny 100k canonical pretrain → **17/17 readiness** → 50k
measured fine-tune이다. 실제 val/test G4도 같은 raw coverage를 다시 검사하며 부족하면
`INCONCLUSIVE`로 막는다.

따라서 현재도 canonical 모델의 실제 덕트 2/4/8kHz 감쇠와 Level-5 unseen 일반화는 숫자로
말할 수 없다. 2026-08-27 high-band 캡처는 clock witness 0개로 Invalid experiment이며,
학습 후 마이크를 다시 연결해 승인된 짧은 OFF/ON raw 실험으로 따로 검증해야 한다.

### 2026-08-28 runtime physical timing audit

현 Jetson의 `runtime_tiny.yaml`/`runtime.yaml`은 legacy artifact의 lead=109만
서로 맞춘 설정이다. 현재 strict P/S capture
`5ac1313488c8434bb4d672a36503df59`의 authoritative lead는
`S.delay 1245 + handoff 256 − P.delay 1386 = 115`다. 6 samples(0.125 ms)의
불일치는 2 kHz에서 약 90°, 4 kHz에서 약 180° 위상 오차가 될 수 있으므로,
legacy 숫자를 115로 고쳐 실행하는 것은 금지한다.

`src/deep_anc/realtime/plant_contract.py`는 digital-reference DL runtime이
sounddevice import·engine 생성·입력 probe보다 먼저 다음을 read-only로 대조하도록
추가됐다.

- same-capture P/S metadata, 48 kHz/256/low, 채널, xrun/repeat/consistency
- immutable raw/analysis SHA 및 paired level evidence(probe=0.003)
- `PlantDelays.lead()`가 유도한 lead=115

따라서 legacy runtime은 정상적으로 fail-closed 되어야 하며, current Tiny/Base의
실제 덕트 ANC 성능 근거로 사용하지 않는다. canonical 115 checkpoint/ONNX와 G4,
natural-crest evidence가 생긴 뒤 별도 deployment config를 만든다.

### 2026-08-28 Jetson storage cleanup

Google Drive의 `gdrive:DeepANC/jetson_data_backup_20260827/data/raw/music`는 local/remote
file count·bytes, fixed manifest SHA, `rclone check --one-way`(0 differences)까지
검증했다. 그 뒤 정확히 `data/raw/music/fma_small`만 삭제했다
(8,002 files, 7,975,472,258 bytes). `fma_metadata/tracks.csv`(260,414,445 bytes)는
Elice transfer lineage 입력이라 보존했다.

`data/raw/noise/esc50/ESC-50-master/audio`도 Drive 내용 대조가
`0 differences / 2,000 matching`인 것을 확인한 뒤에만 삭제했다
(2,000 files, 882,088,000 bytes). `meta/esc50.csv`와 repository metadata는 남겼다.
`raw/speech`는 아직 Drive 전송 중이므로 삭제하지 않는다. 두 검증 삭제 뒤 Jetson의
실제 여유 공간은 11,261,136,896 bytes (약 10.49 GiB)다. strict P/S raw/analysis·82
recorded 세션·RIR·manifest는 삭제 대상이 아니다.

후속 missing-list Drive 전송은 2026-08-28에 log 기준 343/343 파일,
5.570 GiB 전송 완료로 종료됐다. 이것은 전송 프로세스 완료 증거일 뿐 고정 전체 목록의
원격 file count/bytes와 `rclone check` 최종 PASS를 대신하지 않는다. 그 독립 검증 전에는
추가 로컬 원본을 삭제하지 않는다.

### 2026-08-28 Elice v10 — 실제 cache-safe pre-init 증거

Elice `~/Deep_ANC`에서 exact clean commit
`937af1175b3818b00d54f08732d63a9ecf07907a`으로 full bootstrap과 readiness를 다시
실행했다. bootstrap은 exit 0, 전체 pytest 0 FAIL로 끝났고 receipt
`data/manifests/elice_bootstrap_receipt.json`의 SHA-256은
`63a714902401114df9c86c0d3b6604b2a1a58b313e274aa0098f4d46ee4f009c`다.

- readiness artifact는
  `results/training_prerequisites/evidence/readiness_v10_937af11/readiness.json`
  (SHA-256 `27a175bc69e0c4c54f9faf24c1f692dc8a427d974e2356fcfa4773a4ad09743e`)이다.
  **16 gate 중 15 PASS / 1 FAIL**이며, 유일한 FAIL은 의도된
  `completed_init_checkpoint: init_ckpt가 비었습니다`다.
- strict P/S·lead=115, transfer/recorded QA, lineage leakage=0, 통계 검정력과 plant
  confidence ceiling은 모두 PASS했다.
- 이전 결함인 `assets/measured/.design_ceiling_cache.json`의 tracked write는 재발하지
  않았고, readiness 종료 뒤에도 원격 `git status --porcelain`은 비어 있었다.
- 이 evidence는 데이터·환경·pre-init readiness만 증명한다. canonical init, campaign
  ledger, 실제 학습·덕트 ANC 성능은 아직 증명하지 않는다.

이 15/16은 당시 gate schema의 역사적 결과다. 현행 17-gate schema에서는
`recorded_subband_coverage`뿐 아니라 보수적 numeric alias가 밝힌 speech lineage도
복구해야 한다. 따라서 현재 상한은 14/17, speech lineage 복구 후 15/17,
coverage PASS 후 16/17, canonical init 후에만 17/17이다. 역사적 receipt를 새
canonical 학습 개시 근거로 승격하지 않는다.

### campaign prerequisite schema v7

canonical 100k를 열기 전에 수기 NMSE, gradient share, pilot score/winner 또는
`passed=true`를 ledger에 적는 경로를 폐기했다. schema v7은 다음 raw artifact에서
결론을 재계산한다.

1. **각 alpha·lambda_dnh identity별** G0 final model state와 같은 fixed batch의 trusted
   NMSE `< -6 dB`
2. 각 approved G0 checkpoint/batch에서 strict S·settle·150–1600 Hz를 사용해 현재 cfg
   `lambda_dnh`의 model-output `y` gradient share를 재계산한 pre-pilot receipt. 현재 share
   0.2–0.4만 PASS이며 추천값 자체는 PASS가 아니다.
3. 각 identity의 loss pilot `best.pt`/`last.pt`/recorded-val `metrics.npz`/manifest
4. 각 pilot `best.pt`를 exact init으로 결속한 measured 5k probe의
   checkpoint·manifest·raw recorded-val metrics. 최종 winner와 0.2dB/0.85 판정은
   이 probe 점수만 사용한다.
5. 최종 winner pilot `best.pt`와 모든 candidate G0가 공유한 fixed-batch SHA 중
   **winner G0의 authoritative artifact path/SHA 자체**로 strict-S DNH output-y gradient
   share `0.2–0.4` 재검산(20k 동안 출력 분포 drift 확인). post-pilot용 새 batch나
   동일 bytes의 별도 복제 경로도 거부한다.
6. 선택된 `(alpha, lambda_frame, lambda_dnh)`와 같은 A100 exact-resume smoke receipt

issuer와 canonical 100k 명령은 모두 raw measured-probe selection으로 유도된 같은
`loss.nmse_cvar_alpha`와 `loss.lambda_dnh` float를 명시해야 한다. 한 λ를 모든 alpha에
강제하지 않으며, YAML 시작값을 조용히 쓰지 않는다. 실패 G0는 별도 diagnostic kind로
checkpoint/batch/environment를 봉인해 다음 fresh-G0 추천에만 쓸 수 있고, pilot/init 자격을
절대 열지 않는다.

Elice의 nominal A100 80GB PCIe는 PyTorch에서 driver-reserved memory를 뺀 약
79.4GiB로 보인다. smoke runner와 receipt validator는 bootstrap과 동일하게
`79GiB` usable-memory 하한, A100 device name, exact torch/CUDA, world=1,
결정론 backend를 함께 요구한다. 따라서 40GB A100/MIG slice는 계속 거부하며,
실제 device name과 byte 값은 immutable environment receipt에 남긴다.

새 source commit으로 전환하면 bootstrap receipt도 exact commit에 결속되어 바뀐다.
따라서 위 v10 receipt를 다음 campaign의 anchor로 재사용하지 않고, 같은 raw/audit을
새 exact commit에서 다시 full bootstrap한다. speech lineage를 복구해 15/17을
재현하고, coverage를 실제 추가 녹음으로 복구해 16/17이 되기 전에는
G0를 시작하지 않는다. canonical init을 완료한 뒤에만 17/17이다.

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
  public raw 6종을 untouched 상태로 다운로드·manifest 재생성·QA했다. 이때의 bootstrap
  전체 pytest 0 FAIL receipt SHA는
  `f56c3d1042211112627380f74315d5949f05bcf274bdcf3fefc588ea3d3caa7e`다. **이는
  decoder-audit 결속 이전 schema v3 자료의 receipt이므로 diagnostic-only이며,
  canonical v4 readiness나 학습 개시 근거가 아니다.**
- 기존 Elice loss grid 4개(각 20k surrogate pilot)는 폐기된 `lambda_dnh=0.00025`로
  실행되어 모두 diagnostic-only다. 새 `lambda_dnh=0.00075` pilot의 첫 후보도 약
  8.5k/20k에서 `y_rms≈2.2e−5`, trusted NMSE≈0 dB로 영출력 붕괴해 중단·보존했다.
  같은 strict P/S·lead=115 고정 batch에서 NMSE-only는 −12.03 dB, frame-off는
  500 step −8.54 dB, 반면 full loss와 `lambda_frame=0.2`는 각각 약 0 dB로 붕괴했다.
  따라서 signed frame-CVaR 0.5/0.2는 canonical 후보가 아니며, v1은
  `lambda_frame=0` metric-only로 alpha만 비교한다. DNH는 대역 밖 고주파 악화 금지
  장치이므로 끄지 않는다. one-sided/item-wise frame guard v2는 별도 evidence 뒤에만
  도입한다.
- 과거 strict-S gradient budget `0.264`는 `loss_start_sample=0` 측정값이었다.
  실제 Trainer의 strict S + 3549-sample 절단 fixture는 0.130이다. `gradient_budget`과
  campaign ledger는 이제 같은 loss_start_sample을 결속하며, 실제 A100 모델/배치의
  0.2–0.4 증거 없이는 canonical을 fail-closed한다.
- old exact checkout `5979491`에서 frame-off 5k surrogate control 두 개를 끝냈다.
  generic seed `20260802`는 trusted NMSE `−21.668 dB`, matched seed `20260803`/Tiny는
  loss `44.3368 → −23.2893`, trusted NMSE `−24.308 dB`, 3.40 step/s, wrapper exit 0이다.
  둘 다 `secondary_surrogate` 고정-batch 진단이며, 실제 덕트 감쇠·init 자격·model selection
  근거가 아니다. 다만 동일 P/S·lead=115에서 frame-off가 영출력 붕괴 없이 학습되는 것을
  재현한다. generic log의 구형 wrapper exit receipt는 literal `%s`라 별도 verified receipt로
  보존했고, matched control은 올바른 exit receipt를 남겼다.
- Elice music manifest 전체 6,356개 full-decode audit는 접근 방식에 따라 결과가 달라지는
  decoder 결함을 확인했다. 65,536-frame 순차 전수 검사에서는 train의 고유 38개(경고 34,
  실제 read error 3, peak>2 4; 범주는 중복 가능), 262,144-frame 검사에서는 21개가
  검출됐다. val 234개/test 357개는 두 검사에서 0건이지만, 단일 접근 방식만으로 나머지의
  적격성을 주장할 수 없다. `NoisePool`의 retry는 문제 파일을 조용히 다른 파일로
  대체하므로 적격 manifest 증거가 아니다. raw는 불변으로 보존하고, 복수 접근 방식의
  전수 decoder audit에 결속한 신규 manifest 재생성·QA가 끝날 때까지 canonical을 열지 않는다.
- 이 결함을 막는 local 구현은 완료됐다. audit은 65,536/262,144 full sequential과 seek grid,
  C/Python decoder warning·decode error·nonfinite·peak·RMS·raw SHA/size를 기록한다. bootstrap은
  `results/provenance/decoder_audit.json`에서 `data/manifests/canonical_v4/`을 새로 발행하며,
  schema v3는 diagnostic-only다. v4는 decoder runtime fingerprint/raw inventory drift/reject
  content duplicate를 fail-closed하고 runtime `NoisePool`도 fallback하지 않는다. Elice에서
  이 새 exact commit을 checkout한 뒤 실제 전수 audit·v4 QA를 아직 실행해야 한다.
- canonical `recorded_regrouped.jsonl` 전수 QA는 82/82 세션·95.67분·오류/경고 0으로 통과했다.
  불변 `session.json`의 원본 pool group과 재그룹화 manifest의 lineage group을 직접 비교하던
  QA 결함을 수정했으며, 회귀 테스트와 전체 pytest도 0 FAIL이다.
- QA 수정 커밋 `cef615ec40b18e26c1fe3e7fa53a09c715cb7a67`, strict 자산 승격 커밋
  `4c55386`, Elice 이관 상태 문서 커밋 `86c5c45`, 데이터/손실 안정화 커밋
  `2d19f14`, 예산 근거·진행 문서 커밋 `bd2a0cf`/`43563be`가 모두 원격 브랜치에
  push 완료됐다. 현재 준비 기준선은 clean 상태로 push된
  `fix/finetune-readiness-repair`의 HEAD이며, 실행에 사용할 exact SHA는
  `git rev-parse HEAD`로 확인한다. 브랜치별 범위는 `docs/08_dev_workflow.md §7`에 고정했다.
- Elice receipt가 생긴 뒤 `check_finetune.py`의 외부 입력 차단은 해소됐지만,
  speech lineage, strict 부대역 coverage, canonical init checkpoint·campaign ledger가
  아직 없어 readiness는 의도적으로 17/17이 아니다.
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
- 로컬 exact commit `0a492e99d46cd0509dee9718b3f219cbb4406380`에서 전체 pytest는
  **0 FAIL**(경고는 로컬에 없는 downstream public manifest를 진단 fixture가 알리는 것),
  `bash -n`과 `git diff --check`도 통과했다. 현재 실행 중인 Elice pilot은 실행 계약을
  고정하기 위해 소스 변경이 없던 부모 commit `1aaece892ee1ab7bc5f6a224fb1b4f29171019c0`의
  detached checkout에 남겨 두었다. pilot 종료 뒤 ledger/canonical에 사용할 exact SHA를
  한 번 더 명시적으로 고정한다.

- strict-S gradient budget의 정착 절단/ledger 결속 보강 커밋
  `a37ebf53a0e2173745fff5937f6aa6768bbb2179`을 원격 브랜치에 push했고, 로컬 전체 pytest도
  `/dev/shm` basetemp에서 0 FAIL로 재확인했다. 이 commit은 frame=0 metric-only 후보의
  policy 변경 전 기준이며, 진행 중인 Elice diagnostic은 기존 `5979491` detached checkout에
  보존한다.
- 2026-08-28 Elice의 실제 raw decoder audit은 후보 37,761개 중 36,868개를 적격으로,
  893개를 부적격으로 판정했다. 적격 수는 music 7,941, DNS noise 15,553, speech 7,971,
  DEMAND 96, MIMII 3,600, ESC-50 1,707이다. 원본을 삭제하지 않고 audit SHA
  `ceac538487ffe1414d433e3a83fdee11a0d17c204427cf8e7fed92bb73c2940f`와 accepted inventory
  SHA `b665bbb7dd28fc46cdada1d9da9a0535d74ca5ae73a030fbf5612cfcc6e61955`로 보존한다.
  현재 raw 재대조와 `canonical_v4` manifest transaction이 실행 중이므로, bootstrap의
  exit 0·QA·pytest·readiness receipt가 생기기 전에는 이 수치를 readiness PASS로 승격하지
  않는다.
- 현재 실행 중인 full audit은 중단하지 않는다. 완료 뒤 새 exact source commit에서 같은
  raw·decoder 환경을 다시 사용할 때만 `bootstrap_all.sh --reuse-decoder-audit`를 명시할 수
  있다. 이 경로는 외부 전달한 report **file SHA**와 report 내부 `audit_sha256`, canonical
  full-scan recipe, 현재 decoder fingerprint, accepted/rejected를 포함한 raw 전체
  path/SHA/size를 먼저 대조하고, `prepare_noise_pool.py` transaction이 그 전수 대조를
  다시 수행한다. 어느 하나라도 불일치하면 fresh audit으로 자동 fallback하지 않고
  실패한다. 기본값은 새 full decode audit이며, local 전체 pytest 0 FAIL과 shell 문법
  검증을 통과했다.
- 현장 evaluator는 새 source-energy 계약으로 보강했다. OFF/ON은 source를 끄는 것이 아니라
  ANC control만 OFF→ON→OFF로 바꾸며, raw NPZ에는 요청 상태를 저장한다. 각 octave는 OFF
  source의 폭 정규화 PSD 비율이 충분할 때만 감쇠를 발행하고, 그렇지 않으면 `NaN/무효`와
  진단용 원계산값을 분리 저장한다. 따라서 이 필드가 없는 legacy live result는 고주파 또는
  canonical 현장 성능 근거가 아니다.
- canonical ledger가 요구하는 A100 exact-resume smoke와 canonical 계약 사이의 순환을
  `a100_pretrain_smoke` 역할로 분리했다. 이 역할은 init-eligible=false, 200–500 step,
  A100 80GiB·world=1·CUDA/bf16·결정론 조건 및 prerequisite-root 격리를 강제한다. immutable
  `stop.pt`의 실제 resume SHA, 두 arm의 환경·telemetry·model/optimizer/scheduler/RNG/
  best-metric/data-stream 동등성을 검증하고, 첫 evaluation 전 stop checkpoint의 `+inf`
  sentinel만 제한적으로 허용한다. 새 전체 pytest는 `/dev/shm`에서 exit 0이며 기존
  diagnostic manifest-missing RuntimeWarning 2건만 있다.
- smoke는 pilot 선택 **뒤** 선택된 loss와 같은 semantic target으로 실행한다. runner의
  `--loss-alpha {0.7,0.85,1.0}`와 `--loss-lambda-dnh`는 YAML 복제가 아니라 resolved
  config에 float literal을 넣어 target/contract에 결속한다(특히 `1.0`을 정수 `1`로
  바꾸지 않는다). 따라서 기본 alpha/λ의 smoke receipt를 다른 winner identity에 재사용할
  수 없고, 선택된 둘 중 하나라도 달라지면 새 smoke를 실행한다.
- Elice의 decoder audit 뒤 raw SHA 재대조는 생략하지 않는다. 다만 다음 exact bootstrap에는
  `--raw-hash-workers 8`을 명시해 16 vCore A100 노드에서 independent same-FD SHA/size
  검증을 병렬화할 수 있다. 기본값은 1이고 1~32만 허용한다. executor는 입력 순서대로
  결과·예외를 회수하므로 manifest bytes·identity와 첫 실패 경로는 바뀌지 않으며,
  transaction 후 audit inventory와 committed manifest raw 전체를 다시 검증한다.
- public corpus lineage의 DSU는 iterative path compression과 union-by-size를 사용한다.
  이전 lexical-root 재귀 구현은 adversarial reverse merge 2,048개에서 `RecursionError`를
  재현할 수 있었지만, component의 members·identity digest·분할 의미는 root 이름이 아니라
  canonicalized data로 결정되므로 이 변경은 계보 의미를 바꾸지 않고 대형 corpus chain을
  안전하게 닫는다.
- 2026-08-28 exact SHA `70483599c3d2d8f98fe8b36cfbe3a068fa1aa43a`의 Elice v7
  audit-reuse bootstrap은 DNS official marker 대조에서 의도적으로 exit 1로 멈췄다
  (status SHA `37e2617b61f07482be7dd74bfb831861ee2491bbe8adcfd3ea173824f4bc0f85`,
  log SHA `93371453b3e30cafa8c48bb0f0a5711a6345ded3be127ec47a83c7fc23a39d69`).
  원인은 raw 손상이 아니라 구 검증이 `scan_wavs`의 accept 집합만 official archive marker와
  같아야 한다고 가정한 것이다. 실제 decoder audit은 DNS fullband accept/reject
  `15,553/447`, DNS speech `7,971/94`를 기록했고, reject member는 의도적으로 scan에서
  빠지므로 marker의 missing으로 오인됐다. 원격 raw에서 marker의 실제 canonical root가
  `data/raw/noise/dns_fullband`, `data/raw/noise/speech`임을 read-only로 재확인했다.
- 다음 commit 후보는 marker를 **audit accept ∪ audit reject의 exact partition**으로
  검증한다. `speech` tag에 함께 존재할 수 있는 `data/raw/speech/LibriSpeech`는 일반
  lineage/DSU에 계속 포함하되 DNS marker partition에서는 제외한다. transaction
  postcommit도 copied decoder audit에서 accept/reject projection과 marker evidence를
  다시 만들고 exact equality를 요구하므로, sidecar만 바꿔서 raw 누락·reject 누수를
  통과시키지 못한다.
- 같은 작업에서 decoder audit의 raw snapshot은 정상 read가 갱신할 수 있는 `atime`까지
  전체 stat로 비교해 false reject를 만들 수 있음을 재현했다. 동일 raw identity는
  device/inode/size/mtime/ctime과 regular-file mode로만 비교하도록 고쳤다. byte 변경
  탐지는 유지하며, 새 regression과 임시 audit 100회 반복이 통과했다. 이 marker/atime
  변경 후 로컬 전체 pytest도 0 FAIL(로컬 public manifest 부재를 알리는 diagnostic
  warning만 2건), shell 문법·`py_compile`·`git diff --check`도 통과했다. commit/push
  뒤에만 Elice v8 preflight와 full bootstrap을 다시 시작한다.
- Elice v8은 marker 검증을 통과해 canonical_v4를 atomic publish했다
  (build ID `893136241cded3b835daeceafedd43863b8c8a7ecfdcde7e7f29fc5495336707`,
  raw mtime 변경 0, staging 잔재 0). 그러나 [5/6] pytest에서 기존 runtime test 하나가
  실패했다. Elice의 `runs/`에는 학습 log/config snapshot만 있고 Jetson legacy runtime의
  active artifact `runs/pretrain_base_corrected/ckpt/best.pt`,
  `runs/export/tiny_corrected.onnx`는 의도적으로 없는데, test가 `runs/` directory 존재만으로
  artifact가 fetched됐다고 오인했다. v8 log SHA는
  `5b8ad3ea059af82c677946ef6180b0ac02156267b3242d55aca894310e33981c`, status SHA는
  `45d873985c93d13033163f25b882af2974c5242022445f71f27f7428e5b4c760`이다. 이는 data/P/S
  failure가 아니며 bootstrap receipt가 없으므로 published manifest도 아직 학습에 쓰지 않는다.
  다음 commit은 runtime artifact cohort(ONNX/plan 또는 legacy checkpoint)가 실제로 있는지로
  판정하게 고친다. cohort 안에 한 파일이라도 있으면 기존처럼 config path 누락을
  fail-closed하고, A100의 일반 `runs` log만 있는 상태는 unfetched로 skip한다. 해당 Elice
  상태를 재현한 회귀와 로컬 전체 pytest 0 FAIL을 확인한 뒤에만 같은 immutable raw와
  canonical_v4 transaction을 다시 검증한다.
- 2026-08-28 구 `5632c08` Elice bootstrap은 stage 4에서 약 3시간 54분 동안 user CPU만
  증가하고 raw I/O·staging·canonical_v4·로그 진행이 없었다. raw 변경 0, tracked 변경 0,
  staging 0을 재확인한 뒤 worker 하나에만 TERM을 보내 `exit_code=143` receipt를 남겼고,
  원 log/status는 `results/training_prerequisites/evidence/diagnostic/`에 같은 SHA로 보존했다.
  새 exact checkout의 preflight는 기존 기본 경로 `~/Deep-ANC`와 실제 `~/Deep_ANC`의 불일치를
  fail-closed로 잡았다. bootstrap은 이제 호출 위치/clone 이름 대신 script 위치에서 default
  repo root를 유도한다. 이 수정의 pytest·shell 검증과 새 full preflight를 통과한 SHA만 다음
  raw-audit 재사용 실행에 쓴다.
- 새 경로 수정 `ae494ad`의 Elice preflight는 PASS했고, raw audit reuse도 internal/file SHA와
  accept 36,868/reject 893을 PASS했다. 그러나 prepare의 manifest-entry 검증이 cache 없는
  absolute path index를 매 entry마다 다시 만들어 약 `36,868 × 37,761` 경로 계약 검사를 수행하는
  O(N²) 결함을 발견했다. raw hash 뒤 staging/output 0·raw 변경 0 상태에서 worker만 TERM해
  `exit_code=143` receipt를 보존했다. path index는 이제 `(repo root, ordered raw roots)`
  context별 process-local 파생 cache로 한 번만 만들며, entry별 SHA/size·accept/reject 검증과
  transaction 후 전수 raw 재검증은 그대로 유지한다. prepare stdout은 artifact 계약에 영향을
  주지 않는 flush phase log도 남긴다.

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

## 2. 역사적 복구 절차 — 현재 실행 금지

> **2026-08-28 현재 이 절 전체는 일반적인 다음 단계가 아니다.** strict P/S와 그 raw/
> analysis/level evidence는 이미 합격·보존됐고 상단 운영 상태가 권위다. 아래 명령은 그
> artifact가 SHA 불일치·물리 배선 변경 등으로 **명시적으로 invalidated된 경우에만** 새
> 계획·무음 dry-run·사용자 승인 뒤 복구할 때 참고한다. 현재 세션에서 strict P/S를 다시
> 출력하거나 기존 transfer를 재발행하지 않는다.

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

### 2.2 strict P/S 라이브 재측정 — invalidation 시에만

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

## 3. 역사적 strict 복구 뒤 계보·Elice 이관 — 현재 generation 절차 아님

> 아래는 strict capture를 처음 만들던 당시의 82-only 이관 예시다. 현행 82+17
> generation/transfer schema는 `docs/17_recorded_generation.md`와 상단 상태를 따른다.
> 이 명령으로 현재 transfer manifest를 덮어쓰거나 되돌리지 않는다.

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
전체 pytest와 readiness까지 통과해도 speech lineage 복구 전에는 14/17이다.
lineage 복구 후 15/17, 추가 녹음으로 coverage PASS 후 init 하나만 FAIL인
16/17, canonical init 후 17/17 순서를 건너뛰지 않는다.

## 4. 공식 학습 순서

1. family→lineage component→session 균등 sampler와 공통 gain/polarity/EQ, input-only mic
   noise를 사용한다. session mixing과 lead jitter는 0이다.
2. 각 alpha별 현재 `lambda_dnh`로 고정 batch G0를 처음부터 실행해 trusted NMSE
   < −6 dB와 lead metadata를 확인한다. 합격 G0의 같은 checkpoint/batch에서 strict S,
   실제 settle 절단, 150–1600 Hz의 model-output `y` gradient share를 재계산해 현재 λ가
   0.2–0.4일 때만 다음 단계로 간다. 범위 밖 추천 λ는 새 contract/fresh G0용 정보일
   뿐이고 실패 checkpoint를 전이하지 않는다.
3. seed `20260803`, frame-metric-only(`lambda_frame=0`)의 `alpha∈{0.7,1.0}`을 각자의
   approved `lambda_dnh`와 함께 20k
   surrogate + 5k measured probe로 실행하고 probe recorded val만 선택에 사용한다.
   두 probe 점수가 0.2 dB 이내
   동률이면 alpha 0.85를 추가하고, 계속 동률이면 alpha 0.7을 택한다. alpha 1.0의
   non-finite/실행 실패는 immutable pre-forward witness를 재실행하는 failure receipt
   구현 전에는 fallback 근거로 쓰지 않고 canonical을 fail-closed한다.
   170ms frame metric은 candidate마다 기록해 비교·원인 분석에 사용한다. 고정 local
   pass threshold가 생기기 전에는 이 metric으로 성능 PASS를 주장하지 않는다. pilot checkpoint는
   init 자격이 없다.
4. 각 pilot `best.pt`를 init으로 한 measured 5k probe의 completion/provenance와
   raw recorded-val metrics를 확인한다. probe 점수로 선택된 최종 winner의
   pilot `best.pt`와 fixed batch에서 strict S의 `lambda_dnh` output-y gradient 비중
   0.2–0.4를 재계산한다.
5. 선택 계약의 A100 200–500 step exact-resume smoke에서 VRAM, 처리량/ETA, 중단·재개
   수치등가를 먼저 확인한다. schema v7 issuer가 후보별 G0/pre-pilot gradient/pilot/probe와
   selected-20k gradient·smoke raw artifact를
   다시 검증해 canonical ledger를 no-replace 발행한 뒤에만 다음 단계로 간다.
6. 선택 계약의 tiny를 새 run에서 100k 처음부터 사전학습한다.
7. canonical init 지정 뒤 readiness 17/17을 확인하고 open-loop, recorded 70% + synthetic 30%,
   bf16 forward + FP32 loss, 50k fine-tune을 실행한다.
8. checkpoint 선택은 recorded val만 사용한다. 선택을 고정한 뒤 test를 정확히 한 번 연다.
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

- Elice에서 새 exact commit으로 decoder 전수 audit → canonical_v4 manifest → QA/pytest/readiness를
  다시 실행하고, audit binding이 실제 raw/decoder 환경과 일치하는지 확인
- frame-metric-only alpha 2개 각각의 G0+strict-S output-y gradient share를 먼저
  0.2–0.4로 승인한 뒤, 그 `(alpha,frame,lambda_dnh)` identity의 20k pilot+5k measured
  probe를 recorded val만으로 실행·선택(필요 시 alpha 0.85 chain 추가)
- 실제 A100 bf16 중단→resume 수치등가 smoke, 후보별 G0/pre-pilot gradient와
  최종-winner post-pilot gradient
  ledger 작성 및 SHA 결속
- canonical tiny 100k surrogate-pretrain init checkpoint 생성 후 readiness 17/17 확인
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
- Jetson 용량 부족을 이유로 public corpus를 중복 복제하지 않도록 Kaggle/Google Drive를
  사용할 때의 임시 staging·비밀정보·SHA·manifest 재검증 절차를 `docs/14_elice_external_data_staging.md`
  에 기록했다. 현재 Elice raw가 이미 정상적으로 있으므로 pilot 중에는 downloader를
  실행하지 않는다.
- 이 기록을 작성한 당시 branch HEAD는 외부 감사 기록과 후속 정리 변경을 포함한 최신
  commit이었다. 당시 Elice에서 진행 중이던 pilot은 중단하지 않고 종료 뒤 해당 exact
  commit으로 동기화하는 방침이었다. 현재 권위 HEAD와 작업은 이 문서 상단을 따른다.
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

- Jetson `data/` 백업은 Google Drive `gdrive:DeepANC/jetson_data_backup_20260827/data`로
  진행 중이다. 13,428개/17,441,317,063바이트 고정 목록의 SHA는
  `1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2`이며 Drive에 올린
  manifest SHA와 일치했다. 겹치는 전체 복사 프로세스는 중단하고 `raw`, `recorded`,
  `recorded_broken`, `source_pool_v2` 전용 `rclone copy`만 유지한다. 원격 파일 수·바이트와
  `rclone check`가 고정 목록과 일치하기 전에는 원본을 삭제하지 않는다. 백업 절차와 보존/
  삭제 범위는 `docs/15_jetson_drive_backup_20260827.md`에 기록했다.
