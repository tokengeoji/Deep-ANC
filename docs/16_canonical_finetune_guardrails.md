# Canonical 파인튜닝 강제 가드레일

> 상태 기준일: 2026-08-30. 이 문서는 통과를 선언하는 새로운 증거가 아니라, 실제 코드
> authority와 artifact를 한곳에 연결하는 규범적 인덱스다. 문서와 코드가 충돌하면 raw
> artifact와 아래에 명시한 코드 authority가 우선한다.

## 1. 상태의 뜻

| 상태 | 뜻 | 다음 단계 허용 |
|---|---|---|
| **PASS** | 현재 세대의 immutable evidence를 authority가 다시 계산해 모든 조건을 만족했다. | 그 행이 보호하는 다음 단계만 허용 |
| **FAIL** | 유효한 evidence에서 기준 미달 또는 악화가 실제로 확인됐다. | 원인 수정·새 데이터 수집 뒤 새 세대 evidence 필요 |
| **BLOCKED** | 선행 artifact가 없거나, stale/legacy/diagnostic이거나, 통계적으로 결론을 낼 수 없다. | **PASS로 간주 금지**. 학습·test·배포를 열지 않음 |

`INCONCLUSIVE`는 G4의 분석 결과이며 운영 상태로는 **BLOCKED**다. 숫자가 없거나 raw를
검증할 수 없는 경우도 BLOCKED다. FAIL과 BLOCKED를 임계값 완화로 PASS로 바꾸지 않는다.

### 1.1 병목 완화와 pilot-only 경계

2026-08-30 사용자 지시에 따라 기술적으로 본질적이지 않은 운영 병목은 완화할 수 있다.
다만 완화는 **실험을 더 일찍 실행하는 권한**이지, 미달 evidence를 canonical PASS로 바꾸는
권한이 아니다.

| 분류 | 완화 가능 여부 | 적용 |
|---|---|---|
| 예상 감쇠량·초기 수렴 속도·ETA·GPU 사용률 | 가능 | 양의 감쇠와 무증폭을 향하는지 진단하고, 짧은 `a100_pretrain_smoke`를 먼저 실행할 수 있다. |
| coverage·통계력이 아직 부족한 상태의 처리량/finite/resume 확인 | diagnostic-only로 가능 | 200--500 step 상한, `init_eligible=false`, loss winner 선택·weight 전이·test 금지 |
| 2/4/8 kHz 최종 hardware authority가 없는 상태의 Stage-1 개발 | 가능 | 150--1600 Hz Stage-1만 진행하되 광대역 성공·배포 주장은 계속 차단 |
| latency/deadline/xrun/clip, lead·극성·인과성 | 불가 | 한 번이라도 위반하면 해당 runtime/측정은 무효 |
| P/S·coherence·동기 witness·raw SHA, train/val/test lineage 누수 | 불가 | 임계값을 결과에 맞춰 낮추거나 stale artifact를 승격하지 않음 |
| trusted 대역 악화 또는 대역 밖 고주파 증폭 | 불가 | 성능이 작더라도 감소 방향은 요구하며, 증폭을 성능 절충으로 허용하지 않음 |

따라서 녹음 한 행이 coherence 기준에 미달하면 그 행을 억지로 학습 자료로 넣지 않고 더
안정적인 독립 source로 교체한다. 반대로 full-octave 장비 blocker는 Stage-1 smoke 자체를
막지 않으며, 최종 광대역 G4와 배포만 계속 막는다. 이 분리는 GPU를 일찍 검증하면서도
실패 데이터·잘못된 timing이 canonical checkpoint에 섞이지 않게 한다.

### 1.2 실제 출력 전 source admission

15초 recorded addition은 스피커를 울리기 전에 exact rendered `float32` 파형으로 다음
필요조건을 모두 통과해야 한다. 이 검사는 오디오 장치를 열지 않는다.

1. 재생 진폭은 exact `0.06`, 길이는 48 kHz `720,000` frame이다.
2. timeline estimator와 같은 `12,000 + 2×600 = 13,200` frame source span과
   `3,000` frame hop, RMS 하한 `2e-4`를 사용한다. 유효 비율은 실제 capture gate
   `0.90`보다 5 %p 준비 여유가 있는 `0.95` 이상이어야 한다.
3. 공식 meter와 같은 150–1600 Hz RMS 정의를 사용한다. rendered source가 공식 meter
   playback보다 2 dB 넘게 약하거나, quiet ERR ceiling `-64.0 dBFS`에서 coherence²
   `0.90`에 필요한 예측 SNR을 못 채우면 거부한다.
4. DNS selector는 strict-P 상대 대역 density만 보지 않는다. raw PCM16 → repeat/trim
   composite → peak-normalized `0.06` → 0.1초 fade의 실제 파형에서 위 전체 preflight를
   통과한 후보만 receipt에 넣고, selected bytes에서 다시 계산한다.
5. source-plan builder와 `record_session_batch --dry-run`은 같은 공용 validator를 다시
   실행한다. selector receipt가 있어도 이 단계에서 한 행이라도 미달하면 child process와
   실제 stream을 시작하지 않는다.

2026-08-30 첫 추가 수집 실패를 오프라인 재검산했을 때 기존 19행 중 environment 1행과
DNS speech 5행은 source-RMS 필요조건만으로 가능한 비율이 각각 `0.847`,
`0.890/0.814/0.822/0.847/0.682`라서, 물리 경로가 완벽해도 capture gate를 통과할 수
없었다. 이 실패를 이유로 `0.90`을 낮추지 않고 selector와 plan 입구를 수정한다.

## 2. 절대 목표와 주파수 범위

1. Stage-1 canonical 제어 대역은 **150–1600 Hz**다. 저역 150–600 Hz와 고역 600–1600 Hz를
   동시에 통과해야 한다. 공식 G4는 150–300, 300–600, 600–1000, 1000–1600 Hz 네
   부대역을 각 family별로 따로 판정하므로 저역 이득으로 고역 증폭을 숨길 수 없다.
2. 2/4/8 kHz는 현재 strict P/S가 제어 성능을 보증한 대역이 아니다. 이 구간은 먼저 최악
   10% 증폭이 1 dB 미만인 do-no-harm을 통과해야 하며, 별도 광대역·공간 실측 전에는
   “고주파 ANC 성공”으로 승격하지 않는다.
3. source family는 **speech, music, environment, machine 네 계열 모두**다. environment는
   canonical source mix에서 DEMAND와 ESC-50 pool로 구현된다. 한 계열이라도 0 비중,
   coverage 부족, 평균/최악 10%/CI 실패이면 quiet-zone 목표는 미달이다.
4. 처음 듣는 소리는 recorded test와 별도다. 모델·계약·선택을 고정한 뒤 새로 얻은 실제
   덕트 Level-5 speech/music/environment/machine evidence가 있어야 unseen 일반화를
   주장한다. 이 challenge는 학습이나 모델 선택에 재사용하지 않는다.

위 Stage-1은 사용자가 확정한 최종 광대역 목표를 대체하지 않는다. 최종 목표는 2/4/8 kHz
octave를 포함하며, 8 kHz octave 상단 11.314 kHz까지의 유효 P/S·데이터·실측 감쇠와
matched FxLMS 우위를 요구한다. 별도 계약과 현재 blocker는
[`docs/18_broadband_anc_guardrails.md`](18_broadband_anc_guardrails.md)가 권위다. 광대역 v2가
PASS하기 전에는 이 문서의 17/17도 최종 광대역 배포 자격이 아니다.

주파수·family의 코드 단일 출처는
[`src/deep_anc/dsp/invariants.py`](../src/deep_anc/dsp/invariants.py), strict 분할과 통계 하한은
[`src/deep_anc/eval/trusted_subbands.py`](../src/deep_anc/eval/trusted_subbands.py), 대역 밖
옥타브와 1 dB 한계는
[`src/deep_anc/dsp/do_no_harm.py`](../src/deep_anc/dsp/do_no_harm.py)다.

## 3. 현재 상태 지도

아래 표의 상태는 “코드가 존재한다”가 아니라 현재 artifact까지 포함한 판정이다.

| 영역 | 현재 | PASS 조건 | FAIL 조건 | BLOCKED 조건 / 현재 근거 |
|---|---|---|---|---|
| 목표 축소 방지 | **PASS** | 150–1600 Hz와 네 family의 실제 source-mix 비중 유지 | 대역 축소 또는 한 family 비중 0 | `absolute_objective_scope`; 전역 family 불변식을 네 계열로 고정 |
| 최종 광대역 v2 목표 | **BLOCKED** | 150–11.314 kHz P/S·target-d coverage·matched FxLMS·다점 공간 계약 PASS | 한 저/고역 또는 한 family 악화 | 현 strict P/S는 150–1600 Hz뿐이고 82세션은 2.828 kHz 이상 joint coverage group 0. [광대역 가드레일](18_broadband_anc_guardrails.md) |
| strict P/S | **PASS** | same capture, 48 kHz/256/low, xrun·clip 0, 반복 ≥8, 모든 부대역 consistency ≥0.9406, raw/analysis/level SHA 정합 | 지연·채널·캡처·SHA·부대역 중 하나라도 불일치 | [P](../assets/measured/primary_path_il_strict_5dc06fdd.npz), [S](../assets/measured/secondary_path_il_strict_5dc06fdd.npz), [level](../assets/measured/measurement_level_evidence.json) |
| recorded QA·local lineage | **PASS** | 82세션 QA, aligned source, 네 family, split component 교집합 0 | 오디오/정렬/lineage 누수 또는 family/group 부족 | [manifest](../data/manifests/recorded_regrouped.jsonl), [QA](../data/manifests/recorded_qa.json), [holdout](../data/manifests/recorded_holdout.json) |
| addition source preflight | **BLOCKED** | 19/19 exact rendered source가 timeline ratio ≥0.95, 공식 trusted level·예측 SNR PASS이고 selector/plan/dry-run 증거가 같은 bytes에 결속 | 한 source라도 무음 지속성·절대 레벨·SHA 불일치 | 공용 코드와 local source-pool 9행은 PASS. 새 DNS v3 receipt와 새 exact 19행 plan이 아직 없어 전체 세대는 BLOCKED |
| strict 부대역 data coverage | **FAIL** | train/val/test의 family×네 부대역마다 density ≥0.25인 독립 group ≥4 | 유효 감사에서 한 행이라도 미달 | 64-segment 현장 감사에서 train 2, val 5, test 5행 부족. [schema-v2 진단 원본](../results/data_audit/recorded_subband_coverage_fullscan_20260828.json)은 결론 보존용이며 schema-v3 canonical receipt로 승격 금지 |
| Elice bootstrap·public speech lineage | **BLOCKED** | exact clean commit, public manifest 6종, transfer/coverage report SHA 결속과 recorded holdout↔DNS speech numeric alias 교집합 0 | bytes/path/SHA/commit 불일치 또는 cross-public speech lineage 교집합 | 새 schema-v3 coverage/receipt가 없고, 원격 canonical_v4에서 보수적 numeric alias 기준 `dns_book` 340/5946/8201 및 `dns_reader` 422/652 교집합이 발견됨. namespace가 다르다는 이유만으로 PASS 금지 |
| 역할·budget 정책 | **PASS** | 아래 역할표의 role/step/init 자격과 contract SHA exact | role 세탁, step 축소, init 자격 위조 | 코드 정책은 강제됨. 실행 artifact의 완료 여부는 별도 행에서 판단 |
| canonical 100k init | **BLOCKED** | selected contract로 처음부터 100k 완료, completion receipt, `init_eligible=true`, G0 통과 | legacy/pilot/probe 또는 미완주 checkpoint | 해당 canonical checkpoint 없음 |
| canonical 50k fine-tune | **BLOCKED** | readiness 17/17 뒤 canonical init에서 정확히 50k 완료 | readiness 미통과, 다른 init/plant/lead, 자동 resume | coverage와 init이 없어 시작 금지 |
| recorded val/test G4 | **BLOCKED** | val-only 선택 후 single-use test가 G4 PASS | 측정된 증폭 또는 NMSE 기준 실패는 FAIL | canonical 모델/val selection/test raw가 없음. 표본·CI 부족은 INCONCLUSIVE→BLOCKED |
| Level-5 unseen | **BLOCKED** | 고정 모델로 새 네 family 실제 덕트 challenge가 같은 G4 PASS | 한 family/부대역이라도 악화 | 모델 선택 전에는 실행·수집 결과를 최종 unseen 증거로 사용할 수 없음 |
| Jetson latency·안정성 | **BLOCKED** | canonical engine P99 <3.0 ms, max <256/48k=5.333 ms, 공식 session의 xrun·deadline miss·engine error·drop/add·fallback·watchdog·sample slip=0, absolute backlog ≤256·excess backlog=0 | deadline 초과/미스/xrun/불연속 counter, excess backlog 또는 plant lead 불일치 | 고정 1-hop handoff는 plant lead에 모델링하며 callback race로 보이는 정상 absolute backlog 0/256을 지연 불연속으로 오인하지 않는다. legacy benchmark는 canonical 모델의 증거가 아님 |
| ONNX/TensorRT·실제 배포 | **BLOCKED** | canonical G4+Level-5+runtime PASS와 checkpoint/ONNX/engine metadata SHA·lead exact | 어떤 선행 gate라도 FAIL | 기존 export는 legacy/diagnostic이며 promotion 금지 |

canonical readiness의 총수는 17개다. 다만 기존 **15/17** 기록은 새로 발견된
recorded-holdout↔DNS numeric speech overlap을 `corpus_disjoint`가 놓친 false-negative를
포함하므로 더 이상 authoritative하지 않다. 보수적 alias 계약으로 재감사하기 전에는 점수를
확정하지 않으며, 알려진 init·coverage·cross-public lineage 세 blocker를 적용하면 최대
**14/17**이다. overlap을 제거한 manifest와 coverage를 복구하면 init만 남아 **16/17**,
canonical init 완료 뒤에만 **17/17**이다.

## 4. gate ID와 실행 authority

이 표의 gate ID가 문서에서 빠지거나 authority 경로가 바뀌면 문서 회귀 테스트가 실패한다.
`src/deep_anc/ops/gate_registry.py`는 목록 인덱스이고, 실제 판정은 아래 owner가 수행한다.

| 필수 gate ID | 실제 authority |
|---|---|
| `absolute_objective_scope` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `recorded_transfer_snapshot` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `official_secondary_path`, `official_primary_path`, `matched_path_measurement_conditions`, `path_delay_and_lead` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `recorded_dataset_qa`, `recorded_alignment_integrity`, `recorded_statistical_power`, `recorded_subband_coverage` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `corpus_disjoint`, `measured_source_delay_agreement`, `plant_confidence_ceiling`, `completed_init_checkpoint` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `g4_strict_trusted_subbands`, `g4_out_of_band_do_no_harm`, `g4_statistical_power`, `g4_cluster_bootstrap_ci` | `src/deep_anc/eval/recorded.py` ([코드](../src/deep_anc/eval/recorded.py)) |
| `recorded_val_g4`, `recorded_test_g4` | `src/deep_anc/train/finetune_readiness.py` ([코드](../src/deep_anc/train/finetune_readiness.py)) |
| `recorded_selection_test_once_chain` | `src/deep_anc/train/evaluation_contract.py` ([코드](../src/deep_anc/train/evaluation_contract.py)) |
| `runtime_strict_plant_contract` | `src/deep_anc/realtime/plant_contract.py` ([코드](../src/deep_anc/realtime/plant_contract.py)) |
| `runtime_engine_artifact_preflight` | `src/deep_anc/realtime/run_realtime.py` ([코드](../src/deep_anc/realtime/run_realtime.py)) |
| `runtime_deadline_miss_rate`, `runtime_handoff_backlog`, `runtime_pipeline_handoff_budget` | `src/deep_anc/realtime/safety.py` ([코드](../src/deep_anc/realtime/safety.py)) |

추가 계약 authority:

- 역할·optimizer·budget: `src/deep_anc/config.py`
  ([코드](../src/deep_anc/config.py))
- P/S와 runtime handoff pointer: `configs/duct.yaml`
  ([설정](../configs/duct.yaml))
- canonical pretrain/fine-tune: `configs/train_pretrain_tiny.yaml`, `configs/train_finetune.yaml`
  ([pretrain](../configs/train_pretrain_tiny.yaml), [fine-tune](../configs/train_finetune.yaml))
- recorded sampling: `src/deep_anc/eval/recorded_sampling.py`
  ([코드](../src/deep_anc/eval/recorded_sampling.py))
- persisted G4·single-use capability: `src/deep_anc/train/evaluation_contract.py`
  ([코드](../src/deep_anc/train/evaluation_contract.py))
- inference P99 도구: `scripts/bench/measure_inference_latency.py`
  ([코드](../scripts/bench/measure_inference_latency.py))

## 5. 학습 역할과 budget

| 역할 | 정확한 budget | init 자격 | 다음 단계에서 허용되는 용도 |
|---|---:|---|---|
| `a100_pretrain_smoke` | 200–500 step | false | VRAM/처리량/ETA/exact-resume만 확인 |
| `loss_pilot` | 20,000 step | false | alpha 후보 비교. init·배포 금지 |
| `measured_probe` | 5,000 step | false | 같은 pilot에서 measured 70%+synthetic 30% val 비교. test 금지 |
| `canonical_pretrain` | 100,000 step | true | completion receipt와 G0를 통과한 weight-only fine-tune init |
| `canonical_finetune` | 50,000 step | false | recorded val 선택 후보. 그 자체로 배포 자격 없음 |

loss 후보의 권위 identity는 `(nmse_cvar_alpha, lambda_frame, lambda_dnh)` 세 값이다.
`lambda_dnh`는 YAML 시작값을 그대로 승인하지 않고, alpha별 approved G0 checkpoint와 같은
fixed batch에서 strict S·settle 절단·150–1600 Hz를 적용한 model-output `y` gradient share를
실제 재계산한다. 현재 cfg의
`‖lambda_dnh·∂L_dnh/∂y‖ / ‖∂L_nmse/∂y‖`가 0.2–0.4일 때만 그 후보의
20k를 열며, 이미 범위 안이면 0.3에 맞추기 위해 λ를 바꾸지 않는다. 이는 parameter-gradient
증거가 아니다.

범위 밖일 때의 선형 추천 λ는 실제 ANCLoss를 새 값으로 다시 만들어 share를 재계산한
diagnostic 정보다. 추천 receipt 자체는 PASS가 아니며, 새 identity/contract로 G0부터
처음 실행해 NMSE `< -6 dB`와 현재-share 0.2–0.4를 모두 다시 통과해야 한다. 실패 G0
checkpoint는 별도 diagnostic kind로만 봉인되고 weight 전이·pilot·init에 사용할 수 없다.
각 alpha는 서로 다른 approved λ를 가질 수 있고, 후보 identity는
G0→pre-pilot gradient→20k→5k에, 최종 winner identity는 selected-20k drift check→smoke→
100k→50k와 `loss_selection_sha256`에 그대로 결속된다. selected-20k drift check의 batch도
모든 candidate G0의 fixed-batch SHA가 같아야 하고, 그중 winner G0의 concrete artifact
path와 SHA를 authority로 삼는다. selected-20k receipt가 동일 bytes의 다른 경로 복제본이나
새 batch를 가리키면 거부한다.

자동 resume은 금지한다. 명시적 resume도 전체 experiment contract SHA와 stochastic state가
같을 때만 허용한다. pilot/probe/smoke의 파일명을 바꾸거나 metadata를 다시 써서 canonical
init으로 만드는 것은 금지한다.

## 6. one-shot G4

공식 순서는 다음과 같고 어느 단계도 건너뛸 수 없다.

1. recorded val raw metrics를 canonical sampling(max 64, segment 1.5초, edge 0.25초)으로 생성
2. raw segment/family/group/octave/strict-subband 배열을 재검산해 val-only model selection 고정
3. selection SHA에 결속된 test capability를 한 번 발급
4. capability를 한 번 소비하고 independent test raw metrics 생성
5. checkpoint·manifest·selection·capability·consumed marker·metrics SHA와 raw G4 판정을 검증한다.
   **PASS만** `completed.json`을 발행한다. 유효한 FAIL/INCONCLUSIVE raw는 삭제하지 않고
   raw SHA와 판정을 `failed.json`에 no-replace로 봉인하며 readiness·deployment를 열지 않는다.

G4 PASS는 동시에 다음을 요구한다.

- trusted 150–1600 Hz 평균 NMSE `< 0 dB`, fullband 평균 NMSE `≤ 0 dB`
- 네 family 각각 trusted 평균과 최악 10% `< 0 dB`
- family×네 strict 부대역 각각 target density `≥ 0.25`, 독립 group `≥ 4`, 평균·최악 10%와
  group-bootstrap 95% CI 상단 `< 0 dB`
- 125/250/500/1000/1600/2000/4000/8000 Hz 중 trusted 밖 옥타브의 최악 10% 증폭 `< 1 dB`
- manifest의 모든 selected session과 metrics session이 전단사이며 immutable `session.json`
  timeline, checkpoint timing/hop/warmup/feedback, deterministic start exact set이 일치

증명된 악화는 FAIL, 표본/coverage/CI 부족은 INCONCLUSIVE이며 둘 다 test 완료·배포를 막는다.

## 7. 배포 차단 규칙

다음 조건을 **모두** 만족하기 전에는 ONNX export가 성공해도 deployment eligible이 아니다.

1. readiness 17/17
2. canonical 100k pretrain과 50k fine-tune completion
3. recorded val selection과 one-shot test G4 PASS
4. Level-5 unseen 네 family PASS
5. strict runtime plant/checkpoint/ONNX lead·SHA exact
6. Jetson P99 <3.0 ms, max <5.333 ms, 공식 live session xrun/deadline miss/
   engine error/drop/add/fallback/watchdog/sample slip과 excess backlog 0
7. ANC OFF/ON raw와 octave/family 결과 보존

Tiny와 Base의 우열도 같은 checkpoint 세대·plant·lead·source·volume·window에서 latency와 실제
감쇠를 함께 측정하기 전에는 정하지 않는다. legacy Tiny가 빠르다는 사실은 canonical Tiny의
감쇠 증거가 아니며 Base deadline risk도 실제 감쇠 실패의 증거가 아니다.

## 8. 절대 금지

- coverage density 0.25, 그룹 4, consistency 0.9406, G4 0 dB/1 dB 경계를 결과에 맞춰 완화 금지
- Stage-1 150–1600 Hz를 150–600 Hz로 축소하거나 environment/machine을 목표에서 제거 금지
- Stage-1 PASS를 2/4/8 kHz 최종 성공으로 표현하거나 광대역 v2 blocker를 생략 금지
- legacy/corrected checkpoint, old P/S, old ONNX/TensorRT, diagnostic report의 canonical 승격 금지
- summary scalar, 문서 문구, 자체 재봉인 SHA만으로 raw 검증 대체 금지
- INCONCLUSIVE, missing, stale, forged evidence를 PASS로 해석 금지
- test 결과를 본 뒤 모델·loss·threshold·seed를 다시 선택하고 같은 test를 재사용 금지
- G4·Level-5·latency 중 하나라도 BLOCKED/FAIL인 상태에서 closed-loop/ONNX/실제 ANC ON 배포 금지

## 9. Elice 인스턴스 수명·GPU 사용 규칙

Elice는 GPU 사용률과 무관하게 인스턴스가 존재하는 동안 비용이 발생하므로, 단순히 켜 둔
상태를 진행으로 간주하지 않는다.

1. 모든 GPU 작업은 실행 전에 `role`, 가설, step 상한, 산출물 경로를 정한다. 공식 게이트를
   열지 못하는 작업은 파일명과 로그에 `DIAGNOSTIC_ONLY`를 남기고 init/G4 증거로 승격하지
   않는다.
2. exact commit/bootstrap 전의 유휴 시간에는 다음 공식 결정을 실제로 줄이는 bounded 진단만
   허용한다: batch/VRAM/throughput, finite 안정성, G0 loss ablation, deterministic resume.
   사용률 숫자만 높이기 위한 fake workload는 금지한다.
3. 독립 loss-pilot 후보는 자원이 허용되면 병렬 실행할 수 있다. canonical 100k와 canonical
   50k는 timing·telemetry를 오염시키지 않도록 각각 단독으로 GPU를 사용한다.
4. 인스턴스 삭제 전 다음을 모두 만족해야 한다.
   - 실행 중인 공식 작업이 없고 새 공식 작업을 바로 시작할 수 없는 이유가 기록됨
   - checkpoint, completion/evaluation receipt, raw metrics, 환경·telemetry 로그를 외부 저장소로
     복사하고 상대경로·크기·SHA-256을 다시 검증함
   - 공개 raw corpus와 manifest를 다시 받을 위치 또는 검증된 백업이 존재함
   - 삭제 후 재구축 시간이 남은 로컬 blocker 해결 시간보다 짧거나, 더 이상 같은 세대의
     학습을 이어갈 필요가 없음
5. 위 삭제 조건을 아직 만족하지 않았고 수 시간 안에 official bootstrap/학습으로 전환할 수
   있으면 인스턴스를 유지한다. 반대로 유용한 bounded 작업도 없고 blocker가 장기화되면 비용을
   이유로 즉시 삭제 후보로 전환하며, 삭제를 GPU 사용률로 정당화하지 않는다.
