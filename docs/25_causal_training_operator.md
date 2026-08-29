# 25. 연속 causal P/S 학습 연산자와 admission 경계

> 상태: 코드 경로는 구현됐지만 **canonical 학습은 아직 차단**되어 있다. 현재
> `fullband_causal_v4.LIVE_AUTHORITY=None`이고, 현 광대역 v2 계약은 125 Hz 중심 octave의
> 하단 88.388--150 Hz를 포함하지 않는다. fixture나 문서만으로 이 두 차단을 PASS로 바꿀
> 수 없다.

## 1. 해결한 두 코드 결함

기존 광대역 후보는 다음 두 이유로 학습에 연결할 수 없었다.

1. random segment 시작 이전 P/S history 또는 state가 없었다.
2. 합성 `d`는 측정된 full linear P가 아니라 compact diagnostic P generator를 썼다.

새 경로의 단일 byte source는 schema
`fullband_causal_joint_fir_operator_npz_v4`인 **하나의 joint P/S NPZ**다. P를 source-v2용으로
복사한 별도 NPZ는 허용하지 않는다. loader는 NPZ file SHA, 내부 array archive SHA, P/S
float64 FIR raw-byte SHA를 모두 다시 계산한다.

합성 branch는 연속 source를 다음처럼 만든다.

```text
source history | model/S prefix | valid target | digital lead tail
       P를 전체 구간에 한 번 적용
                     | exact crop d와 x_ref
```

- `d=P*n`은 source history를 포함한 전체 신호에 full causal filtering한 뒤 exact crop한다.
- fractional residual은 post-onset FIR 위상에 이미 포함되어 forward에서 다시 지연하지 않는다.
- P에는 handoff가 없다.
- S에는 256-sample handoff를 FIR 밖에서 정확히 한 번 더한다.
- 손실은 prefix를 포함한 `y`에 `S`를 적용한 다음 valid target만 crop한다.
- 극성은 항상 `e=d+S*y`다.

recorded branch도 세션의 실제 연속 prefix를 읽는다. 세션 시작이 너무 가까워 prefix가 없으면
zero padding이나 다른 세션 연결로 대체하지 않고 실패한다. common EQ가 켜졌으면 129-tap
linear-phase EQ의 half-history 64 samples까지 prefix 요구량에 포함한다. `mode="same"`의
오른쪽 zero boundary도 valid target 끝에 닿지 않도록 세션에서 suffix 64 samples를 더 읽고,
EQ 적용 뒤에만 suffix를 exact crop한다.

## 2. prefix와 시간축 계약

valid prefix는 다음 세 값의 최댓값을 256에 올림한 값이다.

- S coarse delay + handoff 256 + FIR support
- 모델 finite branch input span
- 최대 error-feedback delay + recorded common-EQ half-history

GLSTM은 과거 state가 무한하지만, 실제 stream 시작의 zero state에서 **실제 연속 prefix를
순서대로 처리**하는 방식으로 state를 만든다. target 시작에서 모델이나 plant를 새로 0으로
초기화하지 않는다.

P/S coarse delay, fractional residual, FIR peak, handoff와 lead는 authority의
`PlantDelays` 및 `TrainingTimingContract`에서만 유도한다. config의 수동 lead, prefix,
`loss_start_sample`이 유도값과 다르면 admission은 실패한다. resolved config/checkpoint에는
authority SHA, joint NPZ SHA, P/S FIR SHA, timing SHA, prefix 구성과 inline evidence SHA를
함께 남긴다.

## 3. 대역별 batch 자격

한 자연음 segment가 일곱 대역을 동시에 density 0.25 이상 만족하도록 요구하지 않는다.
그 조건은 speech/music에서 사실상 영구 rejection을 만들 수 있다.

- batch planner는 각 대역마다 valid item 4개를 결정적으로 예약한다.
- synthetic batch 4는 네 item 모두에 일곱 대역 동시 자격을 강제하므로 금지하고 최소 5로
  닫는다.
- 각 item은 자신에게 배정된 대역만 post-augmentation에서 다시 검사한다.
- identity-EQ fallback도 배정 대역 자격을 다시 통과해야 한다.
- source-v2의 9개 segment도 각 대역을 독립적으로 세어 **대역별 8/9**를 요구한다.
- equal-subband NMSE, worst-subband guard, actuator-output DNH 임계는 낮추지 않았다.

component의 pre-EQ 증거와 최종 source qualification은 의미가 다르다. 전자는 원본의 실제
native bandwidth/nonzero 정보를 증명하고, 후자는 최종 15초 source/P-applied ERR가 batch
목적함수에 충분한지를 증명한다. 현 source-v2는 이 역할을 일부 함께 검사한다. 임계를 낮추지
말고 향후 schema에서는 두 receipt를 분리해야 한다.

## 4. input channel과 clock-domain 제한

canonical digital-reference에서는 `x_ref` dropout을 exact 0으로 고정한다. `x_ref=0,d!=0`인
item은 causal controller가 풀 수 없는 표본이므로 CVaR에 넣을 수 없다.

error input dropout은 별도 확률로 explicit config/experiment contract에 결속한다. 이를
임의로 0 또는 1로 바꾸지 않는다. synthetic와 recorded branch 모두 global sample index의
순수 함수로 같은 선언을 적용하며, recorded branch가 ERR를 항상 공급해 `err=0` 실험을
거짓으로 만드는 경로는 닫았다.

- error probability 0: ERR context를 항상 사용한다.
- error probability 1: ref-only, ERR input은 항상 0이다.
- 중간값: 강건성 증강일 뿐 장기 clock drift 증명이 아니다.

현재 combined callback의 digital-ref 모델도 ch1 ERR를 읽으므로 output-clock-master DAC와
ADC clock 사이 장기 drift가 2 kHz 이상 위상을 오염할 수 있다. random error dropout은 이
문제를 입증하지 않는다. 따라서 output-clock-master runtime 적격은 다음 중 하나 전까지
항상 false다.

1. ERR를 고정 0으로 한 결정적 G0와 recorded validation이 같은 전 대역 absolute gate를
   모두 통과한다.
2. 검증된 ASRC/clock bridge와 그 raw witness가 있다.

새 degradation 허용값을 만들지 않는다. `err=0` ablation은 같은 대역별 G0/validation 합격선을
그대로 써야 한다.

## 5. 재개와 TOCTOU

synthetic/recorded 선택, family/band reservation, augmentation은 global sample/batch index의
순수 함수다. checkpoint의 `data_stream.global_batch_index`를 보존한다.

frozen causal P/S에는 sequential plant RNG가 없다. checkpoint training-state schema v2는
이를 `plant_rng_kind=not_applicable_frozen_causal_fir`, `plant_rng=null`로 명시하고 nonlinear
RNG는 계속 저장한다. Stage-1은 기존 schema v1과 기존 RNG 동작을 유지한다.

experiment contract는 causal authority, joint NPZ, signal plan, submitted int16 NPY, raw NPZ,
analysis JSON, fit candidates/freezes를 파일 SHA로 결속한다. clock/fit/holdout/stationarity/
change-point inline receipt SHA도 별도 identity로 남긴다. criterion construction과 resume
검증 시 파일을 다시 열어 TOCTOU를 차단한다.

recorded train/val batch receipt도 서로 다른 artifact path/file SHA로 결속한다. receipt는
split과 authority-derived valid prefix를 담고 `edge_trim >= prefix`를 강제한다. causal
fine-tune에서 `recorded_ratio>0`이면 `best.pt` 선택은 증강을 끈 recorded val batch만 사용하며,
val receipt/manifest가 없을 때 synthetic val로 후퇴하지 않는다. Stage-1 validation 경로는
기존 synthetic val 동작을 그대로 유지한다.

## 6. 현재 남은 fail-closed blocker

1. 실제 v4 raw/envelope/publisher가 없고 `LIVE_AUTHORITY=None`이다. signal-only exact
   condition 반증에서도 support 1024의 AᵀA condition이 fit_a 280.3743, fit_b 297.7764로
   사전 선언 상한 20을 넘었다. 더 긴 support는 principal-submatrix interlacing상 이를
   개선할 수 없으므로 현 v4 plan은 임계 완화 없이 envelope를 발행할 수 없다.
2. 현재 `broadband_point_control_150_11314_v2`는 150 Hz에서 시작한다. 따라서 125 Hz octave
   `[88.388,176.777]` 전체 PASS를 주장할 수 없다.
3. v3 contract API는 `BroadbandFullOctaveContractV3.canonical()`과
   `resolve_control_band_contract(payload)`로 확정됐다. 그러나 v4 authority exact
   envelope에는 contract SHA만 있고 immutable payload/reference가 없다. physical 8,
   objective 7, Stage-1 guard 4 band 집합을 독립적으로 복원·검증할 수 없으므로
   현 v4 loader는 v2-only BLOCK을 유지한다. v5 authority가 full immutable contract
   payload/reference와 digest를 제공한 뒤 모든 receipt/loss/batch/eval consumer를 함께
   연결해야 한다. v2 factory/loss Literal을 임의 8-band로 늘리지 않는다.
4. v4 authority가 생겨도 fixture/synthetic envelope는 canonical PASS를 낼 수 없다.
5. v4 diagnostic clock/change-point receipt에는 reviewed exact nested schema와 raw-lineage
   validator가 없다. 따라서 `LIVE_AUTHORITY` 문자열을 임의로 세워도
   `BLOCKED_MISSING_EXACT_CLOCK_CHANGE_POINT_VALIDATOR`로 실패한다. self-sealed
   `passed=true`는 training evidence가 아니다.
6. output-clock runtime은 위 `err=0` absolute G0/val 또는 ASRC witness 전까지 적격이 아니다.

Stage-1 150--1600 Hz 경로와 기존 checkpoint 의미는 변경하지 않았다. v3에서는 Stage-1의 네
strict guard를 그대로 두고, 최종 octave loss band를 별도 계약으로 추가해야 한다.
