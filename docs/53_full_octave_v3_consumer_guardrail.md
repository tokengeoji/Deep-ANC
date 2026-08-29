# 53. Full-octave v3 consumer 가드레일

## 목적

125 Hz부터 8 kHz 옥타브까지의 학습 계산을 준비하되, fixture·surrogate 결과를 실제
덕트 ANC 성능이나 canonical 학습으로 잘못 승격하는 것을 막는다. 이 문서는
`FullOctaveV3TrainerConsumer`와 `FullOctaveV3MatchedFxLMSEvaluator`의 허용 범위를
고정한다.

## 구현된 계산 경로

동일한 verified causal P/S binding에서 다음 순서는 코드와 회귀 테스트로 고정된다.

```text
clean physical playback n
  → P·n
controller prefix + target streaming output y
  → S·y
error target = P·n + S·y
  → 125/250/500/1000/2000/4000/8000 Hz equal-octave loss
```

FxLMS 비교도 같은 P/S, controller reference, zero-reset prefix, target crop, 256-sample
block만 사용한다. 2/4/8 kHz 옥타브에서는 Deep-ANC의 양의 감쇠뿐 아니라 FxLMS 대비
paired mean·worst-10·cluster CI 하한의 양수 우위가 모두 필요하다.

## 절대 금지

다음 중 하나라도 하면 결과는 canonical 학습 또는 physical G4로 사용할 수 없다.

1. `fixture_only=true` P/S binding을 public consumer로 통과시키는 것
2. `P·n` 또는 prefix의 `S·y` tail을 버리고 target만 다시 합성하는 것
3. 서로 다른 reference/P/S/prefix/block으로 FxLMS와 Deep-ANC를 비교하는 것
4. batch item에 임의 session/group ID를 만들어 independent bootstrap group 수를 늘리는 것
5. surrogate matched evaluator의 결과를 실제 덕트 raw ON/OFF 결과로 표시하는 것

현재 evaluator는 하나의 `FullOctaveV3EvaluationIdentity`가 하나의 실제 provenance
단위만 나타내도록 **batch size=1**만 허용한다. per-item identity receipt가 생기기 전에는
batched evaluation을 허용하지 않는다.

## 현재 상태

`V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED=True`는 위의 pure tensor consumer와
surrogate matched evaluator가 존재한다는 뜻뿐이다. 다음 check는 의도적으로 BLOCKED다.

```text
v3_raw_bound_execution_config
```

따라서 `configs/full_octave_v3_admission.yaml`은 Trainer/GPU/DataLoader를 만들지 않으며,
학습 권한을 발행하지 않는다.

`configs/full_octave_v3_execution.yaml`의 형식상 완전해 보이는 non-fixture JSON/SHA chain도
현재는 `BLOCKED_UNATTESTED_EXECUTION_PROVENANCE`다. `declared_sha_structure_valid=true`은
선언된 file SHA/nonce/field 교차검사일 뿐 canonical execution permission이 아니다. 이를
바꾸려면 다음 authority가 각각 독립적으로 필요하다.

- typed P/S operator·raw·analysis validator와 operator/timing exact crosslink
- typed raw/analysis/electrical witness validator
- actual submitted PCM/callback telemetry 및 native↔canonical recipe/equality
- plan nonce·device·session monotonic에 결속된 capture-adapter `O_EXCL` receipt
- canonical finetune init checkpoint·experiment contract·recorded selection을 포함한
  stage-specific training schema

## canonical 학습을 열기 위한 순서

1. synchronized electrical witness를 포함한 raw-bound full-octave causal P/S
2. native high-rate source, lineage-clean population, family-balanced batch receipt
3. strict DNH gradient calibration
4. non-fixture causal binding publisher와 canonical execution config
5. surrogate pretrain 후 recorded fine-tune
6. physical raw one-shot G4 및 unseen-source 평가

위 순서 전에는 checkpoint, ONNX, runtime ANC ON 또는 dB 감쇠를 주장하지 않는다.
