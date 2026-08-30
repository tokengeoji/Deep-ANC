# 실측 플랜트 레벨 보정·도메인 혼합 계약

## 1. 왜 필요한가

기존 82세션의 `source_aligned.wav → mics.wav ch0(ERR)`와 현행 strict primary
`assets/measured/primary_path_il_strict_5dc06fdd.npz`는 같은 48 kHz 정규화 단위지만
플랜트 크기가 같지 않다. 2026-08-30 독립 재계산에서 old82는 strict P보다
150–1600 Hz에서 약 20–25 dB 작았다. 이 상태로 measured synthetic 30%와 recorded
70%를 섞으면 같은 digital reference에 두 개의 상충하는 정답을 준다. 이는 모델 용량,
고역 손실 또는 지연 문제가 아니라 학습 target 단위 불일치다.

원본 WAV는 수정하지 않는다. `recorded_primary_level_calibration_v1` receipt가
2026-08-04/06 cohort별 train split만으로 구한 scalar를 historical ERR와 `d`에만
적용한다. `source_aligned`, REF, acoustic-reference에는 적용하지 않는다.

## 2. 계산과 봉인

계산은 Hann Welch, 48 kHz, `nperseg=8192`, `noverlap=4096`, 5–65초, density scaling,
mean average로 고정한다. 마이크 자기잡음을 플랜트 gain으로 키우지 않도록
`H1 = CSD(source, ERR) / PSD(source)`의 coherent ERR power를 strict P의
`PSD(source)·|H_strict|²`와 비교한다. cohort gain은 train session power-ratio dB의
중앙값에서만 만든다. val/test는 적합에 사용할 수 없고 진단에만 쓴다.

Receipt는 다음을 path/size/SHA로 봉인한다.

- old82 manifest, strict primary NPZ
- 82개 `source_aligned.wav`와 `mics.wav`
- exact Welch/transfer/shape recipe와 구현 파일 SHA
- clean exact 40자리 source commit
- cohort별 train fit session ID/count, gain, held-out residual

공식 `measured_probe`와 `canonical_finetune`은 `recorded_ratio>0`일 때 receipt path와
외부 SHA가 없거나 source commit이 다르면 run directory 생성 전에 실패한다. Receipt는
schema-v2 transfer의 별도 role이며 experiment/checkpoint contract에도 포함된다.

## 3. 결과 맞춤 완화 금지 품질 gate

Scalar 근사를 허용하는 조건은 발행 전에 고정한다.

- cohort별 val/test median residual 절댓값 ≤ 1 dB
- 모든 session residual 절댓값 ≤ 6 dB
- train-only aggregate complex shape agreement ≥ 0.95
- train-only best-delay/complex-scalar 제거 후 normalized error ≤ 0.25
- 보정 후 모든 historical ERR 절댓값 peak ≤ 0.8 (그리고 당연히 < 1.0)

이 기준은 레벨 scalar가 물리 shape 차이를 지나치게 숨기지 않는지 검사하는 것이다.
latency, xrun, provenance, strict P/S consistency 또는 고역 do-no-harm 기준을 완화하지
않는다. 실제 receipt의 shape residual도 반드시 기록하며, scalar만으로 두 도메인이
동일하다고 선언하지 않는다.

## 4. current strict 녹음과 sampler

새 녹음은 `recording_level_campaign` receipt에 결속한다. 공용 meter raw/receipt,
capture ID, probe peak 0.003, ch0 `-50.1 dBFS ±2 dB`, hardware/physical fingerprint,
동일 amplifier 설정, meter→session 600초 이내를 검증한다. 이는 current session이 strict
측정과 같은 운영점에서 수집됐다는 admission이지, 서로 다른 source의 유한 15초 H1을 억지로
strict FIR과 ±2 dB로 맞추는 사후 gate가 아니다. 실제 plant shape 동일성은 아래 단일-domain
ablation과 별도 source→ERR 진단으로 기록하며, 차이가 크면 `current_strict` 표본을 늘리거나
domain conditioning을 검토한다. session 시작은 campaign 발행 시각 이후여야 하므로 이미 끝난
과거 capture를 나중에 만든 campaign으로 소급 승격할 수 없다.

공식 recorded sampler 순서는 다음과 같다.

1. family
2. plant domain (`historical_calibrated`, `current_strict`)
3. lineage component
4. session

Global item index의 `2 × family` 주기에서 각 family의 두 domain을 정확히 한 번씩 뽑아
long-run current 비율을 추정이 아닌 정확한 50%로 만든다. 네 family 각각 train split에
두 domain이 모두 없으면 학습을 차단한다. 서로 다른 plant domain의 session mix 증강은
허용하지 않는다.

현행 19행 additions plan은 current train anchor를 위해 다음 두 행을 추가한다.

- environment: `data/source_pool/environment/environment_006.wav`, 25.75초,
  component `environment-source-lineage-6edef9c5d066`
- music: `data/source_pool_v2/music/music_008.wav`, 31.5초,
  component `music-source-lineage-2ec4cf7992c7`

둘은 parent82와 기존 additions component에 겹치지 않는다. environment 원본 일부는
현재 legacy ESC-50 manifest와 겹치므로, 19행 generation exclusion을 결속해 public
canonical manifest 6종을 다시 발행하고 교집합 0을 검증하기 전에는 학습 입력이 아니다.

## 5. 필수 ablation 경계

성능 선택은 mixed 결과 하나만으로 끝내지 않는다. 같은 checkpoint/lead/P/S/평가
window에서 다음 세 진단을 별도 기록한다.

- `historical_calibrated` only
- `current_strict` only
- exact 50:50 mixed

두 단일-domain 결과가 크게 갈리면 scalar receipt를 재적합하거나 임계값을 낮추지 않는다.
기하·장착·플랜트 shape 차이를 원인으로 보고 current session 보강 또는 명시적 domain
conditioning을 검토한다. val/test로 gain을 다시 추정하거나 test 성능을 보고 scalar를
고치는 것은 금지한다.
