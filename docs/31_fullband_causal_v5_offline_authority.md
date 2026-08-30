# v5 actual raw offline causal P/S authority

> 상태: synthetic fixture 경로만 구현, actual live raw 없음, canonical authority `None`

## 구현 범위

`fullband_causal_v5_offline.py`는 v5 plan과 actual-int16 PCM SHA를 다시 검증하고 다음 순서로
분석한다.

1. raw NPZ full-file SHA, sealed plan raw path, exact key와 PCM/captured/callback SHA 재계산
2. callback 배열의 exact `<i8` 256-frame 저장 accounting 확인. xrun/slip 부재 증거로
   승격하지 않음
3. support-1024 exact Gram condition 재검증
4. raw waveform의 ERR/REF×P/S 공통 affine q 추정
5. 동일 acoustic pilot/optimizer에 대한 linear/cubic 보간 민감도 ≤0.006 sample 검사
6. q 보정된 fit-a와 fit-b를 각각 actual two-input circular joint-LS로 적합
7. support-1024 fit 뒤 unexplained-energy ratio ≤1e-4 및 fit-a/b tap stationarity ≤0.10.
   이 residual을 support 이후 tail energy의 직접 측정이나 상한이라고 부르지 않음
8. frozen 두 fit 평균 operator를 fit-a, fit-b, untouched holdout에서 각각
   P/S×ERR/REF×v3 8대역으로 평가
9. fixture timeline의 zeros-before-FIR=0을 `PlantDelays`와
   `TrainingTimingContract.derive()`로 발행하고, FIR 내부 peak는 diagnostic으로 분리
10. operator NPZ와 authority JSON을 sibling staging에서 함께 fsync한 뒤 atomic
    `RENAME_NOREPLACE` 발행

holdout은 생성, support 선택, fit 평균에 사용되지 않고 마지막 terminal score에만 사용된다.
2048/4096/8192 support는 자동 승격하지 않는다. coarse delay는 FIR peak에서 얻고 현재
fixture operator는 1024 FIR의 pre-peak tap을 자르지 않은 채 그대로 저장하고
zeros-before-FIR=0으로 두어 지연을 이중 계상하지 않는다. 이는 실제 덕트의 약 1200 samples
이상 coarse delay 계약이 아니다. 실제 raw에는 먼저 coarse/fractional marker scan을 수행하고,
그 지연 주위에서 충분한 pre-roll을 보존한 1024 compact window를 별도로 fit해야 한다.
현재 함수는 `synthetic_fixture=false` 입력을 즉시 거부하며, 이 단계가 구현되기 전 live
authority를 열 수 없다.

## 현재 schema와 권위

공식 분석 wrapper는 raw file bytes를 정확히 한 번 읽고 같은 bytes의 SHA를 계산한 뒤
`BytesIO`로 NPZ를 해석한다. plan의 repository-relative sealed path는 lexical containment와
모든 parent/target symlink 검사로 확인한다. canonical writer bytes와 다른 compressed/reordered
NPZ repackage도 거부한다. 순수 array fixture 분석은 `raw_container_bound=false`,
`raw_file_sha256=null`이며 publisher가 반드시 거부한다.

```text
analysis: fullband_causal_v5_offline_analysis_v1
operator: fullband_causal_joint_operator_v5_fixture_only
authority: fullband_causal_training_authority_v5_fixture_only
```

발행물은 sealed plan raw 상대경로와 raw NPZ full-file SHA, plan, submitted PCM, captured raw,
callback accounting, analysis payload, operator file SHA와
`PlantDelays`/`TrainingTimingContract` SHA를 포함한다. 그러나 synthetic fixture이거나 아직
review되지 않은 live raw이므로 다음 값은 항상 유지된다.

```text
authority = null
canonical_training_eligible = false
live_authority = null
```

fixture는 fixed band-limited two-path FIR와 +413.931 ppm affine clock의 positive case,
PCM 변조, callback accounting 변조, piecewise clock, support-1024로 설명되지 않는 에너지,
terminal holdout 한 고역 변조를
포함한다. positive fixture PASS는 실제 덕트 P/S나 파인튜닝 admission이 아니다.

## 실행 형식

```bash
.venv/bin/python scripts/data/analyze_fullband_causal_v5.py \
  --plan-envelope results/data_audit/fullband_causal_v5_signal_plan.json \
  --raw results/fullband_causal_v5/raw_capture.npz \
  --output-directory results/fullband_causal_v5/offline_analysis \
  --synthetic-fixture
```

현 CLI에는 장치 접근이나 live capture 기능이 없다. 실제 raw에 `--synthetic-fixture`를 붙여도
canonical authority가 생기지 않는다.

## 남은 live blocker

- 실제 raw와 독립 xrun/slip/time-info authority가 아직 없음. callback frame accounting만 존재
- independent repeat/Welch coherence가 없으므로 live coherence authority 미구현
- exact raw 경로가 live capture invocation/evidence와 결속되지 않음
- 실제 coarse-marker branch 및 fractional-delay/cubic 독립 receipt 없음
- linear/cubic 검사는 동일 pilot/optimizer의 보간 민감도 검사일 뿐 독립 clock witness가 아님
- fit-a/b 시간구간별 8대역 transfer stationarity 반복 증거 없음
- actual raw xrun/clip/level/hardware/environment/repository SHA envelope 없음
- fixture-only operator schema를 production loader가 의도적으로 받지 않음

이 항목을 모두 해결하고 별도 reviewed live schema를 발행하기 전에는 canonical 학습을
시작하지 않는다.
