# Broadband v3 손실·배치 primitive 계약

## 1. 범위와 현재 판정

이 문서는 `BroadbandFullOctaveContractV3`를 실제 학습 코드에 조용히 연결하기 위한
문서가 아니다. v2의 150 Hz 하단 누락과 넓은 고역 에너지 지배를 반복하지 않도록,
향후 live v5 causal P/S authority가 생겼을 때 사용할 **별도** 손실·배치 primitive의
수학과 입력 SHA를 고정한다.

현재 판정은 다음과 같다.

- v3 loss/batch 수학 primitive: 구현 및 독립 테스트 가능
- v3 physical population: 구조적 audit/receipt만 가능
- live v5 raw P/S operator와 immutable training envelope: 없음
- criterion factory/trainer/checkpoint admission: 연결하지 않음
- canonical training: `BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION`

현재 exact admission blocker는 네 개다.

1. live v5 causal authority envelope 없음
2. output-y gradient share 0.2–0.4 DNH calibration 없음
3. 실제 family-balanced batch receipt와 loss 입력의 결속 없음
4. causal operator/prefix/fractional delay/handoff timing 결속 없음

따라서 상태 문자열은 `BLOCKED_INCOMPLETE_BROADBAND_V3_ADMISSION`이며 어느 한 항목만
해결됐다고 학습을 열 수 없다.

기존 v2 config, checkpoint, batch receipt 또는 150 Hz 시작 P/S를 v3로 자동 승격하는
경로는 없다.

## 2. 서로 다른 세 대역 역할

### 물리 식별·population receipt — 8개

`88.3883476483–150`, `150–300`, `300–600`, `600–1000`, `1000–1600`,
`1600–2828.4271247462`, `2828.4271247462–5656.8542494924`,
`5656.8542494924–11313.7084989848 Hz`이다.

이 8개는 P/S 식별과 source population 자격을 위한 구간이다. loss 가중치 8개가
아니다. batch primitive는 `PopulationAuditV3`의 payload digest를 별도
`physical_identification_population_receipt_sha256`로 받아 다시 계산·대조한다.

### equal-weight octave objective — 7개

`88.3883476483–176.7766952966`, `176.7766952966–353.5533905933`,
`353.5533905933–707.1067811865`, `707.1067811865–1414.2135623731`,
`1414.2135623731–2828.4271247462`, `2828.4271247462–5656.8542494924`,
`5656.8542494924–11313.7084989848 Hz`이다. 중심은 정확히
`125/250/500/1000/2000/4000/8000 Hz`다.

각 octave는 자기 target `d` 에너지로 NMSE를 정규화하고, item valid 수가 4개
이상일 때만 scalar를 만든다. baseline은 정확히 1/7 동일 가중이다. 그 위에
worst-octave guard를 더하므로, 폭과 bin 수가 큰 4/8 kHz 대역이 저역을 에너지 합계로
압도하지 못하고 저역의 큰 성공이 고역 실패를 평균으로 숨기지도 못한다.

실제 식은 다음과 같다.

```text
L_octave = mean(L_125, ..., L_8000) + 0.7 * max(L_125, ..., L_8000)
```

따라서 `octave_worst_guard_weight=0.7`은 convex 혼합의 alpha가 아니라 **동일가중
baseline 위에 더하는** worst guard weight다. baseline에서 각 octave의 계수는 항상
정확히 `1/7`이고, worst octave에만 추가 guard gradient가 생긴다. Stage-1/low-high
guard weight `1.0`과 함께 이 값들은 수학적 반례와 gradient 연결을 검증하기 위한
**diagnostic primitive 설계값**이다. pilot
후보 비교를 거친 campaign hyperparameter authority나 최적값이 아니다. config와
metric은 authority SHA가 없고 diagnostic-only임을 명시하며 이 값만으로 admission을
열지 않는다.

NMSE는 `log(E_error)-log(E_target)`로 계산하며 error energy에는 target energy 대비
`-80 dB`의 상대 하한을 둔다. 고정 absolute epsilon을 더하지 않으므로 동일한 유효
파형을 작게 스케일해도 NMSE가 0 dB 쪽으로 무너지는 quiet-batch 오류가 없어야 한다.
또한 완전상쇄에서 reciprocal overflow와 `0*inf`가 만나 CUDA gradient가 NaN이 되는
경로를 없앤다. `-80 dB` 아래는 현재 물리·평가 목표를 훨씬 넘으므로 더 작은 오차를
보상하지 않고 gradient를 0으로 고정한다.

### Stage-1 보존 guard — 4개

`150–300`, `300–600`, `600–1000`, `1000–1600 Hz`는 objective octave가 아니라
별도 positive-attenuation guard다. 각 구간의 aggregate NMSE가 0 dB보다 크면
독립 `relu`가 생긴다. 네 항은 합하므로 한 구간의 개선이 다른 구간의 증폭을
상쇄하지 못한다.

1600 Hz는 octave 중심이 아니다. exact 1600 Hz 경계는 half-open Stage-1 guard
밖이며 `1414.2135–2828.4271 Hz`의 2 kHz octave objective가 소유한다. 이 구분은
1600 Hz trusted 경계를 1600 Hz octave로 잘못 읽는 일을 막는다.

## 3. target density와 batch 불변식

loss는 각 objective octave의 평균 PSD를 7-octave union의 bin-weighted 평균 PSD로
나눈다. Stage-1 guard의 valid mask는 150–1600만 다시 정규화하지 않고 8개 physical
identification 구간 전체와 같은 denominator를 쓴다. threshold `0.25`, band별
valid item `>=4`는 고정이며 낮출 수 없다.

batch는 네 family를 정확히 균형 배치하고 physical 8개와 objective 7개 각각에서
valid item을 4개 이상 보장한다. 한 clip이 모든 대역을 동시에 통과할 필요는 없다.
batch size 4는 네 item 모두가 모든 대역을 통과해야만 `>=4`가 되므로 명시적으로
거부한다. global sample index를 `(batch index, offset)`으로 나누고 seed, batch index,
population receipt SHA에서 같은 plan을 재생성하므로 중단·재개 후 선택이 같다.
직렬화된 plan은 `split=train|val`을 반드시 포함하고 `selected_item_ids` 중복을
거부한다. family count 숫자만 유지한 채 같은 item을 복제한 forged plan은 유효하지 않다.

손실 primitive 자체는 family-balanced batch receipt를 입력으로 받지 않는다. 오직
전달받은 tensor에서 density와 band별 valid item `>=4`를 다시 계산한다. 실제 receipt
SHA 결속과 family balance는 별도 batch primitive의 책임이다. 두 primitive를 trainer가
동시에 소비하도록 연결하기 전에는 loss 단독 PASS를 batch admission으로 해석할 수 없다.

## 4. 인과 연산자와 부호

loss primitive의 입력 `secondary_output`은 future causal operator가 연속 prefix/state를
포함해 계산한 valid crop의 `S*y`여야 한다. primitive 내부 식은 오직

```text
e = d + S*y
```

이다. 추가 극성 반전은 없다. 이 primitive 자체는 P/S authority나 prefix를 만들거나
검증하지 않는다. 따라서 live v5 envelope, exact P/S bytes, fractional delay, 256-sample
handoff, prefix crop이 admission loader에서 결속되기 전에는 trainer에 연결하면 안 된다.

## 5. do-no-harm

제어 union `88.3883476483–11313.7084989848 Hz` 밖은 actuator output `y_nl`의
단측 DNH가 담당한다. 보호 union bin은 DNH에서 제외하며, union 밖 출력이 G4-consistent
margin을 넘을 때만 hinge가 생긴다. DNH를 0으로 끄거나 margin을 완화할 수 없다.
이 정의의 schema는 `actuator_output_union_margin_hinge_v3`이며, 기존 v2 primitive의
`actuator_output_union_leakage_v1`과 의도적으로 다르다. 서로 다른 수학을 같은
schema로 직렬화하거나 checkpoint에서 자동 호환하면 안 된다.
실제 gradient share 0.2–0.4 calibration은 live operator 이후 별도 admission 조건이다.
따라서 `lambda_dnh`에는 canonical default가 없고 diagnostic 호출자가 값을 명시해야
한다. 현재 config는 calibration receipt를 받을 수 없으며 상태를
`BLOCKED_MISSING_OUTPUT_Y_GRADIENT_SHARE_0P2_0P4_RECEIPT`로 봉인한다.

이 actuator-output leakage 항은 물리 ERR의 증폭 판정이 아니다. 제어 union 안으로 큰
출력을 내면 DNH가 0이면서도 위상/크기가 틀린 `S*y` 때문에 `e=d+S*y`가 증폭될 수 있다.
테스트는 DNH=0과 모든 octave 약 +6.02 dB를 동시에 만드는 반례를 고정한다. 따라서
physical ERR G4를 DNH로 대체할 수 없다.

## 6. 자동 승격 차단

- v3 loss config는 inline `BroadbandFullOctaveContractV3` payload와 exact digest를 모두
  요구한다.
- `ControlBandContract.broadband_point_control()` v2 payload는 v3 loss/batch에서 거부한다.
- 기존 `BroadbandLossConfig`도 v3 inline payload를 extra field로 거부한다.
- v2 checkpoint 자동 승격 허용 필드는 `false`로 봉인되며, 실제 checkpoint loader에는
  이 primitive를 아직 등록하지 않는다.
- batch plan과 global item은 항상 `canonical_training_eligible=false`와 live-v5 blocker를
  기록한다.
- causal authority loader, trainer, checkpoint selection에는 아직 연결하지 않는다.

## 7. 반례 테스트

테스트는 다음 실패를 고정한다.

1. 88–150 Hz 잔차를 v3 첫 octave가 실제로 감지한다.
2. 1600 Hz를 별도 octave 또는 Stage-1 inclusive 끝점으로 오해하지 않는다.
3. 고역 target 에너지와 bin 수가 더 커도 7개 objective baseline이 1/7이다.
4. 저역 평균 성공이 8 kHz 실패를 숨기지 못하고 low/high worst guard가 남는다.
5. band별 valid item이 3개면 loss가 실패한다.
6. partial-band item만으로 family-balanced batch를 만들 수 있으며 all-seven clip을
   강제하지 않는다.
7. batch size 4와 population receipt SHA 변조를 거부한다.
8. 유효한 동일 파형의 레벨만 낮춘 quiet batch에서도 octave NMSE가 scale-invariant다.
9. 보호 union 안 출력에는 DNH gradient가 없고, union 밖 출력에는 실제 autograd
   gradient가 생긴다.
10. peak-normalized 완전상쇄와 근접상쇄에서 CPU 및 사용 가능한 CUDA의 loss와
    gradient가 모두 finite다.
