# 광대역 v3 전 옥타브 계약

> 상태: immutable 계약 코드 기반 구현, downstream 연결·실측 전 `BLOCKED`
> 기준일: 2026-08-28
> 이 문서는 기존 Stage-1과 광대역 v2를 덮어쓰지 않는다.

## 1. 변경이 필요한 이유

최종 평가 문서(`docs/07` §0)는 중심 주파수 125, 250, 500, 1000, 2000,
4000, 8000 Hz의 옥타브 감쇠를 요구한다. 그러나
`broadband_point_control_150_11314_v2`의 하단은 150 Hz다. 따라서 125 Hz
옥타브의 실제 범위인

```text
[125/sqrt(2), 125*sqrt(2)]
= [88.3883476483, 176.7766952966] Hz
```

중 88.388--150 Hz는 v2에서 P/S 식별, source 자격, 손실 최적화, 양의 감쇠
판정을 받지 않는다. 더구나 v2의 대역 밖 DNH는 이 구간의 actuator 출력을 억제할
수 있다. 그러므로 v2 결과로 125 Hz 옥타브 ANC를 주장할 수 없다.

이 누락은 임계값 완화나 125 Hz 단일 톤으로 보충하지 않는다. Stage-1
`stage1_strict_150_1600_v1`과 v2 SHA는 그대로 보존하고, 최종 역할은 별도 v3
계약으로 발행한다.

## 2. v3가 분리해야 하는 세 대역 집합

### 2.1 물리 식별·coverage 구간

P/S가 주파수 축 전체를 빈틈없이 검증했다는 증거에는 다음 8개 연속 구간을 쓴다.

```text
88.3883476483--150
150--300
300--600
600--1000
1000--1600
1600--2828.4271247462
2828.4271247462--5656.8542494924
5656.8542494924--11313.7084989848 Hz
```

첫 구간을 추가하는 이유는 125 Hz 옥타브의 빠진 하단을 실제로 관측하기 위해서다.
이 8개 구간은 P/S residual, complex agreement, SNR, exact-zero noise floor,
fit-a/fit-b stationarity를 빠짐없이 판정하는 **식별 격자**이지 학습 손실의 동일 가중
목록이 아니다.

### 2.2 학습의 equal-weight 목적 구간

손실은 다음 7개 정확한 옥타브를 동일 가중한다.

| 중심 | 하단 Hz | 상단 Hz |
|---:|---:|---:|
| 125 | 88.3883476483 | 176.7766952966 |
| 250 | 176.7766952966 | 353.5533905933 |
| 500 | 353.5533905933 | 707.1067811865 |
| 1000 | 707.1067811865 | 1414.2135623731 |
| 2000 | 1414.2135623731 | 2828.4271247462 |
| 4000 | 2828.4271247462 | 5656.8542494924 |
| 8000 | 5656.8542494924 | 11313.7084989848 |

88.388--150 Hz를 독립 eighth loss band로 동일 가중하면 폭 61.6 Hz가 마지막
5656.9 Hz 폭 구간과 같은 비중을 가져 저역을 과가중한다. 따라서 식별 격자와 학습
목적 격자를 하나의 배열로 재사용하지 않는다. 각 옥타브는 자기 target 에너지로
정규화하고, 평균과 worst-octave CVaR를 함께 최적화한다.

### 2.3 Stage-1 저역 보존 가드

v3 학습·평가는 정확한 옥타브 결과와 별도로 기존 네 구간을 모두 유지한다.

```text
150--300 / 300--600 / 600--1000 / 1000--1600 Hz
```

각 family에서 평균, 최악 10%, 독립 group cluster-bootstrap CI가 모두 양의 감쇠를
보여야 한다. 정확한 옥타브 평균이 이 네 구간 중 하나의 증폭을 가릴 수 없다.

## 3. 측정 신호 계약

v3 P/S 자극은 분석 하단 88.388 Hz의 edge bin과 필터 천이를 보수적으로 덮도록
excitation lower를 80 Hz 이하로 둔다. 상단은 8 kHz 옥타브 전체를 덮도록
11,313.708 Hz 이상이어야 한다. continuous low pilot은 clock 추정용일 뿐이며,
88.388--152 Hz의 plant PE를 대신하지 않는다.

현재 signal-only 감사에서는 PE lower를 80 Hz로 낮춰도 48 kHz, 14.336초,
두 출력 합성 peak 95/32767 이하로 구성 가능했고 152--600 Hz clock comb의 exact-null을
깨지 않았다. 이 값은 **재생 승인이나 실제 P/S PASS가 아니다.** shared v3 계약 SHA,
actual int16 plan SHA, raw-first publisher와 전체 negative fixture가 함께 고정되기 전
live authority는 `None`이다.

## 4. 데이터·학습 admission

v3 source/recorded/synthetic admission은 다음을 동시에 요구한다.

1. speech/music/environment/machine, train/val/test의 각 slot에서 독립 component와
   lineage 교집합 0
2. actual Q15 source와 같은 source에 v3 causal P를 적용한 predicted ERR를 각각
   정확한 7개 옥타브별로 재계산
3. 9개 결정론적 crop 중 각 옥타브가 독립적으로 최소 8개 PASS
4. 실제 recorded target `d`의 Stage-1 네 구간과 정확한 7개 옥타브 coverage
5. causal P/S full prefix, valid crop, timing contract와 artifact SHA의 exact 결속
6. digital-reference `x_ref` dropout 0; error-input ablation은 별도 runtime 적합성
   게이트로 보고

v2의 48-slot 계획과 9x7 구조는 v3 SHA로 재발행되기 전에는 v3 자격이 없다. 기존
0/48 판정을 이름만 바꿔 승격하지 않는다.

## 5. 최종 성능 판정

최종 v3 PASS는 같은 source·SPL·P/S·window에서 다음을 모두 만족해야 한다.

- 정확한 125--8000 Hz 7개 옥타브에서 Deep-ANC 평균·최악 10%·CI 감쇠가 양수
- Stage-1 네 저역 구간도 각각 양의 감쇠
- 2/4/8 kHz 옥타브에서 matched FxLMS 대비 paired delta의 평균·최악 10%·CI 하단이
  모두 0 dB 초과
- 네 source family와 모델 선택 뒤 Level-5 unseen source 통과
- 단일 ERR point 결과와 최소 5개 ERR 위치 spatial 결과를 별도 발행
- 48 kHz/256에서 deadline miss, xrun, fallback, ring drop/add, sample slip 0

단일 마이크 지점의 고역 null은 quiet zone 증거가 아니다. 1.633 kHz 위 고차모드에서
다섯 위치 중 하나라도 증폭되면 spatial quiet-zone PASS가 아니다.

## 6. 현재 판정과 구현 순서

```text
Stage-1 v1: 기존 별도 계약으로 보존
broadband v2: 150--11.314 kHz 구조/진단 역할, 125 Hz 최종 주장 금지
broadband v3: immutable 계약 기반만 구현, 실측·source·checkpoint 없음 -> BLOCKED
```

구현 순서는 다음과 같다.

1. (코드 기반 완료) 기존 `ControlBandContract`의 v1/v2 직렬화는 건드리지 않고,
   별도 v3 모델에 식별 격자·목적 옥타브·Stage-1 guard를 서로 다른 필드로 추가해
   digest를 발행한다.
2. causal-v4 plan/분석기를 contract-driven 8구간으로 실행하고 80 Hz PE negative
   fixture를 통과시킨다.
3. source/recorded/synthetic receipt와 loss/eval을 v3 SHA로 재발행한다.
4. output-clock 기준 runtime 또는 공통-clock 출력 경로와 physical witness를 먼저
   통과시킨다.
5. 그 뒤에만 사용자에게 exact 명령·출력 시간·speaker·volume·raw 경로를 보고하고
   승인을 받아 한 번의 측정 창을 연다.

## 7. immutable 코드 기반과 아직 연결되지 않은 consumer

2026-08-28에 다음 최소 기반을
`src/deep_anc/dsp/control_band_contract.py`에 추가했다.

- schema: `control_band_contract_v3`
- contract id: `broadband_full_octave_88_11314_v3`
- API: `BroadbandFullOctaveContractV3.canonical()`
- 명시 resolver: `resolve_control_band_contract(payload)`
- canonical digest:
  `53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2`

v3는 다음 세 배열을 별도 직렬화한다.

1. `physical_identification_subbands_hz`: 88.388--11,313.708 Hz의 exact 8구간
2. `equal_weight_octave_objective_bands_hz`: 125--8000 Hz 중심 exact 7 octave
3. `stage1_low_guard_subbands_hz`: 기존 150--1600 Hz exact 4구간

각 배열을 다른 역할의 배열로 바꾸거나 한 구간이라도 생략하면 validation이 실패한다.
excitation lower는 `0 < lower <= 80 Hz`, upper는
`11,313.7084989848 Hz <= upper <= Nyquist`를 강제한다.
`legacy_v2_automatic_promotion_allowed=false`와
`requires_exact_v3_contract_sha256=true`도 계약 bytes에 포함한다.

기존 모델에 default v3 field를 넣지 않았으므로 다음 역사적 digest와 canonical JSON은
그대로다.

```text
Stage-1 v1: cf6216ce4bae35fd449b29c726c8b2c7d7d2f9a83adcde1b0b29fead642d0619
broadband v2: 73c8fdf013fec94a3b8697d3be1353a5d59c33f8fd2b5973127fc159328f8047
```

resolver는 schema v2를 기존 `ControlBandContract`로만 복원하고 v3로 바꾸지 않는다.
따라서 v2 payload·plant·checkpoint의 이름을 바꾸는 것으로 v3 자격을 얻을 수 없다.

다음 consumer는 이 단계에서 의도적으로 **미연결**이다.

- causal fullband P/S plan·publisher·plant evidence
- source/recorded/synthetic coverage receipt
- broadband loss와 trainer admission
- G4/raw point·spatial evaluator와 readiness
- checkpoint/export/runtime receipt

이들이 v3 schema와 exact digest를 각각 소비하고 새 raw evidence를 발행하기 전에는
v3 전체 상태가 계속 `BLOCKED`다. focused unit fixture PASS는 실제 P/S 식별, ANC 감쇠,
FxLMS 우위 또는 배포 성능 증거가 아니다. 이 구현 과정에서 오디오 출력은 0회였다.
