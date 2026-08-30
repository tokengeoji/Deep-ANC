# Digital-reference / acoustic-reference 인과성 현장 감사

> 감사 시점: 2026-08-28
> 범위: 현 checkout의 `configs/duct.yaml`, strict P/S NPZ·raw·analysis, 48 kHz/256 runtime
> 방법: **read-only**. 오디오 장치·스피커·마이크에 접근하지 않았다.
> 구현: `src/deep_anc/dsp/reference_mode_causality.py`
> 최종 권위: 구조 감사이며 ANC 감쇠 PASS가 아니다.

## 1. 현재 가장 중요한 결론

"처음 듣는 자연음"이라는 말만으로 reference mode를 정할 수 없다.

| 새 소리의 실제 유입 경로 | reference mode | 현재 인과성 | 125 Hz~8 kHz 물리 ANC |
|---|---|---|---|
| 새 speech/music/environment/machine 파일이나 새 녹음을 Jetson이 먼저 읽고 NS로 재생 | digital-reference | **CONDITIONALLY_CAUSAL** | **BLOCKED** |
| 현장의 대화·팬·기계음을 upstream REF mic가 처음 관측 | acoustic-reference | **BLOCKED** | **BLOCKED** |

첫 행은 모델이 그 파일을 학습했기 때문이 아니다. Jetson이 앞으로 재생할 sample `U_k`를
알고 있어 NS playback을 lead FIFO로 늦출 수 있기 때문이다. 완전히 unseen 파일이어도 같은
인과 구조를 쓸 수 있다. 다만 일반화·감쇠 성능은 별도 실험 대상이다.

둘째 행에는 미래 `U_k`가 없다. REF에 실제 파형이 도착한 뒤 ADC·controller·DAC·CS→ERR
경로를 통과해야 한다. 현재 이 항들을 같은 runtime timeline으로 묶은 측정 receipt가 없으므로
기하 좌표나 legacy 지연을 넣어 PASS할 수 없다.

## 2. strict artifact read-only 확인

### [가설]

현 `duct.yaml`의 P/S가 같은 strict capture이고 timing 계약을 만들 수 있다고 가정한다.

### [근거]

| Artifact | SHA-256 | 핵심 metadata |
|---|---|---|
| `configs/duct.yaml` | `a7091a3ebf4fe37ddd4503ddd84a22e36c0f17acbf978a65f0b30cdbb7fce5ff` | 105×105 mm, 1.190 m, 48k/256 |
| strict P NPZ | `23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598` | delay 1386, bulk 1642, FIR 2048 |
| strict S NPZ | `883c09364c00ad7aecdc038e38d0b6f8a49140fdc4b9788c2f5b0fc4686c2bee` | delay 1245, bulk 1501, FIR 2048 |
| source raw | `31d563b163fe7dcb3f6b85e30e491a6775947e7f1b988690c3668fd13464b347` | observed output S16, input raw int32 |
| source analysis | `064ff82cc5c4ed4febabff856394c22fa4db69510251c1c96eaa9f87789fba94` | joint-LS/crosscheck 결과 |

P/S 공통값:

- `capture_id=5ac1313488c8434bb4d672a36503df59`
- sample rate 48,000, calibration block 256, low latency
- raw/analysis SHA, anchor repeat 29, kept repeat 19개 동일
- xrun 0, output PCM provenance `observed_submitted_int16`
- 입력 APE, 출력 `AB13X USB Audio`
- P/S consistency authority는 모두 **150–1600 Hz**
- P excitation 64–1648 Hz, S excitation 72–1640 Hz

### [확인 방법]

NPZ를 `allow_pickle=False`로 열어 scalar/array metadata를 검증하고, metadata가 가리키는
raw/analysis 파일을 다시 SHA-256 계산했다. P/S 값을 `PlantDelays.from_config()`에 넣고,
P compact FIR을 `TrainingTimingContract.derive()`에 전달했다.

### [결과]

```text
PlantDelays:
  P delay = 1386
  S delay = 1245
  handoff = 256

PlantDelays.lead():
  raw lead = 1245 + 256 - 1386 = 115
  digital-reference lead = 115 samples

TrainingTimingContract:
  P compact FIR peak offset = 245
  P effective delay = 1386 + 245 = 1631
  synthetic total advance = 1631 + 115 = 1746
  digest = 1d6723bbfbad1371fab9d38e827c59789eba35a98fae67e478c44e1fdb0061db
```

NPZ의 P `bulk_delay_samples=1642`와 compact FIR에서 계약이 계산한 P effective 1631은
11 samples 다르다. 둘은 서로 다른 estimator 의미다. bulk metadata를 lead에 대입하지
않고, lead/handoff는 오직 `PlantDelays`와 `TrainingTimingContract` 결과를 사용했다.

### [판정]

**Confirmed** — 현 strict P/S는 같은 capture·raw·analysis에 묶여 있고 Stage-1 timing
계약을 유도할 수 있다.

### [다음 행동]

checkpoint/deployment/runtime이 위 exact timing digest와 v3 계약을 실제로 보존하는지 별도
admission receipt로 검증해야 한다.

## 3. digital-reference 인과성

### [가설]

Jetson이 생성하거나 파일에서 미리 읽은 `U_k`를 NS로 재생한다면 미래 참조를 불법적으로
보지 않고도 인과적 ANC를 구성할 수 있다고 가정한다.

### [근거]

동일 strict capture의 P/S delay와 runtime handoff에서 `lead=115`가 양수로 유도됐다.
`DigitalReferenceBuffer`는 모델에 `U_k`를 즉시 주고 실제 NS playback만 115 samples
늦춘다. output-clock-master 기반은 모델 결과를 다시 정확히 다음 256-sample callback의
CS로 보낸다.

### [확인 방법]

수동 숫자를 쓰지 않고 다음 관계를 immutable contract에서 재검산한다.

```text
digital lead = S.delay + one-block handoff - P.delay
             = 1245 + 256 - 1386
             = 115 samples
```

### [결과]

소스가 새 파일인지, 학습에 없던 화자인지, 음악인지 여부와 별개로 **Jetson이 실제
playback sample을 먼저 소유**하면 timing 구조는 조건부 인과적이다.

하지만 strict P/S의 복소 일관성 권위는 150–1600 Hz뿐이다. 다음이 없다.

- 125 Hz octave 하단 88.388–150 Hz strict consistency authority
- 2/4/8 kHz와 8 kHz octave 상단 11.314 kHz의 multi-panel P/S
- output-clock-master 실제 physical clock witness
- canonical ref-only checkpoint와 G0/val/offline-streaming receipt
- 최소 5개 ERR 위치의 physical OFF/ON raw

### [판정]

- digital timing causality: **Likely / CONDITIONALLY_CAUSAL**
- 125 Hz~8 kHz attenuation: **Inconclusive / BLOCKED**
- 실제 physical attenuation: **Not demonstrated**

### [다음 행동]

새 자연음을 digital playback으로 평가할 때는 actual `U_k`, NS/CS S16, control/gain,
ERR/REF raw와 frame/SHA를 한 session receipt에 보존해야 한다. unseen이라는 이유만으로
acoustic-reference 실험이라고 표기하면 안 된다.

## 4. acoustic-reference 인과성

### [가설]

upstream REF mic가 외부 소리를 ERR보다 충분히 먼저 보고, 전체 시스템 지연보다 선행량이
크면 live acoustic-reference 광대역 ANC가 가능할 수 있다고 가정한다.

### [근거]

config 좌표는 REF X=0.100 m, ERR X=1.100 m라 1.0 m 선행 경로를 암시한다. 단순 기하값은:

```text
1.0 / 343 × 48000 = 139.9417 samples = 2.915 ms
```

그러나 `error_mic` X는 `duct.yaml` 주석 자체가 **잠정값**이라고 명시하고, 105 mm 내경과
마이크 위치도 field-verified receipt가 아니다. strict P/S NPZ는 NS/CS→ERR 경로를
발행했지만 canonical NS→REF→ERR advance를 발행하지 않았다.

### [확인 방법]

다음 항을 서로 다른 field로 두고, 값과 receipt SHA가 모두 있으며 하나의 runtime timeline
receipt에 묶였을 때만 margin을 계산한다.

```text
required latency
  = ADC physical-arrival→REF-available latency
  + fixed one-block handoff 256
  + DAC output latency
  + CS→ERR acoustic propagation latency

causal margin
  = measured REF→ERR advance - required latency
```

inference P99는 256-sample handoff 안에 완료되는지를 판정하는 deadline 항이다. handoff에
이미 숨겨져 있으므로 required latency에 다시 더하지 않는다.

### [결과]

| 항 | 현재 값 | 권위/사용 가능 여부 |
|---|---:|---|
| REF→ERR 기하 추정 | 139.9417 samples | config-only, ERR 위치 잠정, canonical 측정 아님 |
| canonical measured REF→ERR advance | 없음 | **BLOCKED** |
| ADC observation latency | 없음 | **BLOCKED** |
| controller handoff | 256 samples | 계약으로 Confirmed |
| canonical ref-only model inference P99 | 없음 | **BLOCKED** |
| DAC output latency | 없음 | **BLOCKED** |
| 분해된 CS→ERR acoustic delay | 없음 | **BLOCKED** |
| strict S calibration delay | 1245 samples | 알려져 있으나 DAC/ADC/버퍼/S가 섞인 calibration 값 |
| 공통 runtime timeline receipt | 없음 | **BLOCKED** |

strict S=1245를 `DAC latency + acoustic S`로 임의 분해하지 않는다. 여기에 ADC latency를
다시 더하면 중복 계상할 수 있고, 반대로 calibration 절대 delay를 그대로 runtime에 옮기면
서로 다른 clock/frame anchor를 숨길 수 있다.

### [판정]

- live broadband/random acoustic-reference: **BLOCKED**
- 주기음 predictive acoustic-reference: **Inconclusive**
- 기하 추정만 사용한 PASS: **Invalid experiment**

주기음은 미래를 예측할 가능성이 있지만 해당 source의 stationarity/predictability와 실제
latency margin을 측정하기 전에는 PASS가 아니다.

### [다음 행동]

APE 두 입력의 동일 clock에서 NS→REF/ERR 상대 TDOA를 canonical raw로 측정하고, REF physical
arrival→callback availability, AB13X output timing, CS→ERR propagation, ref-only model P99를
같은 runtime frame receipt로 묶어야 한다. 이 결과가 나오기 전 acoustic mode로 실제
광대역 random natural sound를 약속하지 않는다.

## 5. 125 Hz~8 kHz phase-only timing budget

동일 진폭의 신호가 위상만 `phi`만큼 틀릴 때 residual ratio는
`2·sin(|phi|/2)`다. 아래 표는 이 이상 조건에서 10/20 dB 상쇄에 허용되는 timing 오차다.
amplitude mismatch, P/S 오차, 비선형, 고차모드, clock slip은 포함하지 않으므로 성능
예측값이 아니라 **필요 해상도**다.

| Center | Exact octave | 10 dB center / upper samples | 20 dB center / upper samples | 중심 mode 수 | 대역 regime |
|---:|---:|---:|---:|---:|---|
| 125 | 88.388–176.777 | 19.407864 / 13.723432 | 6.114099 / 4.323321 | 1 | plane-wave |
| 250 | 176.777–353.553 | 9.703932 / 6.861716 | 3.057050 / 2.161660 | 1 | plane-wave |
| 500 | 353.553–707.107 | 4.851966 / 3.430858 | 1.528525 / 1.080830 | 1 | plane-wave |
| 1000 | 707.107–1414.214 | 2.425983 / 1.715429 | 0.764262 / 0.540415 | 1 | plane-wave |
| 2000 | 1414.214–2828.427 | 1.212991 / 0.857715 | 0.382131 / 0.270208 | 3 | cutoff 교차 |
| 4000 | 2828.427–5656.854 | 0.606496 / 0.428857 | 0.191066 / 0.135104 | 8 | higher-order |
| 8000 | 5656.854–11313.708 | 0.303248 / 0.214429 | 0.095533 / **0.067552** | 22 | higher-order |

20 dB 위상 허용치는 약 5.732°다. 8 kHz octave 상단에서 0.0675518903 sample이라는
값은 모델 목표가 아니라 clock/path/runtime 전체가 충족해야 할 해상도다.

## 6. 1.633 kHz cutoff와 quiet-zone

### [가설]

1.633 kHz 이상에서 단일 ERR point의 null이 단면 전체 quiet-zone을 나타내지 않을 수 있다고
가정한다.

### [근거]

현 config의 105×105 mm, c=343 m/s를 직접 계산하면:

```text
first transverse cutoff = 343 / (2 × 0.105) = 1633.333 Hz
configured rounded value = 1633 Hz
```

직사각 waveguide mode를 비음수 `(m,n)` cutoff로 계산하면 중심 주파수에서 평면파 포함 mode
수는 2 kHz 3개, 4 kHz 8개, 8 kHz 22개다. 2 kHz octave 자체도 1414–2828 Hz라 cutoff를
가로지른다.

### [확인 방법]

point-control과 spatial claim을 분리하고, exact 7 octave 각각에 중앙과 y/z ±2 mm의 최소
5개 ERR 위치 계약을 붙였다.

### [결과]

단일 CS/ERR로 한 지점의 복소 압력을 낮출 가능성은 1.633 kHz 위에도 있다. 따라서
"1.633 kHz 이상 ANC 불가능"이라는 단정은 틀리다. 하지만 단일 point가 줄었다고 단면
quiet-zone을 주장하는 것도 틀리다.

### [판정]

- 1.633 kHz 위 point-control 가능성: **Likely, 미검증**
- 1-point 결과의 quiet-zone 주장: **Contradicted**
- 현 5-point spatial broadband 결과: **BLOCKED**

### [다음 행동]

중앙+4개 offset 위치 각각에서 family×octave OFF/ON과 matched FxLMS를 별도 보존해야 한다.
다섯 위치 중 하나라도 증폭되면 spatial PASS가 아니다.

## 7. 코드 계약과 negative tests

새 모듈은 다음 API를 제공한다.

- `load_current_causality_snapshot()` — YAML/P/S/raw/analysis SHA read-only 검증
- `phase_error_budget()` — 임의 주파수·감쇠 목표의 phase-only sample budget
- `first_transverse_mode_hz()` / `propagating_rectangular_mode_count()`
- `assess_acoustic_reference()` — 동일 timeline receipt가 없으면 margin 미계산
- `build_current_reference_mode_audit()` — digital/acoustic/natural-source 최종 분리

테스트는 다음 회귀를 차단한다.

- bulk metadata를 lead에 사용하는 회귀
- P/S capture/raw/analysis/timing 불일치
- 기하 REF→ERR 추정을 canonical 측정으로 승격
- latency 값만 있고 receipt SHA가 없는 acoustic PASS
- 공통 runtime timeline 없이 서로 다른 측정 숫자를 합산
- inference를 handoff에 포함하고도 다시 더하는 이중 계상
- 음수 margin 또는 256-sample deadline 초과를 허용
- 단일 point를 5-point quiet-zone으로 승격
- 8 kHz center만 보고 octave 상단 11.314 kHz budget을 누락

현재 read-only audit digest:

```text
c8ed8b01b1df74d8c3fed80bbff961cac94cf4e0eb510d5e62015695810db35b
```

## 8. 최종 판정

### 현재 가장 강하게 말할 수 있는 것

- strict P/S의 동일 capture와 Stage-1 lead 115 samples는 실제 파일에서 재검증됐다.
- Jetson이 미래 playback을 소유하는 digital-reference timing은 조건부 인과적이다.
- 현 strict plant 권위는 150–1600 Hz이며 최종 125 Hz~8 kHz 성능 근거가 아니다.
- 105 mm config geometry에서 첫 횡모드 계산값은 1633.333 Hz다.

### 가능성이 높지만 아직 확정할 수 없는 것

- exact output-clock-master/ref-only 통합이 되면 digital playback의 ADC pacing 문제를 제거할
  가능성이 높다.
- 1.633 kHz 위에서도 한 ERR point의 감쇠는 가능할 수 있다.

### 현재 증거로 말할 수 없는 것

- live upstream acoustic-reference가 광대역 speech/music/random noise를 인과적으로 상쇄한다.
- 2/4/8 kHz에서 실제 감쇠하거나 FxLMS보다 낫다.
- 단일 ERR point의 고역 감쇠가 105×105 mm 단면 quiet-zone이다.

전체 broadband deployment 판정은 **BLOCKED**, physical attenuation PASS는 **False**다.
