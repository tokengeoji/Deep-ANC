# POSTECH CNN ANC PDF 원문 증거 감사

> 감사일: 2026-08-28
> 범위: 사용자가 지정한 로컬 PDF 두 개의 read-only 원문 대조
> 판정 범위: 논문 결과가 현행 105×105 mm 덕트와 125 Hz~8 kHz 목표에 무엇을
> 증명하거나 증명하지 못하는지 구분

## 1. 원본 식별

| 파일 | SHA-256 | 실제 성격 |
|---|---|---|
| `/home/capston/Documents/POSTECH CNN ANC 덕트 구현.pdf` | `7fdab5411db7202f6200695334f8e83540a3f84af0f17702ecbec10f7638482b` | 20쪽 구현 지시서. PDF metadata의 Author/Creator가 `ChatGPT Canvas`/`ChatGPT`이며 논문 원문이 아님 |
| `/home/capston/Documents/포항공대,헤드폰.pdf` | `e97f332ebea7f5611b3b7106c50d59bb031c6d4daaa206e0a16ef00971ed67df` | POSTECH 전기전자공학과 Young-Jae Jang의 2022년 학위논문, *A Convolutional-Neural-Network Feedforward Active-Noise-Cancellation System on FPGA for In-Ear Headphone* |

첫 파일의 숫자는 둘째 파일을 해석해 만든 요구사항이므로 독립 성능 증거로 세지 않는다.
아래 수치는 둘째 파일의 본문에서만 취했다.

## 2. [가설]

논문의 14.3~14.8 dB 결과가 현재 덕트에서 2/4/8 kHz까지 Deep-ANC가 감쇠할 수 있다는
직접 근거일 가능성이 있다고 가정한다.

## [근거]

논문 초록과 5장의 실제 측정 결과에는 다음 수치가 있다.

- 0°/90° 입사에서 total power reduction 14.8/14.3 dB
- attenuation bandwidth 2,000 Hz
- 300 Hz 아래에서 20 dB 초과 감쇠
- FxLMS 기반 선행 결과보다 total power와 bandwidth가 우수하다는 비교

## [확인 방법]

측정 자극, 구조, reference mode, latency, 학습 분포, bandwidth 정의를 본문의 모델·학습·
simulation·measurement 절에서 다시 대조했다.

## [결과]

논문의 물리 실험은 현재 덕트와 다른 조건이다.

| 항목 | POSTECH 논문 | 현행 `Deep_ANC` 최종 목표 |
|---|---|---|
| 구조 | 사람 귀에 장착한 in-ear headphone | 105×105 mm, 길이 1.190 m 사각 덕트 |
| reference | 외부 reference mic가 먼저 관측 | 1차 digital playback, 최종 live acoustic-reference도 별도 필요 |
| sample rate | 46,875 Hz | 48,000 Hz |
| controller latency | 8 samples, 170.6 μs | 256-sample handoff와 실제 P/S·ADC/DAC/runtime 계약 |
| 물리 자극 | 100–2,000 Hz band-limited pink noise | 88.388–11,313.708 Hz 식별, 125/250/500/1k/2k/4k/8k octave |
| 물리 고역 결과 | ANC ON/OFF spectrum의 crossover가 2 kHz | 2/4/8 kHz에서 양의 감쇠와 matched FxLMS 우위 |
| 공간 | 귀 내부 error mic 한 위치 | 1.633 kHz 위 최소 5개 ERR 위치 |

논문에서 attenuation bandwidth는 ANC ON/OFF error spectrum의 **교차 주파수**로 정의된다.
즉 2 kHz까지 모든 대역에서 큰 감쇠라는 뜻도, 2 kHz 위를 측정했다는 뜻도 아니다. 실제
자극 자체가 100~2,000 Hz였으므로 4/8 kHz 근거가 될 수 없다. 본문은 ANC OFF에서도
헤드폰 기계 구조 때문에 약 300 Hz 위가 이미 감쇠한다고 설명한다. 따라서 total power
14.8/14.3 dB를 controller 단독의 broadband 감쇠로 해석해서도 안 된다.

## [판정]

**Contradicted** — 이 논문은 저지연 CNN feedforward ANC가 인이어 헤드폰에서 2 kHz
crossover를 만들 수 있다는 사례지만, 현재 덕트의 4/8 kHz 감쇠를 입증하지 않는다.

## [다음 행동]

논문 수치를 목표 prior로 복사하지 않는다. 현행 v3 P/S, 같은 덕트·볼륨·source·window의
Deep-ANC/FxLMS A/B, 5개 ERR 위치 raw만 최종 증거로 사용한다.

## 3. [가설]

3,232-parameter 논문 CNN을 그대로 사용하면 현행 tiny보다 고역 ANC가 좋아질 가능성이
있다고 가정한다.

## [근거]

논문 모델은 작은 parameter 수에도 327 ms의 긴 과거를 보고, FxLMS보다 넓은 2 kHz
bandwidth를 보고했다.

## [확인 방법]

3.2절의 구조와 latency, 3.3절의 학습 분포를 현행 `HybridANCNet-tiny`와 비교했다.

## [결과]

논문 모델의 확인 가능한 구조는 다음과 같다.

- 10-layer dilated causal 1-D CNN
- layer마다 16 kernels, kernel size 16, dilation `2^(n-1)`
- 각 layer의 fully connected 연산과 residual/skip 구조
- 마지막 512-tap FIR
- `(256+16)×10+512 = 3,232` parameters
- receptive field 15,355 samples, 약 327 ms
- FPGA가 3 samples를 한 단위로 처리하며 CNN processing 47.5 μs
- input collection을 포함한 system latency 8 samples, 170.6 μs

학습은 60시간의 일상 소음과 0~2,000 Hz random sine을 사용했다. 긴 과거는 주기·준주기
신호의 미래 예측에 유리할 수 있지만, 완전히 random한 광대역 고역의 미래를 만들어내지는
못한다. 또한 논문의 지연은 현행 USB DAC/APE ADC/256-block 지연보다 두 자릿수 이상 작다.
구조만 복사해도 현재 timing causality가 해결되는 것이 아니다.

현행 tiny는 약 1.16M parameters, 약 0.16초 TCN 수용영역과 GLSTM을 사용하고 Jetson ORT
deadline 여유를 우선한다. 현재 canonical v3 checkpoint와 동일 P/S physical A/B가 없으므로
두 구조 중 ANC attenuation 우승자를 말할 수 없다.

## [판정]

**Inconclusive** — 논문형 CNN은 유용한 offline baseline 후보지만 현재 모델을 교체할
근거는 없다. parameter 수나 receptive field만으로 고역 감쇠를 예측할 수 없다.

## [다음 행동]

먼저 현행 tiny로 v3 causal G0·validation·physical runtime을 통과시킨다. 이후 같은 seed,
P/S, 데이터, loss, source, latency 조건에서 논문형 dilated CNN을 offline baseline으로
학습하고 실제 deadline과 octave attenuation을 함께 비교한다. 모델 구조 비교가 plant·lead·
seed 차이를 대신하지 못하게 한다.

## 4. [가설]

논문이 Deep-ANC의 비선형 고역 우위를 matched FxLMS보다 이미 입증했을 가능성이 있다고
가정한다.

## [근거]

논문은 ReLU CNN으로 primary/acoustic secondary path를 사전 모델링하고, CNN controller가
FxLMS 선행 결과보다 우수하다고 설명한다.

## [확인 방법]

path model, controller training, 물리 비교 조건을 분리해 확인했다.

## [결과]

ReLU를 썼다는 사실은 path의 비선형 표현 가능성을 제공하지만 실제 시스템의 어느
비선형 항에서 얼마만큼 이득이 났는지를 분해하지 않는다. 물리 실험은 100~2,000 Hz pink
noise이며, 현재 덕트의 같은 source/SPL/P/S/lead/window에서 튜닝된 FxLMS와 CNN을 교대
측정한 raw가 아니다. 논문의 table comparison은 현재 프로젝트의 matched physical A/B를
대체할 수 없다.

## [판정]

**Inconclusive** — CNN feedforward ANC의 가능성은 보여주지만 현재 덕트의 고역·비선형
FxLMS 우위는 독립적으로 다시 입증해야 한다.

## [다음 행동]

현행 최종 평가는 같은 실제 source와 볼륨에서 OFF→FxLMS→Deep-ANC를 동일 window로 기록하고,
2/4/8 kHz 각 octave의 평균·worst10·paired cluster-bootstrap CI를 계산한다. THD/IMD로 실제
plant 비선형성이 측정된 경우에만 "비선형 우위"라는 원인 해석을 허용한다.

## 5. 현행 설계에 채택할 것과 채택하지 않을 것

### 채택

- 미래 sample 없이 긴 과거만 보는 causal streaming 원칙
- primary, acoustic secondary, electrical/runtime delay를 분리하는 원칙
- 작은 모델도 실제 deadline과 수용영역이 맞으면 유효한 후보라는 점
- 물리 ANC ON/OFF raw spectrum으로 bandwidth를 판정하는 방식

### 채택하지 않음

- 14.8/14.3 dB를 현재 덕트 예상 감쇠로 사용
- 2 kHz crossover를 4/8 kHz 감쇠 증거로 사용
- 3,232 parameters라는 숫자만으로 tiny를 교체
- FPGA 170.6 μs를 Jetson+USB+APE runtime latency로 사용
- 다른 구조·source·SPL의 FxLMS 비교를 matched physical A/B로 사용

최종 결론은 단순하다. 논문은 "긴 과거와 매우 낮은 전기 지연을 가진 CNN ANC가 2 kHz
부근까지 확장될 수 있다"는 설계 근거다. 현재 프로젝트의 더 어려운 4/8 kHz 덕트 목표는
v3 plant, 0.0676-sample급 clock 증거, 다점 raw, matched FxLMS A/B로 새로 증명해야 한다.
