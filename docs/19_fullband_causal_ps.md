# Fullband causal P/S 식별 계약

## v1 판정

기존 32,768/30,720-sample periodic prototype은 diagnostic-only다. periodic
deconvolution과 geometric remainder로 finite causal support를 증명할 수 없으므로 어떤
입력에서도 `canonical_training_eligible=False`다.

## 역할

이 측정은 기존 100–11.314 kHz multi-panel 측정을 대체하지 않는다.

- multi-panel: 제어 대역의 실제 복소 전달함수·위상·일관성 authority
- fullband causal: `d=P*n`, `e=d+S*y`에 사용할 causal history, prefix, support,
  tail authority

두 증거가 각 일곱 부대역에서 agreement와 relative-error gate를 함께 통과해야만
광대역 학습 plant로 사용할 수 있다. partial-band compact FIR을 fullband causal plant로
승격하는 것은 금지한다.

## Signal-only 계획

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal.py --dry-run
```

현재 `FULLBAND_CAUSAL_LIVE_AUTHORITY=None`이므로 `--execute-live`는 오디오 모듈 import와
장치 open 이전에 실패한다. 실제 출력은 plan file SHA, canonical payload SHA, submitted
PCM SHA를 검토해 authority로 고정하는 별도 변경 전에는 실행할 수 없다.

계획은 다음과 같다.

| 항목 | 값 |
|---|---:|
| sample rate / block | 48 kHz / 256 |
| fit period | 32,768 samples, 1.46484375 Hz grid |
| holdout period | 30,720 samples, 1.5625 Hz grid |
| fit | warmup 1 + analysis 16 repeats/path |
| holdout | warmup 1 + analysis 8 repeats/path |
| peak / RMS / crest | PCM 98 / 0.00140–0.00143 / 6.3–6.7 dB |
| 총 stream | 37.232초 |
| active slot | 34.731초, path별 약 17.365초 |

P-only와 S-only slot에서 반대 DAC channel은 exact zero다. fit과 holdout은 모든 실제
int16 rFFT bin이 nonzero이고 condition number가 1.10 이하여야 한다. holdout의 100–
11312.5 Hz 중 fit grid와 8 Hz panel grid에 없는 6,646개 tone은 별도로 평가한다.

## Offline gate

offline analyzer는 raw ADC index를 그대로 받지 않는다. callback의 DAC/ADC timestamp와
repeat clock witness로 absolute DAC sample `q`에 resample된 response 및
`absolute_dac_q_timewarp_v1` receipt가 필요하다.

- fit valid repeat ≥12/16, holdout ≥6/8
- adjacent score ≥0.995, sample slip 0
- 11.314 kHz의 20 dB-grade clock residual ≤0.0675519 sample
- causal delay branch 0–4800 samples
- candidate post-onset support 1024, 2048, 4096 samples
- repeat-vector bootstrap tail L1 95% upper ratio ≤0.03
- tail L1로 계산한 heldout induced-output upper ratio ≤0.03
- 마지막 tail block의 감쇠 상한이 수렴하지 않으면 BLOCKED
- 각 path final 0.1초가 input-only noise floor의 ±1 dB 이내
- 일곱 부대역 및 off-grid complex agreement ≥0.995, relative error ≤0.10
- 기존 panel raw/analysis SHA·capture ID·repeat consistency ≥0.95를 결속하고, actual
  tone에 FIR DTFT를 직접 계산하여 같은 gate 적용

가장 짧은 candidate가 모든 조건을 통과할 때만 선택한다. 4096도 실패하면 임계값이나
ridge를 완화하지 않고 더 긴 support/capture가 필요하다.

P operator에는 thread handoff가 없다. S FIR에도 256 samples를 bake하지 않으며 timing
contract가 handoff를 정확히 한 번 적용해야 한다.

## v2 aperiodic zero-tail 계약

v2는 길이가 서로 다른 `fit_a=131072`, `fit_b=130816`, `holdout=130560` 단발 burst를
사용한다. 세 burst는 seed도 다르며 sign/swap 반복이 아니다. 각 burst 직후 65,536
samples(1.365초)의 exact-zero guard를 둔다. P/S 전체 stream은 25.045초, active output은
16.352초, zero guard는 8.192초다.

목표 자극 대역은 100–11.314 kHz이고 64–100 Hz와 11.314–14 kHz에는 식별 안정성만을
위한 -20 dB skirt가 있다. DC/Nyquist는 canonical identification에서 제외한다. 실제
int16 burst의 peak는 98 이하, RMS는 0.00140–0.00143, target-band condition은 1.10
이하여야 한다.

v2 검증은 `n_fft >= len(x)+len(h)-1`을 강제하는 linear FFT convolution만 허용한다.
zero guard 전체의 residual RMS/L1/induced peak가 각각 active response의 3% 이하여야 하며,
guard 마지막 16,384 samples 전체가 input-only noise floor +1 dB 이하여야 한다. geometric
tail extrapolation은 사용하지 않는다. 16k samples 뒤 delayed echo와 delay를 taps에도 넣는
double-delay fixture는 모두 BLOCKED여야 한다.

현재 v2 production raw/time-warp publisher와 live authority는 아직 없다. 따라서 synthetic
known-FIR 검증이 PASS해도 canonical 학습 자격은 False다.

## v3 reserved-pilot 시간축 계약 — 현재 signal-only

v3는 65,536-sample causal PE burst의 152--600 Hz를 비우고, 그 자리에 실제 int16로
양자화되는 6,000-sample 저역 pilot을 중첩한다. clock 분석은 고역 응답을 보지 않으며,
각 분석창의 **actual submitted total int16 spectrum**만 분모로 사용한다. intended-float
pilot이나 결과에 맞춘 high-band phase repair는 사용할 수 없다.

기존 방식처럼 한 6,000-sample ADC 창을 rigid shift로 근사하지 않는다. 413.931 ppm이면
한 창 안에서도 약 2.48 samples가 늘어나거나 줄어든다. v3 estimator는 후보 clock ratio마다
submitted DAC-q 좌표 전체를 ADC 좌표로 되사상한 뒤 다음을 독립적으로 검사한다.

- fit_a/fit_b의 짝수 cycle만 clock ratio fit에 사용
- 홀수 cycle과 holdout은 leave-out validation에만 사용
- ERR/REF × P/S 네 view가 캡처 전체 0.05 sample 이내의 같은 ratio에 동의
- linear fit과 cubic interpolation crosscheck 차이 0.006 sample 이하
- actual int16 pilot로 다시 계산한 role/cycle 전달함수의 coherence 0.995 이상
- 0.1-sample trajectory step, one-sample slip, low-SNR, marker alias는 즉시 거부
- high-band plant response를 바꾸어도 clock map은 바뀌지 않아야 함

각 262,144-sample exact-zero tail 뒤에는 3주기 저역 clock anchor와
`maximum_delay + maximum_support` 길이의 응답 guard를 둔다. anchor는 causal candidate
데이터에서 제외한다. 이 보강 뒤 전체 signal 길이는 약 47.061초이며 peak는 PCM 98이다.

다만 exact-zero tail 내부에서 발생했다가 두 endpoint 사이에서 상쇄되는 비선형 clock
excursion은 음향 raw만으로 정보론적으로 관측할 수 없다. callback time_info는 monotonic/slip
witness일 뿐 sub-sample clock authority가 아니다. 따라서 electrical loopback 같은 독립
연속 witness가 생기기 전까지 `LIVE_AUTHORITY=None`, `canonical_training_eligible=False`를
유지한다. affine synthetic fixture의 PASS를 실측 권위로 승격하지 않는다.
