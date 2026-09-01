# Stage-2 2 kHz same-capture P/S 측정 경로

## 현 장비 판정

2026-08-31 read-only `/proc/asound` inventory는 다음과 같다.

- AB13X USB Audio playback: S16_LE, 48 kHz, 2채널, adaptive endpoint
- AB13X USB Audio capture: S16_LE, 48 kHz, 1채널, async endpoint
- APE capture: S32_LE, 48 kHz, 2채널 ERR/REF

현 장비는 Stage-2의 **single-point P/S 상대 plant와 lead**를 조건부로 식별할 수 있다.
두 AB13X 출력에 알려진 독립 aperiodic code를 동시에 보내고 APE ERR/REF를 한 캡처로
받으면, 두 입력×두 출력 fixed-LTI와 capture 전체에 공통인 affine `q` 하나를 fit-a/fit-b로
교차 검증할 수 있다. playback→ERR의 고정 음향지연은 제거할 nuisance가 아니라 각 P/S
plant delay에 포함한다.

acoustic raw만으로 `실제 ADC/DAC q`와 두 microphone에 공통인 시간가변 음향지연의 의미는
구분할 수 없다. 그러나 둘은 같은 time gauge로 작용하며 P-S 상대지연에서 소거된다. 따라서
다음 범위를 명시적으로 나눈다.

| 주장 | 현 장비에서 가능 여부 |
|---|---|
| single-point P/S 상대 FIR·상대 delay·`PlantDelays.lead()` 입력 | 엄격한 cross-fit을 통과하면 가능 |
| absolute DAC/ADC hardware-frame identity | 불가 |
| callback 시작 전 drop 및 hardware counter 기준 slip=0 | 불가 |
| 다점 quiet-zone 또는 4/8 kHz ANC 성능 | 이 측정만으로 주장 불가 |

AB13X mono async 입력은 APE ERR/REF와 같은 capture frame이 아니므로 absolute-frame 증거로
쓰지 않는다. 반대로 이 한계만으로 P-S 상대 lead 측정을 거부하지도 않는다.

## 무음으로 봉인된 경로

`src/deep_anc/dsp/stage2_2khz_measurement.py`는 오디오 backend를 import하지 않고 다음을
고정한다.

- exact 48 kHz / block 256
- meter 20초 + signal 24초 = 최대 audible 44초
- fit-a / fit-b / untouched transfer holdout 각 8초
- NS/CS 독립 PCG64 aperiodic code를 signal 전체 24초 동안 동시에 제출
- actual submitted interleaved int16 dtype/shape/bytes SHA-256
- 88.388--150, 150--300, 300--600, 600--1000, 1000--1600,
  1600--2828.427 Hz의 여섯 물리 식별 구간
- full-capture 88.388--600 Hz known-code likelihood에서 shared affine `q`를 선선택
- P/S×ERR/REF 네 view 모두 같은 `q`; view별 `q`와 high-band residual repair 금지
- P/S×ERR/REF×6 = 24개 row의 fit-a↔fit-b 및 untouched-holdout consistency 0.95 이상
- 모든 row response-to-noise 20 dB 이상
- timing residual과 q ambiguity envelope 각각 0.270208 sample 이하
- 256-frame callback 연속성은 software transport 증거로 검사하고, acoustic shared-q는
  비중첩 2초 epoch의 모든 경계에서 nonaffine/change-point/1-sample insert-drop을 검사
- fit-a/fit-b/untouched-holdout마다 비중첩 1초 independent epoch 8개(총 24개)를 실제
  known-code likelihood로 판정하며 periodic repeat index를 만들지 않음
- xrun/clip/callback-status 및 signal detector 기준 slip/drop/add exact 0
- raw/analysis no-replace, raw SHA와 plan/actual PCM SHA 결박
- holdout은 shared-q nuisance의 사전 선언된 full-capture likelihood에만 포함하며 FIR, delay,
  support, threshold 선택에는 사용하지 않음
- holdout 실패 뒤 refit 금지

위의 signal detector `slip=0`을 hardware counter의 절대 slip=0으로 다시 표기할 수 없다.
receipt는 absolute frame identity/callback-before-start/hardware-counter 주장을 모두 false로
강제한다.

## live 실행과 authority 범위

기존 v1 80--2828 Hz 자극은 unrestricted 1024-tap actual-int16 Gram condition이 약
`1e6`으로 실패한다. near-Nyquist near-white v2는 2048×2048 Gram을 통과하지만 제조사
근거 없는 speaker/amp에 출력할 수 없다. 따라서 현 CLI는 80--2828 Hz time-separated PE를
112-DOF/path DPSS subspace로 제한한 signal-only fallback만 무음 검증하고,
`--execute-live`와 `--write-plan`을 audio import 전에 무조건 BLOCK한다.

DPSS basis SHA를 physical raw analyzer, P/S artifact, training/eval consumer가 모두 소비하고
actual-int16 projected Gram rank/condition, untouched holdout, level/route/THD preflight를
통과하기 전에는 기존 v1 adapter나 analyzer를 새 plan에 재사용하지 않는다.

config의 `canonical_training_eligible=false`는 아직 raw/analysis/admission PASS가 없기
때문이다. absolute frame clock 부재 자체가 relative P/S training admission을 영구 차단한다는
뜻이 아니다. typed admission은 이 문서의 24-row·shared-q·holdout·nonaffine gate를
모두 통과한 immutable native raw에서 재계산된 결과만 relative plant 후보로 연다.

독립 전기 tap 또는 APE shared-frame 출력은 absolute hardware clock 주장이 필요할 때의
선택적 강화 경로다.

1. NS/CS를 APE I2S1/RT5640으로 보내고 APE I2S2 ERR/REF와 동일 hardware frame임을 봉인
2. USB DAC를 유지하면서 ERR/REF와 같은 capture clock의 세 번째 전기 playback tap 추가

## 무음 실행 명령과 출력 경로

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz.py --dry-run
```

이 명령은 24초 PCM을 메모리에서 렌더링하고 DPSS projected Gram과 live BLOCK을 출력하지만
파일을 쓰거나 audio backend를 import하지 않는다. 다음 명령은 현재 의도적으로 exit 2다.

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz.py --write-plan
```

아래 v1 출력 경로는 diagnostic legacy이며 새 physical artifact 발행 대상으로 쓰지 않는다.

- plan: `results/stage2_2khz_ps_v1/measurement_plan.json`
- immutable native raw: `results/stage2_2khz_ps_v1/native_raw_capture.npz`
- production raw: `results/stage2_2khz_ps_v1/raw_capture.npz`
- production analysis: `results/stage2_2khz_ps_v1/analysis.npz`
- P/S: `results/stage2_2khz_ps_v1/{primary,secondary}_path_candidate.npz`
- final binding commit marker: `results/stage2_2khz_ps_v1/plant_binding.json`

아래 명령은 현재 소리를 내지 않고 exit 2여야 한다. 따라서 현 단계 예상 audible은 0초다.
향후 reviewed v2 통합이 완료돼도 별도 승인된 meter20초+signal24초, 최대44초 계약을 새로
발행하기 전에는 이 문서만으로 출력 authority를 주지 않는다.

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz.py --execute-live \
  --confirm-user-present --confirm-volume-fixed --confirm-routing-and-geometry
```

이 결과는 **125 Hz 옥타브부터 2 kHz 옥타브까지의 단일 ERR 지점 P/S/lead 후보**다.
1.633 kHz 위 PASS를 spatial quiet-zone PASS로 승격하지 않으며, 실제 ANC는 별도 physical
source-valid 평가에서 1.6 kHz sentinel 최소 6 dB와 2 kHz 옥타브 증폭 방지(감쇠 >0 dB)를
통과해야 한다.
