# Jetson 실시간 ADC/DAC clock-domain 감사

> 감사 시각: 2026-08-28 KST
> 범위: Jetson에 이미 존재하는 설정·장치 descriptor·raw receipt·실시간 코드의
> read-only 감사
> 금지한 작업: PCM 장치 open, 음향/전기 출력, 시스템 설정 변경,
> `DeepANC_CRN_n_codex` 접근·변경

> **2026-08-29 정정:** 이 문서의 외부 I2S2 DAC 우선 결론은 온보드 RT5640을 누락한
> 당시 감사 결과다. 후속 실제 장치 감사에서 `/sys/bus/i2c/devices/8-001c`의 RT5640,
> I2S1↔ADMAIF1 route, `hw:APE,0` playback, I2S1/I2S2의 공통 `PLL_A_OUT0` 계열을
> 확인했다. 따라서 1차 후보는 `I2S2 마이크 입력 + I2S1/RT5640 출력`이고 외부 I2S2
> DAC는 2차 fallback이다. 다만 J511 배선·동시 duplex·sample-slip 0은 아직 미확인이라
> shared-clock authority는 계속 BLOCK이다. 현행 설계는 [40](40_fullband_v7_nonaffine_clock.md)
> §5를 우선한다.

## 결론

현재 `APE hw:1,1` 입력과 `AB13X hw:2,0` 출력은 **서로 다른 PCM/clock
domain**이다. strict 측정에서 상대 rate가 `413.9314767411314 ppm`, 즉 48 kHz
기준 `19.86871088357431 samples/s`로 관측됐다. 현재 실시간 코드는 이 rate
차이를 ASRC/rate matcher로 보정하지 않고, PortAudio가 callback 전에 버린 입력
period도 계수하지 못한다. 따라서 다음 두 문장은 동시에 참이다.

1. NS/CS 두 출력은 같은 AB13X DAC에 있으므로 `413.931 ppm`을 곧바로
   NS 대 CS의 연속 상대 위상 오차라고 해석하면 안 된다.
2. APE의 ERR/REF를 읽는 실시간 모델의 시간축·상태·lead가 장시간 연속이라는
   증거는 없으며, 현 상태에서 2 kHz 이상 실시간 ANC authority는 **BLOCK**이다.

| 게이트 | 현장 판정 | 이유 |
|---|---|---|
| 실제 장치 topology 식별 | **PASS / Confirmed** | APE capture와 AB13X playback이 별도 ALSA card |
| USB endpoint sync type 식별 | **PASS / Confirmed** | OUT adaptive, IN asynchronous, explicit feedback 없음 |
| strict capture의 clock-rate 증거 | **PASS / Confirmed** | raw/metadata SHA와 `413.931 ppm`, xrun 0 보존 |
| AB13X capture의 독립 loopback authority | **BLOCK / Inconclusive** | mono mic endpoint만 있고 internal loopback·공통 oscillator 증거 없음 |
| 현 배선의 APE full-duplex 공통 clock | **BLOCK / Not installed** | I2S2 DIN은 있으나 pin 40 DOUT과 외부 2ch I2S DAC가 없음 |
| 실시간 rate matching·sample-slip 관측 | **BLOCK / Contradicted** | ASRC 없음, `_time_info` 폐기, PortAudio silent input-period drop 가능 |
| 2–11.3 kHz 실시간 timing authority | **BLOCK** | 20 dB급 허용 오차보다 작은 continuous witness가 없음 |
| 연속 acoustic pilot만으로 독립 clock authority | **BLOCK / Conditional only** | LTI plant 가정과 clock map이 서로를 설명할 수 있어 독립적이지 않음 |

## 1. 실제 topology

### [가설]

Jetson I2S ADC와 USB DAC가 같은 48 kHz 설정을 사용하더라도 실제 sample clock은
공유하지 않을 가능성이 있다고 가정한다.

### [근거]

- `configs/hardware_jetson.yaml` byte SHA-256:
  `45232a45e51fd76c7b88db338b9cf4f3840a88299b4d452e259064c0ee559351`
- `/proc/asound/cards`: card 1 `APE`, card 2 `Audio` = `AB13X USB Audio`
- `/proc/asound/pcm`: `01-01` APE ADMAIF2 capture/playback, `02-00` USB
  capture/playback
- strict raw:
  `results/calibration_interleaved/strict_20260827/20260827_112608_5ac13134/raw_measurement.npz`
  - SHA-256:
    `31d563b163fe7dcb3f6b85e30e491a6775947e7f1b988690c3668fd13464b347`
- strict acquisition metadata:
  `results/calibration_interleaved/strict_20260827/20260827_112608_5ac13134/metadata.json`
  - SHA-256:
    `08aff6e3f2dcedf3d582989bc18819799d6c2a9cac1209ab472b86eaaa5b3574`
  - input PCM 1, output PCM 0, 48 kHz, 600,000 frames
  - callback status 0, xrun 0, output elapsed `12.521571584999947 s`
- strict analysis metadata:
  `results/calibration_interleaved/strict_20260827/20260827_112608_5ac13134/analysis_metadata.json`
  - SHA-256:
    `bdd64d28bc23e3686dcf9056bf04f76ba7660569c2079f5434bdd02425fc7bd3`
  - `drift_ppm=413.9314767411314`
  - `drift_samples_per_period=2.483588860446788`, period `0.125 s`

### [확인 방법]

다음 명령만 사용했다. PCM은 열지 않았다.

```bash
cat /proc/asound/cards
cat /proc/asound/pcm
cat /proc/asound/card2/stream0
lsusb -v -d 001f:0b21
fuser -v /dev/snd/*
cat /proc/asound/card*/pcm*/sub*/status
sha256sum configs/hardware_jetson.yaml \
  results/calibration_interleaved/strict_20260827/20260827_112608_5ac13134/{raw_measurement.npz,metadata.json,analysis_metadata.json}
```

감사 시 모든 PCM status는 `closed`였고 PulseAudio는 `controlC0`과 `controlC2`
control node만 열고 있었다.

### [결과]

실제 stream map은 다음과 같다.

```text
INMP441 × 2
  └─ I2S2 DIN / Jetson APE card 1, PCM 1 / S32_LE / 2 ch / 48 kHz
       └─ ERR ch0, REF ch1

Jetson USB host
  └─ AB13X card 2, PCM 0 / S16_LE / 2 ch / 48 kHz
       ├─ OUT ch0: noise speaker
       └─ OUT ch1: cancel speaker
```

같은 명목 sample rate는 공통 oscillator를 뜻하지 않는다. 보존 raw가 그 차이를
직접 관측했다. 진단 capture들의 rate도 `203.760–654.321 ppm` 범위였으므로 하나의
고정 상수로 영구 hard-code할 수 없다.

### [판정]

**Confirmed.** 현재 기본 실시간 경로는 별도 clock-domain이다.

### [다음 행동]

실시간 receipt에 ADC/DAC 절대 sample counter, callback `time_info`, ring occupancy,
rate estimate, ASRC ratio, 모든 discontinuity를 저장하기 전에는 장시간 실시간 ANC를
canonical로 발행하지 않는다.

## 2. AB13X USB endpoint와 capture loopback 후보

### [가설]

AB13X의 capture endpoint를 playback clock witness로 사용할 수 있을 가능성이 있다고
가정한다.

### [근거]

`lsusb -v -d 001f:0b21`과 `/proc/asound/card2/stream0`의 실제 descriptor는 다음과
같다.

| 방향 | Endpoint | 형식 | Sync type | 채널 | 명시적 sync address |
|---|---:|---|---|---:|---:|
| playback | `0x03 OUT` | S16_LE, 48 kHz | **Adaptive** | 2 | 0 |
| capture | `0x83 IN` | S16_LE, 48 kHz | **Asynchronous** | 1 | 0 |

AudioControl topology는 playback의 `USB Streaming → Feature Unit → Speaker`와
capture의 `Microphone → Feature Unit → USB Streaming` 두 체인뿐이다. 내부 mixer,
selector 또는 playback-to-capture loopback terminal이 없다. 장치 serial은 strict
metadata에 `20210926172016`, VID:PID는 `001f:0b21`로 결속돼 있다.

### [확인 방법]

descriptor의 terminal ID/source ID와 endpoint `bmAttributes`, `bSynchAddress`를
직접 추적했다. `ASYNC capture`라는 이름만으로 DAC와 ADC가 같은 oscillator라고
가정하지 않았다.

### [결과]

capture endpoint 자체는 존재하지만 다음이 확인되지 않았다.

- DAC와 ADC가 같은 crystal/PLL을 공유하는지
- mic input에 line-level 신호를 넣어도 안전한지
- mic bias 전압, 입력 임피던스, clipping headroom
- driver가 playback/capture sample index의 관계를 보존하는지

외부 전기 tap을 쓰려면 AB13X 출력 중 하나를 **power amplifier 앞에서** 뽑아
고임피던스 감쇠기와 DC blocking을 거쳐 mono mic input으로 넣어야 한다. 출력 두
채널은 이미 NS/CS에 모두 사용되므로 항상 켜진 전용 pilot 채널도 남아 있지 않다.
현재 `sounddevice.Stream`은 APE 입력 하나만 열므로 AB13X capture를 함께 기록할
3-stream 구조도 없다.

### [판정]

**Inconclusive / BLOCK.** AB13X capture는 유망한 **진단 후보**이지, 현재 clock
authority가 아니다. endpoint 이름이나 sync type만으로 공통 clock을 선언할 수 없다.

### [다음 행동]

물리 부품과 사용자 승인을 받은 뒤, 스피커·파워앰프를 분리한 electrical-only
loopback에서 다음을 먼저 증명한다.

1. DC bias와 최대 입력 peak를 계측하고 감쇠비를 확정한다.
2. AB13X playback/capture sample ratio를 60초 이상 측정한다.
3. APE와 AB13X capture가 각각 같은 submitted PCM을 볼 때 one-sample slip을 전부
   기록한다.
4. AB13X capture가 playback과 실제 공통 clock임이 입증될 때만 DAC-q witness로
   승격한다.

## 3. APE/I2S full-duplex 공통 clock 후보

### [가설]

I2S2에 2채널 외부 DAC를 추가하면 ADC와 DAC를 한 APE sample clock에 둘 수 있을
가능성이 있다고 가정한다.

### [근거]

- `amixer -c APE cget name='ADMAIF2 Mux'`: `values=18` = I2S2
- `amixer -c APE cget name='I2S2 Mux'`: `values=2` = ADMAIF2
- active DT:
  `/proc/device-tree/bus@0/aconnect@2900000/ahub@2900800/i2s@2901100`
  - `status=okay`, `assigned-clock-rates=1536000`, prefix `I2S2`
- active pinmux:
  `/proc/device-tree/bus@0/pinmux@2430000/exp-header-pinmux`
  - pin 12 = I2S2 BCLK
  - pin 35 = I2S2 FS/WS
  - pin 38 = I2S2 DIN
  - **pin 40 DOUT node 없음**
- 이 JetPack source의 NVIDIA UDA1334A overlay:
  `/home/capston/FxLMS/kernel_work_r36_4_4/source/r36_4_0_rt/hardware/nvidia/tegra/nv-public/overlay/jetson-audio-adafruit-uda1334a.dtsi`
  는 pin 12/35/40을 I2S로 구성하고 pin 40을 output으로 둔다.
- 현재 boot entry:
  `/boot/extlinux/extlinux.conf`의 `real-time`이
  `/boot/dtb/kernel_tegra234-p3737-0000+p3701-0005-nv-user-custom.dtb`를 사용한다.

### [확인 방법]

active DT property와 mixer route를 읽고 NVIDIA가 제공한 동일 보드용 overlay와
대조했다. pinmux나 boot file은 변경하지 않았다.

### [결과]

APE PCM 1은 논리적으로 capture/playback 양방향을 제공하고 I2S2도 같은 audio PLL
계열을 사용할 수 있다. 그러나 현재 물리 구성에는 DOUT pinmux와 2채널 DAC가 없다.
따라서 지금 명령만 바꿔 공통-clock playback을 얻을 수는 없다.

### [판정]

**Likely feasible, currently not installed.** 세 후보 중 고주파 실시간 ANC에 가장
강한 구조이지만 현재 상태는 **BLOCK**이다.

### [다음 행동]

되돌릴 boot entry/DTB 백업을 먼저 만들고, pin 38 DIN을 보존한 채 pin 40 DOUT을
추가하며, 3.3 V I2S slave 2채널 DAC를 BCLK/WS/DOUT에 연결한다. 이후에는 old
AB13X P/S를 재사용하지 않고 level, P/S, delay/lead, THD/IMD, end-to-end latency를
모두 새 경로에서 다시 측정해야 한다.

## 4. 현재 realtime callback의 drift·queue·slip 처리

### [가설]

현재 코드의 xrun/ring-buffer telemetry가 별도 clock의 장기 sample slip까지 검출할
가능성이 있다고 가정한다.

### [근거]

- `src/deep_anc/realtime/run_realtime.py` SHA-256:
  `95a405cf0218843dff2b566754c1f928d0af11f231f50960bd101310d5f63a58`
- `src/deep_anc/realtime/ring_buffer.py` SHA-256:
  `ab72b5164ae0562c67886f021be3435c4f483a7ac43805c3d1e1e664968788a7`
- installed PortAudio:
  - Debian `libportaudio2 19.6.0-1.1`
  - runtime revision `396fe4b6699ae929d3a685b3ef8a7e97396139a4`
  - `/usr/lib/aarch64-linux-gnu/libportaudio.so.2.0.0` SHA-256:
    `ec3228bdc94ec27d1636516357c95aaef5fe9fa8c8d7e8d3f8aa5607c5743ece`
- exact upstream revision의
  `src/hostapi/alsa/pa_linux_alsa.c:3900-3921`은 playback이 ready가 아니고
  `neverDropInput`이 false면 capture **한 period를 callback 없이 버린다**.
- installed `/usr/include/portaudio.h:668-677`은 `paNeverDropInput`이
  `framesPerBuffer=0`인 full-duplex callback에서만 유효하다고 명시한다.
- current runtime은 block 256을 고정하고 `never_drop_input`을 설정하지 않는다.

### [확인 방법]

`run_realtime.py:367-368, 429-434, 534, 550-568, 626-635`와 exact PortAudio
revision source를 대조했다.

### [결과]

현재 runtime의 관측 범위는 다음과 같다.

| 사건 | 현재 감지 여부 | 이유 |
|---|---|---|
| PortAudio가 callback status로 보고한 xrun | 예 | `if status: xruns += 1` |
| 추론 input ring의 `pop_latest` drop | 예 | app-level ring counter |
| output ring underrun/drop | 예 | app-level ring counter |
| PortAudio가 callback 전에 버린 capture period | **아니오** | callback 자체가 호출되지 않음 |
| ADC/DAC rate ratio 변화 | **아니오** | estimator/ASRC 없음 |
| callback ADC/DAC timestamp 궤적 | **아니오** | `_time_info`를 사용하지 않음 |
| ALSA negotiated period/queue depth | **아니오** | receipt에 저장하지 않음 |
| one-sample 또는 one-period discontinuity | **아니오** | physical sample index 없음 |

`results/session_20260804_125538/metrics.csv`
(SHA-256 `e47b84f8ddf87e0c6f1d7b494849f1eb8c7e7ed503201f0f796fc9b8cf4d0c05`)
에는 35초 protocol에서 `underruns=9`, `xruns=87`가 남아 있다. legacy 모델 실험이라
현재 모델 성능 증거는 아니지만, combined path가 장기 안정적이라는 주장에는 반증이다.
반대로 strict 12.5초 측정의 xrun 0도 silent input-period drop 0을 증명하지 않는다.

strict rate에서 256 samples의 상대 축적 시간은 `12.884580257878639 s`다. 단,
PortAudio가 실제 협상한 ALSA period를 receipt에 보존하지 않았으므로 이것은
`period=256`일 때의 예상치이지 실제 drop 시각의 증거가 아니다.

### [판정]

**Contradicted / BLOCK.** 현재 telemetry는 app inference backlog를 다루지만
physical ADC/DAC clock drift와 silent capture drop을 다루지 않는다.

### [다음 행동]

AB13X를 유지한다면 출력 clock을 master로 하는 별도 OutputStream과 APE InputStream,
continuous rate matcher/ASRC, 절대 sample index receipt가 필요하다. hard sample drop으로
맞추지 말고 fractional resampling을 하며, discontinuity가 하나라도 생기면 model state를
reset하고 해당 실험을 invalid 처리해야 한다. 이 변경은 buffering과 timing을 바꾸므로
P/S와 lead도 재측정한다.

## 5. 고주파 위상 오차 예산

### [가설]

413.931 ppm의 미보정 ADC/DAC time-map 오차가 고주파에서 허용 가능한 위상 예산을
매우 빨리 소모한다고 가정한다.

### [근거]

48 kHz에서 상대 rate는 다음과 같다.

```text
delta_samples_per_second = 48000 × 413.9314767411314 × 1e-6
                         = 19.86871088357431 samples/s
```

동일 진폭의 두 파형이 위상 `delta`만큼 어긋날 때 normalized residual amplitude는
`2 sin(|delta|/2)`다. 이상적인 20 dB 상쇄의 허용 위상은
`2 asin(0.1/2) = 5.731967965°`, 10 dB는 `18.194872339°`다.

### [확인 방법]

각 주파수에서 `degrees_per_sample = 360 f / 48000`을 계산하고 위상 한계를
sample 수와 rate 누적 시간으로 환산했다.

### [결과]

| 주파수 | 1 sample 위상 | 256-frame 동안 rate 누적 위상 | 20 dB 허용 sample | 미보정 시 예산 소진 | 10 dB 허용 sample | 미보정 시 예산 소진 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 kHz | 15.000° | 1.5895° | 0.382131 | 19.2328 ms | 1.212991 | 61.0503 ms |
| 4 kHz | 30.000° | 3.1790° | 0.191066 | 9.6164 ms | 0.606496 | 30.5252 ms |
| 8 kHz | 60.000° | 6.3580° | 0.095533 | 4.8082 ms | 0.303248 | 15.2626 ms |
| 11.314 kHz | 84.8528° | 8.9916° | 0.0675519 | 3.39991 ms | 0.214429 | 10.7923 ms |

1초 동안의 unwrapped ADC/DAC mapping phase는 각각 `298.031°`, `596.061°`,
`1192.123°`, `1685.916°`다. 이 표는 **미보정 ADC-vs-DAC time-map** 예산이다.
NS/CS 출력이 같은 AB13X DAC에서 동시에 나온다는 사실을 무시하고 곧바로 실제
상쇄 위상 손실로 해석하면 안 된다. 문제는 APE ERR를 함께 소비하는 모델 상태와
측정/평가 time-map에 discontinuity가 생겨도 현재 검출하지 못한다는 점이다.

### [판정]

**Confirmed as a timing budget; realtime impact not yet directly quantified.** 고주파일수록
continuous witness/ASRC residual 기준이 엄격해진다. 11.314 kHz에서 20 dB-grade
clock residual gate `0.0675519 sample`은 수학적으로 일관된다.

### [다음 행동]

30초 이상 실제 runtime receipt에서 rate residual과 모든 slip을 보존하고, 2/4/8/11.3
kHz 각 대역에서 residual timing을 attenuation과 함께 보고한다. 숫자가 없으면
고주파 실시간 성능을 발행하지 않는다.

## 6. 152–600 Hz 연속 reserved pilot + actual-sequence FIR 대안

### [가설]

electrical loopback이나 I2S DAC가 당장 없을 때 zero-tail을 없애고, 152–600 Hz
reserved pilot을 capture 전체에 계속 출력하면서 **실제 submitted int16 전체 sequence**로
clock map과 FIR을 공동 fit하면 clock authority를 정보론적으로 닫을 수 있을 가능성이
있다고 가정한다.

### [근거]

기존 `docs/19_fullband_causal_ps.md` v3의 blind spot은 exact-zero tail 동안 acoustic
관측이 사라져, 두 anchor 사이에서 생겼다가 상쇄되는 clock excursion을 볼 수 없다는
점이다. 연속 pilot은 이 특정 blind interval을 제거한다. 그러나 관측식은 여전히
개념적으로 다음과 같다.

```text
z_i[k] = sum_j sum_l h_ij[l] x_j(q(k)-l) + nonlinear_i[k] + noise_i[k]
```

여기서 `x_j`만 알고 `q(k)`와 acoustic `h_ij`를 함께 구한다. 고정 LTI FIR, 매끄럽고
단조인 clock map, 충분한 SNR/PE라는 제약을 두면 `q`를 추정할 수 있다. 반대로 plant
delay/phase가 시간에 따라 변하도록 허용하면 작은 `q` 변화와 plant 변화가 같은
관측을 설명할 수 있다. 즉 acoustic pilot은 clock과 무관한 독립 witness가 아니다.

### [확인 방법]

다음 네 설계를 식별 가능성, finite support, 비선형 누설, P/S 상대 지연 관점에서
비교했다.

| 설계 | clock 관측 | finite-memory 증거 | 핵심 한계 |
|---|---|---|---|
| 전체 exact-zero tail | tail 내부 없음 | 전 대역 직접 관측 가능 | 숨은 clock excursion |
| 전 구간 acoustic pilot | LTI 가정 아래 연속 | deconvolution/holdout의 통계 증거만 | clock–plant confounding |
| low pilot + highband zero tail | 저역은 연속 | 고역 tail은 조건부 직접 관측 | 저역 tail 미증명, pilot nonlinear leakage |
| electrical/common clock | plant와 독립 | zero-tail과 동시에 가능 | 추가 배선/하드웨어 필요 |

### [결과]

#### 6.1 clock authority

연속 pilot은 **관측 공백을 없애지만 독립 clock authority를 닫지는 못한다**. 다음
조건을 모두 강제한 경우에만 `conditional acoustic clock map`으로 사용할 수 있다.

- fit/holdout마다 다른 deterministic aperiodic code를 사용한다. 단순 periodic
  multisine은 cycle alias 때문에 충분하지 않다.
- intended float가 아니라 actual submitted two-channel int16 PCM SHA를 분모로 쓴다.
- global `q(k)` 하나를 ERR/REF × P/S 전체에 공유하고 role별 sample offset이나
  high-band 결과 기반 phase repair를 금지한다.
- fit-a/fit-b만 ratio fit에 사용하고 odd segment와 새 holdout code는 validation에만
  쓴다.
- linear/cubic crosscheck, monotonicity, trajectory step, one-sample slip, 네 view
  agreement를 각각 gate한다.
- holdout에서는 clock map과 FIR을 동시에 자유롭게 refit하지 않는다. 그러면 검증이
  아니라 같은 자유도로 자료를 다시 설명하는 것이 된다.

이 조건을 통과하면 별도-clock acoustic raw를 유용한 diagnostic P/S로 만들 수 있다.
그러나 “clock 자체가 independently witnessed”됐다는 주장은 할 수 없다.

#### 6.2 zero-tail과 finite memory

pilot을 전 대역에 계속 출력하면 실제 입력이 0인 구간이 없으므로, 특정 support 뒤의
출력이 실제로 noise floor로 돌아왔다는 direct proof를 잃는다. long FIR을 먼저 fit한 뒤
post-support tap의 bootstrap upper bound와 independent actual-sequence holdout induced
error를 계산할 수는 있지만 이는 **모델 기반 통계 증거**다. 아주 늦은 약한 echo가
ongoing excitation과 겹치거나 regularization에 흡수되는 경우를 배제하지 못한다.

절충안은 152–600 Hz pilot만 tail에 남기고 600 Hz 초과 main PE burst를 exact zero로
만드는 것이다. 그러면 fitted low-pilot contribution을 뺀 뒤 600 Hz 초과 tail을 검사할
수 있다. 다만 다음 이유로 여전히 conditional이다.

- 저역 FIR은 zero-tail direct proof가 없다.
- 152–600 Hz pilot의 2차 성분은 304–1200 Hz, 3차 성분은 456–1800 Hz에 들어오며,
  intermodulation도 600 Hz 위 식별 대역을 오염시킬 수 있다.
- linear pilot subtraction은 speaker/amp/mic의 nonlinear product를 지우지 못한다.

따라서 이 hybrid는 고역 causal support의 강한 diagnostic이 될 수 있지만 low+high
canonical finite-memory authority를 혼자 완성하지 못한다.

#### 6.3 low-band 식별과 nonlinear gate

152–600 Hz를 “비운다”는 것은 그 대역을 포기한다는 뜻이어서는 안 된다. pilot code
자체가 두 path의 해당 대역을 persistently excite해야 한다. 두 출력이 같은 frequency
bins를 단순 공유하면 P와 S가 섞이므로, 독립 aperiodic code의 2×2 input Gram matrix와
holdout condition number를 gate해야 한다. disjoint tone만 나누면 각 path가 빠진
frequency를 갖게 되므로 full low-band P/S authority가 되지 않는다.

최소 nonlinear gate는 다음을 포함해야 한다.

- actual PCM peak/RMS/crest 및 두 DAC channel의 cross-correlation
- 동일 code의 최소 두 amplitude 단계에서 complex transfer의 amplitude invariance
- pilot fundamental 밖 2차/3차 harmonic과 IMD residual
- highband FIR을 바꿔도 inferred `q(k)`가 변하지 않는 negative-control
- pilot amplitude를 바꿔도 clock slope가 gate 범위 안에서 같은지 확인

이 gate 없이 pilot harmonic을 high-frequency plant response나 clock correction으로
오인할 수 있다.

#### 6.4 P/S 전환

P-only와 S-only를 순차 실행하면 switch gap에서 continuous acoustic witness가 끊기고,
role별 unknown delay가 `q` offset을 흡수할 수 있다. PortAudio가 그 사이 capture period를
버려도 현재 callback은 검출하지 못한다. 그러므로 sequential switch + role별 clock
fit은 canonical clock authority로 **거부**해야 한다.

하드웨어 추가 없이 가장 강한 acoustic-only 설계는 NS와 CS 두 output에 서로 독립인
aperiodic pilot/full sequence를 **동시에** 보내고, ERR/REF 두 input에서 2×2 MIMO FIR과
global `q(k)` 하나를 공동 식별하는 것이다. 이 방식은 P/S switch를 없애고 relative delay를
한 capture에 묶는다. 대신 inactive channel exact-zero 조건을 포기하고, superposition과
nonlinearity를 직접 gate해야 한다. fit-a, fit-b, holdout은 서로 다른 sequence여야 한다.

### [판정]

**Conditional diagnostic: Likely. Independent/canonical clock authority: BLOCK.** 연속
reserved pilot은 v3 zero-tail의 “완전 blind interval”을 실제로 개선한다. 하지만
acoustic LTI 가정을 이용해 clock을 추정한 뒤 같은 raw로 LTI causal plant를 승인하는
순환성을 완전히 없애지 못한다. 특히 저역 direct finite-memory proof와 nonlinear leakage가
남는다. 따라서 electrical loopback/common clock 게이트를 낮추는 근거로 쓰지 않는다.

### [다음 행동]

소리 출력 전에 signal-only v4를 먼저 구현·검증한다.

1. actual two-channel int16 fit-a/fit-b/holdout SHA와 MIMO condition receipt 생성
2. simultaneous two-output aperiodic pilot, global q only, role offset 금지
3. synthetic known-FIR + injected affine/curved clock + one-sample slip + nonlinear
   harmonic adversarial fixture
4. holdout no-refit, negative-control, amplitude-ladder gate
5. highband pilot-only tail과 lowband statistical-tail 판정을 서로 다른 필드로 발행
6. 결과 schema가 `clock_authority=conditional_acoustic_only`와
   `canonical_training_eligible=false`를 강제

이 dry-run과 전체 test가 통과한 뒤에도 실제 출력 시간·speaker·volume·artifact 경로를
먼저 사용자에게 보고하고 명시적 승인을 받아야 한다.

## 7. 최소 변경 우선순위

### [가설]

clock 문제를 숨긴 채 모델/손실만 바꾸는 것보다 time-base authority를 먼저 닫는 것이
고주파 ANC 성능을 가장 직접적으로 개선할 가능성이 높다고 가정한다.

### [근거]

8–11.3 kHz의 20 dB-grade sample budget은 `0.0955–0.0676 sample`인데 current runtime은
one-period silent drop조차 직접 계수하지 못한다. 모델 capacity를 늘려도 callback 전에
사라진 physical sample은 복원되지 않는다.

### [확인 방법]

추가 latency, 기존 하드웨어 재사용, authority 강도를 비교했다.

### [결과]

| 우선순위 | 방안 | 장점 | 비용/위험 | 판정 |
|---:|---|---|---|---|
| 1 | APE I2S2 + 외부 2ch slave DAC | 공통 clock, ASRC 불필요 | pin40/DT/배선, P/S 전부 재측정 | **권장** |
| 2 | 안전한 AB13X electrical loopback | 현 DAC의 독립 q witness 가능성 검증 | mic bias/감쇠/3-stream recorder | 진단 선행 |
| 3 | APE→AB continuous ASRC/rate matcher | 현 output hardware 유지 | latency·estimator 복잡도, witness 필요 | 차선 |
| 4 | acoustic-only simultaneous MIMO pilot | 추가 DAC 없이 diagnostic 개선 | conditional, nonlinear/finite-tail 한계 | canonical 대체 불가 |

### [판정]

**현재 high-frequency realtime gate는 BLOCK.** 가장 작은 코드 변경은 telemetry
추가지만, 가장 작은 **물리적으로 강한** 해결은 I2S2 DOUT + 2ch DAC로 ADC/DAC clock을
통일하는 것이다.

### [다음 행동]

#### 모든 실기 전 공통 확인

- `fuser -v /dev/snd/*`와 `/proc/asound/card*/pcm*/sub*/status`에서 PCM 무점유
- `/home/capston/DeepANC_CRN_n_codex` 쪽 병행 측정이 없는지 확인
- 사용자가 현장에 있고 speaker/amp volume 최저인지 확인
- 실행 명령, 어떤 speaker가 동작하는지, audible/electrical 시간, raw 경로를 먼저 보고
- callback/error/xrun/slip 발생 시 즉시 무음·중단, 같은 자리에서 반복하지 않음

#### electrical loopback 추가 확인

- 스피커와 TPA3116D2 입력을 물리적으로 분리
- DMM/oscilloscope로 DC bias, peak, ground potential 확인
- 계산된 고임피던스 divider + DC-block capacitor 사용, 직결 금지
- clipping/rail/USB mic gain을 무출력 상태에서 먼저 확인

#### I2S DAC 추가 확인

- 현재 DTB와 `extlinux.conf`의 복구 entry를 별도로 보존
- pin 38 DIN, pin 12 BCLK, pin 35 WS를 유지하고 pin 40만 DOUT으로 추가
- DAC가 3.3 V logic, I2S slave, 48 kHz stereo를 지원하는지 확인
- 전원/GND 연결 후 amplifier와 speaker를 연결하기 전에 electrical output만 검증
- 새 path의 level/P/S/lead/latency/THD를 old artifact와 분리된 capture ID로 측정

## 최종 PASS/BLOCK

### [결과]

- **PASS:** 실제 topology, USB sync descriptor, strict drift 수치, PortAudio silent-drop
  경로, phase budget을 artifact/source에서 독립 확인했다.
- **BLOCK:** AB13X capture의 공통 oscillator 여부, 장시간 slip-free runtime,
  continuous ASRC residual, 2 kHz 이상 실제 attenuation은 아직 현장 evidence가 없다.
- **BLOCK:** acoustic-only continuous pilot은 진단력을 높이지만 독립 clock authority와
  전 대역 direct finite-memory proof를 동시에 제공하지 않는다.

### [판정]

현재 Jetson은 2 kHz 이상을 포함한 broadband 실시간 ANC를 **아직 입증하지 못했다**.
이 판정은 모델 성능이 나쁘다는 뜻이 아니라, 물리 time-base와 discontinuity가 관측·제어되지
않아 실제 고주파 attenuation 숫자를 신뢰할 수 없다는 뜻이다.

### [다음 행동]

1. runtime sample/time/queue receipt와 slip fail-closed를 먼저 구현한다.
2. AB13X electrical witness 가능성을 무음·speaker-disconnected 조건에서 검증한다.
3. 병렬로 I2S2 2ch DAC 공통-clock 전환안을 준비한다.
4. acoustic-only v4 MIMO continuous-pilot은 canonical 대체가 아닌 diagnostic으로
   dry-run/fixture까지만 먼저 완성한다.
5. time-base gate를 통과한 뒤에만 같은 P/S로 actual 2/4/8/11.3 kHz OFF/ON raw를
   측정하고 attenuation을 발행한다.
