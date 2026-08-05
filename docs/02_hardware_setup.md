# 02. 하드웨어 구성과 점검 절차

## 1. 장치·채널 맵 (anc_project 에서 실기 검증된 구성)

```
[Jetson AGX Orin]
  입력  hw:APE,1 (ADMAIF2) · S32_LE · 48kHz · 스테레오     sounddevice 예시 idx 5
    ch0 = 에러 마이크   (INMP441, L/R핀 → GND)   덕트 X≈1.10m 벽면
    ch1 = 레퍼런스 마이크(INMP441, L/R핀 → 3.3V)  덕트 X=0.10m 벽면
  출력  AB13X USB Audio (hw:2,0) · S16_LE · 48kHz · 스테레오  idx 24
    ch0 = 소음 스피커   (좌)  덕트 X=0 폐단, 축방향
    ch1 = 상쇄 스피커   (우)  덕트 X=1.05m 상면, side-branch
  앰프  TPA3116D2 (12~24V) — 시작 전 볼륨 최소로!
```

### J30 40핀 헤더 물리 배선 (2026-08-03 사용자 확정)

두 INMP441은 `VDD/GND/SCK/WS/SD`를 공유하고, `L/R`만 서로 다른 전원 레벨에 연결해
하나의 스테레오 I²S 프레임을 만든다.

| INMP441 신호 | 선 색 | Jetson J30 물리 핀 | 역할 |
|---|---|---:|---|
| 두 마이크 `VDD` | 빨강 | **1** | 3.3V 전원 |
| 두 마이크 `GND` | 검정 | **6** | GND |
| 두 마이크 `SCK` | 주황 | **12** | I²S2 SCLK |
| 두 마이크 `WS` | 노랑 | **35** | I²S2 FS/word-select |
| 두 마이크 `SD` | 갈색 | **38** | I²S2 DIN, 두 마이크의 SD를 같은 선에 연결 |
| 레퍼런스 마이크 `L/R` | 초록 | **17** | 3.3V(high) → 오른쪽 프레임 → 입력 ch1 |
| 에러 마이크 `L/R` | 파랑 | **39** | GND(low) → 왼쪽 프레임 → 입력 ch0 |

> [!CAUTION]
> 전원을 끈 상태에서 배선하고, PCB의 흰색 삼각형으로 J30 pin 1 방향을 먼저 확인한다.
> pin 2/4는 5V이므로 INMP441 전원이나 `L/R`에 사용하지 않는다. INMP441 허용 전원은
> 1.8–3.3V다. 두 마이크의 SD 출력은 공통선으로 묶는 구성이며, 데이터시트는 공통 SD에
> 100kΩ pull-down을 권장한다. 사용 중인 breakout에 저항이 이미 있는지 먼저 확인한다.

이 매핑은 NVIDIA의 J30 표와 I²S2 표(`SCLK=12`, `FS=35`, `DIN=38`) 및 INMP441의
`L/R low=left`, `high=right` 규약을 따른다. 근거:
[NVIDIA carrier-board specification](https://developer.nvidia.com/assets/embedded/secure/jetson/agx_orin/jetson_agx_orin_devkit_carrier_board_specification_sp),
[NVIDIA audio guide](https://docs.nvidia.com/jetson/archives/r38.2.1/DeveloperGuide/SD/Communications/AudioSetupAndDevelopment.html#board-interfaces),
[TDK INMP441 datasheet](https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/inmp441.pdf).

- 장치 해석은 `deep_anc.audio_io.resolve_alsa_portaudio_device` (fxlms_core 이식)가
  `/proc/asound/cards` 의 짧은 ID(`APE`, `Audio`)로 자동 매핑한다.
- 장치 목록 확인: `.venv/bin/python -m deep_anc.realtime.run_realtime --list-devices`
- **USB 오디오(AB13X)가 꽂혀 있어야 카드 `Audio` 가 보인다.** 2026-08-03 현재
  APE 입력 `hw:1,1`과 AB13X 출력 `hw:2,0`은 인식되며 48kHz/2채널 설정도 수락된다.

## 2. 시스템 정책 (중요)

Jetson 의 **핀 설정(pinmux/I2S)과 RT 커널 구성은 의도된 것이므로 절대 변경하지 않는다.**
전원모드(30W), pulseaudio/pipewire, RT priority limit 등 시스템 상태도 건드리지 않는다.
이 저장소의 모든 도구는 유저 공간(venv)에서만 동작하도록 만들어졌다.
(성능 튜닝 여지가 있는 항목들은 docs/06 §5 에 "참고"로만 기록)

NVIDIA 일반 문서상 J30 I²S2는 pinmux가 필요한 인터페이스지만, 이 Jetson의 현재 APE/I²S
구성은 이미 의도적으로 설정된 실험 환경이다. `Jetson-IO`나 device tree를 다시 적용하지 말고,
아래의 장치 목록과 무음 녹음으로 기존 구성을 읽기 전용 검증한다.

이 정책은 입력 문제를 진단할 때도 예외가 없다. `sudo`, Jetson-IO, pinmux/device-tree,
RT 커널, `nvpmodel`/`jetson_clocks`, pulseaudio/pipewire, apt 설치를 변경하지 않는다.

## 3. 하드웨어 점검 순서 (USB 오디오 연결 후)

### 현재 정상 입력 상태 (2026-08-03 19:29 KST)

19:11의 무출력 캡처에서는 ERR/REF raw sample이 전부 `-1`이었다. 19:25 재배선 뒤 두 채널에
동적 데이터는 들어오지만, 손을 뗀 상태의 5초 캡처가 ERR −9.19dBFS/6.79% clip,
REF −8.77dBFS/7.60% clip, 양쪽 peak 1.0으로 실패했다. 0.1초별 clip이 최대 약 38%로
튀었다. 원인은 빠져 있던 pin17(REF L/R→3.3V)이었다.

pin17 재연결 뒤 5초 검사는 ERR −46.33dBFS/peak 0.0609/clip 0%,
REF −46.64dBFS/peak 0.0572/clip 0%로 모두 PASS했다. 현재 배선을 유지하고, 매 출력 실험 전에
probe를 다시 실행한다. 이 결과를 이유로 Jetson 시스템 설정을 바꾸지 않는다.

### 안전한 점검 순서

```bash
# 1) 장치 인식
.venv/bin/python -m deep_anc.realtime.run_realtime --list-devices     # APE(hw:1,1), Audio(hw:2,0) 확인
# 2) 출력 장치를 열지 않는 입력 probe — FxLMS/digital-ref는 ERR ch0가 필수
.venv/bin/python scripts/bench/check_audio_input.py
# recorded/acoustic-ref는 두 채널 모두 필수
.venv/bin/python scripts/bench/check_audio_input.py --require-both
# 3) 위 probe PASS 후 세션 도구 자체 점검 (noise/cancel 스피커는 무음)
.venv/bin/python scripts/data/record_duct.py --program silence --seconds 10
# 4) 무음 전체 루프 (스피커 소리 없음, 3-스레드 검증)
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set noise.type=silence --set engine.type=ort --run-seconds 10
# 5) 실효 지연 측정 (처프 재생 — 사용자 입회·볼륨 최저!)
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate
# 6) I/O 지연 스윕 (선택)
.venv/bin/python scripts/bench/measure_io_latency.py
```

probe는 raw code 다양성, float 변환 RMS, peak, clipping을 함께 검사한다. 19:29 기준 두 채널
모두 PASS했지만 출력 직전마다 다시 확인하고, 실패 시 `--force`나 재생으로 우회하지 않는다.

입력 복구 전에도 noise speaker ch0의 출력 채널·누설·주관적 공진만 확인하려면, 사용자 입회와
앰프 볼륨 최저를 실제로 확인한 뒤 다음 정성 진단을 사용할 수 있다.

```bash
.venv/bin/python scripts/bench/playback_duct_probe.py --confirm-volume-minimum
```

이 도구는 설정의 축방향 공진과 300Hz를 peak 0.002 단계 톤으로 재생하고 cancel ch1은 항상
무음으로 둔다. 마이크 데이터가 없으므로 생성되는 JSON은 자극 로그일 뿐 P(z), S(z), 지연,
감쇠 dB 또는 `duct.yaml` 갱신 근거가 아니다.

### 레퍼런스 마이크(ch1) 이력

2026-08-01 진단에서 ch1은 사실상 무신호였고 2026-08-03 19:11에는 ch0까지 raw −1로
고착됐다. 빠져 있던 pin17을 복구한 뒤 둘 다 정상 PASS했다. acoustic-ref와 recorded 수집도
각 세션 직전 두 채널 probe가 계속 PASS할 때만 진행한다.

## 4. 2차경로 S(z) 보정

### 현재 자산 (assets/measured/)

| 파일 | delay | 검증 대역 | 그 대역 일관성 | 전대역 | 유지/전체 | 방식 | 비고 |
|---|---:|---|---:|---:|---:|---|---|
| `primary_path_il.npz` | **1602** | 150–1600Hz | **0.9993** | **0.9988** | 18/48 | interleaved | **채택 P(z)** (재발행 2026-08-05) |
| `secondary_path_il.npz` | **1462** | 150–1600Hz | **0.9990** | **0.9984** | 18/48 | interleaved | **채택 S(z)** — P 와 같은 capture·앵커 |
| `*_il.npz.orig` | 1608 / 1465 | 150–600Hz | 0.973 / 0.956 | 0.920 / **0.781** | 16/16 | interleaved | **오염본 백업** (아래 경고) |
| `secondary_path_4s.npz` | 1342 | 150–600Hz | 0.40 | — | — | 순차 ESS | 폐기 (2026-08-05) |
| `secondary_path_legacy_512high.npz` | 2613 | — | 0.27 | — | — | 순차 ESS | 구버전 기록용 (block 512/high) |

`P − S = 140`, `lead = 116`, 앵커 반복 13, `capture_id = f7b0fecd…`.
채택본 두 개는 **한 번의 재생으로 동시에** 측정했고 `capture_id` 가 일치한다. 순차 ESS 는
두 측정 사이에 일어난 **출력 버퍼 프레임 슬립**이 P/S 상대 지연에 실려 lead 를 틀리게
만든다 — **클록 드리프트가 아니다.** 두 클록의 상대 드리프트는 **+0.4 ppm**(10분에 12샘플)
으로 lead(116샘플)를 틀리게 만들 크기가 아니며, 둘 다 +17 ppm 으로 같은 Tegra 발진기를
공유한다(USB 싱크 ADAPTIVE).

> [!CAUTION]
> **`.orig` 백업본(전대역 S 0.781 / P 0.920)은 오염된 반복 5개를 포함한 값이다.**
> `alignment_scores` 반복 11–15 가 0.750–0.758 로 별도 무리인데 기각 임계 0.5 때문에
> `rejected_repeats: 0` 이었다. 결정적 증거는 **P−S 상대 τ** — 두 채널은 같은 DAC·같은
> 출력 스트림의 인터리브라 설계 원리상 상수여야 하는데 반복 11 에서 **1.4 → 32 샘플
> 점프**한다(출력 버퍼 프레임 슬립). 게이트는 요약 스칼라 `delay_spread_samples 32` 를
> 허용치 48 과 비교해 **통과시켰다.**
> 오염 반복을 버린 재계산에서 600–1000Hz 는 S 0.837 → **0.999**, 1000–1600Hz 는
> S 0.737 → **0.999**, 전대역 0.782 → **0.999** 로 회복한다.
> **→ "600 Hz 위는 스피커 물리 한계" 는 틀렸다. 진짜 한계는 80–150Hz 뿐이다**
> (클린 후에도 S 0.706~0.758). 상세: [docs/12 §2.3](12_system_summary.md#23-실측-경로-자산).
>
> 스피커를 울리지 않고 저장된 캡처를 재분석하려면:
> `.venv/bin/python scripts/data/reanalyse_paths_interleaved.py <세션 디렉터리> --dry-run`

`excitation_band_hz` 는 **두 경로가 다르다** — 인터리브라 두 채널이 인접 FFT 빈을 번갈아
쓰기 때문이다: **P(noise) 64–1648Hz / S(cancel) 72–1640Hz.**
`consistency_band_hz`(검증 **150–1600Hz**, P/S 동일)와도 다른 값이다.
학습 손실과 평가는 **검증 대역**을 쓴다 — 재현되지 않는 대역까지 최적화하면 그 잘못된 위상이
gradient 를 지배해 신뢰 구간 성능까지 잃는다.

주의: 기존 anc_project 는 512/high 로 측정된 모델을 256/low 런타임에 쓰고 있었다
(지연 26ms 어긋남 — appendix 참조). 측정 latency 는 런타임과 반드시 같아야 한다.

### 광대역 재보정 (풀밴드 학습의 선행 게이트)

```bash
# S(z) 재보정: 상쇄 스피커(ch1) → 에러 마이크, 80–8000Hz ESS 스윕
.venv/bin/python scripts/data/calibrate_wideband.py --output-channel cancel \
    --out assets/measured/secondary_path_wb.npz
# 반복 일관성 ≥0.9 확인 후 duct.yaml secondary_path.npz 교체 → 파인튜닝
# digital-ref 1차경로 지연 실측: 소음 스피커(ch0) → 에러 마이크
.venv/bin/python scripts/data/calibrate_wideband.py --output-channel noise \
    --out assets/measured/primary_path_wb.npz
# → duct.yaml digital_reference.primary_path_npz에 위 NPZ 경로,
#   d_noise_delay_samples에 출력된 delay를 함께 기입
```

## 5. 안전 수칙

1. 모든 실행은 **ANC OFF 로 시작**한다 (A 키로 수동 ON).
2. TPA3116D2 볼륨은 최소에서 시작해 점진적으로 올린다.
3. 런타임 안전장치: 출력 리미터(0.2) / 클립 스트릭 자동 mute / 발산 워치독(+6dB·0.5s)
   / 추론 데드라인 워치독 — 이상 시 자동으로 상쇄 채널이 꺼진다.
4. 스피커에 소리를 내는 스크립트(`record_duct.py`, `calibrate_wideband.py`,
   `measure_io_latency.py`, `evaluate_session.py`, `run_realtime.py`)는 반드시 사람이
   현장에 있을 때 실행한다.
