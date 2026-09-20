# 02. 하드웨어 구성과 점검 절차

기존 Jetson AGX Orin·덕트·마이크·USB DAC를 유지한다.
이 문서의 배선·초기 진단 기록은 현재 연결·입력 정상 여부를 보장하지 않는다.
실행 환경과 최근 검증은 [HANDOFF.md](../HANDOFF.md),
위치별 작업 조건은 [docs/14](14_pc_jetson_workplan.md)를 따른다.

## 1. 장치·채널과 배선

[configs/hardware_jetson.yaml](../configs/hardware_jetson.yaml)의 논리 맵은 다음과 같다.

| 구분 | 구성 | 채널 |
|---|---|---|
| 입력 | APE ADMAIF2, `hw:APE,1`, 48kHz 스테레오 S32_LE | ch0=ERR, ch1=REF |
| 출력 | AB13X USB Audio, 카드 ID `Audio`, 48kHz 스테레오 S16_LE | ch0=NS, ch1=CS |
| 앰프 | TPA3116D2 | 출력 시험 전 볼륨 최소 확인 |

2026-08-03에는 APE가 `hw:1,1`, USB 출력이 `hw:2,0`으로 열거됐다.
이 숫자와 당시 PortAudio index를 현재 장치 번호로 복사하지 않는다.
`deep_anc.audio_io.resolve_alsa_portaudio_device`는 ALSA 카드의 짧은 ID를 사용하며,
USB 연결 상태와 현재 목록을 매 세션 확인한다.

### 사용자 확인 배선 — 2026-08-03 기록

두 INMP441은 VDD/GND/SCK/WS/SD를 공유하고 L/R 선택을 달리하는 구성이다.
아래는 당시 사용자 확인값이며 이번 문서 정리에서 실물 배선을 재검사한 것은 아니다.

| 신호 | 선 색 | Jetson J30 물리 핀 | 역할 |
|---|---|---:|---|
| 공통 VDD | 빨강 | 1 | 3.3V |
| 공통 GND | 검정 | 6 | GND |
| 공통 SCK | 주황 | 12 | I²S2 SCLK |
| 공통 WS | 노랑 | 35 | I²S2 FS |
| 공통 SD | 갈색 | 38 | I²S2 DIN |
| REF L/R | 초록 | 17 | high → 오른쪽 프레임 → ch1 |
| ERR L/R | 파랑 | 39 | low → 왼쪽 프레임 → ch0 |

물리 접촉을 점검할 때는 전원을 끄고 J30 pin 1 방향부터 확인한다.
pin 2/4의 5V를 마이크 전원이나 L/R에 연결하지 않는다.
배선 기록의 근거는 [NVIDIA carrier-board specification](https://developer.nvidia.com/assets/embedded/secure/jetson/agx_orin/jetson_agx_orin_devkit_carrier_board_specification_sp)과
[INMP441 datasheet](https://product.tdk.com/system/files/dam/doc/product/sw_piezo/mic/mems-mic/data_sheet/inmp441.pdf)다.
덕트 좌표·장착 미확정값은 [docs/09](09_duct_structure.md)를 따른다.

## 2. 시스템과 실행 범위

Jetson의 pinmux/I²S, device tree, RT 커널, 전원모드,
`nvpmodel`/`jetson_clocks`, 오디오 데몬과 priority limit을 변경하지 않는다.
호스트 apt 설치도 하지 않는다. Docker 환경 관리는 [docker/README.md](../docker/README.md)를 따른다.
코드·Python·테스트·장치 점검 도구는 컨테이너 안에서 실행한다.

기본 개발 컨테이너에는 오디오 장치가 노출되지 않는다.
실제 장치 점검은 해당 장치 접근을 준비한 Jetson 컨테이너에서 수행하며,
장치가 보이지 않는 문제를 호스트 시스템 설정 변경으로 우회하지 않는다.

## 3. 현재 세션의 입력 점검

다음은 장치 목록과 입력만 읽는 점검이다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m deep_anc.realtime.run_realtime --list-devices
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_audio_input.py --require-both --max-clip-ratio 0
```

acoustic-ref에는 ERR/REF 두 채널이 모두 필요하다. raw code 다양성, RMS, peak,
clipping을 확인하고 clip 0의 반복 PASS를 확보한다.
입력이 실패하면 강제 진행이나 스피커 재생으로 우회하지 않는다.
`record_duct --program silence`나 무음 런타임도 출력 장치를 여는 도구이므로
입력 전용 probe와 구분한다.

### 과거 입력 진단 — 현재 상태로 사용하지 않음

| 시점 | 관측과 범위 |
|---|---|
| 2026-08-01 | REF ch1 무신호 기록 |
| 2026-08-03 19:11 | ERR/REF raw sample이 −1에 고착 |
| 2026-08-03 19:25 | 재배선 후 동적 입력은 있으나 ERR/REF clip 약 6.79%/7.60% |
| 2026-08-03 19:29 | pin17 재연결 뒤 5초 검사에서 ERR/REF 약 −46.33/−46.64dBFS, clip 0% |
| 2026년 8월 초의 22:39 이후 기록 | 간헐 과클리핑 재발. 2초 검사 ERR/REF clip 약 2.47%/5.03%, 재검사도 실패하여 출력 측정 중단 |

잠깐의 PASS 이후에도 실패가 재발한 이력이다.
이 표만으로 현재 정상·고장을 판단하거나 pin17만이 모든 실패의 원인이라고 단정하지 않는다.
현재 세션의 무출력 진단과 실물 접촉 확인이 필요하다.

## 4. 채택한 경로와 재측정

현재 `duct.yaml`이 참조하는 자산은 다음과 같다.
파일 채택과 현재 장치 조건에서의 재검증 완료는 다른 판단이다.

| 파일 | 순수지연 | 반복 일관성 | 검증 대역 |
|---|---:|---:|---|
| `primary_path_il.npz` | 1608샘플 | 약 0.973 | 150–600Hz |
| `secondary_path_il.npz` | 1465샘플 | 약 0.956 | 150–600Hz |

두 파일은 같은 capture의 interleaved P/S다.
자극 대역은 P 64–1648Hz, S 72–1640Hz이지만 검증 대역은 양쪽 모두 150–600Hz다.
Jetson 목표 1000–1600Hz는 별도 반복 검증이 필요하다.
총지연·digital/acoustic 정렬은 [docs/01](01_physics_limits.md)이 기준이다.

`secondary_path_4s.npz`(1342샘플)와
`secondary_path_legacy_512high.npz`(2613샘플)는 과거 비교 자료다.
채택 S를 대신하지 않으며, 당시 품질·설정 불일치는
[legacy 부록](appendix_legacy_fxlms.md)에 한정해 기록한다.

재측정에서는 블록·latency·샘플레이트·채널·게인·볼륨을 기록하고 기존 S의 조건과 대조한다.
CS→ERR의 S, CS→REF의 F, 외부 소리의 REF→ERR 선행 시간을 구분한다.
측정 도구는 `measure_paths_interleaved.py`, `calibrate_wideband.py`,
`measure_duct_transfer_map.py`이며 구체적 순서는 [docs/14](14_pc_jetson_workplan.md)를 따른다.
새 측정은 새 산출물로 남기고, 반복 일관성·지연 안정성·대역 검증 전에 채택 NPZ를 교체하지 않는다.

소리가 나는 측정과 실행은 사용자 입회·볼륨 최소 상태에서만 한다.
런타임은 항상 ANC OFF로 시작한다. 출력 제한은 선택한 설정을 따른다
(acoustic 기준선은 `control_limit=0.10`). 페이드·클리핑·발산·출력 누락 보호가
측정 조건 확인이나 사용자 입회를 대체하지 않는다.
