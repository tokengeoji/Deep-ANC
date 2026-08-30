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

전원모드(30W), RT priority limit 등 시스템 상태는 기본적으로 건드리지 않는다.
이 저장소의 도구는 유저 공간(venv)에서 동작하도록 만들어졌다.
(성능 튜닝 여지가 있는 항목들은 docs/06 §5 에 "참고"로만 기록)

**핀먹스와 디바이스 트리는 변경해도 된다** (2026-08-06 사용자 명시 허용, [AGENTS.md](../AGENTS.md) §4).
이전 판본의 "절대 변경 금지" 는 삭제됐다 — 아래 §2.1 이 그 이유다.

### 2.1 마이크는 **병합 DTB의 핀먹스 오버레이에 의존한다** (2026-08-06 확인)

**이 사실을 모르면 마이크 무신호를 절대 못 고친다.** J30 40핀 헤더의 I²S2 핀은 기본
디바이스 트리에서 muxed 되어 있지 않다. 오버레이를 **사전 병합한 DTB** 로 부팅해야 한다.

```
설치본:  ~/FxLMS/realtime_fxlms/boot_fix/
           apply_boot_fix.sh
           extlinux.conf.new
           kernel_tegra234-p3737-0000+p3701-0005-nv-user-custom.dtb   (md5 9823e1cb7b7c…)
```

`/boot/extlinux/extlinux.conf` 의 `LABEL real-time` 항목이 `FDT /boot/dtb/…-nv-user-custom.dtb`
를 가리켜야 하고, `DEFAULT real-time` 이어야 한다. 부트로더의 OVERLAYS 처리에 의존하지
않게 병합해 둔 것이라 어느 항목으로 부팅해도 마이크가 산다.

**살아 있는지 확인하는 한 줄** (읽기 전용, sudo 불필요):

```bash
find -L /proc/device-tree -type d -name exp-header-pinmux -print -quit
```

노드가 나오면 그 아래에 `hdr40-pin12`(SCK) · `hdr40-pin35`(WS) · `hdr40-pin38`(SD) 이 있고,
각각 `nvidia,function = i2s2` 여야 한다. 노드가 **없으면** 그 부팅은 핀먹스 없이 올라온
것이고, 마이크는 무슨 짓을 해도 0 만 낸다.

### 2.2 마이크 무신호 진단 사다리 (2026-08-06 에 실제로 탄 순서)

위에서부터 확인한다. 각 단계는 **앞 단계가 통과했을 때만** 의미가 있다.

| # | 확인 | 명령 | 통과 기준 |
|---|---|---|---|
| 1 | USB DAC 인식 | `cat /proc/asound/cards` | `2 [Audio] USB-Audio AB13X` |
| 2 | 핀먹스 오버레이 | 위 `find -L …` | `exp-header-pinmux` + pin12/35/38 = i2s2 |
| 3 | 부트 경로 | `uname -r` / `md5sum /boot/dtb/…` | `5.15.148-rt-tegra` / md5 일치 |
| 4 | XBAR 라우팅 | `amixer -c APE cget name='ADMAIF2 Mux'` | `values=18` (=I2S2) |
| 5 | PulseAudio 간섭 | `pactl set-card-profile alsa_card.platform-sound off` | 44.1kHz sink 가 PLL_A 를 흔드는 것 차단 |
| 6 | **I²S2 컨트롤러** | `I2S2 Loopback` on → 재생/캡처 → off | 재생한 톤이 그대로 돌아옴 |
| 7 | 실제 마이크 | `arecord -D hw:APE,1 -f S32_LE -r 48000 -c 2 -d 2` | 0 이 아닌 샘플 존재 |

6번이 통과하고 7번이 실패하면 **패드 바깥(선·마이크 모듈)** 이다. 6번은 컨트롤러 안에서
TX→RX 로 도는 것이라 패드까지는 증명하지 못하므로 2번과 함께 봐야 한다.

⚠ **두 마이크가 SD 선(pin38)을 공유하므로 소프트웨어로는 둘을 가를 수 없다.** 공통 SD 에는
100kΩ 풀다운이 있어 (a) SD 선 빠짐 (b) 한 모듈이 low 로 물음 (c) 둘 다 무전원 이 **전부
정확히 0** 으로 똑같이 보인다. 마이크 하나만 남기고 다시 재는 물리 시험이 유일한 판별법이다.

살아 있을 때의 모습(참조, `~/anc_project/diagnostics/mic_stats.txt` 2026-08-01):
`ch0 RMS −4.78 dBFS, 0인 샘플 0.0017%`. 죽은 채널은 `raw = [-1,-1,-1,0,0,-1,0]` 처럼
LSB 만 흔들리거나, 완전히 끊기면 **전 비트 0** 이다.

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
# 3) 세션 도구의 무변경 dry-run 뒤 무음 점검 (noise/cancel 스피커는 무음)
.venv/bin/python scripts/data/record_duct.py --program silence --seconds 10 --dry-run
.venv/bin/python scripts/data/record_duct.py --program silence --seconds 10 \
    --confirm-user-present --confirm-volume-minimum --confirm-routing-and-geometry
# 4) 무음 전체 루프 (스피커 소리 없음, 3-스레드 검증)
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml \
    --set noise.type=silence --set engine.type=ort --run-seconds 10 \
    --confirm-speaker --confirm-user-present --confirm-volume-minimum
# 5) 실효 지연 측정 (처프 재생 — 사용자 입회·볼륨 최저!)
.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml --calibrate \
    --confirm-speaker --confirm-user-present --confirm-volume-minimum
# 6) I/O 지연 스윕 (선택)
.venv/bin/python scripts/bench/measure_io_latency.py \
    --confirm-speaker --confirm-user-present --confirm-volume-minimum
```

probe는 raw code 다양성, float 변환 RMS, peak, clipping을 함께 검사한다. 19:29 기준 두 채널
모두 PASS했지만 출력 직전마다 다시 확인하고, 실패 시 `--force`나 재생으로 우회하지 않는다.

### 짧은 보충 세션의 안전 수집

`record_session_batch.py`의 CSV는 각 행에 `path,seconds,source_family,group_id`를 가지며,
`lineage_key`가 없으면 `group_id`를 사용한다. split은 녹음 전에 각 행의 `split` 또는
명시적 `--preassigned-split`로 정해야 한다. 15초를 포함해 길이와 대상 수는 CSV/`--limit`에서
계획하며 코드에 고정하지 않는다. 먼저 아래처럼 dry-run으로 source-list exact SHA, 각 행의
source SHA·lineage·split, audible/output-open/connected 예상 시간과 hard timeout을 확인한다.
dry-run은 출력·입력 장치를 열지 않고 디렉터리나 progress 파일도 만들지 않는다.

```bash
.venv/bin/python scripts/data/record_session_batch.py \
    --sources data/<사전_split_계획>.csv --limit <이번_연결창_세션수> \
    --amplitude 0.06 --dry-run

# 위 계획을 그대로 실행할 때만 세 확인을 붙인다.
.venv/bin/python scripts/data/record_session_batch.py \
    --sources data/<사전_split_계획>.csv --limit <이번_연결창_세션수> \
    --amplitude 0.06 \
    --confirm-user-present --confirm-volume-minimum --confirm-routing-and-geometry
```

file playback 기본 amplitude는 기존 82세션과 같은 **0.06**이다. canonical additions는
이 값을 exact로 강제하여 실측 브랜치의 단일 레벨 계약을 보존한다. 일반 diagnostic batch만
명시적으로 다른 값을 줄 수 있으며 공용 peak 안전 상한 0.15는 그대로 적용된다.
기본값은 **재시도 없음**이다. 동일 소스를 다시 울려야 할 근거를 확인한 경우에만
`--retry-once`를 명시한다. 각 자식 세션은 `seconds + settle +
--session-timeout-overhead-seconds` hard timeout을 가지며 stdout/stderr가 터미널과
`results/recording_logs/record_session_batch/`에 동시에 기록된다. file source의 선택 구간은
CSV의 optional `start_seconds`(기본 0)까지 source-list SHA/행에 exact 결속되고, 해당 window가
파일 길이를 넘으면 출력 전에 실패한다. noise-output peak는 공용 안전 상한 0.15보다 완화할 수
없다. `record_duct.py`는 출력 stream이 닫히는 즉시
분리 안내를 먼저 내고, xrun·길이·정렬·저장 실패 때 메모리에 남은 raw와 metadata를
`results/recording_staging/record_duct/`에서 WAV SHA/size와 fsync를 마친 뒤 active tree로
atomic no-replace 발행한다. 실패 staging은
`results/recording_failures/record_duct/`의 새 no-replace 경로에 통째로 봉인한다. 다만 프로세스
강제 종료·전원 손실·첫 callback 전 실패는 메모리 raw가 없어 metadata/log만 남을 수 있으므로,
실패 직후 자동 재생하지 말고 해당 증거부터 오프라인 분석한다.
`record_duct.py`가 durable `failure.json`을 발행한 실패는 배치가 기계 판독 포인터를 검증해
`batch_progress.csv`의 `failure_stage`, `detail`, `failure_artifact`, `failure_receipt` 및 receipt
SHA에 기록한다. 따라서 stdout 마지막 안내 문구를 실패 원인으로 사용하지 않는다.

### Official P/S 연결 창 순서

긴 녹음은 선행하지 않는다. 먼저 전체 pytest, 장치 무점유 확인, 입력 probe와 아래 무음
dry-run을 끝낸다. 그 뒤 사용자 입회와 볼륨 최소 상태에서만 연속 운영 절차를 시작한다.
meter는 input-only preflight 1.5초 뒤 nominal **20.0초**/hard-max **21.0초**, strict P/S는
input-only preflight 총 3.0초 뒤 nominal **12.5초**/hard-max **13.5초**다. nominal audible
합계는 **32.5초**지만 장치 기동·명령 인계·분리/재연결을 포함한 wall-clock 시간은 고정값이
아니다. 각 출력 close 안내 즉시 물리 분리하고, 앰프 노브는 바꾸지 않은 채 다음 명령 직전에만
다시 연결한다. 분석 중에는 스피커가 필요 없다.

최초 1회에는 paired evidence가 아직 없으므로 정상 live gate가 `BLOCKED`인 것이 맞다. 이를
우회하는 일반 옵션은 없고, 아래의 명시적 bootstrap 두 명령만 허용된다. 첫 명령이 fresh
meter raw+SHA receipt를 새 경로에 보존하고, 두 번째 명령이 그 경로·10분 freshness·동일
logical hardware/channel과 ALSA physical fingerprint(`/proc/asound` PCM info, sysfs
realpath/uevent/안정 속성), recipe/status/target 및 운영자의 같은 노브 확인을 검증한다.
이 fingerprint를 live에서 수집할 수 없거나 evidence/meter/strict 중 하나와 다르면 출력 전에
실패한다. 이어서 **추가 자극 없이**
기존 12.5초 strict raw를 interleaved half로 사용해
`assets/measured/measurement_level_evidence.json`을 원자 생성·재검증한 뒤에만 P/S 분석과
official NPZ 승격으로 간다.

```bash
# 1) 소리 없음: 코드와 exact official 파라미터/출력 경로 검증
.venv/bin/python -m pytest -q
.venv/bin/python scripts/data/measure_paths_interleaved.py --dry-run \
    --primary-out results/path_measurement_next/p.npz \
    --secondary-out results/path_measurement_next/s.npz

# 2) 최초 1회 bootstrap: 사용자 입회·시작 전 볼륨 최소(출력 20초)
.venv/bin/python scripts/data/set_amp_level.py --bootstrap-level-evidence \
    --confirm-speaker --confirm-user-present --confirm-volume-minimum

# 위 명령이 출력한 meter_raw.npz 상대경로를 그대로 복사한다. 노브를 바꾸지 않는다.
METER_RAW=results/calibration_interleaved/level_bootstrap/<session>/meter_raw.npz

# 3) 같은 노브 확인 후 strict P/S(출력 스트림 12.5초, 별도 level probe 없음)
.venv/bin/python scripts/data/measure_paths_interleaved.py \
    --bootstrap-level-evidence --meter-raw "$METER_RAW" \
    --confirm-same-amplifier-setting --confirm-user-present \
    --confirm-volume-minimum \
    --confirm-routing-and-geometry \
    --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \
    --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
```

meter PASS가 출력한 fresh raw 경로와 충돌 없는 `<capture-id>` 출력명을 포함한 **strict 명령
전체를 그대로 복사**한다. meter 세션에는 `meter_raw.npz`와 `meter_raw.receipt.json`, strict
세션에는 `raw_measurement.npz`, `metadata.json`, `analysis_results.npz`,
`analysis_metadata.json`이 생긴다. 최초 bootstrap PASS 때만 paired evidence JSON이 생기고,
모든 분석 gate PASS 뒤 새 P/S NPZ가 no-replace로 승격된다. 기존 legacy P/S는 덮어쓰지 않는다.
canonical evidence가 이미 있는 이후에도 모든 strict live는 fresh meter가 필수다.
`set_amp_level.py`를 bootstrap 옵션 없이 같은 세 confirmation으로 실행하고, 출력된
`--meter-raw` strict 명령(bootstrap 옵션 없음)을 그대로 쓴다. 영구 evidence는 calibration
근거일 뿐 현재 앰프 노브 증거가 아니다.

`[스피커 출력 종료]`가 표시되는 즉시 스피커/앰프를 분리한다. meter→strict 시작 간격은
10분 이하여야 하며 receipt/SHA, 장치/채널, exact peak 0.003 recipe, xrun/close/완료 상태,
meter 목표와 strict ERR noise-bin 중 하나라도 어긋나면 raw만 보존하고 승격하지 않는다.
실패했더라도 보존 raw를 오프라인 분석하기 전에는 재측정하지 않는다. strict P/S 전 항목이
PASS하기 전에는 기존 82세션을 대체하는 장시간 재녹음을 시작하지 않는다.

입력 복구 전에도 noise speaker ch0의 출력 채널·누설·주관적 공진만 확인하려면, 사용자 입회와
앰프 볼륨 최저를 실제로 확인한 뒤 다음 정성 진단을 사용할 수 있다.

```bash
.venv/bin/python scripts/bench/playback_duct_probe.py \
  --confirm-volume-minimum --confirm-speaker --confirm-user-present
```

이 도구는 설정의 축방향 공진과 300Hz를 peak 0.002 단계 톤으로 재생하고 cancel ch1은 항상
무음으로 둔다. 마이크 데이터가 없으므로 생성되는 JSON은 자극 로그일 뿐 P(z), S(z), 지연,
감쇠 dB 또는 `duct.yaml` 갱신 근거가 아니다.

### 레퍼런스 마이크(ch1) 이력

2026-08-01 진단에서 ch1은 사실상 무신호였고 2026-08-03 19:11에는 ch0까지 raw −1로
고착됐다. 빠져 있던 pin17을 복구한 뒤 둘 다 정상 PASS했다. acoustic-ref와 recorded 수집도
각 세션 직전 두 채널 probe가 계속 PASS할 때만 진행한다.

## 4. 2차경로 S(z) 보정

### 역사적 자산 (assets/measured/, 현재 training-ready 아님)

| 파일 | delay | 검증 대역 | 그 대역 일관성 | 전대역 | 유지/전체 | 방식 | 비고 |
|---|---:|---|---:|---:|---:|---|---|
| `primary_path_il.npz` | **1602** | 150–1600Hz | **0.9993** | **0.9988** | 18/48 | interleaved | legacy diagnostic P(z), 재사용 금지 |
| `secondary_path_il.npz` | **1462** | 150–1600Hz | **0.9990** | **0.9984** | 18/48 | interleaved | legacy diagnostic S(z), 재사용 금지 |
| `*_il.npz.orig` | 1608 / 1465 | 150–600Hz | 0.973 / 0.956 | 0.920 / **0.781** | 16/16 | interleaved | **오염본 백업** (아래 경고) |
| `secondary_path_4s.npz` | 1342 | 150–600Hz | 0.40 | — | — | 순차 ESS | 폐기 (2026-08-05) |
| `secondary_path_legacy_512high.npz` | 2613 | — | 0.27 | — | — | 순차 ESS | 구버전 기록용 (block 512/high) |

`P − S = 140`, `lead = 116`, 앵커 반복 13, `capture_id = f7b0fecd…`는 역사적 진단값이다.
두 파일은 한 번의 재생에서 나왔지만, 그것만으로 현재 official 계약을 충족하지 않는다.
순차 ESS 는
두 측정 사이에 일어난 **출력 버퍼 프레임 슬립**이 P/S 상대 지연에 실려 lead 를 틀리게
만들 수 있다. 여기에 더해 USB DAC와 Tegra I²S ADC는 비동기다. 보존된 interleaved raw A의
adjacent-cycle 시간영역 관측은 주기당 약 **+620 ppm**을 보였고, 1600 Hz 톤을 약 0.124 bin
옮겨 guard=1 정수 FFT의 1--1.6 kHz P/S 교차성분을 약 15%까지 만들었다. 따라서 새 official
측정은 ERR/REF 공통 ``q`` 관측, 실제 제출 int16 PCM 기반 fractional-frequency joint LS,
cubic playback-grid 교차검증을 모두 통과해야 한다. ``+0.4 ppm/같은 Tegra 발진기``라는 이전
결론은 이 raw 증거와 모순되어 폐기한다.

현재 training-ready P/S는 2026-08-27의 같은 strict capture
`5ac1313488c8434bb4d672a36503df59`에서 나온다.

| 경로 | delay / bulk | 150–1600Hz consistency | repeats | 현재 용도 |
|---|---:|---:|---:|---|
| `primary_path_il_strict_5dc06fdd.npz` | 1386 / 1642 | 0.999821 | 19 | official P |
| `secondary_path_il_strict_5dc06fdd.npz` | 1245 / 1501 | 0.999716 | 19 | official S |

두 NPZ는 48k/256/low, ERR0/REF1/NS0/CS1, xrun/clip 0, observed submitted
int16 PCM, 공통 raw/analysis SHA, clock-q witness, fractional joint-LS, cubic
crosscheck와 compact round-trip을 같이 통과했다. handoff 256을 포함한
lead는 두 NPZ에서 115로 유도된다. 이 측정은 150–1600Hz 플랜트
식별을 증명하지만 2/4/8kHz 실제 ANC 감쇠를 증명하지는 않는다.
옛 raw를 `--dry-run`으로 분석해도 누락 provenance를 official로
승격할 수 없다.

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
> 스피커를 울리지 않고 저장된 legacy 캡처를 **진단용으로만** 재분석하려면:
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
# → 새 strict primary/secondary NPZ 경로만 고정한다. 지연/lead는 NPZ provenance를 읽는
#   TrainingTimingContract가 유도하며 duct.yaml에 숫자를 손으로 옮기지 않는다.
```

## 5. 안전 수칙

1. 모든 실행은 **ANC OFF 로 시작**한다 (A 키로 수동 ON).
2. TPA3116D2 볼륨은 최소에서 시작해 점진적으로 올린다.
3. 런타임 안전장치: 출력 리미터(0.2) / 클립 스트릭 자동 mute / 발산 워치독(+6dB·0.5s)
   / 추론 데드라인 워치독 — 이상 시 자동으로 상쇄 채널이 꺼진다.
4. 스피커에 소리를 내는 스크립트(`record_duct.py`, `calibrate_wideband.py`,
   `measure_io_latency.py`, `evaluate_session.py`, `run_realtime.py`)는 반드시 사람이
   현장에 있을 때 실행한다.
