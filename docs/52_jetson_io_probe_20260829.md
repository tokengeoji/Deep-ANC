# 52. Jetson I/O 무출력 확인 기록 — 2026-08-29

이 문서는 실제 덕트 ANC 성능이나 스피커 출력을 주장하지 않는다. 스피커를 분리하기 전
수행한 **무출력** I/O 판정의 한계를 명시적으로 보존한다.

## [가설]

J30 물리 pin 17 재연결 뒤 REF 마이크(ch1)가 다시 살아났고, 그 변화가 J511 스피커
출력 경로도 복구했을 수 있다.

## [근거]

- [하드웨어 배선표](02_hardware_setup.md#j30-40핀-헤더-물리-배선-2026-08-03-사용자-확정)는
  J30 pin 17을 `REF INMP441 L/R → 3.3 V → I2S right frame → ch1`로 정의한다.
- 출력은 별도 AB13X USB DAC 또는 RT5640/J511 경로를 쓴다. 따라서 pin 17과 J511
  plug detect는 같은 신호가 아니다.

## [확인 방법]

스피커를 열지 않는 다음 두 검사를 실행했다.

```bash
# J511 codec 상태만 세 번 읽는다. PCM/스피커 출력은 열지 않는다.
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect HP --samples 3
amixer -c 1 cget name='CVB-RT Jack-state'

# APE 입력만 2초 열어 ERR/REF를 검사한다. 스피커 출력은 열지 않는다.
.venv/bin/python scripts/bench/check_audio_input.py --seconds 2 --require-both
```

## [결과]

### J511 codec plug detect

```text
expected='HP', observed=['None', 'None', 'None']
CVB-RT Jack-state: values=0 (None)
```

동시에 읽은 `/proc/asound/card*/pcm*/sub*/status`는 모두 `closed`였다. 이 검사는
스피커 출력이나 설정 변경을 하지 않았다.

### APE ERR/REF input-only capture

```text
ERR ch0: RMS -67.40 dBFS, peak 0.001593, clip 0.000%, unique 16290
REF ch1: RMS -58.56 dBFS, peak 0.004949, clip 0.000%, unique 36079
```

두 채널은 `--require-both` 조건을 PASS했다. 이 terminal probe는 raw artifact를 발행하지
않으므로 P/S, ANC 감쇠, 입력 레벨 calibration authority에는 사용할 수 없다.

## [판정]

| 주장 | 판정 | 이유 |
|---|---|---|
| J30 pin 17 재연결 뒤 REF 입력이 살아 있다 | Confirmed | ERR/REF input-only 2채널 PASS |
| J511에 HP/HS가 연결되어 있다 | Contradicted | read-only jack-state 3회가 모두 `None` |
| 실제 앰프·스피커에서 소리가 난다 | Inconclusive | 스피커 분리 전까지 출력 시험을 실행하지 않음 |
| 현재 결과로 ANC 감쇠 dB를 주장할 수 있다 | Contradicted | ON/OFF raw session, plant, runtime receipt가 없음 |

## [다음 행동]

스피커를 다시 연결하는 연결 창에서만 아래 순서로 진행한다.

1. Jetson 쪽 출력 경로가 AB13X USB DAC인지, RT5640 J511인지 물리적으로 확인한다.
2. J511 경로를 쓰는 경우 소리를 내지 않고 `CVB-RT Jack-state`가 연속 `HP` 또는 `HS`인지
   다시 읽는다. USB DAC 경로라면 J511 `None`은 출력 불능 증거가 아니므로 AB13X device
   preflight를 별도로 한다.
3. 사용자 입회와 앰프 볼륨 최소가 명시된 뒤에만, noise speaker ch0에 300 Hz·0.001
   3초(정착 1초 포함) 출력과 ERR/REF recording을 수행한다. ANC는 OFF로 유지한다.
4. stream close 직후 스피커를 분리하고, raw/metadata가 생긴 뒤에만 실제 출력 여부를
   판정한다.

이 gate가 열리기 전에는 fullband P/S, ANC ON, 성능 수치 또는 학습 plant authority를
발행하지 않는다.
