# RT5640 J511 소프트웨어 연결 상태 게이트

## [가설]

Jetson 온보드 RT5640의 ALSA `CVB-RT Jack-state` control을 반복해서 읽으면, J511에
3.5 mm plug가 감지됐는지 여부를 소리를 내지 않고 확인할 수 있다고 가정한다.

## [근거]

실제 Jetson APE 카드에는 다음 enumerated control이 있다.

```text
numid=1156,iface=MIXER,name='CVB-RT Jack-state'
Item #0 'None'
Item #1 'HP'
Item #2 'MIC'
Item #3 'HS'
```

v8 transport 실행 직전에는 이 control이 세 번 연속 `None`이었다. 따라서 J511에
headphone/headset plug가 검출되지 않은 상태에서 exact-zero 동시 입출력 검증을 수행했다.

## [확인 방법]

다음 명령은 `amixer cget`만 읽으며 PCM, Pulse profile, mixer, pinmux, 장치 트림을 바꾸지
않고 어떠한 음도 내지 않는다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect None --samples 3
```

반대로 실제 J511→앰프 line cable을 연결한 뒤에는 먼저 관측값을 확인한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect HP --samples 3
```

`HP`가 아닌 상태이면 그 관측값을 바탕으로 배선·TRRS adapter를 확인하고, 임의로 `HP`를
가정하거나 실기 출력을 시작하지 않는다. 측정 전에는 이 gate 외에도 PCM 전역 무점유, 입력
preflight, operator confirmation, 앰프 최소 볼륨 확인이 계속 필요하다.

## [결과]

2026-08-29 실제 실행 결과:

```json
{
  "expected_state": "None",
  "observed_states": ["None", "None", "None"],
  "passed": true,
  "authority": {
    "j511_unplugged_detected": true,
    "j511_plug_detected": false,
    "amplifier_end_connected": false,
    "amplifier_power_state": false,
    "electrical_output_witness": false,
    "acoustic_output_witness": false
  }
}
```

## [판정]

**Confirmed — J511의 software-visible plug state는 `None`이다.** 이는 Jetson 잭에서
plug가 감지되지 않았다는 증거다.

다음은 이 control만으로는 판단할 수 없다.

- 케이블 반대편이 앰프 입력에 실제로 연결됐는지
- 앰프 전원·볼륨·스피커 출력 상태
- J511의 아날로그 전압, 실제 음향 출력, DAC↔ADC frame identity
- ERR/REF 물리 위치·극성·주파수응답

따라서 J511 state PASS를 P/S, ANC 감쇠, 공통 clock 또는 학습 적격성으로 승격하지 않는다.

## [다음 행동]

실제 광대역 P/S 연결 창에서는 J511 cable의 감지 상태를 먼저 raw receipt에 결속한다. 그 뒤
낮은 레벨의 channel/polarity 확인과 fullband clock/P/S gate를 순서대로 수행한다.
