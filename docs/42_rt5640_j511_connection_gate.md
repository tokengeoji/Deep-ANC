# RT5640 J511 소프트웨어 연결 상태 게이트

## [가설]

Jetson 온보드 RT5640의 ALSA `CVB-RT Jack-state` control을 반복해서 읽으면, J511에
결합된 Intel HD Audio front-panel breakout의 headphone/headset jack plug가 감지됐는지
여부를 소리를 내지 않고 확인할 수 있다고 가정한다. J511 자체는 3.5 mm socket이 아니라
10-pin HD Audio header다.

## [근거]

실제 Jetson APE 카드에는 다음 enumerated control이 있다.

```text
numid=1156,iface=MIXER,name='CVB-RT Jack-state'
Item #0 'None'
Item #1 'HP'
Item #2 'MIC'
Item #3 'HS'
```

v8 transport 실행 직전에는 이 control이 세 번 연속 `None`이었다. 따라서 J511 HDA
header에 결합된 front-panel breakout의 headphone/headset plug가 검출되지 않은 상태에서
exact-zero 동시 입출력 검증을 수행했다.

`HP`는 headphone detect만, `HS`는 RT5640 HDA-header mode에서 headphone과
mic/presence detect가 함께 active인 Linux jack state다. `HS` 자체는 TRRS plug나 앰프
반대편 연결의 증거가 아니므로, 실제 P/S 전 gate는 안정적인 `HP` **또는** `HS`를 허용한다.

## [확인 방법]

다음 명령은 `amixer cget`만 읽으며 PCM, Pulse profile, mixer, pinmux, 장치 트림을 바꾸지
않고 어떠한 음도 내지 않는다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect None --samples 3
```

반대로 J511에 keyed Intel HD Audio front-panel breakout을 결합하고, 그 breakout의
headphone 3.5 mm jack에서 앰프 line input까지 stereo TRS line cable을 연결한 뒤에는 먼저
관측값을 확인한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect HP --samples 3
```

위 명령의 observed state가 `HS`라면, HDA sense wiring이 유효한 상태일 수 있으므로
`--expect HS --samples 3`으로 다시 확인한다. 실제 Stage-2 gate는 두 상태를 모두 허용한다.

`HP` 또는 `HS`가 아닌 상태이면 HDA breakout/harness의 keyed 결합과 headphone-jack
detect 배선을 확인하고, 임의로 상태를 가정하거나 실기 출력을 시작하지 않는다. 단순
TRS/TRRS 변환 젠더만으로는 J511 header의 dedicated detect pin을 대체할 수 없다. 측정
전에는 이 gate 외에도 PCM 전역 무점유, 입력 preflight, operator confirmation, 앰프 최소
볼륨 확인이 계속 필요하다.

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

J511이 `None`인 같은 상태에서 APE PCM0(=I2S1/RT5640 codec 방향)을 **입력 전용**으로
1초 settle + 3초 probe했다. 이 작업은 output stream을 열지 않았고 종료 후 모든 PCM은
다시 `closed`였다. 하지만 ch0은 RMS `-4.767 dBFS`, peak `1.0`, clip `1.191%`로 rail
gate에 실패했고, ch1은 RMS `-189.645 dBFS`, raw `[-1, 0]`, unique code 2로 stuck이었다.
이 probe는 raw artifact를 발행하지 않은 일회성 hardware capability 진단이며 P/S나
clock evidence가 아니다.

## [판정]

**Confirmed — J511 HDA header 경로의 software-visible plug state는 `None`이다.** 이는
결합된 front-panel breakout의 headphone/mic sense가 RT5640에 감지되지 않았다는 증거다.

다음은 이 control만으로는 판단할 수 없다.

- 케이블 반대편이 앰프 입력에 실제로 연결됐는지
- 앰프 전원·볼륨·스피커 출력 상태
- J511의 아날로그 전압, 실제 음향 출력, DAC↔ADC frame identity
- ERR/REF 물리 위치·극성·주파수응답

따라서 J511 state PASS를 P/S, ANC 감쇠, 공통 clock 또는 학습 적격성으로 승격하지 않는다.
특히 현재 `None` 상태의 PCM0 capture는 정상적인 2채널 electrical tap이 아니므로 이를
J511 전기 witness 입력으로 쓰지 않는다.

## [다음 행동]

실제 광대역 P/S 연결 창에서는 J511 HDA front-panel breakout의 headphone-jack 감지 상태를
먼저 raw receipt에 결속한다. 독립 electrical witness가 필요하면 J511 pre-amp tap을 안전한
감쇠/DC-block 회로를 거쳐 2채널 동기 ADC 또는 검증된 RT5640 capture에 넣고, 그 입력
health부터 다시 PASS시켜야 한다. 그 뒤 낮은 레벨의 channel/polarity 확인과 fullband
clock/P/S gate를 순서대로 수행한다.
