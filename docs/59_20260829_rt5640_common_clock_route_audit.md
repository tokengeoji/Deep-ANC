# 2026-08-29 RT5640/J511 common-clock 출력 후보 감사

## [가설]

USB AB13X DAC 대신 `APE PCM0 → ADMAIF1 → I2S1 → RT5640 → J511`을 쓰면,
APE PCM1/I2S2의 ERR/REF 입력과 같은 SoC 오디오 clock 계열에서 동작할 수 있는
고주파 P/S의 출력 후보가 될 가능성이 있다.

## [근거]

2026-08-29 KST read-only 감사, source commit
`ad08f64885e85122bae1ceccd99b53a02422b2f4`에서 다음을 직접 확인했다.

- `/sys/bus/i2c/devices/8-001c`는 `realtek,rt5640`, driver `rt5640`, DT status
  `okay`로 노출된다.
- APE mixer routing은 `I2S1 Mux=ADMAIF1`, `ADMAIF1 Mux=I2S1`,
  `ADMAIF2 Mux=I2S2`, `I2S2 Mux=ADMAIF2`다.
- `configs/hardware_jetson_rt5640_fullband_v10.yaml` (SHA-256
  `5fe219b4e2026d09fffc276aa5ad7e99a84e46e47bdcdefe08284e7af83ecfa4`)은
  48 kHz/256/low, input APE PCM1 S32 ERR/REF와 output APE PCM0 S32
  `ADMAIF1_I2S1_RT5640_J511`을 정확히 선언한다.
- runtime DT의 `/proc/device-tree/sound/clock-names`에는 `pll_a`, `plla_out0`,
  `extern1`이 있고, `src/deep_anc/audio_io.py`는 I2S1–I2S6가 APE `PLL_A`를
  공유한다고 명시한다.
- 반대로 현재 AB13X USB device는 playback `ADAPTIVE`, capture `ASYNC` endpoint다.
  APE I2S microphone BCLK/WS와 absolute frame이 같다는 artifact는 없다.

## [확인 방법]

ALSΑ mixer/DT/codec binding/config만 읽었다. PCM stream, output, mixer, pinmux,
device tree와 시스템 설정은 바꾸지 않았다. J511 state는 프로젝트의 read-only gate로
세 번 확인했다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect None --samples 3
```

static route config checker도 output을 열지 않는 조건에서 통과했다.

## [결과]

J511 gate의 실제 결과는 다음과 같다.

```json
{
  "expected_state": "None",
  "observed_states": ["None", "None", "None"],
  "passed": true,
  "authority": {
    "j511_plug_detected": false,
    "j511_unplugged_detected": true,
    "electrical_output_witness": false,
    "acoustic_output_witness": false
  }
}
```

APE PCM0/PCM1은 모두 `closed`였고 idle I2S1 rate/frame/master 값도 0이었다.
`CVB-RT HP Channel Switch`는 `off,off`다. 이는 idle 상태의 관측일 뿐 active stream에서
어떤 자동 설정이 되는지, J511의 실제 전압, 앰프 입력 연결 또는 DAC↔ADC sample identity를
보여 주지 않는다.

기존 RT5640 ADC input-only probe도 ch0 rail/clip 및 ch1 stuck으로 실패했다. 이는
RT5640 **출력**이 실패했다는 증거는 아니지만, 현 상태의 RT5640 ADC를 electrical
witness로 바로 사용할 수 없다는 반대 증거다.

## [판정]

| 항목 | 판정 |
| --- | --- |
| RT5640/J511 output route가 Jetson에 노출됨 | Confirmed |
| APE 내부 common-PLL topology 후보 | Likely |
| USB DAC보다 timing 구조상 유리한 후보 | Likely |
| J511 plug/cable을 현재 software가 감지 | Contradicted (`None`) |
| 실제 앰프까지의 아날로그 출력 | Inconclusive |
| ADC↔DAC absolute hardware-frame identity | Inconclusive |
| high-band P/S·학습·ANC authority | BLOCKED |
| 8-input spatial quiet-zone acquisition의 대체 | Contradicted |

## [다음 행동]

1. J511에서 앰프 input으로 가는 line cable이 물리적으로 연결됐을 때만, 위 read-only
   gate가 세 번 연속 `HP` 또는 `HS`가 되는지 먼저 확인한다. `None`이면 TRS/TRRS
   adapter·cable·jack detection을 해결하기 전 mixer/pinmux를 추측으로 바꾸지 않는다.
2. plug gate가 통과한 뒤에도 실제 sound를 내기 전에 disarmed S32 device-open/transport
   admission을 통과시킨다. zero-duplex PASS는 physical output, common frame 또는 P/S
   authority가 아니다.
3. 그 뒤에만 낮은 level output/electrical witness/fullband P/S를 별도 raw-first 계획으로
   검증한다. 최종 quiet-zone에는 여전히 8-input same-frame acquisition이 필요하다.
