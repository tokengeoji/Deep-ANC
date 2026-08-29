# 외부 동기 electrical witness admission v1

이 문서는 현재 Jetson에서 125 Hz–8 kHz canonical P/S를 열기 전에 필요한 **외부
전기 witness의 최소 경계**를 고정한다. 구현은
`deep_anc.dsp.external_electrical_witness_admission_v1`이고, 정적 확인 명령은 다음과
같다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/jetson/check_external_electrical_witness_static.py
```

이 명령은 PCM·ALSA·sounddevice를 열지 않고 `BLOCKED`를 반환하는 것이 정상이다.
`static_gate_pass=true`는 요구사항 JSON/YAML의 정합성만 뜻하며, physical PASS가 아니다.

## [가설]

외부 4-input ADC 하나를 추가하면 현재 덕트의 ERR·REF·noise/cancel 출력 전압을 동기화해
fullband P/S를 만들 수 있다.

## [근거]

- 현재 ERR/REF는 APE I2S2의 INMP441 두 채널이다.
- AB13X는 2ch playback이지만 capture는 1ch이고 APE와 clock domain이 다르다.
- RT5640/J511 S32 transport는 정적 계약과 disarmed primitive까지만 있으며, 실제 output
  voltage와 ERR/REF를 하나의 sample frame으로 잡는 recorder는 없다.
- USB/OS timestamp나 둘 다 “48 kHz”라는 표시는 physical frame identity가 아니다.

## [확인 방법]

아래 두 topology 중 하나만 이후 raw adapter의 후보가 된다.

| topology | 최소 조건 |
|---|---|
| `single_acquisition_clock_all_four` | ERR, REF, NOISE_TAP, CANCEL_TAP을 하나의 acquisition clock에서 동시 수집. 최소 4ch. |
| `ape_external_hardware_frame_bridge` | INMP441/APE ERR·REF를 유지하되 external tap recorder와 BCLK, WS, absolute frame counter의 연속 hardware witness를 보존. |

두 경우 모두 raw는 48 kHz, `<i4`, 연속 frame counter, xrun/drop/add 0이어야 한다.
8 kHz octave 상단 11.314 kHz에서 20 dB grade timing budget은 `≤0.0675518903 sample`이다.

## [결과]

현재 config는 topology, actual S32 callback SHA, native/canonical/analysis raw SHA 모두
`null`로 유지한다. 따라서 현재 판정은 의도적으로 다음과 같다.

```text
status = BLOCKED
electrical_witness_pass = false
fullband_plant_identification_pass = false
canonical_training_eligible = false
deployment_eligible = false
```

## [판정]

**Blocked.** 스피커를 물리적으로 연결한 상태는 실험을 무효화하지 않는다. 차단 원인은
연결 여부가 아니라 전기 출력, microphone, clock 계보를 함께 증명할 acquisition interface가
없다는 점이다.

## [다음 행동]

1. 아래 둘 중 하나를 실제 하드웨어로 확정한다.
   - ERR/REF도 external ADC로 옮겨 네 역할을 하나의 frame clock에서 취득한다.
   - INMP441를 유지하되 APE↔external recorder의 continuous BCLK/WS/absolute-frame bridge를 추가한다.
2. noise/cancel tap은 고임피던스·절연·DC block·감쇠된 안전 tap이어야 한다. ADC를 스피커
   단자에 직접 연결하면 안 된다. pre-amplifier line tap은 DAC 입력 witness일 뿐 amplifier/
   speaker transfer witness를 뜻하지 않는다.
3. 새 raw adapter는 planned/actual S32 callback SHA, aperiodic command→두 tap frame map,
   native raw·canonical raw·analysis SHA와 no-replace publication을 하나의 receipt로 결속한다.
4. 그 raw가 생긴 뒤에만 fullband P/S, THD/IMD, recorded v3 coverage, Elice pretrain을 순서대로 연다.

## spatial quiet-zone 범위

P/S 식별의 최소 profile은 4 input이다. 최종 quiet-zone 주장은 `REF + NOISE_TAP +
CANCEL_TAP + ERR 5 positions`의 최소 8 input 동시 profile이 필요하다. ERR를 한 위치씩
옮겨 얻은 순차 측정은 현재 final evidence로 허용하지 않는다.
