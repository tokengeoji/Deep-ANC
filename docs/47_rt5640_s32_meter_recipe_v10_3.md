# RT5640 S32 fresh level-control meter recipe v10.3

## [가설]

기존 USB AB13X/S16 meter raw를 RT5640/J511 S32 fullband P/S의 level evidence로
재사용할 수 있다.

## [근거]

기존 level evidence와 meter raw에는 AB13X USB/S16 output bytes가 봉인되어 있다.
S16 PCM peak `98`을 S32에 단순 cast하면 normalized output이 약 96.3 dB 작아진다.
또한 기존 meter의 `-50.1±2 dBFS`는 150–1600 Hz control-band level이지 2/4/8 kHz
plant consistency 증거가 아니다.

## [확인 방법]

새 pure recipe는 같은 official float probe를 먼저 exact Q15로 quantize하고, `int64`
multiply + 16-bit signed left shift로 S32를 만든다. 20초는 960,000 frame = 3,750개의
256-frame callback이고 ch0만 nonzero, ch1은 exact zero다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/jetson/check_rt5640_s32_meter_static_v10_3.py
```

이 명령은 ALSA, sounddevice, PCM, GPU, result file을 열지 않는다.

## [결과]

- config: `configs/hardware_jetson_rt5640_fullband_s32_v10_3.yaml`
- recipe: `src/deep_anc/dsp/rt5640_s32_meter_v10_3.py`
- planned PCM: `<i4 [960000,2]`, ch1 zero, lower 16 bits zero
- Q15 peak: 98; S32 peak: 6,422,528; declared normalized probe peak: 0.003
- legacy low-band compatibility target: `-50.1±2 dBFS`
- v3 health band declaration: exact 125, 250, 500, 1k, 2k, 4k, 8k octaves

## [판정]

**Confirmed — no-audio recipe only.** static PASS는 amplifier level, J511 cable end,
actual negotiated S32 hw_params, output voltage, full-octave health, P/S, ANC attenuation
또는 canonical training을 증명하지 않는다.

## [다음 행동]

1. disarmed S32 stream에서 actual hw_params/route/J511/occupancy를 확인한다. arm 전
   callback은 exact zero여야 한다.
2. 사용자 입회 아래 20초 fresh S32 meter raw를 no-replace로 저장한다. capture 종료
   직후 반드시 `출력 종료 — 지금 스피커 분리`를 알린다.
3. meter raw는 10분 이내의 same-amplifier fullband P/S에만 level compatibility로
   결속한다. 고역 적격성은 88.388–11,313.708 Hz P/S raw와 electrical witness로 따로
   판정한다.
