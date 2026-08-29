# RT5640 full-octave v10 무음 static gate

상태: **구현·테스트 완료, live P/S는 BLOCKED**

이 문서는 USB AB13X/S16 기반 v6 P/S를 RT5640/J511으로 이름만 바꾸는 일을 막기 위한
별도 세대의 시작점이다. 이 단계는 오디오 장치·ALSA mixer·결과 파일을 열지 않는다.

## 고정한 것

- input: `APE,1` / I2S2 / ERR·REF / `S32_LE`, 48 kHz, 2 ch
- output: `APE,0` / I2S1 / RT5640·J511 / `S32_LE`, 48 kHz, 2 ch
- callback block: 256, latency: `low`
- control band: `broadband_full_octave_88_11314_v3`
  (`53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2`)
- excitation requirement: 80 Hz 이하부터 11,313.7084989848 Hz 이상
- live J511 상태 허용값: `HP` 또는 `HS`만. `None`/`MIC`는 live preflight 실패다.
- signal scale: Q15 actual-int16 plan을 S32에 **정확히 16 bit left shift**한다.
  `int16 -> int32` 단순 cast는 정규화 레벨을 약 96.3 dB 낮추므로 금지한다.

실행해도 무음인 확인 명령:

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_fullband_static_v10.py
```

출력의 `[PASS]`는 config/계약 PASS일 뿐이며 receipt 상태는 의도적으로 `BLOCKED`다.

## 이 gate가 증명하지 않는 것

- J511 케이블 반대편 앰프 연결·전원·아날로그 전압
- S32 전이중 stream의 hardware frame identity, sample drop/add/reorder 부재
- electrical/clock witness
- P(z), S(z), 125 Hz--8 kHz 식별·일관성·stationarity
- ANC attenuation, FxLMS 우위, quiet zone, canonical training/deployment eligibility

특히 현재 J511 software-visible 상태는 `None`이고, 그 상태에서 시도한 APE PCM0 입력은
2채널 electrical tap health gate에 실패했다. 따라서 이를 loopback witness로 사용하지
않는다.

## v10.1 S32 signal-only plan

`src/deep_anc/dsp/rt5640_fullband_s32_plan_v10.py`는 v5의 **signal-only Q15 waveform만**
입력으로 받아 S32 plan을 만든다. v5/v6 live authority·raw publisher·meter·USB identity는
상속하지 않는다. 이 module도 device/file write 없이 synthetic fixture만 통과한다.

- Q15→S32은 `int64(q15) * 65536` 후 signed int32 range 검사를 하는 exact 16-bit
  left shift다. float, simple cast, saturation은 금지한다.
- current deterministic plan: 557,056 frames × 2ch, 2,176 callbacks, 11.605333초,
  planned S32 SHA `dc897bde69b60e8df81d4d677cd68ae030b375e704d5082b214c1dbacd40c6ce`.
- 이 SHA는 **planned application PCM**일 뿐 DAC 전기 출력/physical sample identity의
  증거가 아니다. actual submitted PCM은 future duplex raw에서 별도 이름·SHA로 기록한다.

## 다음 순서

1. fail-closed S32 duplex primitive를 synthetic fixture로 검증한다. v6 raw/authority/path를
   재사용하지 않는다.
2. 안전한 external tap + 동기 ADC 또는 검증된 RT5640 TRRS capture를 준비해 electrical
   witness health를 통과시킨다. 이 장비가 없으면 v3 문서의 fixed-LTI conditional
   acoustic clock/stationarity evidence를 별도 raw로 충족해야 한다.
3. 앰프/스피커 연결 전 무음 dry-run, meter, output-close receipt를 모두 통과시킨다.
4. 사용자 입회·최소 볼륨의 한 연결 창에서 fresh P/S raw를 수집한다. 예상 시간과 raw
   경로를 사전에 고지하고, capture 종료 즉시 물리적으로 출력 연결을 해제한다.
5. 그 raw가 v3 8개 physical subband와 7개 objective octave 및 Stage-1 guard를 모두
   통과한 뒤에만 manifest와 학습 계약을 새 SHA에 재결속한다.
