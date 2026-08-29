# 2026-08-29 2/4/8 kHz 단일점 결합 진단

## 목적과 범위

이 기록은 현재 AB13X USB DAC→앰프→덕트→APE ERR/REF 경로에서 2, 4, 8 kHz의
단일 tone이 실제 마이크까지 도달하는지를 확인한 짧은 **결합·배선 진단**이다. ANC와
FxLMS는 켜지 않았고, P(z)/S(z), 절대 지연, 위상, THD/IMD, 감쇠 또는 quiet-zone을
측정하지 않았다. raw는 `.gitignore`된 `results/`에 immutable no-replace artifact로
보존하며 canonical 데이터, 학습, ONNX, runtime 성능 근거로 사용하지 않는다.

실행 source commit은 `685b826569441d3742669d52f850102e1284e702`이다. 이 commit은
실제 출력 전에 같은 PCM 계획을 장치 open 없이 검증하는 `--dry-run`을 추가했다.

## [가설]

현재 연결의 noise speaker(NS)와 cancel speaker(CS)는 2/4/8 kHz tone을 덕트의
ERR/REF 두 마이크까지 전달할 수 있다.

## [근거]

- 실제 설정: `configs/hardware_jetson.yaml` — APE `hw:1,1` 2채널 S32 48 kHz 입력,
  AB13X `hw:2,0` 2채널 S16 48 kHz 출력, ERR/REF=`0/1`, NS/CS=`0/1`.
- 현재 앰프 위치는 같은 날 공식 level meter의 `-48.197 dBFS` PASS 조건으로 고정됐다.
  meter raw SHA-256은
  `ed6fddae136f468f7b44539874e35b996a71db1968760f5eb1495970e15f7028`이다.
- 출력 직전 PCM은 모두 `closed`였고, PulseAudio는 control node만 열고 있었다.
- 각 주파수의 live 전 dry-run은 48 kHz/256/low, peak `0.003`, raw/summary
  no-replace 경로와 silence-only channel을 검증했다. dry-run은 audio backend나
  ALSA/PortAudio 장치를 열지 않았고 결과 파일도 만들지 않았다.

## [확인 방법]

각 주파수마다 다음을 정확히 한 번만 실행했다.

1. ERR/REF S32 입력-only preflight 2초.
2. NS(ch0): 1초 무음 → 2초 tone → 1초 무음.
3. CS(ch1): 1초 무음 → 2초 tone → 1초 무음.
4. 두 출력에 zero block 47개(약 0.251초)를 쓰고 stream을 닫았다.

따라서 주파수 하나의 장치 점유는 약 10.251초, 실제 tone은 4초이고 세 주파수의
실제 tone 합계는 12초다. 각 raw의 callback xrun/clip/flush와 pre-tone 대비 Hann
단일-bin tone 상승을 검사했다. 결과가 종료된 뒤에는 스피커 분리 안내를 냈다.

## [결과]

모든 capture는 48 kHz/256/low, 2입력/2출력, `xrun=0`, steady 구간 clip ratio `0`,
zero flush underflow `0`, `stream_closed=true`였다.

| Tone | NS→ERR | NS→REF | CS→ERR | CS→REF | 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2 kHz | +50.05 dB | +60.63 dB | +41.52 dB | +58.08 dB | 네 경로 DETECTED |
| 4 kHz | +37.90 dB | +30.38 dB | +22.44 dB | +31.81 dB | 네 경로 DETECTED |
| 8 kHz | +23.30 dB | +44.33 dB | +25.06 dB | +43.72 dB | NS→ERR UNRESOLVED, 나머지 DETECTED |

8 kHz NS→ERR은 증가량이 +23.30 dB였지만, detector의 절대 tone floor
`-100 dBFS`보다 낮은 `-101.775 dBFS`여서 `DETECTED`로 승격하지 않았다. xrun이나
clipping 실패가 아니며, 이 진단을 반복하지 않고 raw를 보존한다.

별도로 raw int32를 직접 읽어 0.1초의 tone 양 끝을 제외한 직사각-bin projection도
계산했다. 8 kHz NS→ERR은 `-97.49 dBFS`, pre-tone 대비 `+18.34 dB`였다. USB DAC와
APE ADC가 독립 clock인 조건에서는 이 수치와 callback-window estimator의 차이 자체가
phase/시간축을 control authority로 사용할 수 없다는 추가 이유다. 둘 다 "tone 에너지의
도달"까지만 뒷받침한다.

| Artifact | UTC 생성 시각 | SHA-256 |
| --- | --- | --- |
| `results/channel_paths/coupling_only_20260829_diagnostic_2khz.npz` | 13:29:18 | `80d816031b51e1c0652de15365f93498786cecf298a369ffde9c42298f1021de` |
| 같은 2 kHz JSON | 13:29:18 | `de326be864c2cd181c1f763151cef9ddb706c6f1fd2f57b586d45c9aff93c610` |
| `results/channel_paths/coupling_only_20260829_diagnostic_4khz.npz` | 13:29:47 | `72b71d6185cb5f6d7fd3627c3c666270c280c5390d232d9b40c81d319b3c36af` |
| 같은 4 kHz JSON | 13:29:47 | `cfc5b0953b0657fee5c1f954fd0e14a400bcb04aa1bbcec31c6dcc45fad8bd95` |
| `results/channel_paths/coupling_only_20260829_diagnostic_8khz.npz` | 13:30:14 | `8d4e08f5977b40aea6539b743a89cd1b29541af560c55bc8d1f060e594d720a0` |
| 같은 8 kHz JSON | 13:30:14 | `753e43e6d7ecd79ea8654d89018f09bba0abf759639557e327aee1fcc79ecd4d` |

## [판정]

- **Confirmed (제한적):** 2/4 kHz에서 NS와 CS 모두 ERR/REF로 전달됐고, 8 kHz에서도
  CS→ERR 및 양 REF 경로가 detector 기준으로 관측됐다.
- **Inconclusive:** 8 kHz NS→ERR은 raw에 tone 상승은 있으나 현재 conservative absolute
  detector floor를 통과하지 못했다.
- **Invalid experiment:** 이 결과를 고주파 P/S, high-band Deep-ANC 감쇠, FxLMS 비교,
  THD/IMD, latency, spatial quiet-zone 또는 모델 성능으로 해석하는 모든 주장은 무효다.
  summary metadata도 `performance_claim_allowed=false`,
  `duct_identification_complete=false`로 봉인한다.

## [다음 행동]

1. 이 raw를 재생·재측정하거나 임계값을 완화하지 않는다.
2. 고주파 ANC 식별에는 actual submitted PCM과 ADC↔DAC 공통 clock/전기 witness,
   time-separated P/S, holdout 분석을 갖춘 새 raw-first capture adapter가 필요하다.
3. 최종 125 Hz–8 kHz quiet-zone은 `REF + NOISE_TAP + CANCEL_TAP + ERR_0..ERR_4`의
   same-frame 검증이 필요하다. 현 2채널 APE 입력은 그 대체가 아니다.
