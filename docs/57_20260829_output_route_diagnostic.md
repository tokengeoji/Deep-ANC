# 2026-08-29 출력 경로 진단 기록

## 목적과 범위

이 기록은 앰프의 작은 gain 조정 뒤 `noise` 출력 채널에서 300 Hz가 실제 덕트의
ERR/REF 입력까지 도달하는지만 확인한 짧은 **진단**이다. ANC를 켜지 않았고,
P/S를 식별하지 않았으며, 모델·고역·감쇠 성능을 검증하지 않았다. 아래 raw는
collection plan이 없는 `unbound_diagnostic`이고 canonical 데이터나 학습 자료로
사용하지 않는다.

## [가설]

앰프 minimum 위치에서는 출력이 마이크 잡음 바닥에 묻혔고, 사용자 조정 한 단계 뒤에는
noise speaker → 덕트 → ERR/REF의 물리 경로가 살아 있을 가능성이 있다.

## [근거]

둘 다 48 kHz, 3초, 300 Hz, DAC amplitude 0.001, ANC OFF로 실행했다. 첫 번째는
앰프 minimum 위치, 두 번째는 사용자 승인으로 그 위치에서 gain 한 단계를 조정한 뒤였다.
출력은 noise channel뿐이고 cancel channel은 무음이었다. 원본은 실패 raw로 no-replace
보존됐다. 따라서 두 번째는 strict level 측정의 최소-volume 조건을 충족했다고 주장하지
않으며, route 진단 외 용도로 사용하지 않는다.

| 조건 | raw 디렉터리 | `mics_raw.wav` SHA-256 | `source_raw.wav` SHA-256 | 수집 판정 |
| --- | --- | --- | --- | --- |
| minimum | `results/recording_failures/output_route/20260829_211318_815792_timeline_gate_98e54e5e/` | `cc6ed10bc6b437d6570f276761294ae780e0402d03cf54c2c29af470b0d266a3` | `c3f845d3719b15e55f23deb8422fedc22c5213b03bee36a1a54e456241f9f78c` | `timeline_gate` 실패 |
| 사용자 조정 한 단계 | `results/recording_failures/output_route_step1/20260829_211514_123793_timeline_gate_aa6785cb/` | `2665f4d3995bf5312d605b24556057332a0905091429b84069d5dbefc955e300` | `427e06200ab1abaec367b55c4fe697a591bc997a33260d959cf91f963254c49f` | `timeline_gate` 실패 |

두 번째 raw의 `failure.json` SHA-256은
`764b7c305820073320109b62b8cb1546e257fbc43972e7592e457ae235d2f6d0`이다.
첫 번째의 것은 `b77edd2afd6be138a29d471c5ea51b43d65f32a429a4e847d7e394e0419b5e2d`다.

## [확인 방법]

두 raw의 중앙 3초(48,000–192,000 frames)를 독립적으로 읽었다. source의 300 Hz
FFT bin 및 8192-frame Welch coherence(4096 overlap)를 비교했다. 두 번째 raw에서
300 Hz의 coherence²는 source→ERR `0.741641`, source→REF `0.730343`, ERR↔REF
`0.992065`였다. minimum raw의 같은 값은 각각 `0.012588`, `0.060609`,
`0.002447`였다.

공식 recorder의 150–700 Hz timeline witness도 그대로 다시 판정했다. 두 번째 raw는
`coh2=0.018797`, `valid_window_ratio=0.0`, REF→ERR wide coherence `0.358308`으로
기록되어 있다. 이 판단을 변경하거나 임계값을 낮추지 않는다.

## [결과]

gain을 한 단계 조정한 두 번째 raw에는 300 Hz 전송과 ERR↔REF의 강한 좁은대역
상관이 나타났다. 따라서 이 짧은 조건에서는 physical output-to-microphone path가
없는 상태라는 가설과 맞지 않는다.

그러나 단일 300 Hz tone은 150–700 Hz에 걸친 비주기 timeline 정렬 witness가 아니다.
따라서 recorder가 `timeline_gate`를 거부한 것은 정상이며, 이 raw로 session alignment,
P/S, lead, ANC OFF/ON 감쇠, 2/4/8 kHz 또는 broad-band 성능을 주장할 수 없다.

## [판정]

**Likely — 낮은 gain에서 생겼던 무신호 문제는 해소됐고, 한 단계 조정 조건의 noise
speaker→ERR/REF route는 좁은 300 Hz 진단에서 관측됐다.**

**Invalid experiment — canonical recording/P/S/ANC 성능 검증 관점에서는 timeline witness가
없으므로 실패 raw로만 보존한다.**

## [다음 행동]

1. 이 raw를 canonical manifest, P/S, training, ONNX/runtime 또는 성능 표에 넣지 않는다.
2. 다음 음향 창 전에는 strict P/S 또는 persistently-exciting broadband 계획을 무음
   dry-run으로 완결하고, 장치 점유·배선·볼륨·예상 출력 시간·저장 경로를 다시 확인한다.
3. 8-input synchronous acquisition blocker가 남아 있으므로, 이 2-channel diagnostic은
   최종 quiet-zone/125 Hz–8 kHz physical claim의 대체 증거가 아니다.
