# Stage-2 output-master P/S v3 경계

## 현재 판정

`scripts/data/measure_paths_stage2_2khz.py`의 v2 경로는 USB DAC output과 APE
input을 legacy combined callback의 같은 frame index로 묶었다. 두 장치의 hardware
clock이 다르므로 이 raw는 diagnostic-only다. P/S, plant binding, 학습 권한으로 사용할
수 없다. v2 `--execute-live`는 backend import 전에 차단된다.

실측 보존 raw
`results/stage2_2khz_ps_v2/diagnostic_phase_raw.npz`
(`a68d13b52eefaf1ccf7b439fda0f5d95353d4a8bd1a91dd2778ac9918ef7805b`)는
xrun/clip 0이지만 slot clock이 약 `-0.73~+3.38 kppm`으로 바뀌었다. nominal-bin
비선형 판정을 P/S 권한으로 승격할 수 없다.

## v3가 강제하는 것

`stage2_2khz_output_master_ps_v3.py`는 다음을 강제한다.

- 독립 `InputStream`/`OutputStream`에서 output callback만 제출 cursor와 pacing 소유
- input pre-roll과 post-roll 보존
- output PCM SHA와 가변 길이 input raw SHA를 별도 clock 축으로 기록
- 같은 index/같은 길이를 hardware frame identity로 해석하지 않음
- durable diagnostic clock JSON만 보지 않고 결속된 raw NPZ를 path/SHA로 다시 열어 재검증
- input start/complete marker를 callback axis·pre/post-roll과 exact 결속
- global q로 corrected 49/98 linearity를 재계산하고 receipt를 no-replace 발행·재계산
- P/S stream의 local q를 별도 추정하고 cubic/linear bounded resampling 교차검증
- P/S success raw를 analysis 전 no-replace 발행·재로드하며, 실패는 partial raw로 보존
- fresh meter raw/receipt, calibration, hardware config, resolved device, ALSA identity,
  exact origin/dev authority가 diagnostic→P/S 사이에 같은지 재검증
- 저장소 bytes에 결속된 physical level evidence만 DPSS fit/holdout analyzer가 소비
- legacy combined raw가 어떤 v3 blocker도 충족하지 못함

소프트웨어 adapter와 synthetic 2×2 LTI/q=250 ppm end-to-end는 PASS했다.
다만 실제 output-master diagnostic/P/S raw는 아직 없다. 실측 diagnostic
clock/linearity가 PASS하기 전에는 `ps_stream_may_open=false`이며, 실측 P/S
raw·DPSS holdout·3 dB feasibility가 PASS하기 전에는 training은 계속 차단된다.

## 무음 검증 명령

기본 실행은 sounddevice를 import하지 않고, 파일을 쓰지 않으며, 소리를 내지 않는다.

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz_v3.py
```

Diagnostic capture는 별도 diagnostic-only CLI로 실행한다. 아래 명령은
스피커 출력 11.605333초이며 P/S를 자동 실행하지 않는다.

```bash
.venv/bin/python scripts/data/capture_stage2_output_master_diagnostic.py \
  --execute-live --meter-raw <fresh-meter-raw.npz> \
  --expected-meter-raw-sha256 <64SHA> \
  --confirm-speaker --confirm-user-present --confirm-volume-fixed \
  --confirm-routing-and-geometry --confirm-same-amplifier-setting
```

Clock PASS 후 corrected linearity receipt를 무음 offline으로 발행한다. 이 단계는
speaker/backend을 열지 않는다.

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz_v3.py \
  --publish-diagnostic-linearity \
  --diagnostic-session results/stage2_2khz_output_master_diagnostic/<session> \
  --diagnostic-clock-sha256 <clock_receipt.json의-64자리-SHA256>
```

Linearity PASS receipt SHA를 받은 뒤에만 P/S 12.394667초 출력을 한 번 열다.

```bash
.venv/bin/python scripts/data/measure_paths_stage2_2khz_v3.py \
  --execute-live \
  --diagnostic-session results/stage2_2khz_output_master_diagnostic/<session> \
  --diagnostic-clock-sha256 <clock-64SHA> \
  --diagnostic-linearity-sha256 <linearity-64SHA> \
  --meter-raw <같은-fresh-meter-raw.npz> \
  --expected-meter-raw-sha256 <meter-64SHA> \
  --confirm-speaker --confirm-user-present --confirm-volume-fixed \
  --confirm-routing-and-geometry --confirm-same-amplifier-setting
```

세 live boundary 모두 clean attached `dev`, `HEAD == origin/dev`, tracked critical bytes를
요구한다. clean해도 push되지 않은 HEAD와 다른 branch는 backend import 전 실패한다.
Library의 exported `assess/run_stage2_output_master_ps_v3_if_admitted` 함수는
항상 audio 0회로 차단된다. 실제 capture state machine은 위 tracked CLI의
repository·ALSA·device·fresh-meter pre-open 검증을 통과한 경우에만 호출된다.

## v3를 실제 P/S 측정으로 열기 위한 합격 조건

- diagnostic과 P/S를 각각 고유 no-replace session으로 raw-first 발행
- partial/failure raw 보존 및 자동 재측정 금지
- input/output callback/status/timestamp/valid mask를 독립 보존
- global affine q가 ±1000 ppm interior, ambiguity, view/endpoint residual gate PASS
- cubic/linear resampling 결과 일치
- corrected diagnostic 49/98 level, SNR, THD/IMD, phase gate PASS
- P/S fit-a/fit-b와 untouched holdout의 모든 80–2828 Hz subband gate PASS
- DPSS representation만 lower-edge를 0 Hz까지 guard하며, authority/excitation
  80–2828.427 Hz와 upper edge는 늘리지 않음. guard 대역은 training/eval에 사용 금지
- physical operating level과 3 dB actuator feasibility를 actual raw SHA에 결속
- 그 뒤에만 plant binding과 training admission을 별도 no-replace artifact로 발행

이 조건 전에는 새 P/S 측정을 반복하거나 Stage-2 학습을 시작하지 않는다.
