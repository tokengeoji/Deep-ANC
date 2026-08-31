# 광대역 Deep-ANC 강제 가드레일

> 상태 기준일: 2026-08-28. 이 문서는 `docs/16`의 150–1600 Hz Stage-1 계약을 삭제하지
> 않고, 사용자가 최종 목표로 확정한 2/4/8 kHz까지의 실제 ANC와 matched FxLMS 우위를
> 별도 세대로 정의한다. 문구가 아니라 raw artifact와 코드 검사가 권위다.

## 1. 최종 목표

최종 성공은 다음을 동시에 만족해야 한다.

1. **저역 유지**(`positive_attenuation`): 150–1600 Hz의 모든 부대역에서 실제 감쇠가 양수다. FxLMS보다 반드시
   우수할 필요는 없지만 한 저역의 이득으로 다른 저역의 증폭을 숨길 수 없다.
2. **고역 우위**: 2/4/8 kHz octave와 고역이 포함된 speech/music/environment/machine에서
   Deep-ANC 감쇠가 같은 조건의 튜닝된 FxLMS보다 크고, paired bootstrap 95% CI 하단이
   0 dB보다 크다.
3. **모든 소리**: speech, music, environment, machine 네 family와 모델 선택 뒤 새로 얻은
   Level-5 실제 덕트 소리를 모두 통과한다.
4. **실시간성**: 48 kHz/256 sample에서 xrun, deadline miss, engine exception,
   ring drop/add, fallback, watchdog, sample slip은 exact 0이다. ring의 absolute maximum backlog는 정상 one-hop 허용량
   256 samples 이하여야 하고, 그 허용량을 넘은 maximum excess backlog는 exact 0이어야
   한다. 고정 지연은 모델링할 수 있지만 비결정적 1-sample slip은 허용하지 않는다.
5. **공간 판정 분리**: 단일 ERR 지점 감쇠를 quiet zone으로 부르지 않는다. 1.633 kHz 위
   고차모드 구간은 최소 5개 ERR 위치에서 별도로 검증한다.

8 kHz는 이 프로젝트의 octave **중심**이다. 따라서 octave 전체 성능 주장을 위한 식별·데이터
상단은 `8000×sqrt(2) = 11,313.708... Hz`다. 8 kHz에서 자극을 끝내거나 8 kHz 단일 톤만
통과한 결과를 8 kHz octave 성공으로 해석하지 않는다.

코드 단일 출처는
[`src/deep_anc/dsp/control_band_contract.py`](../src/deep_anc/dsp/control_band_contract.py)다.

Jetson runtime 증거의 별도 단일 출처는
`src/deep_anc/eval/broadband_runtime.py`다. 최소 30초 raw runtime log에서
P99 `<3 ms`, max `<5.333 ms`, plant/checkpoint/deployment/runtime lead exact 일치와
miss/engine-error/xrun/drop/add/fallback/watchdog/sample-slip exact 0, absolute backlog `≤256`,
maximum excess backlog `=0`을 동시에 요구한다. producer/consumer 관측 race에 따라
정상 absolute backlog는 0 또는 256일 수 있으므로 absolute 0을 강제하지 않는다.
8 kHz의 1 sample은 60°이므로 평균 latency가 빠르다는 사실만으로 이 gate를 대신할 수 없다.
첫 callback의 의도된 handoff prime은 stream open 전에 exact zero 한 block을 한 번만
발행하고 별도 counter로 보존한다. 이 prime은 fallback으로 세지 않지만 두 번 발행하거나
prime 이후 실제 fallback이 한 번이라도 생기면 runtime PASS가 아니다.

## 2. Stage-1과 최종 광대역 계약의 관계

| 역할 | Point-control subband | 고역 의미 | 최종 배포 자격 |
|---|---|---|---|
| `stage1_strict_150_1600_v1` | 150–300 / 300–600 / 600–1000 / 1000–1600 Hz | 2/4/8 kHz 증폭 방지 진단만 | 없음 |
| `broadband_point_control_150_11314_v2` | 위 네 구간 + 1600–2828 / 2828–5657 / 5657–11314 Hz | 모든 구간 bilateral cancellation 목표, 2/4/8 kHz matched FxLMS 우위 | 모든 후속 게이트 통과 시에만 가능 |

기존 strict P/S를 삭제하거나 숫자만 늘리지 않는다. v1 checkpoint/P/S/G4/ONNX를 v2로 이름만
바꾸는 것은 금지한다. v2 checkpoint, init/resume, export, runtime은 control-band contract SHA,
P/S SHA, timing SHA가 모두 정확히 같아야 한다.

## 3. 현재 실제 상태

### 3.1 확인된 것

- strict P/S capture `5ac1313488c8434bb4d672a36503df59`는 같은 raw/analysis에서 나온
  48 kHz/256/low 자산이다.
- P/S effective delay는 1386/1245, handoff 256이며 `PlantDelays.lead()`는 115 samples다.
- 150–1600 Hz consistency는 P 0.999821, S 0.999716, kept repeat 19, xrun 0이다.
- NS와 CS는 같은 AB13X USB DAC stereo stream을 사용하므로 두 출력은 같은 DAC clock이다.
  USB DAC와 APE I2S ADC 사이만 비동기다.
- runtime callback은 이제 PortAudio의 ADC/DAC/current timestamp, exact frame 수, callback
  완료, engine-step budget 초과, output fallback, xrun, ring drop/add, absolute/allowed/excess
  backlog와 watchdog을
  별도 counter로 immutable session receipt에 결속한다. 이 구현의 무음 회귀 100개는
  통과했지만 실제 acoustic session 증거는 아직 없다.

### 3.2 차단된 것

- strict P/S excitation upper는 P 1648/S 1640 Hz다. 11.314 kHz 광대역 plant가 아니다.
- 2026-08-27의 60–8000 Hz 진단 raw는 고역 에너지는 관측됐지만 저장된 clock 계약에서
  valid repeat가 0/64다. threshold를 낮춰 승격하지 않는다.
- current 82 recorded 세션은 실제 ERR target-d 기준 1600 Hz 위 coverage가 부족하다.
  2.828 kHz 이상은 family×split 독립 group이 사실상 0이다.
- recorded-v2 source 계약은 compressed decode 전체 계보, 최소 3개 독립 short-component
  조합, actual Q15 source 9×7 및 physical causal P 적용 predicted-ERR 9×7 재계산까지
  구현·회귀됐다. 그러나 실제 causal P와 검증된 source bytes가 없어 현재 확보 수는
  **0/48**이며 issuer/live authority는 `None`이다.
- 현재 canonical 광대역 checkpoint, matched physical FxLMS A/B, 다점 quiet-zone raw,
  Level-5 unseen raw는 없다.
- Python callback에 도달하기 전에 PortAudio/ALSA가 버린 ADC period는 위 software
  telemetry만으로 0건임을 증명할 수 없다. 따라서 구조가 정상이어도 clock authority는
  최대 `INCONCLUSIVE`이며, 공통-clock electrical witness 또는 동등한 물리 증거 전에는
  runtime PASS로 승격하지 않는다.

따라서 현재 판정은 다음과 같다.

```text
150–1600 Hz Stage-1 준비: 별도 기존 게이트로 판정
2/4/8 kHz point-control 학습 준비: BLOCKED
2/4/8 kHz quiet-zone: BLOCKED
FxLMS 고역 우위: Not yet demonstrated
```

현재 자산의 실제 차단 사유는 다음 read-only 명령으로 재계산한다.

```bash
.venv/bin/python scripts/data/audit_broadband_prerequisites.py --plant-only

# 82개 WAV까지 실제로 읽는 diagnostic audit
.venv/bin/python scripts/data/audit_broadband_prerequisites.py \
  --output results/data_audit/broadband_prerequisite_<generation>.json
```

이 보고서는 현재 blocker를 찾는 diagnostic evidence다. campaign readiness receipt로 승격하지
않는다.

## 4. 광대역 P/S 측정 계약

기존 dense fullband 자극과 v4 panel별 phase stitch를 반복하지 않는다. v4는 각 drive가
16 Hz grid라 bulk delay가 3000 samples마다 alias되고, 실제 strict drift
2.4835888604 samples/0.125 s가 panel 간 약 181 samples씩 누적되는데도 ±16 samples
결과 기반 stitch를 사용했다. 따라서 v4 plan/artifact는 diagnostic-only다.

v5 측정기는 한 continuous capture 안에서 P-only/S-only 0.25초 비주기 marker와 guard를
먼저 재생하고, 각 panel에 150–600 Hz pilot과 독립 high tone을 함께 넣는다. 중요한 차이는
intended float pilot을 clock authority로 믿지 않는다는 것이다. panel별 high tone을 더한 뒤
int16 양자화하면 실제 pilot-bin spectrum이 panel마다 달라진다. 분석기는 **각 period의 실제
submitted int16 복소 spectrum**을 분모로 사용해 이 차이를 exact 제거한 뒤, ERR/REF×P/S
네 trajectory가 하나의 global ADC→DAC map에 동의하는지를 검증한다. highband transfer는
map fit이나 phase repair에 사용하지 않는다.

이 나눗셈은 같은 pilot bin의 반대 DAC channel이 실제 int16 DFT에서도 null일 때만
식별 가능하다. 따라서 각 panel과 분석에 사용한 모든 period에서 main magnitude가 0이
아니고, 반대 channel의 absolute magnitude가 `1e-8` 이하이며 main 대비 ratio가 `1e-12`
이하인지 actual PCM으로 재계산한다. 하나라도 깨지면 joint P/S 분리로 간주하지 않고
authority를 즉시 차단한다. intended float의 직교성을 이 증거로 대체할 수 없다.

```text
100–1800
1400–3200
2800–6000
5400–8500
7800–11400 Hz
```

panel당 분석 반복은 63회다. panel 사이에는 첫 period를 tail guard로 버리고 뒤 9개에서
8개 exact adjacent witness를 얻도록 0.125초 저역 anchor를 10회 삽입한다. marker/guard와
block padding을 포함한 v5 signal-only 출력은 **49.627초**, hard maximum은 50초다.

현재 v5 live authority는 의도적으로 `None`이다. 새 plan의 exact file/payload/PCM SHA는
root 최종 검토 뒤에만 고정한다. 기존
`broadband_measurement_signal_plan_live_authority_v4_20260828.json`과 그 SHA들은 보존하지만
live에서 거부한다. 즉 현재 가능한 것은 signal-only dry-run뿐이며 실제 출력 명령을
발행할 수 없다.

live 경로는 exact saved plan, 10분 이내 fresh meter raw, paired level evidence, 사용자/
speaker/볼륨최저/배선·기하/same-amp 확인을 모두 요구한다. audio lock과 PCM gate 뒤
3초 input-only preflight를 통과해야만 NS ch0와 CS ch1을 열며, 종료 즉시 stream을 닫고
스피커 분리 안내를 먼저 출력한다. 정상·partial 어느 경우도 분석보다 immutable raw를
먼저 no-replace 발행하고, partial/PCM mismatch는 `INVALID`로 보존한다.

광대역 직전 레벨 미터는 legacy strict 후속 명령을 재사용하지 않는다. v5 authority가
고정된 뒤에만 saved plan의 exact recipe와 새 `results/` raw target을 **미터 장치 import
전에** 검증하고, PASS 뒤 fresh meter raw와 다섯 확인을 포함한 live 명령을 출력한다.

```bash
.venv/bin/python scripts/data/measure_paths_broadband_interleaved.py --dry-run
```

기본 `--followup-mode strict`와 bootstrap 안내는 역사적 strict 복구 호환을 위해 그대로다.
광대역 모드에서는 `--bootstrap-level-evidence`, plan 누락·변조, 저장소 밖 plan, 기존/저장소 밖
raw target을 모두 출력 전에 거부한다.

광대역 followup은 JSON 의미 비교만으로 plan을 승인하지 않는다. 새 v5 authority의 exact
repository-relative path와 file SHA, canonical payload SHA, PCM SHA를 모두 meter raw의
`broadband_meter_followup_v1`에 봉인한다. 같은 계약에는 실제 `--hardware` 상대경로/파일 SHA,
paired level evidence 상대경로/실제 SHA, 예정 raw session 경로도 들어간다. meter raw의
`calibration_evidence.mode`는 반드시 `verified_existing`이며 broadband live가 자신이 받은
evidence·hardware·plan·raw target과 이 metadata를 exact 교차검증한다.

TOCTOU를 막기 위해 meter는 최초 CLI preflight뿐 아니라 stream open 직전, capture 종료 직후,
후속 명령 출력 직전에 plan의 세 SHA와 path, hardware/evidence SHA, raw target freshness를 다시
읽는다. capture 뒤 변경은 meter raw에 FAIL 사유를 보존하고 명령을 출력하지 않는다. live도
output open 직전과 stream close·스피커 분리 안내 직후에
authority·meter/evidence/hardware/physical fingerprint를 다시 검증한다. 캡처 중 변경을 발견하면
raw를 버리거나 PASS로 남기지 않고 `post_capture_binding_invalid`인 immutable `INVALID` raw로
먼저 보존하며, offline publisher는 이 post-capture receipt가 exact PASS가 아니면 거부한다.

- signal-only와 무음 dry-run
- `/dev/snd` 점유와 PCM status 확인
- 사용자 입회, 앰프 볼륨 최저, ERR/REF 및 NS/CS 배선 확인
- expected artifact path와 no-replace 확인
- 출력 종료 즉시 stream close 및 스피커 분리 안내

각 panel과 일곱 point-control subband에서 다음을 모두 요구한다.

- observed submitted int16 PCM, raw/analysis/level SHA
- xrun/clip/status 0
- P/S 비주기 marker의 0–4800 search에서 width <3000인 alias branch가 정확히 1개
- warmup부터 마지막 panel까지 global affine map residual ≤0.0675518903 sample
- transition 4개가 guard 뒤 각각 8 adjacent-valid, callback time_info monotonic/slip 0
- actual submitted int16 pilot의 ERR/REF×P/S trajectory agreement ≤0.0675518903 sample
- actual submitted pilot bin의 반대 channel null: absolute ≤1e-8, ratio ≤1e-12
- panel clock valid repeat ≥8, adjacent score ≥0.995
- fractional joint-LS와 cubic crosscheck agreement ≥0.999, relative error ≤0.01
- measured complex tone의 7-band every-other holdout agreement ≥0.995,
  relative error ≤0.10
- P/S consistency ≥0.95
- 같은 capture/stream/hardware fingerprint
- subband별 fractional timing budget
- highband 결과 기반 per-drive phase repair는 모든 panel에서 exact 0

global clock row 경계는 drift의 부호를 가정하지 않는다. 새 row 첫 period만 버리면 양의 drift에는
맞아도 음의 drift에서 이전 row 마지막 period가 반대 row head를 포함할 수 있으므로, 각 row 경계의
직전·직후 interval을 모두 clock authority에서 제외한다. 이때도 각 10-period transition의 내부
8개 exact adjacent witness는 그대로 남아야 한다. `+413.931 ppm`, `0 ppm`, `-413.931 ppm`을 모두
actual submitted int16 분모로 통과하고, 비선형/구간별 drift와 경계의 영구 1-sample slip은
임계값 완화 없이 거부해야 한다. 같은 이유로 각 63-period analysis panel의 첫/마지막 period는
highband P/S 평균과 consistency에서도 authority로 세지 않으며, 내부 61개 중 기존 strict gate를
통과한 repeat만 사용한다.

raw 캡처 후 발행되는 `BroadbandPlantEvidence` authority는
`broadband_interleaved_plant_evidence_v4`다. v4는 P/S marker, actual submitted pilot
trajectory SHA, global-map SHA/slope/intercept/residual, callback witness, transition count,
zero phase-repair와 P/S bulk fractional+integer delay, pre-roll, effective delay,
256-sample handoff, `PlantDelays.lead()`, 다섯 panel의 P−S fractional delay와 final broadband
delay 대비 deviation을 필수로 결속한다. 저장된
deviation 숫자를 믿지 않고 panel/final delay에서 다시 계산해 panel 상단의 20 dB
timing 예산과 대조한다.
exact authority plan의 file/payload/PCM SHA, measurement-level evidence SHA,
fresh meter raw/receipt SHA도 canonical embedded evidence의 필수 필드다.

`scripts/data/analyse_broadband_interleaved.py`는 `sounddevice`를 import하지 않는 offline
publisher다. exact raw/plan/PCM/hardware/level/fresh-meter SHA와 meter→stream 10분 freshness를
재검증한 뒤, analysis NPZ SHA를 포함한 canonical evidence JSON과 그 payload SHA를
P/S NPZ 안에 결속한다. analysis/P/S/final receipt는 모두 no-replace이며, 중간 실패
rollback은 발행 시의 device/inode/size/SHA가 같은 자신의 파일만 제거한다. 학습
canonical plant 표현은 측정된 complex tone response다. 1024-tap compact FIR은 실제 설계행렬
rank가 부족하거나 condition이 나빠도 measured response 발행을 막지 않으며 항상
`compact_role=diagnostic_only`, `compact_training_eligible=false`다. 또한 제한된 tone만으로
exact finite causal history operator를 정할 수 없으므로 publisher는
`measured_band_training_eligible=false`와
`blocked_until_fullband_persistently_exciting_causal_history`를 저장한다. 별도 fullband causal
P/S 식별 전에는 광대역 학습을 열지 않는다. criterion은 embedded evidence와 외부
raw/analysis/level SHA 및 delay/lead를 다시 대조한다.

10 dB 위상 해상도에서 1-sample 오차는 2/4/8 kHz에서 15°/30°/60°이고, 허용 timing
오차는 약 1.213/0.606/0.303 sample이다. 20 dB에서는 약
0.382/0.191/0.0955 sample이며 8 kHz octave 상단 11.314 kHz에서는
0.0675518903 sample이다. 이 값은 모델 성능 약속이 아니라 측정·런타임의 위상
해상도 조건이다.

현 AB13X를 먼저 유지한다. software clock pilot, q correction, 장시간 no-slip이 실패할 때만
rollback 가능한 I2S2 DOUT/공통-clock DAC 경로를 연다. 비동기 ADC를 곧바로 NS/CS 실제 출력
위상 drift로 해석하지 않는다.

## 5. THD/IMD와 비선형 단계

고주파와 비선형은 같은 개념이 아니다. 고역 신호가 있다고 비선형인 것도 아니고, 신경망을
썼다고 비선형 우위가 자동으로 생기지도 않는다. 현 Stage-1 `eta=10`, drive 1,
hardclip 0은 사실상 선형이다.

광대역 P/S가 통과한 뒤 NS/CS 각각에 대해 다음을 별도 raw로 측정한다.

- THD ESS: 100–11300 Hz, 안전한 4개 상대 레벨, 약 18초
- IMD pairs: 1.8/2.2, 3.6/4.4, 7.2/8.8 kHz, 4개 레벨, 약 30초
- 합산 peak는 사전 승인된 0.003을 넘지 않음
- nominal audible 합계 약 48.5초, P/S와 별도 연결 창

signal-only 계획은 시간 회계를 포함한
`results/data_audit/broadband_nonlinearity_signal_plan_strict_v2_20260828.json`에
48.000초로 고정했다. 파일 SHA-256은
`2d40ee05a1587f470a0de67eec35e765a58c8929cd07b89037be62aeb515e590`, PCM SHA-256은
`59e8845f4ea898afbacba8afd030bec13f0005b571b78f20ac8329e695615d14`다. P/NS와 S/CS는
같은 slot에서 동시에 구동하지 않으며, THD/IMD window는 18/30초, P/S 각 24초,
실제 nonzero 출력은 29.724초다. 현재 live 경로는 분석 gate가 없어서 잠겨 있다.

`THD/IMD > -30 dBc` 또는 1 dB 이상 compression이 관측되면 small-signal linear P/S만으로
학습하지 않는다. threshold는 raw를 보기 전에 코드 계약으로 고정한다.

## 6. 광대역 데이터 게이트

source 파일의 스펙트럼이 아니라 실제 ERR target `d`의 에너지를 본다. source에는 고역이
있어도 스피커·덕트·마이크를 지난 ERR pressure가 없으면 ANC 학습 표본이 아니다.

각 split×family×subband마다 다음을 모두 요구한다.

- source→ERR coherence ≥0.60
- target-d energy-density ratio ≥0.25
- 독립 lineage group ≥4
- 한 transient로 group을 채우지 못하도록 group별 최소 segment 수와 coverage fraction
- sample rate/native Nyquist 검증. 16 kHz 원본을 upsample해 8 kHz octave coverage로 세지 않음
- session/source/mics/manifest/P/S/timing/alignment SHA exact
- low/high 중 한쪽만 통과하거나 global 평균이 family 실패를 숨기면 FAIL

campaign receipt 구조는 `src/deep_anc/data/broadband_coverage_receipt.py`가 검증한다.
각 group×band는 joint PASS segment 8개 이상이면서 전체 segment의 50% 이상이어야 하고,
각 split×family×band에는 그런 독립 group이 4개 이상 있어야 한다. 동일 source SHA나
lineage를 여러 group으로 세거나, 16 kHz 원본을 upsample해 8 kHz octave로 세거나,
segment count/median을 raw metric row와 다르게 적으면 실패한다. 8/50%는 v2 pilot 결과를
보기 전에 고정한 hard floor이며 이후 결과에 맞춰 낮추지 않는다.

현재 v1 `source_aligned.wav`는 1.6 kHz 기준 정렬이다. 광대역 v2에는 electrical playback
loopback 또는 robust pilot+fractional warp가 결속된 새 alignment receipt가 필요하다.
역사적 이름이 `highband-coverage-v1`인 82+17 generation도 Stage-1의 600--1600 Hz
부족분 보충용이므로 이 광대역-v2 receipt나 2/4/8 kHz 증거로 승격하지 않는다.

합성 public corpus도 48 kHz tensor라는 이유만으로 통과시키지 않는다. 원본 sample rate와
lineage를 먼저 검사하는 diagnostic gate는 다음과 같다.

```bash
.venv/bin/python scripts/data/audit_synthetic_broadband_native.py \
  --manifest-dir data/manifests/canonical_v4
```

`src/deep_anc/data/synthetic_broadband_coverage.py`는 split×family×7 subband마다 native
Nyquist가 band 상단 이상인 독립 public lineage group을 최소 4개 요구한다. built-in
synthetic tone은 public group으로 세지 않고, 동일 content SHA를 여러 group으로 세거나
한 group을 split에 나누면 실패한다. 이 검사는 source spectrum을 읽지 않는 1차 구조
감사이므로 PASS여도 후속 source spectral-density receipt와 실제 ERR target-d coverage가
필수다.

현재 MIMII fan 원본은 16 kHz라 native Nyquist가 8 kHz다. 48 kHz로 업샘플해도
8 kHz 중심 octave의 상단 11.314 kHz를 담지 못한다. 따라서 machine 계열에는 native
sample rate가 최소 22.628 kHz보다 큰 실제 고역 원본을 별도로 확보해야 하며, ESC-50
전체를 machine으로 중복 분류해 이 blocker를 우회하지 않는다. Jetson에는
`canonical_v4`가 없으므로 현재 로컬 진단 결과도 `BLOCKED`다.

## 7. 손실·모델 선택·평가

광대역 loss는 일곱 제어 subband 모두에 cancellation gradient를 준다.

- equal-subband normalized NMSE와 worst-band guard
- family/item CVaR
- DNH는 point-control union **밖**에만 적용하고 overlap은 0
- 새 P/S와 출력 분포에서 `lambda_dnh`를 다시 교정
- 저역/고역을 단일 평균으로 합쳐 서로의 실패를 숨기지 않음

2/4/8 kHz 단일 tone은 phase sanity check다. 최종 우위의 권위는 high-band가 포함된
speech/music/environment/machine 및 held-out nonlinear/nonstationary source다.

Matched physical 비교는 다음 조건을 exact하게 공유한다.

- source bytes, SPL/gain, P/S, lead, sample rate, block, limiter, window
- validation에서 미리 튜닝·고정한 FxLMS filter length/step size
- controller 순서 ABBA 또는 counterbalancing
- OFF 10초 → ON 30초 → OFF 5초, source는 전 구간 계속 ON
- raw source/output/ERR/REF와 P50/P95/P99/max, miss/drop,
  absolute/allowed/excess backlog, xrun/clip 보존

고역 각 subband에서 `attenuation_DL - attenuation_FxLMS` paired bootstrap 95% CI 하단이
0 dB보다 커야 우위를 주장한다. 숫자가 없거나 조건이 다르면 `INCONCLUSIVE`다.

이 판정의 pure raw-segment 구현은
`src/deep_anc/eval/broadband_point_control.py`다. OFF/DL/FxLMS가 같은 길이인 matched
segment만 받고, target-d density ≥0.25와 독립 group ≥4를 먼저 확인한 뒤
family×7 subband 각각의 Deep-ANC 평균·worst 10%·cluster-bootstrap CI를 검사한다.
150–1600 Hz 네 구간은 모두 양의 감쇠여야 하고, 1600–11314 Hz 세 구간은 양의 감쇠에
더해 DL−FxLMS의 평균·worst 10%·CI 하단도 모두 0 dB보다 커야 한다. 여러 ERR 위치는
단일 point 결과로 평균낼 수 없고 `spatial=True`에서 위치별로 같은 판정을 반복한다.
이 함수는 acoustic metric만 계산하므로 latency/xrun/test-once/provenance를 통과시키는
최종 receipt로 단독 사용할 수 없다.

## 8. 공간 quiet-zone

105×105 mm 정사각 덕트의 첫 횡모드 cutoff는 약 1,633 Hz다. 전파 모드는 2/4/8 kHz에서
평면파 포함 대략 3/8/22개다. 단일 CS·ERR는 한 점의 복소 압력을 줄일 수 있지만 단면 전체를
독립 제어할 rank가 없다.

따라서 다음을 분리한다.

- **point-control PASS**: 중앙 ERR 한 위치의 일곱 subband 감쇠
- **quiet-zone PASS**: 중앙과 y/z ±2 mm 최소 5개 ERR 위치의 family×subband 결과

현재 2채널 APE 입력으로 REF와 5개 ERR를 동시에 수집할 수 없다. 다채널 ADC/작은 probe
array가 가장 좋은 해법이며, 정밀 이동 반복은 시간 drift와 audible time을 추가하므로 별도
불확도 계약이 없으면 공식 공간 증거로 쓰지 않는다.

## 9. 순서와 중단 조건

1. 광대역 코드 계약·negative/positive fixture
2. P/S multi-panel 측정기 signal-only/무음 dry-run
3. 사용자 승인 뒤 50초 미만 P/S 실제 캡처
4. raw 오프라인 분석; 실패하면 재출력하지 않음
5. P/S PASS 뒤 THD/IMD
6. v2 alignment와 네 family×split×subband recorded coverage
7. synthetic broadband/nonlinear curriculum과 loss G0
8. 새 A100 80GB exact bootstrap 및 canonical pretrain/fine-tune
9. fixed test, Level-5 unseen, Tiny/Base/FxLMS matched physical ABBA
10. 다점 quiet-zone과 Jetson latency/안정성

어느 단계든 raw/SHA/clock/coverage/phase가 없으면 다음 단계로 넘어가지 않는다.
**Stage-1 PASS를 2/4/8 kHz 최종 성공으로 해석하지 않는다.** legacy checkpoint, 문서 주장,
이론 상한도 광대역 blocker를 해제하지 않는다.
