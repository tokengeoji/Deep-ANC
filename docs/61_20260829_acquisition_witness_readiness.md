# 2026-08-29 실제 acquisition witness readiness 감사

이 문서는 현재 Jetson의 실제 ALSA 장치와 repository의 fail-closed gate를 함께
대조한 기록이다. 정적 설정 파일이 존재한다는 사실을 하드웨어 통과로 읽지 않으며,
스피커 출력·ALSA PCM open·믹서 변경은 수행하지 않았다.

## [가설]

현재 Jetson APE, RT5640, AB13X USB Audio만으로 125 Hz--8 kHz canonical P/S와
최종 quiet-zone 실험에 필요한 동기 acquisition을 만들 수 있다.

## [근거]

2026-08-29 KST에 read-only로 확인한 실제 장치 정보는 다음과 같다.

- APE PCM1은 현재 ERR/REF 두 입력이다. ADMAIF 20개 표기는 20개의 물리 ADC 입력을
  뜻하지 않는다.
- AB13X USB Audio는 48 kHz/S16 playback 2채널이지만 capture는 48 kHz/S16 **mono
  1채널**이며 endpoint가 `ASYNC`다.
- RT5640/J511 후보는 `APE PCM0 -> ADMAIF1 -> I2S1 -> RT5640 -> J511`이지만, 최근
  read-only probe의 J511 state는 세 번 모두 `None`이었다. RT5640 input-only probe도
  ch0 rail/clip 및 ch1 stuck으로 실패했다.
- 고주파 P/S의 최소 electrical witness는 `ERR`, `REF`, `NOISE_TAP`, `CANCEL_TAP` 네
  역할을 같은 acquisition clock에서 동시에 취득하거나, APE--외부 recorder 사이의
  연속 `BCLK + WS + absolute frame counter` bridge를 보존해야 한다.
- 최종 quiet-zone은 위 tap과 `ERR_0..ERR_4`를 포함한 8개의 같은-frame 입력이
  필요하다. ERR 위치를 순차 이동해 기록한 raw는 최종 공간 성능 증거가 될 수 없다.

관련 정적 계약은
[`configs/external_electrical_witness_admission_v1.yaml`](../configs/external_electrical_witness_admission_v1.yaml),
[`configs/full_octave_v3_physical_session_bundle.yaml`](../configs/full_octave_v3_physical_session_bundle.yaml),
그리고 [외부 동기 witness admission](49_external_synchronous_adc_witness_admission.md)에
고정돼 있다.

## [확인 방법]

source commit `e474e2975c336846bd7c9e5a7840a9827938aa3d`에서 다음 **무출력** 검사를
실행했다. 두 명령 모두 ALSA/PCM/GPU/network를 열지 않는다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/jetson/check_external_electrical_witness_static.py \
  --config configs/external_electrical_witness_admission_v1.yaml

PYTHONPATH=src .venv/bin/python \
  scripts/data/check_full_octave_v3_physical_session_bundle.py \
  --config configs/full_octave_v3_physical_session_bundle.yaml
```

또한 다음 focused test와 최신 전체 pytest를 실행했다.

```bash
.venv/bin/python -m pytest -q \
  tests/test_external_electrical_witness_admission_v1.py \
  tests/test_full_octave_v3_physical_bundle.py \
  tests/test_full_octave_v3_execution.py \
  tests/test_measure_channel_paths.py \
  tests/test_measure_paths_interleaved_dry_run.py

.venv/bin/python -m pytest -q
```

## [결과]

첫 static checker는 exit 0이지만 다음을 반환했다.

```text
static_gate_pass = true
status = BLOCKED
electrical_witness_pass = false
fullband_plant_identification_pass = false
canonical_training_eligible = false
deployment_eligible = false
```

이는 YAML 요구사항이 변조되지 않았다는 뜻일 뿐 hardware PASS가 아니다. 8-input physical
bundle checker는 exit 1로 `BLOCKED`를 반환했다. plan/native raw/canonical raw/session
sidecar가 없고 `required_input_channels=8`이다. focused test와 전체 pytest는 모두
**0 FAIL**이었다. 전체 pytest의 RuntimeWarning 두 건은 local `canonical_v4` public
manifest 부재를 알려 주는 의도된 diagnostic warning이다.

현재 장비로 할 수 있었던 2/4/8 kHz 단일점 coupling raw는 스피커에서 해당 주파수 에너지가
나오는지까지만 확인한다. AB13X와 APE의 독립 시간축, mono capture, 전기 tap 부재 때문에
이를 high-band P/S, lead, 학습 또는 ANC attenuation 근거로 승격할 수 없다.

문서 commit 직후 `9c8687ce84f9c86250252923e6262733cbb4db1c`에서 같은 read-only
inventory를 다시 확인했다. 모든 PCM stream은 `closed`였고 PulseAudio는 control node만
열고 있었다. AB13X descriptor도 앞의 2ch adaptive playback/mono asynchronous capture와
일치했다. J511 checker의 세 표본은 다시 모두 `None`이었다.

```json
{
  "observed_states": ["None", "None", "None"],
  "j511_plug_detected": false,
  "j511_unplugged_detected": true,
  "electrical_output_witness": false
}
```

같은 commit에서 `check_finetune.py --config configs/train_finetune.yaml --set
data.digital_primary_path_mode=measured`도 실행했다. exit 2로, canonical
`data.bootstrap_receipt`와 외부 `bootstrap_receipt_sha256`가 없다는 설정 admission에서
멈췄고 run directory를 만들지 않았다. 따라서 지금 GPU 학습을 시작하지 않는 것은 추정이
아니라 실제 admission 결과다.

## [판정]

**Contradicted -- 현재 장비 조합만으로 canonical high-band P/S timing authority를 만들 수
없다.**

RT5640/J511은 향후 common-clock **출력 후보**로는 `Likely`지만, plug detection·전기 출력
witness·hardware-frame identity가 아직 없고, 어떤 경우에도 8-input quiet-zone acquisition을
대체하지 않는다. 일반 USB 2/4채널 ADC를 단순 추가하는 방식도 APE의 ERR/REF와 hardware
frame bridge가 없으면 충분하지 않다.

## [다음 행동]

1. 최종 광대역을 열 hardware topology를 먼저 확정한다. 최소 P/S용 4입력인지, 바로 최종
   quiet-zone용 8 simultaneous input인지 사용자가 결정해야 한다. 물리 tap은 고임피던스,
   절연, DC block, 감쇠, fixed gain, AGC/limiter off를 만족해야 하며 ADC를 speaker terminal에
   직접 연결하지 않는다.
2. 실제 장비를 추가한 뒤에는 sound 출력 전에 channel map, `BCLK/WS/frame` witness, tap
   polarity/clip/stuck과 O_EXCL raw-first adapter를 read-only/무음으로 먼저 검증한다.
3. 이와 독립된 Stage-1 150--1600 Hz 경로는 새 Elice A100에서 exact `dev` bootstrap과 DNS
   speech selection receipt를 발행하면 재개할 수 있다. 그 뒤 no-replace 17행 plan과 PASS
   dry-run을 만들고, 예상 audible 255초(연결 상한 약 5분 48초)의 한 번의 수집 창에서만
   추가 session을 녹음한다.
4. 현재 344파일/4,689,042,188-byte transfer manifest는 82세션 schema v1이다. 새 Elice의
   `canonical_v4` 및 DNS selection을 여는 데는 쓸 수 있지만, canonical pretrain/fine-tune
   입력으로 재사용하지 않는다. 17세션 수집 뒤에는 99세션 schema v2 transfer manifest를
   no-replace로 재발행하고 additions를 같은 Elice에 검증 전송한 뒤 receipt를 다시 확인한다.
5. 17세션은 Stage-1 coverage 보강용일 뿐 2/4/8 kHz 또는 125 Hz--8 kHz final claim으로
   승격하지 않는다.
