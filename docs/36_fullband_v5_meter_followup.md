# fullband-v5 입력 preflight와 meter followup

## 입력-only preflight

`deep_anc.audio_io.capture_measurement_preflight_raw()`는 출력 Stream이나 출력 device를
열지 않는다. official input/output PCM 무점유와 capture clock을 먼저 확인하고, resolved
input device가 48 kHz·2채널·int32 입력을 지원하는지 확인한 뒤 입력만 녹음한다.

`seconds`는 기존 `assert_measurement_preconditions()`와 동일하게 0.5초 settle을 포함한
총 capture 시간이다. 반환 raw는 settle을 제거한 owned C-contiguous exact `<i4
[frames,2]`이며, 채널 통계와 valid 판정은 `analyze_int32_input_probe()`가 단일 출처다.
기존 assert 함수는 이 API에 delegate하고 기존 channel별 rail-ratio list 반환을 유지한다.

## `--followup-mode fullband-v5`

이 모드는 sounddevice import/open 전에 다음 committed bytes와 의미를 exact 검증한다.

- `assets/contracts/fullband_causal_v5_signal_plan.json`의 file/payload/PCM SHA
- `assets/contracts/fullband_causal_v5_live_capture_authority.json`의 file/payload SHA
- `configs/hardware_jetson.yaml`의 exact path/file SHA
- `assets/measured/measurement_level_evidence.json`의 v2 bootstrap-pair schema/file SHA
- `results/fullband_causal_v5/raw_capture.npz`가 아직 존재하지 않는다는 sealed freshness

old broadband plan/raw argument, v4/legacy schema, 기존 raw, 다른 hardware/evidence path는
거부한다. 같은 검증을 output Stream open 직전, capture 종료 직후, command 출력 직전에
반복한다.

Meter recipe는 exact 48 kHz·20.000초·peak 0.003이며 noise speaker ch0만 구동하고 cancel
speaker ch1은 exact silent다. 사용자 입회, speaker output, 시작 전 volume minimum,
routing/geometry, same amplifier setting의 다섯 confirmation을 요구한다. Stream close 직후
물리 분리 안내를 먼저 출력한다.

PASS raw와 receipt는 plan/authority/hardware/evidence, resolved devices, followup canonical
payload SHA와 raw file SHA를 no-replace atomic/fsync writer로 보존한다.

현재 committed authority는 의도적으로 다음 상태다.

```text
capture_only = true
plan_live_capture_enabled = false
status = blocked_until_v5_live_adapter_implementation
```

따라서 출력되는 `measure_paths_fullband_causal_v5.py --execute-live` 명령은 다음 adapter
묶음과 exact 인자를 맞추기 위한 capture-only handoff다. 현재 adapter는 exit 2로 닫혀 있고,
이 meter가 live 실행 권위나 준비 완료를 주장하지 않는다.
