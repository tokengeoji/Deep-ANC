# Stage-2 RT5640/J511 S32 실제 P/S 어댑터

## 현재 판정

`scripts/jetson/measure_stage2_2khz_actual_ps_s32.py`는 `dev`의 한 exact commit에서만
실행하는 현장 어댑터다. 기본 `--dry-run`은 48 kHz/256-frame/S32 full-PE 계획만 만들며
backend·ALSA PCM·스피커 출력·파일 쓰기를 하지 않는다. dry-run 성공은 물리 P/S나 학습
권한이 아니다.

실제 실행은 다음 조건을 모두 만족할 때만 허용한다.

1. `amixer -c APE`의 J511 상태가 `HP` 또는 `HS`로 세 번 동일하다.
2. APE PCM 전역 점유가 없고 APE mux가 `I2S1↔ADMAIF1`, `I2S2↔ADMAIF2`다.
3. 사용자 다섯 확인(스피커, 입회, 최소 볼륨, 배선·덕트, 동일 앰프 설정)이 true다.
4. stream 시작 뒤 `pcm1c`/`pcm0p`의 실제 `hw_params`가 `S32_LE/48000/2/256`이고,
   route·J511·PCM owner receipt가 다시 PASS한다.
5. pre-arm callback은 exact zero만 내보내며 위 receipt가 PASS한 뒤에만 sealed 24초
   계획을 arm한다.

실패 raw도 삭제·재측정하지 않고 partial schema로 no-replace 보존한다. 성공 raw 역시
`CAPTURED_RAW_UNANALYZED` 상태로만 발행하며, clock/phase/fixed-LTI·P/S 분석 PASS 전에는
physical P/S authority, plant binding, pretrain/fine-tune 권한을 만들지 않는다.

## 무음 확인

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/jetson/measure_stage2_2khz_actual_ps_s32.py --dry-run
```

예상 결과는 `status=DRY_RUN_NO_AUDIO`, `audio_backend_imported=false`,
`speaker_output=false`, `raw_written=false`다. 계획은 24초(1,152,000 frames)이며,
실제 출력은 이 명령으로 발생하지 않는다.

## 물리 gate와 한 번의 출력 창

먼저 소리 없이 J511을 확인한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/check_rt5640_j511.py \
  --expect HP --samples 3
# 상태가 HS라면 --expect HS 사용
```

그 gate가 PASS한 뒤 사용자 입회와 앰프 최소 볼륨에서 아래를 한 번만 실행한다.

```bash
PYTHONPATH=src .venv/bin/python \
  scripts/jetson/measure_stage2_2khz_actual_ps_s32.py --execute-live \
  --confirm-speaker --confirm-user-present --confirm-volume-minimum \
  --confirm-routing-and-geometry --confirm-same-amplifier-setting
```

full-PE audible 계획은 24초다. 종료 즉시 프로그램이 `출력 종료 — 지금 스피커 분리`를
출력한다. raw를 오프라인 분석하기 전에 speaker/amp를 물리적으로 분리하고, 실패 시
자동 재출력하지 않는다. 현재 J511 상태가 `None`이면 위 live 명령을 실행하지 않는다.

## 학습 gate

다음 순서는 고정한다.

```text
J511/PCM read-only PASS
  → fresh S32 raw (no-replace)
  → clock/phase/P/S analysis PASS
  → 47-slot Stage-2 recorded coverage/lineage PASS
  → Elice exact bootstrap
  → canonical 100k surrogate pretrain
  → measured 50k fine-tune
```

기존 USB AB13X, output-master split-clock, S16 raw 및 legacy checkpoint는 이 어댑터의
실측·학습 근거로 승격하지 않는다.
