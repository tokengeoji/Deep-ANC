# RT5640 exact-zero 동시 입출력 검증

이 단계의 목적은 USB DAC와 I2S ADC의 서로 다른 클록으로 실패한 v6 경로를 반복하지 않고,
Jetson APE의 `I2S1` 출력과 `I2S2` 입력을 동시에 여는 최소 안전 경로를 검증하는 것이다.
출력 payload는 모든 callback에서 S32_LE bitwise zero로 고정한다. 이 단계는 P/S 측정이나
ANC 성능 평가가 아니다.

## [가설]

`hw:APE,0` playback과 `hw:APE,1` capture를 48 kHz, 2채널, block 256, S32_LE로 동시에
열 수 있고, PortAudio application callback buffer에 nonzero sample을 한 번도 제출하지
않으면서 60초 transport를 완료할 수 있다고 가정한다.

## [근거]

2026-08-29 read-only 현장 감사에서 다음을 확인했다.

- APE의 모든 PCM status가 `closed`였다.
- PulseAudio의 `alsa_card.platform-sound` active profile은 `off`였다.
- `I2S1 Mux=ADMAIF1`, `ADMAIF1 Mux=I2S1`, `ADMAIF2 Mux=I2S2`,
  `I2S2 Mux=ADMAIF2`였다.
- PortAudio는 `hw:1,0`과 `hw:1,1`을 각각 하나의 장치로 열거했다. 장치 index는
  동적이므로 숫자를 config에 저장하지 않는다.
- `/sys/bus/i2c/devices/8-001c`의 codec name과 driver는 `rt5640`이었다.
- APE pcm1 capture/pcm0 playback physical fingerprint SHA-256은
  `a5f20b5dad1e3eee11b5275ef0a2f2531a461e93dc01410d0868ebb8b34c2957`였다.

이 fingerprint와 route는 실행 때 다시 계산한다. 위 값은 실행 admission을 대신하지 않는다.

## [확인 방법]

### 안전 경계

live 실행 전에 아래 물리 조건을 모두 만족해야 한다.

1. J511과 앰프 사이 케이블 분리
2. 앰프 전원 OFF
3. 앰프 입력 케이블 분리
4. AB13X와 앰프 사이 케이블 분리
5. 사용자 입회

의도된 audible signal은 **0초**지만 codec/PCM open·close pop은 application zero buffer만으로
증명할 수 없다. 따라서 위 물리 분리 없이 Stream을 열지 않는다.

스크립트는 ALSA mixer, Pulse profile, pinmux, device tree를 변경하지 않는다. 현재 Pulse
profile이 이미 `off`가 아니거나 route가 다르면 자동 수정하지 않고 실패한다. 실행 전후
`alsactl store APE`와 `amixer -c APE contents`를 read-only snapshot으로 보존하며, 둘 중
하나라도 허용 범위 밖에서 달라지면 `STATE_UNCERTAIN`으로 종료하고 자동 restore하지 않는다.
실제 장치 미개방 double snapshot에서 ALSA가 `read volatile`로 선언한
`Lane1..6 Ratio Int/Frac` 12개 값은 읽는 사이에도 변했다. 두 raw snapshot과 SHA는 그대로
보존하되, exact numid/name/type/access/count allowlist에 들어 있는 이 12개 control의 value
line만 sentinel로 정규화한다. 그 밖의 control, metadata 또는 allowlist 집합이 한 바이트라도
달라지면 실패한다. `alsactl`과 `amixer` parser도 같은 12개 identity를 독립 확인해야 한다.

### 장치 없는 dry-run

```bash
PYTHONPATH=src .venv/bin/python scripts/jetson/audit_rt5640_zero_duplex.py --dry-run
```

dry-run은 `sounddevice`를 import하지 않고 PCM을 열지 않으며 result generation도 만들지
않아야 한다.

### clean commit에서 live 1회

```bash
EXPECTED_COMMIT=$(git rev-parse HEAD)
PYTHONPATH=src .venv/bin/python scripts/jetson/audit_rt5640_zero_duplex.py \
  --execute-live \
  --expected-commit "$EXPECTED_COMMIT" \
  --confirm-j511-disconnected \
  --confirm-amplifier-power-off \
  --confirm-amplifier-input-disconnected \
  --confirm-ab13x-amplifier-disconnected \
  --confirm-user-present
```

예상 stream 시간은 정확히 60초이고 callback은 11,250회다. generation은
`results/rt5640_zero_duplex/v1` 한 곳뿐이며 성공·실패와 무관하게 한 번 소비하면 재실행하지
않는다. unique raw를 먼저 no-replace 보존하고 final receipt도 덮어쓰지 않는다.

## [결과]

코드 단계에서는 다음 fault를 모의 검증한다.

- Stream constructor/start/watchdog/callback/stop/abort/close 실패
- INT/TERM/HUP가 stream 및 cleanup 중 도착
- callback dtype/shape/frame/status/xrun/시간축 위반
- nonzero output, partial valid mask, callback sequence/frame accounting 손상
- 잘못된 commit/PYTHONPATH/module/device/route/Pulse/PCM 점유
- 실행 전후 ALSA snapshot 불일치와 no-replace generation 재사용
- receipt에 shared-clock/P/S/감쇠 권위를 다시 서명해 주입하는 시도

실제 live 결과와 artifact SHA는 실행 후 이 절과 `HANDOFF.md`에 추가한다. live 전에는 결과를
미리 PASS로 기록하지 않는다.

## [판정]

최대 성공 상태는 `ZERO_DUPLEX_TRANSPORT_SMOKE_PASS`다. 다음 항목은 PASS 이후에도 모두
미확인이다.

- 두 PCM의 물리 frame identity와 sample drop/add/reorder
- shared-clock 권위
- J511의 전기적 zero와 physical output route
- P(z), S(z), lead, 2/4/8 kHz plant consistency
- 실제 덕트 ANC 감쇠 dB와 학습 자격

따라서 all-zero 결과만으로 canonical fullband P/S나 파인튜닝을 시작하지 않는다.

## [다음 행동]

1. transport smoke가 PASS하면 raw 입력의 ERR/REF health와 negotiated hw_params를 확인한다.
2. 별도 전기적 frame witness 또는 같은 APE clock-domain을 직접 검증하는 짧은 nonzero 실험을
   설계한다. 이때도 앰프 연결 전 무음 dry-run을 먼저 끝낸다.
3. 짧은 level/channel/polarity 확인 뒤 새 fullband P/S를 한 번 측정한다.
4. 150–1600 Hz뿐 아니라 2/4/8 kHz consistency와 out-of-band do-no-harm를 독립 검증한다.
5. 유효한 fullband P/S·lead·계보 manifest가 생긴 뒤에만 Elice canonical pretrain과
   fine-tune을 연다.
