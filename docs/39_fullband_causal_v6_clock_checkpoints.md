# 39. Fullband causal v6 — 시간 분리 clock checkpoint 실측 계약

## 1. 목적과 현재 권한

v6는 기존 고주파 진단 캡처의 공통-clock witness 실패를 해결하기 위한 **P(z)/S(z)
식별 전용 실측 계약**이다. 저역만 좁혀 측정하지 않고 다음 8개 물리 부대역을 모두
독립 gate로 검사한다.

| index | 대역 (Hz) |
|---:|---:|
| 0 | 88.388–150 |
| 1 | 150–300 |
| 2 | 300–600 |
| 3 | 600–1000 |
| 4 | 1000–1600 |
| 5 | 1600–2828.427 |
| 6 | 2828.427–5656.854 |
| 7 | 5656.854–11313.708 |

이 대역을 식별하는 것과 ANC 감쇠를 입증하는 것은 다르다. 특히 덕트의 평면파 cutoff
약 1,633 Hz 위에서는 단일 ERR 지점의 상쇄 가능성과 단면 전체 quiet zone을 구분해야
한다. v6 P/S PASS만으로 2/4/8 kHz 감쇠 dB, 처음 듣는 소리 일반화, Tiny/Base 우위를
주장하지 않는다.

v6 live authority는 `capture-only`, `canonical_training_eligible=false`다. offline
clock/P/S/compact/8-band 분석까지 PASS한 뒤 별도 검토·승격 절차를 거치기 전에는
학습 자산으로 사용하지 않는다.

## 2. 봉인된 신호

- 48,000 Hz, block 256, exact 1,179,648 frame, 24.576초
- ch0: primary/noise speaker, ch1: secondary/control speaker
- 두 출력은 절대 동시에 활성화하지 않고 시간 분리한다.
- 8개 clock block과 6개 near-white PE slot을 고정 순서로 배치한다.
- terminal clock은 q 검증에만 쓰며 P/S 선택·noise 추정에 사용하지 않는다.
- operator holdout은 최종 고정 평균 식을 hash한 뒤 처음 연다.
- actual submitted peak는 int16 98이며 기존 20초 meter보다 active-block power가 높지 않다.

봉인값:

| 항목 | SHA-256 |
|---|---|
| signal-plan payload | `8b37213a13131a071e10527c948580c906dfd914a1134e98a640ead259ba42f7` |
| actual submitted PCM | `4e8a66b983af872192624bd6759282058cfe4a845460111a24bcd684b22551a3` |
| exact shifted-condition receipt | `211f581296d9d99927241a08c7a1096615246d68fe6702db8ff241cf1f582034` |
| plan envelope file | `500b93d1a5289ac0d467683088ea2d72181810f45872faf0bcb29265bb13cf3b` |
| live authority file | `7a795e4e780004d4260fd85abab5c73e6d46858b3ab99c551997a2337fd15b75` |

plan의 `publisher_contract.raw_npz_schema`는 실제 writer 상수
`fullband_causal_live_raw_v6_v1`과 exact 일치한다. 신호 모듈은 오디오 장치를 열거나
raw를 발행하지 않는다.

## 3. 실제 출력 전 fail-closed 순서

1. clean exact Git checkout과 실행 script blob을 검증한다.
2. plan/authority/hardware/paired level evidence SHA와 현재 ALSA 물리 fingerprint를 검증한다.
3. sealed raw·receipt·분석 경로가 아직 없는지 no-follow dirfd로 확인한다.
4. 저장된 20초 v6 meter가 10분 이내이고 같은 commit/branch/`set_amp_level.py` SHA인지
   검증한다.
5. `/dev/snd` 독점 잠금과 실제 PCM 무점유를 확인한다.
6. 입력 전용 1.5초 preflight를 실행한다. 이 구간의 speaker output은 0초다.
7. `pre_open_check` 완료 및 Stream 생성 뒤, `Stream.start()` 직전에 watchdog 기준시각을
   잡는다. 긴 무음 준비시간은 26.576초 hard maximum에 포함하지 않는다.
8. 24.576초 duplex를 정확히 한 번 실행하고 스트림을 닫는다.
9. 분석·저장보다 먼저 `출력 종료—지금 스피커 분리`를 출력한다.
10. 성공/실패와 무관하게 단 하나의 immutable raw와 외부 receipt를 보존한다. 실패한
    동일 plan을 즉시 재실행하지 않는다.

## 4. 실행 명령과 시간

먼저 무음 계획 검증을 수행한다.

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal_v6.py --dry-run
```

레벨 미터는 noise/primary speaker(ch0)만 exact 20초 출력한다.

```bash
.venv/bin/python scripts/data/set_amp_level.py \
  --mode fullband-v6 \
  --confirm-speaker \
  --confirm-user-present \
  --confirm-volume-minimum \
  --confirm-routing-and-geometry \
  --confirm-same-amplifier-setting
```

PASS meter가 출력한 follow-up 명령을 그대로 실행한다. 이 명령은 입력 전용 1.5초 뒤
ch0와 ch1을 순차 사용해 exact 24.576초 출력한다. 두 단계의 총 audible time은
44.576초다. contract 계산·device gate·fsync 시간은 소리가 없는 준비/보존 시간이다.

raw가 PASS하면 adapter가 SHA와 capture-id가 결속된 offline 명령을 출력한다. 그 명령을
그대로 실행한다. offline publisher는 외부 receipt와 raw를 다시 읽고, current clean exact
adapter identity를 확인한 뒤 raw에서 분석을 **독립 재실행**한다. caller 결과와 analysis
canonical JSON 및 operator 6개 ndarray의 dtype·shape·bytes가 모두 같을 때만
`analysis.json`과 `operator.npz`를 no-replace로 발행한다.

## 5. P/S 분석 PASS 조건

- actual submitted PCM exact match, callback 256-frame 연속성, valid mask 전부 true
- xrun/status 0, clipping 0, pre-open/capture monotonic 순서와 hard maximum PASS
- 세 preterminal clock epoch와 terminal clock의 fixed-line SNR·basin·cubic/linear
  endpoint·phase gate PASS
- terminal clock을 q 선택/fit/noise에 사용하지 않음
- P/S fit_a·fit_b bulk/fractional peak stationarity PASS
- shifted support 1024 exact Gram condition ≤20 및 quadratic crosscheck PASS
- q-corrected broadband repeat half-difference noise 사용
- fixed-average의 fit/cross/terminal 96개 score row와 8개 부대역 전부 PASS
- P/S compact FIR와 서로 다른 zero delay가 timing receipt, formula, raw SHA에 결속

PASS 뒤에도 산출물은 우선 `diagnostic/capture authority`다. 실제 학습 승격 여부는
기존 strict 150–1600 Hz P/S와의 저역 일치, 1.6 kHz 이상 SNR/안정성, duct geometry,
G0·G4 계약을 함께 검토해 결정한다.

## 6. v5와 구형 고주파 결과의 관계

v5 transport/raw writer의 검증 primitive는 재사용하지만 v5 telemetry schema를 v6로
허용하지 않는다. v5 schema v4 필드 집합은 변경하지 않았고, v6만 pre-open telemetry가
포함된 schema v2를 쓴다.

2026-08-27 `experimental_high_band` raw는 xrun/clip 0이어도 유효 clock 주기가 0개라
`Invalid experiment`다. v6 결과와 합치거나 2–8 kHz 증폭/감쇠 수치의 근거로 재사용하지
않는다. 기존 strict P/S와 legacy checkpoint도 자동 교체·resume하지 않는다.
