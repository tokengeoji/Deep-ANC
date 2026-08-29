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

## 7. 2026-08-29 v6 실제 실행 결과

이 섹션은 capture commit
`872e59322527880330acd989a435cd31a2d16387`
(`work/v6-clock-checkpoints`), capture-id
`232a4e53a4eaa024d54b740a01c95fe1`에서 발행된 불변 raw와 receipt를
기준으로 한다. `POST_CAPTURE_PASS`와 P/S PASS는 다른 권한이며, 아래 시계
재검산은 임계를 우회하는 승격 경로가 아니다.

### 7.1 raw transport와 level이 유효한가

#### [가설]

v6 raw가 planned 출력과 실제 입력 transport를 완전히 보존했고, offline
clock 실패가 출력 누락, 마이크 단선, callback 누락, xrun 또는 clipping
때문이 아닐 가능성을 검사한다.

#### [근거]

| Artifact | 경로 | SHA-256 |
|---|---|---|
| immutable raw | `results/fullband_causal_v6/raw_capture.npz` | `f153c8664106b0c341b67db940fb2fb1d76cb7e58c2fa9a6e49558e1dba50a63` |
| external post-capture receipt | `results/fullband_causal_v6/raw_capture.npz.post_receipt.json` | `6372cfdec4ce15013f7bdc958f47c25fa1055f1e368adaeaa1a8d5627608dbda` |
| offline failure | `results/fullband_causal_v6/failure_232a4e53a4eaa024d54b740a01c95fe1.json` | `10856999254a8dc70c3696b02aed239db1b80f217a3dfd771442cedb2aacc75d` |

failure 내부 payload SHA는
`8f30be9aff2c4f3d63ab0b08208eec5731962e16ce66f251c66fee12e2d3405d`이며 재계산과
일치한다. fresh meter ch0는 −48.234 dBFS로 계약 −50.1±2 dBFS 안이었다.

#### [확인 방법]

raw ndarray의 dtype·shape·SHA, planned/actual PCM, valid mask, callback sequence/start/frame
count/status, preflight, monotonic/watchdog, stop telemetry를 독립 재계산했다. 현재 clean
checkout에서 external receipt loader로 raw와 현재 plan·authority·hardware·meter 바이트를
다시 결속했다.

#### [결과]

- 1,179,648/1,179,648 frame, 4,608×256 callback, input/output valid mask 전부 true
- actual submitted PCM이 planned PCM과 byte-exact, 반대 출력 channel은 exact zero
- callback status bitmask 전부 0, xrun/status count 0, normal stop/output stop true
- 실측 capture elapsed 24.596525초로 24.576+2.0초 watchdog 안에 있음
- input preflight 양 channel non-stuck/PASS, 본 capture peak ch0 0.05609, ch1 0.07635,
  clip 0
- preterminal fixed-line SNR 192/192 PASS(최저 24.174 dB), terminal 64/64
  PASS(최저 26.963 dB), 기준 20 dB

#### [판정]

**Confirmed.** raw의 소프트웨어 transport와 level은 PASS했고,
transport·level·완전 단선이 offline clock 실패의 주원인이라는 해석은
raw와 receipt에 반한다. 다만 telemetry는 명시적으로
`hardware_sample_slip_authority=false`이므로 transport PASS를 hardware clock PASS로
승격하지 않는다.

#### [다음 행동]

이 raw를 연결·transport 진단에는 보존하되, clock/P/S/학습 권한으로 쓰지
않는다. 별도 hardware clock witness가 없는 상태에서 callback frame count로
sample slip을 추정하지 않는다.

### 7.2 하나의 stationary affine q로 설명할 수 있는가

#### [가설]

전체 preterminal clock checkpoint에 하나의 유일한 DAC/ADC rate ratio `q`가 존재하고,
각 path·microphone view도 같은 q를 선택할 가능성을 검사한다.

#### [근거]

최초 failure는 `failure_stage=global_grid_basin_search`, `optimizer_started=true`,
`global clock objective가 multimodal ambiguous`를 기록했다. 초기 구현이 전체 basin
receipt를 예외 경로에서 버리던 결함을 복구한 뒤, 위 immutable raw를
`require_unique=false`로 **진단 재계산**했다. 이 option은 receipt를 보존할
뿐 후속 gate를 열지 않으며, `unique=false`면 즉시 fail-closed한다.

#### [확인 방법]

1 ppm grid의 ±1,000 ppm 전체 interior basin을 전부 refine하고, runner-up/best
objective ratio를 기준 4.0과 비교했다. 같은 관측을 primary/secondary×ERR/REF
네 view로 분리해 동일한 진단 search를 실행했다. 추가로 8,192-sample
Hann, 1,024-sample hop, 8 line×2 microphone median의 short-time rate를 8 block에서
56개씩 총 448 step 계산했다. short-time mode는 진단일 뿐 admission gate가
아니다.

#### [결과]

전체 search는 interior basin 20개를 찾았고 best q는
`0.9996450923072727`(−354.9077 ppm), runner-up/best ratio는
`1.029125322639433`이었다. 4.0에 크게 못 미치므로 이 best q는
선택된 clock이 아니라 단지 최저 objective basin이다.

| View | 최저-basin q | ppm | basin 수 | runner/best | unique |
|---|---:|---:|---:|---:|---|
| primary/ERR | 1.0005620496902223 | +562.0497 | 16 | 1.008394 | false |
| primary/REF | 1.0005646970175202 | +564.6970 | 16 | 1.002325 | false |
| secondary/ERR | 0.9995229549130533 | −477.0451 | 15 | 1.089759 | false |
| secondary/REF | 0.9995263607599849 | −473.6392 | 16 | 1.078258 | false |

같은 path 내 ERR/REF는 약 2.65–3.41 ppm 차이지만, primary와 secondary 사이는
약 1,038–1,042 ppm 반대 방향으로 분리됐다.

short-time 448 step의 mode 요약은 다음과 같다.
이 수치는 현재 dirty v7 진단 draft에서 재현된 **non-authoritative 경향**이다.
script provenance, raw loader/cross-binding, dirfd no-replace, failure association의 독립
검토와 clean commit 기반 final artifact 발행 전이므로 draft 경로·SHA를 최종
증거로 고정하지 않는다.

| Histogram center (ppm) | membership | membership median (ppm) |
|---:|---:|---:|
| −4,625 | 100 | −4,743.581 |
| −875 | 129 | −851.638 |
| +3,125 | 193 | +3,149.531 |

전체 중 기존 affine search 경계 ±1,000 ppm 안의 step은 132/448
(29.46%)이었다. 전체 median은 −674.456 ppm, 5–95 percentile은
−4,818.908–+3,403.853 ppm이었다.

#### [판정]

**Contradicted.** 이 capture를 하나의 유일한 stationary affine q로 승인할 수
없다. 20-basin/4-view 결과는 정식 uniqueness gate 실패이며, short-time 3-mode는
경로·시간에 따른 non-affine 동작 가능성을 강하게 지지하는 diagnostic이다.
다만 `hardware_sample_slip_authority=false`이므로 이 결과만으로 USB DAC, APE ADC,
driver 중 어느 구성요소가 원인인지는 확정하지 않는다.

#### [다음 행동]

기존 4.0 uniqueness, ±1,000 ppm, SNR, endpoint, 8-subband 임계를 낮추지
않는다. 먼저 current raw에서 block/path 별 non-affine clock hypothesis를 분리
진단하고, 다음 실측은 하나의 q를 전제하지 않는 새 plan/authority/raw
경로와 hardware clock witness를 먼저 설계한다. 전체 무음 dry-run과 새 fresh meter
PASS 전에는 출력하지 않는다.

### 7.3 P/S와 학습 권한을 발행할 수 있는가

#### [가설]

transport와 SNR이 PASS했으므로 이 raw에서 P/S를 발행하거나 파인튜닝을
개시해도 된다는 가설을 검사한다.

#### [근거]

failure artifact는 `operator_published=false`, `analysis_published=false`,
`canonical_training_eligible=false`를 명시한다. clock uniqueness가 P/S LS, compact FIR,
96-row/8-subband consistency, delay 발행보다 먼저 실패했다.

#### [확인 방법]

`results/fullband_causal_v6/`와 `assets/measured/`에서 해당 capture-id에 결속된
analysis/operator/P/S 산출물을 검색하고 failure flag와 대조했다. clock
failure 이후 view optimizer, P/S operator, holdout score publisher가 열리지 않는지
확인했다.

#### [결과]

analysis/operator/P/S는 하나도 발행되지 않았다. 따라서 current fullband P/S,
lead, 2/4/8 kHz plant consistency, ANC 감쇠 dB는 모두 미확정이며, 이 capture로
canonical pretrain/파인튜닝을 열 수 없다.

#### [판정]

**Invalid experiment.** raw transport 진단은 유효하지만 plant identification과 학습
권한은 무효다.

#### [다음 행동]

이 capture-id와 동일 v6 signal-plan을 다시 출력하지 않는다. 실패 raw를
불변 진단 자료로 보존하고, offline 원인 분석·새 clock/measurement 계약·전체
무음 검증을 먼저 완료한다. 새 sealed plan이 기존 임계를 그대로 통과한
다음에만 **새 capture-id와 새 raw 경로**로 한 번 실행한다. 유효 P/S가
발행되기 전에는 학습을 계속 차단한다.
