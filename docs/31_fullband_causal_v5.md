# 광대역 causal P/S v5 — time-separated near-white PE

> 상태: signal-only fixture PASS, 실제 raw 없음, `live_authority=None`, `BLOCKED`
> 기준일: 2026-08-28

## 1. 결론

v4의 pilot-line comb-null PE는 support 1024의 exact joint Gram condition이 280 이상이라
finite causal P/S 학습 operator의 지속 여기 조건을 만족하지 못했다. v5는 이 결과를
덮어쓰지 않고, P와 S main PE를 시간 분리한 actual-int16 Rademacher near-white 입력으로
바꿨다. 실제 recipe의 fit-a/fit-b 중앙 period를 그대로 사용한 dense Gram 끝 고유값 계산은
다음과 같다.

| 범위 | exact condition | 고정 임계 | 판정 |
|---|---:|---:|---|
| fit-a, support 1024 | 11.5717140215 | ≤20 | PASS |
| fit-b, support 1024 | 12.5752910925 | ≤20 | PASS |
| fit-a+fit-b, support 1024 | 9.0580335309 | ≤20 | PASS |
| support 2048/4096/8192 | 계산하지 않음 | ≤20 | `NOT_AUDITED_NO_CLAIM` |

긴 support의 condition이 좋아진다고 주장하지 않는다. v5에서 authority 후보로 선택 가능한
길이는 현재 exact audit를 통과한 1024뿐이다. 실제 holdout residual이 실패하면 임계를
낮추거나 더 짧은 history로 자동 후퇴하지 않는다.

이 PASS는 신호 행렬의 식별 가능성만 뜻한다. 실제 덕트 P/S, ANC 감쇠, finite-memory의
물리적 참을 증명하지 않는다. 유한 capture는 1024 samples 안의 후보를 반증하거나 유지할
수 있을 뿐, 뒤늦은 echo나 무한 impulse response가 없음을 증명할 수 없다.

## 2. immutable 신호 계약

- sample rate/block: 48 kHz / 256
- 총 길이: 557,056 frames, 11.605333초
- actual submitted peak: 69 PCM count, 제한 98 이하
- whole active-slot two-output total power: `2.4505060310e-6`, 공식
  `MeasurementLevelContract` actual-int16 meter recipe `2.4719371771e-6` 대비
  `-0.037817 dB`로 초과하지 않음. 두 값 모두 `/32768` 변환을 사용한다.
- P/S 각각 `fit_a`, `fit_b`, terminal `holdout`의 서로 다른 seed
- 각 slot: 16,384-sample cyclic prefix + 32,768-sample central period +
  16,384-sample suffix
- path switch와 period boundary 양쪽 16,384 samples는 분석에서 제외
- P/S main PE는 동시에 켜지지 않음
- continuous 152--600 Hz disjoint pilot은 모든 frame에 유지
- near-white PE는 actual int16 `±49`; qualification 80--11,313.7084989848 Hz,
  실제 spectrum은 DC--Nyquist까지 존재

`BroadbandFullOctaveContractV3.canonical()`의 **전체 payload**를 plan에 inline하고 그
digest를 별도 저장한다. 현재 고정 값은 다음과 같다.

```text
control contract SHA-256:
53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2

actual submitted PCM SHA-256:
c18416e4066556479fd317659d908c215e6662d08f5bfa9d50e4ac63971c4aff

signal plan canonical payload SHA-256:
32a79b3700b457dc40373dc4dd0969301287baea7100b1ec5edd86ea907ee127
```

따라서 88.388--150 Hz를 빠뜨린 v2/7-band plan은 v5로 승격할 수 없다. 물리 식별은
v3의 연속 8대역 전부를 별도로 점수화한다.

## 3. clock 식별과 PE contamination

v5는 “파일럿만 분모”라고 가정하지 않는다. active path의 pilot line에는 near-white PE
성분도 존재한다. clock transfer의 분모는 해당 중앙 row에 **실제로 제출된 int16 전체
입력의 DFT**다. 반대 DAC channel만 그 path의 선택 line에서 actual DFT `≤1e-8`이어야 한다.
즉 PE contamination을 무시하거나 빼지 않는다.

P와 S의 pilot line 집합은 다르므로 view별 주파수 배열을 사용한다. ERR/REF×P/S 네 view는
고정 LTI amplitude와 constant phase를 각 intercept로 profile out하고, 시간 phase slope로
하나의 affine ADC→DAC rate를 추정한다. 단순 phase-array 단위 회귀뿐 아니라, fixed two-path
FIR를 통과한 전체 waveform을 affine ADC grid로 합성한 뒤 candidate q마다 raw를 cubic
재표본화하고 각 cyclic row에서 `captured FFT / matching actual submitted FFT`를 다시 만드는
회귀를 둔다. 이 raw 회귀에서 `-413.931/+413.931 ppm`을 0.01 ppm 이내로 복원했고,
`0 ppm` phase-array 회귀도 통과했다. actual PCM을 pilot-only 분모로 바꾸면 plan PCM SHA에서
거부된다. 중간부터 rate가 바뀌는 piecewise raw/phase fixture는 validation/change-point가
고정 0.0675518903-sample budget을 넘어서 거부됐다. 고역 결과 기반 phase repair는 정확히
0 sample로 receipt에 기록한다.

이 식별성은 fixed-LTI 또는 관측 가능한 구간별 stationarity를 전제로 한다. 시간에 따라
plant phase가 clock slope와 정확히 같은 형태로 변하면 유한 acoustic 관측만으로 둘을
구분할 수 없다. 그래서 실제 authority는 fit-a/fit-b의 P/S×ERR/REF×8대역 transfer agreement와
change-point가 모두 통과할 때만 열 수 있다. 임의 time-varying plant를 가정해 영구 차단하지는
않지만, raw에서 관측된 nonstationarity는 즉시 차단한다.

## 4. plant score와 terminal holdout

`score_candidate_on_role_v5`는 공통 q로 DAC grid에 재표본화된 raw와 actual submitted PCM을
받아 다음 32행을 role마다 발행한다.

```text
P/S 2 × ERR/REF 2 × v3 physical identification subband 8 = 32 rows
```

각 행에는 response bin 수, pilot-only lead/tail의 두 입력 exact-zero noise bin 수,
noise-conditioned relative residual, complex agreement, response-to-noise dB가 들어간다.
모든 행이 각각 residual ≤0.10, agreement ≥0.995, SNR ≥20 dB를
통과해야 한다. global residual로 저역 에너지가 고역 실패를 숨길 수 없다.

단일 deterministic FFT의 complex agreement를 별도 coherence로 복제하지 않는다. 독립
repeat/Welch coherence가 필요한 live authority gate는 future raw schema가 제공할 때까지
명시적으로 미구현·차단 상태다.

fit-a와 fit-b만 생성·지원 선택에 사용하고 holdout은 마지막 한 번의 terminal validation에만
사용한다. 현재 모듈은 score primitive와 합성 fixed-LTI 회귀까지만 제공한다. 실제 raw-derived
fit-a/fit-b stationarity, clock receipt, frozen 1024-tap P/S, terminal holdout receipt를 하나의
immutable training authority envelope로 발행하는 publisher는 아직 없으므로 canonical
training admission은 계속 fail-closed다.

## 5. raw publisher와 안전

plan은 정확한 raw 상대경로(기본
`results/fullband_causal_v5/raw_capture.npz`)를 미리 포함한다. offline publisher는 해당
경로만 허용하고 lexical repository containment와 모든 parent symlink를 검사한다. raw는
같은 directory의 exclusive sibling staging 파일에 쓰고 `fsync`한 뒤 atomic hard-link
no-replace로 발행하며 staging을 제거하고 parent directory도 `fsync`한다.
NPZ에는 `submitted_pcm`, `captured_pcm`, `callback_frames`, uint8 canonical JSON metadata만
저장하며 plan/actual PCM/captured/callback SHA를 결속한다. 기존 target과 symlink parent를
거부하는 회귀가 있다.

signal-only 명령은 다음이다. 이 명령은 소리를 내지 않는다.

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal_v5.py \
  --dry-run \
  --output results/data_audit/fullband_causal_v5_signal_plan.json \
  --raw-session-relative-path results/fullband_causal_v5/raw_capture.npz
```

`--execute-live`는 signal 생성이나 장치 backend 접근 전에 exit 2로 닫힌다. 실제 측정 명령,
스피커 연결 시간, volume 조건은 raw/analysis/training authority publisher가 모두 완성되고
사용자 승인을 받은 뒤에만 별도로 발행한다.

### 2026-08-29 실행 경계 재검증

tracked `assets/contracts/fullband_causal_v5_live_capture_authority.json`의
`plan_live_capture_enabled`는 현재 `false`다. 따라서
`measure_paths_fullband_causal_v5.py --execute-live`는 meter 경로나 다른 인자를 주어도
`_execute_live()`·audio primitive·`sounddevice` import보다 먼저 exit 2로 중단한다. 내부
`_execute_live()` 직접 호출도 같은 gate를 통과해야 한다. 이 상태에서 이 명령을 실제
출력 명령으로 해석하거나 호출하면 안 된다.

## 6. 남은 blocker

1. actual raw capture와 독립 xrun/slip telemetry 증거 없음. callback frame 배열은
   저장 frame accounting일 뿐 xrun/slip authority가 아니다.
2. actual-input spectral common-q 및 cubic crosscheck의 raw receipt 없음
3. fit-a/fit-b P/S×ERR/REF×8대역 stationarity/change-point PASS 없음
4. support-1024 joint causal FIR fit과 immutable operator NPZ 없음
5. terminal holdout 32행 PASS 없음
6. 실제 finite-memory가 1024 안에 충분하다는 독립 반복 증거 없음
7. 실제 level/SNR/clip/xrun/clock witness 없음

따라서 v5의 현재 권위 판정은 `BLOCKED`이며 `live_authority=None`이다. 테스트 PASS를 실제
P/S 또는 파인튜닝 자격으로 사용하지 않는다.
