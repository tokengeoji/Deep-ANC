# Output-clock-master / ref-only 런타임 기반 계약

> 상태: **구조 기반 구현 완료, 실제 런타임 통합·물리 검증 BLOCKED**
> 구현: `src/deep_anc/realtime/output_clock_master.py`
> 이 문서는 AB13X/APE 장치를 열거나 소리를 내서 얻은 결과가 아니다.

## 1. 해결하려는 문제와 해결하지 않은 문제

현행 `run_realtime.py`는 APE 입력과 AB13X 출력을 하나의 combined callback으로
묶는다. 그러면 PortAudio 이전에 APE가 입력 period를 조용히 버려도 callback telemetry가
정상처럼 보일 수 있고, 별도 ADC/DAC clock의 상대 오차가 digital-reference 시간축에
들어올 수 있다. 11.314 kHz에서 20 dB 상쇄를 유지하기 위한 timing budget은 약
`0.0675518903 sample`이므로 이 불확실성을 문서 추정으로 PASS할 수 없다.

새 기반은 다음 구조만 강제한다.

1. AB13X `OutputStream` callback만 global output frame과 NS/CS 두 채널을 소유한다.
2. callback `k`에서 생성한 future source `U_k`는 기존
   `DigitalReferenceBuffer`의 `TrainingTimingContract.digital_reference_lead_samples`
   FIFO를 거쳐 NS로 재생된다.
3. 별도 worker는 모델에 `[U_k, exact-zero ERR]`만 넣는다.
4. worker의 `y_k`는 정확히 callback `k+1`의 CS에만 제출된다. handoff는 기존 계약의
   정확히 256 samples이고, digital-reference lead와 합쳐 쓰지 않는다.
5. APE `InputStream`은 ERR/REF raw 보존, 안전 감시, 사후 평가와 physical clock
   witness 전용이다. 출력 pacing과 모델 feature가 될 수 없다.

이 구조는 ADC clock이 출력 제어를 당기는 경로를 제거하지만 다음은 해결하지 않는다.

- AB13X 실제 callback deadline/xrun 안정성
- APE raw와 DAC 사이의 physical ADC↔DAC clock witness
- ref-only로 학습한 checkpoint의 성능
- P/S 또는 덕트가 88.388 Hz~11.314 kHz에서 식별됐다는 사실
- 저·고역 물리 ANC 감쇠나 FxLMS 대비 우위

따라서 admission/receipt 모두 `physical_performance_pass=false`,
`run_realtime_integrated=false`이며 물리 PASS로 승격될 수 없다.

## 2. 출력 callback 시간축

### 2.1 정상 시작

1-block 인과 handoff에는 시작 시 이전 `y`가 존재하지 않는다. 이를 fallback silence로
숨기지 않고 protocol state로 분리한다.

| callback | NS용 future source | CS | 상태 | 성능 구간 |
|---:|---|---|---|---|
| 0 | `U_0`를 lead FIFO에 등록 | exact zero | `startup_prime=true` | 제외 |
| worker | `[U_0, ERR=0] → y_0` | 출력 callback 밖에서 실행 | completed receipt 필수 | 제외 |
| 1 | `U_1` 등록 | 오직 `y_0` | ANC ON | 포함 |
| 2 | `U_2` 등록 | 오직 `y_1` | ANC ON | 포함 |

callback 1 전에 `y_0`가 없으면 CS=0을 대신 내지 않는다. scheduler는 영구 BLOCKED가
된다. 즉 prime은 fallback counter를 0으로 보이게 만드는 예외가 아니라, 애초에 ANC
성능 구간에 포함되지 않는 명시적 ANC-OFF protocol block이다.

### 2.2 reset과 ANC OFF→ON

- reset은 과거 pending/result를 성능 구간 밖 경계에서 폐기하고 epoch를 증가시킨다.
- reset 뒤 ANC 요청이 유지돼도 다음 callback은 `reset_prime`, gain exact zero다.
- 정상 ANC OFF 뒤 다시 ON을 요청하면 다음 callback은 `rearm_prime`, gain exact zero다.
- 그 prime의 `y`가 completed된 뒤에만 다시 ANC ON이 된다.
- ANC OFF→ON도 같은 재-prime을 건너뛸 수 없다.
- 이미 worker가 claim한 job이 있는 동안 reset/OFF를 요청하면 race를 숨기지 않고
  BLOCKED한다. 실제 통합은 먼저 worker를 quiesce해야 한다.
- source lead FIFO는 reset하지 않는다. 물리 NS timeline이 계속되는데 source FIFO만
  되감으면 frame identity가 깨지기 때문이다.

## 3. immutable admission

`OutputClockMasterAdmission`은 다음을 inline payload 또는 immutable receipt SHA로
결속한다.

| 항목 | 강제 조건 |
|---|---|
| 제어대역 | exact canonical broadband v3 payload와 재계산 digest |
| 최종 목적 | 125/250/500/1000/2000/4000/8000 Hz exact octave objective |
| 물리 식별 grid | 88.3883476483 Hz부터 11,313.7084989848 Hz까지 8구간 |
| timing | inline `TrainingTimingContract`와 digest, 48 kHz, handoff=256, exact lead |
| 실험/모델 | experiment contract, checkpoint, deployment artifact SHA |
| 모델 입력 | inline `digital_reference_only_err_exact_zero` payload와 digest |
| dropout | `reference_dropout=0` |
| ERR 제거 | `error_dropout=1`, 또는 canonical train item 전부 ERR exact-zero receipt |
| ref-only 검증 | ablation, absolute G0, validation receipt SHA 모두 필수 |
| 수치 등가 | offline-streaming receipt와 max absolute error `<=1e-5` |
| APE 역할 | raw/safety/evaluation witness only, pacing/feature 둘 다 false |

v2 계약을 v3로 자동 승격하거나 150~1600 Hz 결과를 이 admission에 대입할 수 없다.
SHA 형식만 맞는 외부 receipt의 실제 파일 존재·내용·상호 결속은 향후 deployment gate가
검증해야 한다. 현재 클래스는 그 receipt를 만들어 주거나 성능을 자기 선언하지 않는다.

## 4. scheduler fail-closed 조건

다음 중 하나면 해당 scheduler instance는 영구 BLOCKED다.

- callback index 또는 global output frame의 drop/add/reorder/sample slip
- `y_k`가 callback `k+1`보다 한 block 빠르거나 늦음
- callback 시점 inference underflow, queue overflow
- 이전 epoch/job의 stale 또는 reused control
- worker가 job의 `U_k`와 다른 reference를 사용
- ERR feature에 한 sample이라도 exact zero가 아닌 값 사용
- OFF/prime callback에서 gain이 한 sample이라도 0이 아님
- source 또는 gain/control의 dtype/shape/finite/무클립 조건 위반
- reset/OFF 시 worker job이 inflight인 race
- 실제 어댑터 counter의 xrun, callback status, deadline miss, queue under/overflow,
  fallback, drop/add, sample slip, stale/reuse, nonzero ERR 중 하나라도 0이 아님

output callback 메서드는 모델/engine 객체를 받지 않는다. worker는
`claim_inference_job()`으로 immutable-byte-backed `U_k`와 zero ERR를 받은 뒤,
실제로 사용한 두 입력을 `submit_inference_result()`에 다시 제시해야 한다.

현재 Python scheduler는 상태·SHA·receipt를 함께 검증하는 **실행 가능한 규격**이며
callback deadline이 검증된 production adapter가 아니다. 실제 adapter는 같은 상태 머신을
보존하되 callback에서는 미리 확보한 버퍼 제출과 lock-free enqueue만 수행하고, SHA 및
Pydantic receipt 구성은 비실시간 thread로 옮긴 뒤 raw byte identity를 재검증해야 한다.

## 5. raw/receipt frame identity

각 output frame은 다음을 SHA-256으로 보존한다.

- generated future source `U_k`: float32와 동일 변환의 S16
- model reference: float32
- lead FIFO 뒤 실제 NS playback: float32와 S16
- model control `y_(k-1)`: float32
- callback별 ANC gain: float32, min/max
- gain 적용 뒤 실제 CS control: float32와 S16
- 최종 NS/CS interleaved S16 payload
- generated/reference/playback/control/output의 callback와 global frame 시작·끝
- source job ID, epoch, target callback `k+1`

SHA는 contiguous raw payload byte를 계산하고 dtype/shape는 schema가 별도로 고정한다.
실제 sounddevice receipt에서는 이 S16 payload SHA가 PortAudio 제출 raw NPZ와 다시
일치해야 한다. 마지막 callback이 만든 `U_k`는 유한 관측 창 밖 `k+1` tail로 명시되며,
중간 drop이나 fallback으로 간주하지 않는다.

## 6. `run_realtime` 통합 전 남은 BLOCKED 항목

이번 변경은 기존 runtime을 수정하지 않았다. 다음을 모두 별도 구현·검증하기 전에는
새 구조로 실제 소리를 출력하면 안 된다.

1. AB13X 전용 2-channel S16 `OutputStream` adapter를 만들고 그 callback만 global frame을
   증가시킨다. callback 안에서는 배열 복사/제출만 하고 모델 추론, 파일 I/O, APE 대기를
   금지한다.
2. 기존 engine `step(ref, err)`를 worker에서만 호출하되 `err`가 receipt의 exact-zero
   block인지 결속한다. 현재 legacy/corrected checkpoint를 ref-only 자격으로 간주하지
   않는다.
3. APE를 별도 `InputStream`으로 열고 ERR/REF raw를 보존한다. APE callback 정지/지연이
   출력 callback을 막아서는 안 된다.
4. APE 안전 경로가 비동기여도 즉시 AB13X CS를 mute할 수 있는 watchdog/atomic gate를
   설계한다. 입력 stream이 죽었을 때 과거 control을 계속 출력하면 안 된다.
5. gain ramp, limiter, reset, user ANC OFF/ON을 scheduler epoch/prime과 한 경계로 묶는다.
6. 실제 callback마다 xrun/status/deadline/fallback/drop/add/slip/queue counter를 exact-zero
   receipt로 남긴다. backlog는 허용 256 이내이고 excess backlog가 0이어야 한다.
7. submitted S16, actual source/control/gain, APE ERR/REF raw, config/checkout/checkpoint/
   admission receipt를 no-replace NPZ/JSON 및 SHA로 결속한다.
8. 30초 이상 reserved pilot physical witness로 ADC↔DAC q/phase slope, change point, slip,
   stationarity를 재검산한다. NS/CS가 같은 DAC clock이라는 사실을 ADC↔DAC witness로
   오인하지 않는다.
9. exact v3/ref-only로 새로 학습된 checkpoint의 absolute G0, validation,
   offline-streaming equivalence가 실제 artifact로 존재해야 한다.
10. read-only/dry-run 테스트를 먼저 끝낸 뒤 사용자 승인, 볼륨 최소, 장치 무점유 확인을
    거쳐야만 실음 실험을 설계한다.

## 7. 최종 판정

현재 말할 수 있는 것은 다음뿐이다.

> AB13X 출력 clock 하나로 NS/CS와 global frame을 소유하고, digital reference `U_k`의
> ref-only 결과를 정확히 다음 callback에 제출하는 구조를 순수 코드로 표현하고
> fail-closed 단위 테스트할 수 있다.

현재 말할 수 없는 것은 다음이다.

> 이 구조가 Jetson 실시간 deadline을 만족한다, 125 Hz~8 kHz octave를 감쇠한다,
> unseen speech/music/environment/machine에서 ANC가 된다, 또는 고주파에서 FxLMS보다
> 낫다.

그 결론에는 위 통합 항목과 실제 덕트 OFF/ON raw가 필요하다.
