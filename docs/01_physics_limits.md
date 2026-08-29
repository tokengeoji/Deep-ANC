# 01. 물리 한계와 지연 예산 — 정직한 기대치

이 문서는 "왜 어떤 소음은 지워지고 어떤 소음은 물리적으로 지울 수 없는지"를 수치로 고정한다.
여기 있는 숫자가 데이터 합성(`data_sim.yaml`)과 학습 플랜트(`duct.yaml`)의 근거다.

## 1. 역사적 실측 진단값 (현재 official/training-ready 아님)

| 항목 | 값 | 출처 |
|---|---|---|
| I/O 왕복지연 (block 256/low) | **30.6ms** (1470샘플) | calibration_4s.log 상호상관 피크 |
| I/O 왕복지연 (block 512/high) | 57.1ms | calibration_4s_512.log |
| legacy S(z) 진단 지연 | **1462샘플 = 30.46ms** | secondary_path_il.npz (당시 150–1600Hz 일관성 **0.9990**) |
| legacy P(z) 진단 지연 | **1602샘플 = 33.38ms** | primary_path_il.npz (당시 동 대역 **0.9993**), 같은 capture·같은 앵커 |
| **P − S** | **140샘플** | 이 측정의 **유일한 물리 불변량** — 독립 캡처 9건에서 139~141 |
| **lead** | **116샘플** | `S 1462 + handoff 256 − P 1602`. 캡처 9건에서 115~117 |
| S/P 신뢰 대역 (`consistency_band_hz`) | **150–1600Hz** | 동 npz. **2026-08-05 재발행에서 150–600 → 150–1600 으로 확대** |
| 구동 대역 (`excitation_band_hz`) | P **64–1648Hz** / S **72–1640Hz** | 인터리브라 두 채널이 인접 FFT 빈을 번갈아 쓴다 — **두 경로가 다른 값이다** |
| **실제 저역 재현 한계** | **80–150Hz** | 클린 재측정 후에도 S 부대역 일관성 **0.758** — 스피커 저역 SNR 8–10dB. **여기만 진짜 물리 한계다**. (80 Hz 는 `band_consistency_hz` 의 최저 부대역 하한이고 구동은 64~72 Hz 부터다) |
| 3-스레드 런타임 핸드오프 | **+256샘플 = 5.33ms** | 콜백→추론→다음 콜백 1 hop (설계 C1) |
| digital `D_noise` legacy 기준선 | **1602샘플 = 33.38ms** | primary_path_il.npz. 기하 예측 1612 와 10샘플 차; 신규 official 근거로 재사용 금지 |

주의: S(z) 지연 30.46ms 의 대부분은 **USB/ALSA 버퍼 지연**이다 (덕트 내 음향 전파는
CS→ERR 50mm = 0.15ms 에 불과).

위 수치는 과거 캡처에서 찾은 물리 진단값이다. 해당 NPZ는 observed submitted PCM,
q+joint-LS/cubic witness, immutable source SHA와 두 운영자 확인이 없어 readiness가 거부한다.
새 strict 48k 캡처 전에는 어떤 값도 현재 official P/S로 간주하지 않는다.

### 현재 strict P/S (2026-08-27, official)

현재 학습·runtime timing의 단일 출처는
`capture_id=5ac1313488c8434bb4d672a36503df59`이다. P/S는 같은 raw·analysis SHA,
48 kHz/256/low, xrun 0, 19 kept repeats를 공유한다.

| 항목 | 현재 strict 값 |
|---|---:|
| P `delay_samples` / `bulk_delay_samples` | 1386 / 1642 |
| S `delay_samples` / `bulk_delay_samples` | 1245 / 1501 |
| handoff | 256 |
| **lead** | **115** (`1245 + 256 − 1386`) |
| P/S 150–1600 Hz consistency | 0.999821 / 0.999716 |

이 150–1600 Hz 식별은 모델 감쇠나 2–8 kHz ANC 성능을 증명하지 않는다.

> [!CAUTION]
> **절대 지연(1602 / 1462)은 캡처 간 재현되지 않는다.** 저장된 캡처 11건을 전수 재분석하면
> 유효 9건 전부에서 `P−S = 139~141`, `lead = 115~117` 이지만 절대 지연은
> low-latency 1565~1659 / high-latency 2858~2888 로 흩어진다(캡처별 타임베이스 드리프트
> 364~729 ppm + 앵커 반복 선택 의존). **P 와 S 는 반드시 같은 캡처·같은 앵커의 값을
> 함께 써야 한다** — 아티팩트에 `capture_id` · `anchor_repeat` · `kept_repeat_indices`
> 가 박혀 있고, 게이트가 일치를 검사한다.
>
> 그리고 **`S/P 신뢰 대역 = 150–600Hz` 라는 이전 판의 값은 물리가 아니라 측정 결함이었다.**
> 오염 반복 5개를 게이트가 통과시킨 결과였고, 기각 후 재계산하면 1000–1600Hz 에서도
> P 0.999 / S 0.999 가 나온다. 상세와 재현 명령은
> [README §7.5](../README.md#75-sz-가-33-틀려-있었다--게이트가-오염-반복-5개를-통과시켰다) ·
> [docs/12 §2.3](12_system_summary.md#23-실측-경로-자산).

## 2. 음향 전파 시간 (duct.yaml 기하, c=343m/s)

| 구간 | 거리 | 시간 | 샘플(@48k) |
|---|---|---|---|
| NS(0) → REF(0.100) | 100mm | 0.29ms | 14 |
| REF(0.100) → CS(1.050) | 950mm | **2.77ms** | 133 |
| CS(1.050) → ERR(1.100) | 50mm | 0.15ms | 7 |
| NS(0) → ERR(1.100) | 1100mm | **3.21ms** | 154 |

## 3. 두 레퍼런스 모드의 지연 물리 (핵심)

### digital reference — 1차 릴리스 기본 모드

소음을 Jetson 이 직접 생성해 ch0 으로 재생한다. 소음 경로와 상쇄 경로가
**같은 USB 출력 장치**를 지나므로 전기/버퍼 지연 δ_out 이 양쪽에 공통으로 들어가 **소거**된다.

```
d 도달:  δ_out + t_ac(NS→ERR)=3.21ms      y 도달: δ_out + t_ac(CS→ERR)=0.15ms (+핸드오프 5.33ms)
→ 예측 여유 = 3.21 − 0.15 = +3.06ms (147샘플) − 핸드오프 5.33ms = −2.27ms
```

새 학습의 정렬은 strict P/S NPZ에서만 유도한다. `TrainingTimingContract`는 P/S bulk
`delay_samples`, compact FIR peak 지연, runtime 256-sample handoff,
`PlantDelays.lead()`와 합성 총 선행량을 서로 다른 필드로 보존한다. recorded branch는 같은
총 선행량에서 세션별 정렬 잔여를 빼서 lead를 유도한다.

정상 정렬은 수치가 무엇이든 다음 등식으로 검증한다.

```text
reference[t]에 대응하는 synthetic d 도달 시각
  == reference[t]로 만든 y가 S+handoff를 지나 도달하는 시각
  == recorded session residual alignment + session-derived lead
```

이는 미래 입력 참조가 아니다. Jetson이 앞으로 재생할 자기생성 소스를 먼저 모델에 공급하고
실제 playback을 FIFO로 늦추는 인과적 스케줄이다. checkpoint/ONNX/runtime의 timing contract
SHA가 다르면 오디오 시작 전에 거부한다. 과거 수동 lead artifact는 diagnostic-only다.

현재 `configs/duct.yaml`은 위 `capture_id=5ac13134…`의 strict P/S를
가리킨다. `delay_samples`나 lead를 문서/명령에 수기 입력하지 않고
NPZ와 `TrainingTimingContract`/`PlantDelays.lead()`에서만 유도한다.
`assets/measured/*_path_il.npz`처럼 strict suffix가 없는 구형 파일과 구형
`D_noise=1602`는 legacy 진단용으로만 보존한다. 옛 raw를
`scripts/data/reanalyse_paths_interleaved.py --dry-run`으로 읽는 것은 누락
provenance를 되살리거나 official로 승격하지 못한다.

### 현재 Stage-1의 P(z) 대용 정책

실측 `P(z)`가 없을 때 1D `p_err` RIR은 절대 장치 gain이 없고, 측정 장치 스케일의
`S(z)`와 직접 비교할 수 없다. 실제로 이 조합은 필요한 y가 limiter ±0.2를 크게 넘어
영출력(약 0dB)이 유리한 잘못된 목적을 만들었다. 현행 `secondary_surrogate`는
`P(z)`의 FIR/gain에 `S(z)`를 빌리고 P bulk delay는 strict primary NPZ에서 읽는다. 따라서
P/S 스케일이 맞는 역매핑을 학습할 수 있지만 다음 제한이 있다.

- checkpoint `physics_status=secondary_surrogate_representation_pretrain`
- 실제 noise 스피커 경로의 주파수응답·극성·비선형을 재현하지 않음
- surrogate val dB로 실제 감쇠나 FxLMS 대비 우위를 주장하지 않음
- 실측 `P(z)` + 실측/재보정 `S(z)` + 독립 recorded val/test 이후에만 물리 성능 판정

### acoustic reference — 2단계 목표

외부 소음을 레퍼런스 마이크로 수음한다. 마이크가 소음을 들은 시점부터
상쇄음이 에러 마이크에 도달할 때까지:

```text
필요 예측 지평 = strict S bulk delay + handoff − REF→ERR의 측정/기하 선행분
```

현 하드웨어에서는 이 지평이 광대역 비주기 소음의 상관시간보다 훨씬 길다. 정확한 수치는
strict capture 이후 계약에서 계산하며 legacy 절대 지연을 재사용하지 않는다. 따라서:

| 잡음 유형 | 상쇄 가능성 | 이유 |
|---|---|---|
| 톤/고조파/회전기계(주기성) | **가능 (강한 감쇠)** | 주기 신호는 무한히 예측 가능 — LSTM/MHSA 가 주기 학습 |
| 협대역/준정상 색잡음 | 부분 가능 | 상관시간이 P 에 근접한 만큼만 |
| 광대역 랜덤(백색) | **물리적으로 불가** | 상관시간 ≪ 33ms — 어떤 알고리즘도 불가 |

### 고전 인과성 예산과 3단계

acoustic-ref 광대역이 되려면 전기적 총지연 < REF→CS 음향 전파 **2.77ms** 가 필요하다.
현 USB/ALSA 경로에서는 불가능하다. 덕트 구조 문서의 하드웨어 개선(USB 폐기 → I2S DAC
직결, 더 작은 버퍼)이 후속 acoustic-ref의 선결 조건이다. 지연이 바뀌면 새 strict NPZ를
측정해 timing contract를 다시 만들며 YAML에 수동 delay를 쓰지 않는다.

## 4. 주파수 상한

- **평면파 컷오프 f_cut = c/(2a) = 343/(2×0.105) = 1,633Hz.**
  그 이상은 고차 모드가 전파되어 단일 CS/ERR 로는 단면 전체를 제어할 수 없다
  (마이크 1개가 단면을 대표하지 못함 → 제어 붕괴 위험).
- Stage-1 실험 목표 대역은 **80–1600Hz**(`realistic_target_band_hz`, 2026-08-05 에
  80–800 에서 확대)이고, `S(z)` 가 신뢰되는 범위는 **150–1600Hz**(`consistency_band_hz`)다.
  Stage-1은 두 범위의 교집합인 **150–1600Hz** trusted NMSE를 최적화하고 fullband NMSE를
  do-no-harm 관측값으로 함께 기록한다.
- **이전 판의 "신뢰 범위는 150–600Hz 뿐" 은 물리 한계가 아니라 측정 결함이었다.**
  오염 반복 5개를 게이트가 통과시킨 결과였고, 기각 후 재계산하면 **S 부대역 일관성**이
  300–600Hz **0.9995** / 600–1000Hz **0.9981** / 1000–1600Hz **0.9994** 다
  (P 는 각각 0.9995 / 0.9982 / 0.9995). 대역 확대는 이 재발행을 근거로 한 것이다.
  재현: `np.load('assets/measured/secondary_path_il.npz')['band_consistency']`
- **80–150Hz 는 클린 재계산 후에도 S 부대역 일관성 0.706~0.758 로 회복되지 않는다
  — 이것이 진짜 물리 한계다** (스피커 저역 SNR 8–10dB, 독립 캡처 간 `|H|` 편차 27.8%).
  이 대역 성능은 주장하지 않는다.
- ⚠ **대역을 넓힌 것과 고역이 좋아지는 것은 다르다.** 손실에 대역 밖 do-no-harm 항이
  없던 상태에서 실측 모델은 2–8kHz 를 **15–22dB 증폭**한다(docs/12 §4.7).
  손실 수정이 선행되지 않으면 대역만 넓히는 것은 오히려 150–600Hz 를 해칠 수 있다.
- 사용자가 확정한 최종 광대역-v2는 2/4/8kHz까지의 point-control과 matched FxLMS 우위를
  요구한다. 8kHz octave 상단 11.314kHz까지 ① P/S multi-panel 재식별 ② sub-sample clock/
  phase ③ 실제 ERR target-d data coverage ④ 다점 공간 실측을 통과한 뒤에만 연다. 상세는
  `docs/18_broadband_anc_guardrails.md`다. 모델·데이터 shape가 48kHz를 지원한다는 사실은 이
  네 물리 게이트를 대체하지 않는다.
- 축방향 공진 70/210/350/489/629Hz — 이 주파수로 가진하면 큰 SPL 을 얻어
  시연 효과가 좋다 (70Hz 는 스피커 f_s 미만이라 재생 곤란, **210/350Hz 권장**).

## 5. 요약 — 시나리오별 기대치와 판정 단계

| 시나리오 | 모드 | 판정 |
|---|---|---|
| 현재 surrogate 체크포인트 | digital | 표현 사전학습 결과. 실제 감쇠 수치로 사용 금지 |
| 톤/멀티톤/기계음 150–1600Hz | digital measured | 실측 P/S 파인튜닝과 독립 OFF/ON/OFF 평가 후 판정. **현재 G4 FAIL** |
| 대역잡음 80–1600Hz | digital measured | 밴드별 판정. **현재 2–8kHz 를 15–22dB 증폭한다 = 절대 목표 1 위반** |
| 80–150Hz | — | **주장하지 않는다.** 클린 측정 후에도 S 일관성 0.758 (진짜 물리 한계) |
| 비선형(고조파+포화) | digital measured | THD/IMD 실측 후 점진적 비선형 커리큘럼에서 DL 우위 검증 |
| 외부 팬/모터 소음 | acoustic | 2단계 — 주기성분 위주 감쇠 |
| 외부 백색소음 | acoustic | 불가 (정직하게 명시) — 3단계 하드웨어 개선 후 재도전 |
