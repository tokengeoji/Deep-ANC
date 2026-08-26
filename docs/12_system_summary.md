# 12. 시스템 총정리 — 하드웨어 · 데이터 · 아키텍처 · 결과 · 개선방안

> **보존된 2026-08-06 forensic snapshot이다.** 아래 legacy P/S·lead·checkpoint·격리 판정과
> 실행 명령을 현행 학습 계약으로 재사용하지 않는다. 2026-08-26 이후의 authoritative 상태와
> 다음 명령은 [HANDOFF.md](../HANDOFF.md), timing은 `TrainingTimingContract`, 학습 판정은
> current readiness report가 단일 출처다. 특히 기존 82세션은 전량 폐기가 아니라 historical
> 계보 복구와 component regrouping 뒤 사용하며, 새 strict P/S 전에는 official delay가 없다.

> 최종 갱신 **2026-08-06**. 이 문서는 **측정된 것만** 담는다. 각 수치의 출처 명령을 함께 적었고,
> 재현이 불가능한 값은 넣지 않는다. 진행 중인 작업 상태는 [HANDOFF.md](../HANDOFF.md)가 단일 출처다.

> [!CAUTION]
> **2026-08-05~06 에 이 문서의 결론 다수가 뒤집혔다.** 이전 판은 "측정된 것만 담는다"고
> 선언했으면서도 (a) 오염된 측정을 물리 법칙으로 오독했고, (b) 자기 저장소의 원시 산출물보다
> 낙관적이었다. 뒤집힌 것 목록:
>
> | 이전 판의 서술 | 실제 (검증됨) |
> |---|---|
> | "600 Hz 위 `S(z)` 재현 불가 = 상쇄 스피커 물리 한계" | **측정 후처리 결함.** 오염 반복 기각 후 신뢰대역 **150–1600 Hz** ([§2.3](#23-실측-경로-자산)) |
> | "−2 dB 정체 = 용량 부족 → `tiny_wide` 실험" | **철회.** 데이터 시간축 붕괴 + 33~54% 틀린 플랜트 ([§4.5](#45-진단--용량-부족이-아니었다-철회)) |
> | "1 kHz 위는 변하지 않는다 / 증폭하지 않는 것이 성공 기준" | **2–8 kHz 를 15–22 dB 증폭한다** ([§4.7](#47-실기-anc-사전학습-tiny--ort-cpu)) |
> | "파인튜닝으로 1.30 dB 개선" | **무효.** 전후가 서로 다른 플랜트 ([§4.4](#44-stage-2-파인튜닝-tiny-50k-step--g4-는-fail)) |
> | "재생·녹음은 사실상 동기(+0.4 ppm)" | **철회.** raw adjacent-cycle 관측은 약 **+620 ppm**이며, 버퍼 프레임 슬립과 별도로 fractional-bin P/S 누설을 만든다 ([§1.2](#12-실시간-오디오-체인)) |
> | "max 40–58 ms 는 데스크톱 세션 잡음" | **커널 RT 스로틀링** (`sched_rt_runtime_us=950000`) ([§4.6](#46-추론-지연--30w-vs-maxn)) |
> | GPU 지연 수치 | **듀티 100% 연속 실행** 값. 실제 듀티 6% 에서 3.7배 나빠진다 ([§4.6](#46-추론-지연--30w-vs-maxn)) |
> | "전수 QA 80/80 PASS" | 형식만 봤다. **시간축이 붕괴**해 있었고 지금은 전량 격리 ([§2.1](#21-실측-덕트-녹음-파인튜닝용-70)) |
> | "진입 게이트 9개 전부 PASS" | **초록불 9개가 아무것도 보증하지 못했다** ([§5.0](#50-근본-원인--게이트-9개가-pass-인데-전부-무의미했다)) |

---

## 1. 하드웨어

### 1.1 연산 플랫폼

| 항목 | 값 | 확인 |
|---|---|---|
| 보드 | NVIDIA **Jetson AGX Orin Developer Kit** | `/proc/device-tree/model` |
| JetPack | R36 rev 4.4 (2025-06-16), OOT 커널 | `/etc/nv_tegra_release` |
| CPU | **12 × Cortex-A78AE**, 3클러스터 × 4코어 | `lscpu` |
| CPU 클록 | **2.20 GHz** (30W 시절 1.73 GHz) | `scaling_max_freq` |
| GPU | Orin (nvgpu), compute capability **8.7** (Ampere) | `torch.cuda.get_device_capability()` |
| CUDA | 12.6, 드라이버 540.4.0 | `nvidia-smi` |
| 메모리 | **61 GiB 통합** (CPU/GPU 공유) | `free -h` |
| **전원 모드** | **MAXN (ID 0)** — 2026-08-05 30W 에서 전환 | `nvpmodel -q` |

> [!IMPORTANT]
> **2026-08-05 MODE_30W → MAXN 으로 전환했다(재부팅 필요).** 30W 에서는 CPU 클록이
> 2.20 → 1.73 GHz (79%) 로 묶여 있었고, **그 이전의 모든 지연 수치가 그 제한 아래**에서
> 측정된 것이었다. 전환 후 GPU P50 이 절반으로 줄었다([§4.6](#46-추론-지연--30w-vs-maxn)).
> 되돌리려면 `sudo nvpmodel -m 2` 후 재부팅.
>
> ```bash
> sudo nvpmodel -m 0      # MAXN (재부팅 확인 프롬프트에 YES)
> sudo nvpmodel -m 3      # MODE_50W
> ```

### 1.2 실시간 오디오 체인

```
        ┌─ 재생 (USB, ADAPTIVE 싱크) ──────────────────┐
n(t) ──►│ AB13X USB Audio (card 2) → TPA3116D2 앰프 → NS/CS 스피커
        └──────────────────────────────────────────────┘
                                                    ↓ 덕트 음향
        ┌─ 녹음 (Tegra APE I²S) ───────────────────────┐
        │ INMP441 ×2 → I²S2 → APE ADMAIF2 (card 1)     │
        └──────────────────────────────────────────────┘
```

USB DAC와 Tegra APE I²S ADC는 별도 클록 도메인이다. 보존된 raw A의 adjacent-cycle
시간영역 관측은 주기당 약 **+620 ppm**이었으며, 고정 정수-bin FFT로 무시할 수 없는 값이다.

| 항목 | 값 |
|---|---|
| 샘플레이트 / 블록 | 48 kHz / 256 샘플 (5.33 ms) |
| 입력 | `hw:APE,1` S32_LE 2ch — ch0 ERR, ch1 REF |
| 출력 | `hw:Audio,0` S16_LE 2ch — ch0 소음, ch1 상쇄 |
| RT 우선순위 | `@audio rtprio 95` (`ulimit -r` = 95) |
| 마이크 잡음 바닥 | **−67.4 dBFS** (기동 트랜지언트 0.5초 제외 후) |

> [!NOTE]
> **두 클록을 동기라고 가정하면 안 된다.** raw A에서 실측한 약 +620 ppm은 0.125초/6000
> sample 주기의 길이를 약 3.72 sample 바꾸며, 1600 Hz를 약 0.124 bin 이동시킨다. 이때
> guard=1 정수 FFT의 1--1.6 kHz 교차성분은 P/S 모두 약 15%였다. 새 official 측정은 원시
> ERR/REF adjacent-cycle clock witness로 ``q=N/(N+d)``를 구하고, 실제 제출 int16 PCM의
> 199개 톤을 fractional-frequency joint real LS로 동시에 분리한다. cubic affine
> playback-grid 재표본화+정수 FFT와 전대역 및 네 부대역에서 일치하지 않으면 실패한다.
>
> `results/clock_drift/20260804_222644/clock_drift.json` 의 `drift_ppm: 92.9` 를 근거로 쓰지
> 말 것 — `residual_rms_samples: 276`, `residual_max: 824` 로 1차 적합이 성립하지 않으며,
> **스크립트 자신이 `verdict` 에 "기울기는 작은데 잔차가 크다 — 무작위 점프다. 버퍼
> 드롭/중복을 의심하고 ALSA 직접 경로로 재확인한다" 라고 판정해 두었다.** 두 번째 세션
> `20260804_224225` 는 `drift_ppm 1400.1` / "상관 자체가 낮다" 로 역시 무효다.
> 이 두 무효 세션만으로 단일 원인을 정할 수 없다. 별도로 확인된 **출력 버퍼 프레임
> 슬립**([§2.3](#23-실측-경로-자산), P−S 상대 τ 1.4 → 32 샘플 점프)과 지속적인 비동기
> sample-rate offset은 모두 fail-closed로 검출해야 한다.
>
> ```bash
> .venv/bin/python -c "
> import json; d=json.load(open('results/clock_drift/20260804_222644/clock_drift.json'))
> print('ppm', d['drift_ppm'], 'residual_rms', d['residual_rms_samples']); print(d['verdict'])"
> ```

### 1.3 지연 예산 — **99.6%가 버퍼다**

| 항목 | 샘플 | ms | 비중 |
|---|---:|---:|---|
| 음향 CS→ERR (실제 소리가 가는 시간) | 7 | 0.15 | 0.4% |
| **I/O 왕복 (USB DAC + I²S ADC 버퍼)** | **1455** | **30.31** | **85%** |
| 3-스레드 handoff (= block 256) | 256 | 5.33 | 15% |
| 추론 (tiny + ORT CPU, P99 @MAXN) | 69 | 1.44 | **4.0%** |
| **루프 총지연 (S 1462 + handoff 256)** | **1718** | **35.79** | |

소리가 공기를 가로지르는 데 0.15 ms인데 루프는 35.79 ms다. **최적화 여지가 가장 큰 곳은
추론이 아니라 I/O 버퍼**이며, 블록을 256 → 128로 줄이면 왕복이 대략 절반이 된다
(실측 근거: block 512 → 57.1 ms, block 256 → 30.6 ms).

### 1.4 덕트

| 항목 | 값 |
|---|---|
| 내부 길이 / 단면 | 1.190 m / 105 × 105 mm 사각 (PMMA 10 mm) |
| 경계 | closed–open (폐단 반사 0.80, 개방단 −0.45) |
| **평면파 차단** | **1633 Hz** = `c / (2 × 0.105)` |
| 축방향 공진 | 70 / 210 / 350 / 489 / 629 Hz |
| 배치 | NS `x=0` · REF `x=0.100` · **CS `x=1.050` 상면 Ø40** · ERR `x=1.100` · 개구 `x=1.200` |

`ERR x=1.100`은 **잠정값**이다(CS 마운트 구간 0.990–1.110과 겹침). 확정 시 `duct.yaml`과
RIR 뱅크를 함께 갱신해야 한다.

### 1.5 학습 플랫폼

| 항목 | 값 |
|---|---|
| 인스턴스 | Elice `central-02:58626`, **A100 80GB PCIe × 1** |
| CPU / 디스크 | 16 vCPU / 128 GB (사용 70 GB) |
| 요금 | ₩2,000/시간 |
| torch | 2.5.1+cu121 |

---

## 2. 데이터

### 2.1 실측 덕트 녹음 (파인튜닝용, 70%)

| 항목 | 값 |
|---|---|
| 규모 | 80세션 · 93.3분 · 64그룹 — **형식** QA 80/80 PASS |
| 계열 | speech / music / environment / machine 각 20세션 |
| 분할 | train 64 / val 9 / test 7 (**그룹 단위**) |
| 채널 | ERR + REF 2ch `mics.wav` + 재생한 `source.wav` |
| 재생 레벨 | peak 0.06, 크레스트 10 dB 제한 |
| **현재 상태** | **전량 격리됨** — `data/recorded_broken/` (되돌릴 수 있음: `quarantine_ledger.json`) |

> [!CAUTION]
> **이 데이터셋은 시간축이 붕괴돼 있다 — 재녹음 필요.** 8세션 직접 측정
> (nperseg 8192, 150–600 Hz 중앙값):
>
> | 측정 | 값 | 의미 |
> |---|---|---|
> | coh²(`source.wav` → ERR ch0) | **0.021 ~ 0.126** | 학습이 배워야 할 바로 그 관계 — 사실상 없다 |
> | coh²(REF ch1 → ERR ch0) | **0.959 ~ 0.991** | 음향 자체는 멀쩡하다 |
> | source→ERR 지연 표준편차 | **248 ~ 4813 샘플** | 시간축이 세션 안에서 요동친다 |
> | 창별 최적정렬 후 coh² | 1.5s 0.430 / 0.5s 0.541 / **0.1s 0.745** / 25ms 0.518 | 창을 줄여도 1.0 에 수렴하지 않음 = **빠른 위상 점프** |
>
> **전수 QA 80/80 PASS 는 RMS·클리핑·길이·그룹 누수만 본 결과**이며 정렬은 검사하지
> 않았다. 새 QA 를 같은 80세션에 돌리면 **0/80 PASS** 다:
>
> ```
> $ .venv/bin/python scripts/data/validate_recorded_sessions.py
> [FAIL] 실측 QA: 0/80 세션, 93.33분
>   '재생→캡처 150-600Hz 결맞음 0.040 < 0.60 — 학습이 배워야 할 관계가 없습니다
>    (음향 대조군은 0.989 로 정상 = 음향이 아니라 녹음 소프트웨어 타임베이스 문제)'
> ```
>
> **원인**: `source.wav` 는 **재생 배열이지 방출 시각이 아니다.** USB DAC 의 PLL 헌팅
> (주기 4~5초, 진폭 **259~407 샘플**)이 그 둘을 벌려 놓는데,
> [`record_duct.py`](../scripts/data/record_duct.py) 가
> `sd.Stream(device=(in_dev, out_dev))` 로 **서로 다른 두 장치**(USB AB13X 재생 /
> Tegra APE I²S 캡처)를 duplex 로 묶고 콜백에서 출력 커서와 입력 커서를 **인덱스로만**
> 정렬했다 — "두 커서가 같은 물리 시각"이라고 **단언**하고 측정하지 않았다.
> 수정본은 `src/deep_anc/data/timeline.py` 에서 실제로 **측정**하고, 저장 시점에 게이트한다.
>
> **오프라인 재정렬로 80세션 중 47개를 복구했다** (`results/timeline/realign_full.json`,
> `ref_witness_warp_v1` — REF 마이크를 시간축 증인으로 써 L(t) 를 추정하고 ERR 로만
> 검증한다. 게이트 `coh² ≥ 0.9` 그리고 `유효창 비율 ≥ 0.9`):
>
> | 지표 (47세션) | 전 | 후 |
> |---|---|---|
> | coh²(source→ERR) 150–600 Hz | 0.025 ~ 0.182 (p50 **0.078**) | 0.905 ~ 0.973 (p50 **0.947**) |
> | coh²(source→ERR) 600–1600 Hz | 0.007 ~ 0.071 (p50 0.019) | 0.596 ~ 0.920 (p50 **0.824**) |
> | 선형 Wiener 하한 (중앙) | **−0.23 dB** | **−12.09 dB** |
> | 잔여 지연 중앙값 | — | 142.02 ~ 143.37 샘플 (p50 **142.53**, 세션 간 산포 1.35) |
> | 잔여 robust-std / p95−p5 | — | 1.22 ~ 2.99 / 4.83 ~ 11.98 샘플 |
>
> 잔여 지연 **142.53** 은 덕트 기하 예측 **139.9 샘플**과 일치한다 — 재정렬이 물리적으로
> 옳다는 독립 증거다.
>
> **그래도 33세션은 재녹음해야 한다** — 47세션으로는 게이트(`≥80세션 그리고 ≥90분`)를
> 못 채운다. **게이트를 낮추지 않는다.** 다만 실패분만 다시 받으면 되므로 스피커 발음
> 시간이 93.3분 → **약 38.5분**으로 줄었다.
>
> 그리고 **파인튜닝 val −0.07 dB 정체가 이 데이터의 선형 하한 −0.23 dB 와 사실상 같다**
> — 모델 용량 문제가 아니었다는 직접 증거다([§4.5](#45-진단--용량-부족이-아니었다-철회)).

**계열별 그룹 수가 고르지 않다** — 이것이 G4 판정의 약점이다.

| 계열 | 전체 그룹 | val | test | 원본 |
|---|---:|---:|---:|---|
| speech | 20 | 2 | 2 | LibriSpeech |
| music | 20 | 2 | 2 | FMA-small |
| environment | 16 | 2 | 1 | ESC-50 (환경 16종) |
| **machine** | **8** | **1** | **1** | ESC-50 (기계 **8종뿐**) |

`machine`은 성능이 가장 나쁜데 val/test가 각각 **그룹 하나**로 판정된다. ESC-50의 기계
카테고리가 8종뿐이라 생긴 구조적 한계다.

### 2.2 합성 소스 풀 (30%)

| 소스 | 비율 | 담당 |
|---|---:|---|
| DNS noise_fullband | 30% | 광범위 실환경 소음 |
| synthetic | 25% | 톤·고조파·AM/FM·협대역·chirp |
| DNS speech | 15% | 대화 — **기능 2** |
| FMA-small | 10% | 음악 — **기능 2** |
| DEMAND | 8% | 주방·세탁기·사무실·지하철 |
| MIMII fan | 7% | 저역 회전기계 |
| ESC-50 | 5% | 비정상 환경음 |

원격 인스턴스 기준 **37 GB**, manifest 7종.

> [!CAUTION]
> **코퍼스 누수 — 실측 `music` 60트랙 전부가 합성 풀에도 있다.**
> `data/source_pool/sources.csv` 의 `clips` 열과 `data/raw/` 를 직접 대조한 결과:
>
> | 계열 | 실측이 쓴 원본 클립 | 합성 풀과 교집합 |
> |---|---:|---:|
> | **music** | 60 | **60 (100%)** |
> | speech | 218 | 0 |
> | machine | 188 | 0 |
> | environment | 225 | 0 |
>
> `assign_splits(ratios={train:0.9,val:0.05}, seed=20260802)` 를 재현하면 실측 music
> 60트랙 중 **55개(92%)가 합성 train split** 에 있다.
>
> **기전**: 같은 오디오에 **상충하는 정답**이 간다. 합성 브랜치는 이상적 P/S 라 −18 dB 까지
> 상쇄 가능하고, 실측 브랜치는 정렬 붕괴로 천장이 −0.4 dB 다. 모델이 같은 음악에서 반대
> 방향 gradient 를 받는다. **`music` 만 이 조건에 있고, `music` 만 개선되지 않았다**
> (+0.09 vs 나머지 −0.85 ~ −2.05).
>
> 게이트에 **"합성 매니페스트 ∩ 실측 소스 = ∅"** 항목이 없었다(전형적 군집 B).
> 현재는 `corpus_disjoint` / `invariant_corpus_disjoint` 게이트와 active 82세션 historical
> 재현 held-out 목록(`data/manifests/recorded_holdout.json`, 682 unique clip)이 있고,
> `prepare_noise_pool.py --expected-holdout-sha256 <64hex>`가 구성 단계에서 차단한다.
> holdout는 report SHA/CSV row 합집합, manifest schema v2는 raw content SHA를 결속한다.
> **다만 실데이터 양성 확인은
> manifest 격리 때문에 미완**이다.
>
> **부수 결함 (D4)**: `music` 의 `group_id` 는 `fma_small/<3자리 디렉터리>` = **FMA 트랙 ID
> 버킷**이지 아티스트도 앨범도 아니다. 세션/그룹 비율이 music 1.00 · speech 1.00 이라
> 교차 세션 누수를 전혀 막지 못한다. 고치려면 `fma_metadata`(tracks.csv)가 필요하다
> — 다운로드 후 매니페스트만 다시 쓰면 되고 재녹음은 불필요하다.
>
> **부수 결함 (D5)**: `build_recording_sources.py` 의 `TARGET_CREST_DB = 10.0` tanh
> 소프트클리핑이 네 계열 크레스트를 **9.61~9.98 dB(폭 0.37 dB)** 로 균질화해 계열 간
> 실제 차이를 지웠다. 현장 소음 크레스트는 15–25 dB 다. 그 결과 `music` 이 측정 가능한
> 모든 축에서 가장 쉬운 신호가 됐다(제어대역 에너지 비중 0.901 최고, 1633 Hz 초과 유출
> 0.009 최저). **레벨 상향 여유는 감사가 말한 22 dB 가 아니라 13.4 dB** 다 — 클리핑하는
> 것은 소스가 아니라 REF 마이크이고 실측 12세션 최악 peak 가 0.2123 이다.
> 안전한 조합은 **크레스트 15 dB + 레벨 +3 dB**(REF peak 0.533, 여유 +5.4 dB).

### 2.3 실측 경로 자산

**현행 채택본 (2026-08-05 재발행, 캡처 `225546_f7b0fecd`)**

| 파일 | 벌크지연 | 검증 대역 | 그 대역 일관성 | 전대역 | 유지/전체 반복 | P−S spread |
|---|---:|---|---:|---:|---:|---:|
| `primary_path_il.npz` | **1602** | 150–1600 Hz | **0.9993** | **0.9988** | 18/48 | 1 |
| `secondary_path_il.npz` | **1462** | 150–1600 Hz | **0.9990** | **0.9984** | 18/48 | 1 |
| `secondary_path_4s.npz` | 1342 | 150–600 Hz | 0.40 | — | — | 순차 ESS (폐기) |

`P − S = 140`, `lead = S 1462 + handoff 256 − P 1602 = 116`, 앵커 반복 13.
두 채택본은 **한 번의 재생으로 동시 측정**했고 `capture_id` 가 일치한다.

채택본 부대역 일관성 (`band_consistency` / `band_consistency_hz`):

| 대역 Hz | 80–150 | 150–300 | 300–600 | 600–1000 | 1000–1600 |
|---|---:|---:|---:|---:|---:|
| **P** | 0.9273 | 0.9994 | 0.9995 | 0.9982 | 0.9995 |
| **S** | **0.7584** | 0.9983 | 0.9995 | 0.9981 | 0.9994 |

```bash
.venv/bin/python -c "
import numpy as np
for f in ['assets/measured/primary_path_il.npz','assets/measured/secondary_path_il.npz']:
    d=np.load(f)
    print(f, d['delay_samples'], d['consistency_band_hz'], d['excitation_band_hz'],
          np.round(d['band_consistency'],4), 'rejected', int(d['rejected_repeats']),
          'spread', int(d['delay_spread_samples']), 'anchor', int(d['anchor_repeat']))"
```

> [!CAUTION]
> **이전 판의 출하본(S 전대역 0.781 / P 0.920)은 오염된 반복 5개를 포함한 값이었다.**
> `alignment_scores` 반복 11–15 가 0.750–0.758 로 별도 무리인데 기각 임계 0.5 때문에
> `rejected_repeats: 0` 이었다. 결정적 증거는 **P−S 상대 τ** — 두 채널은 같은 DAC·같은
> 출력 스트림의 인터리브라 설계 원리상 상수여야 하는데 반복 11 에서 **1.4 → 32 샘플
> 점프**한다(출력 버퍼 프레임 슬립). 게이트는 요약 스칼라 `delay_spread_samples 32` 를
> 허용치 48 과 비교해 **통과시켰다** — 진짜 판별자는 파일 안에 있었는데 아무도 읽지 않았다.
>
> ```bash
> .venv/bin/python -c "
> import numpy as np
> p=np.load('assets/measured/primary_path_il.npz.orig')['repeat_tau_samples']
> s=np.load('assets/measured/secondary_path_il.npz.orig')['repeat_tau_samples']
> print(np.round(p-s,2))"
> ```
>
> **파이프라인 효과만 분리한 대역별 비교** (같은 캡처 `03f4c088`, 출하규약 → 신 파이프라인):
>
> | 대역 Hz | P 전 → 후 | S 전 → 후 |
> |---|---|---|
> | 80–150 | 0.868 → 0.910 | 0.748 → **0.706** ← 회복 안 됨 |
> | 150–300 | 0.996 → 0.999 | 0.964 → **0.998** |
> | 300–600 | 0.961 → 1.000 | 0.970 → **1.000** |
> | 600–1000 | 0.895 → 0.999 | 0.837 → **0.999** |
> | **1000–1600** | **0.752 → 0.999** | **0.737 → 0.999** |
> | 150–1600 | 0.921 → 0.999 | **0.782 → 0.999** |
>
> → **600 Hz 위는 덕트·스피커 물리 한계가 아니라 오염된 반복 때문이었다.**
> → **진짜 물리 한계는 80–150 Hz 뿐이다** (클린 후에도 S 0.706~0.758, 독립 캡처 간 `|H|`
> 편차 27.8% — 스피커 저역 SNR 8–10 dB).
>
> **형상 오차**(벌크지연 제거, `‖Δ‖/‖new‖`, 150–1600 Hz): **P 17.0% / S 54.1%**
> (S FIR 시간영역 54.2%). **출하 npz 로 설계한 최적 필터를 클린 플랜트에 적용하면
> −0.54 dB** 밖에 못 낸다(올바른 설계 −6.53 dB) — 결함 3·4 의 플랜트 측 원인이다.

**절대 지연은 재현되지 않는다 — `P − S = 140` 만이 물리 불변량이다.** 저장된 캡처 11건을
전수 재분석한 결과 유효 9건 전부에서 P−S = 139~141, lead = 115~117(중앙 **116**)이지만,
절대 지연은 low-latency 1565~1659 / high-latency 2858~2888 로 흩어진다(캡처별 타임베이스
드리프트 364~729 ppm + 앵커 반복 선택 의존). 그래서 P 와 S 는 반드시 **같은 캡처·같은
앵커** 값을 함께 써야 하고, 아티팩트에 `capture_id` · `anchor_repeat` ·
`kept_repeat_indices` 를 박아 두었다.

**신설 게이트 (전부 강화 방향).**

| 게이트 | 이전 | 현재 | 근거 |
|---|---|---|---|
| P−S 상대 τ 편차 | 없음 (요약 스칼라만) | **≤3.0 샘플, 궤적 전수** | 정상 최대 1.99 / 오염 최소 4.32 |
| 국소 타임베이스 드리프트 | 없음 | **≤2.0 샘플/주기** | 정상 ≤0.83 / 이상 ≥2.63 |
| 정렬 신뢰도 하한 | 0.5 | **0.95** | 유지 0.9845~0.9995 / 오염 최고 0.966 |
| 유지 반복 하한 | 3 | **8** | — |
| 부대역 일관성 | 총계 1개 | **요구 대역 안 모든 부대역** | 총계가 80–150 Hz 0.706 을 숨겼다 |
| readiness `delay_spread` | 아티팩트 신고값 | **상수 3 (신고값 무시)** | 자기증명 구조 차단 |
| 슬립 과반 | 없음 | **실패 폐쇄** | 유지 무리가 첫 분석 주기와 프레임 정렬이 다르면 판별 불가 |
| `--warmup-periods` / `--repeats` | 4 / 16 | **32 / 64** | 정상상태 워밍업과 최소 8개 유지 반복의 여유 확보. 기본 자극 12.0초 |

> **`excitation_band_hz` 는 두 경로가 다르다** — 인터리브라 두 채널이 인접 FFT 빈을 번갈아
> 쓰기 때문이다: **P(noise) [64, 1648] Hz / S(cancel) [72, 1640] Hz.**
> `consistency_band_hz`(검증 **150–1600 Hz**, P/S 동일)와는 다른 값이고, 학습 손실과 평가는
> **검증 대역**을 쓴다.

---

## 3. 아키텍처

### 3.1 모델 계보

`HybridANCNet` = Conv-TasNet 파형 encoder/decoder + WaveNet dilated causal TCN +
GCRN grouped LSTM. 원논문을 그대로 쓰지 못한 이유는 [README §2.3](../README.md#23-왜-원논문-구조를-그대로-쓰지-않는가)에 있다.

| 변형 | 파라미터 | C | 블록 | dilations | MHSA | 상태 |
|---|---:|---:|---:|---|---|---|
| `tiny` | 1.16M | 128 | 8 | 1,2,4,8 ×2 | — | **배포 후보** |
| `tiny_long` | 1.30M | 128 | 10 | 1,2,4,8,16 ×2 | — | 유의하지 않음 |
| `tiny_attn` | 1.23M | 128 | 8 | 1,2,4,8 ×2 | 4h | **실격** (do-no-harm) |
| `tiny_long_attn` | 1.37M | 128 | 10 | 1,2,4,8,16 ×2 | 4h | **실격** |
| `base` | 5.99M | 256 | **15** | 1,2,4,8,16 ×3 | 4h | 배포 탈락 |
| `tiny_wide` | **12.8M** | **512** | 8 | 1,2,4,8 ×2 | — | 지연 실측 완료 · 사전학습 착수했으나 **근거 철회**([§4.5](#45-진단--용량-부족이-아니었다-철회)) |

### 3.2 end-to-end인 부분과 아닌 부분

**end-to-end** — 모델은 `(x_ref, err) → y`를 파형에서 파형으로 직접 매핑하고, 손실이
**에러 마이크 신호 자체**다. 미분 가능한 `S(z)`가 그래프 안에 있어 gradient가 `e`에서
가중치까지 흐른다.

```python
# losses/anc_loss.py:_forward_fp32
e = d + self.plant(y_nl, perturb)
```

**단계적(piecewise)** — 다음은 학습하지 않고 **측정해서 고정**한다.

| 요소 | 왜 고정하나 |
|---|---|
| `P(z)`, `S(z)` | 함께 학습하면 틀린 플랜트를 틀린 `y`로 벌충하고 **아무도 모른다** |
| timing contract | strict P/S bulk delay·compact FIR peak·handoff에서 유도. 아래 수동 lead는 legacy 진단값 |
| 학습 단계 | Stage-1 합성 → **Stage-2 실측(진행 중)** → Stage-3 closed-loop. 각 단계 가정을 G1–G4로 검문 |
| 런타임 3스레드 | 마감 때문. 오프라인↔스트리밍 등가 `3e-8`이 분할이 수학을 바꾸지 않음을 보장 |

### 3.3 되먹임

에러 마이크는 **매 hop(2.67 ms) 모델 입력 ch1로 들어간다**.

```python
y = self.engine.step(ref, err)        # run_realtime.py
x = np.stack([ref, err])              # [1, 2, hop]
```

자기회귀(자기가 만든 `y`를 되먹임)와 다르다 — 후자는 `x_ref`로 이미 결정된 값이라 새
정보가 없고, 48 kHz에서 초당 48,000회 순차 forward가 필요해 예산을 26배 초과한다.

단, **현재 Stage-2는 `open_loop`**이라 모델이 "내 출력이 다음 `e`를 바꾼다"는 관계를 아직
학습하지 않는다. 그것은 Stage-3의 몫이다. 안전망은 `divergence_ratio 4.0 / hold 0.5 s`.

---

## 4. 결과

### 4.1 G1 — 실측 P(z)/S(z)

2026-08-04 에는 **분석 창 길이**를 원인으로 지목했다(주기 1.0 → 0.125 s 에서 일관성
0.535 → 0.955). 창 단축이 효과를 낸 것은 사실이지만, **그것이 지운 것은 연속적인 클록
warp 가 아니라 창 안에 버퍼 프레임 슬립이 들어갈 확률**이었다. 당시의 후보 배제표는
"클록 도메인 독립성" 자체를 검증 항목에 넣지 않고 전제로 깔았다는 한계가 있다([§1.2](#12-실시간-오디오-체인)).

아래는 저장된 legacy 캡처 11건을 당시 기각 규칙으로 전수 재처리한 **역사적 진단**이다.
P−S 불변량을 찾는 데는 유용했지만, 실제 submitted PCM과 q+joint-LS/cubic provenance가 없어
현재 official/training-ready P/S로 승격할 수 없다([§2.3](#23-실측-경로-자산)).

```
$ for d in results/calibration_interleaved/2026*/; do \
    .venv/bin/python scripts/data/reanalyse_paths_interleaved.py "$d" --dry-run; done
 225325_2ffe142b  kept=12/16  584ppm  P=2858 S=2717 P-S=141 lead=115
 225441_ed4db22c  kept=12/24  545ppm  P=1565 S=1425 P-S=140 lead=116
 225546_f7b0fecd  kept=18/48  542ppm  P=1602 S=1462 P-S=140 lead=116   ← 당시 진단 기준선
 225650_6950a52c  kept= 9/12  364ppm  P=1659 S=1519 P-S=140 lead=116
 225844_460f5205  kept=12/24  552ppm  P=1655 S=1515 P-S=140 lead=116
 225856_a9ab3c4c  kept=10/32  617ppm  P=1645 S=1505 P-S=140 lead=116
 225952_ca5fbb58  kept=13/24  528ppm  P=1584 S=1445 P-S=139 lead=117
 230150_af80aa1f  kept= 9/16  687ppm  P=2888 S=2748 P-S=140 lead=116
 235822_03f4c088  kept= 8/16  729ppm  P=1646 S=1506 P-S=140 lead=116
[기각] 132812 슬립 과반 / 225641 min_kept 미달 / 225700 xrun
```

**교차검증** — `P−S` **140** 샘플(기하 `(1.100−0.050)/343` = 147, 캡처 9건 139~141) ·
`lead` **116**(캡처 9건 115~117).
`d_noise` = P = **1602**(기하 예측 1612). 단 절대 지연은 캡처 간 재현되지 않는다([§2.3](#23-실측-경로-자산)).

**채택 캡처 선택 근거.** 반복 간 일관성 0.999 는 *재현성*이지 *정확도*가 아니다 —
근접장·마운트 같은 계통 오차는 반복 간 공통이라 잡히지 않는다. 그래서 **독립 캡처 간
`|H|` 일치도**를 leave-one-out 으로 따로 쟀다(low-latency 7건, 150–1600 Hz):

| 캡처 | P rel-RMS | S rel-RMS (최악 편차) |
|---|---:|---:|
| 225441 | 4.32% | 4.22% (13.9%) |
| **225546** | **1.96%** | **2.54% (13.7%)** ← 최선, 채택 |
| 225650 | 3.55% | 5.31% (19.5%) |
| 225844 | 2.33% | 2.75% (7.7%) |
| 225856 | 2.98% | 4.00% (12.8%) |
| 225952 | 3.08% | 3.33% (9.3%) |
| **235822 (`03f4c088`, 옛 출하본)** | 5.37% | **8.21% (44.8%)** ← 7건 중 최악 |

`225325` / `230150` 은 `latency: high` 라 배포 구성(`low`)과 다르고 캡처 간 편차도 8~10% 로
커서 제외했다.

### 4.2 파인튜닝 진입 게이트 — "9개 전부 PASS" 였고, **전부 무의미했다**

2026-08-04 시점의 게이트 목록:

```
config_fail_closed_flags · measured_primary_mode · recorded_mix_ratio
official_secondary_path · official_primary_path
matched_path_measurement_conditions · path_delay_and_lead
completed_init_checkpoint · recorded_dataset_qa
```

> [!CAUTION]
> **이 9개가 전부 초록불인 상태에서 다음 셋이 동시에 참이었다.**
> ① `official_secondary_path` 가 통과시킨 `S(z)` 의 **형상이 54% 틀려 있었다**
> ② `recorded_dataset_qa` 가 통과시킨 데이터셋의 **재생↔녹음 시간축이 붕괴해 있었다**
> ③ 그 위에서 학습한 모델은 신뢰대역 밖 2–8 kHz 를 **15–22 dB 증폭**하고 있었다
>
> **게이트가 초록불이라는 사실은 검증이 아니라 게이트의 시야에 대한 진술일 뿐이다.**
> 근본 원인은 [§5.0](#50-근본-원인--게이트-9개가-pass-인데-전부-무의미했다).
> 현재 게이트는 72개로 늘었고 **전부 실패 fixture 와 짝**으로 선언돼 있다
> (`src/deep_anc/ops/gate_registry.py`, 메타 테스트가 1:1 대응을 강제).
> **단, 짝이 강제하는 것은 "발동시킬 수 있는가" 뿐이고 "정상 데이터에서 발동하지
> 않는가"(위양성)는 아직 강제하지 못한다** — 실제로 새 QA 정렬 게이트가 올바르게 재정렬된
> 세션의 27~44% 를 오검출로 떨어뜨리는 것이 확인됐다([§5.4](#54-남은-발생기--다음-세션이-반드시-처리할-것)).

### 4.3 파인튜닝 전 기준선 — **아직 증폭한다**

사전학습 `tiny`를 **실제 덕트 녹음** val에 걸었다. NMSE는 낮을수록 좋고 **양수는 증폭**이다.

| 계열 | 세션/그룹 | trusted 평균 | 최악 10% |
|---|---|---:|---:|
| machine | 3 / **1** | **+1.86** | +3.91 |
| environment | 2 / 2 | +1.72 | +3.45 |
| speech | 2 / 2 | +0.51 | +4.05 |
| music | 2 / 2 | +0.49 | +2.89 |
| **전체** | 9 / 7 | **+1.23** | +3.79 |

합성 데이터에서 −18.66 dB를 내던 모델이 실측에서 +1.23 dB다. **이것이 sim-to-real
격차의 크기**이며 파인튜닝이 메우는 대상이다.

### 4.4 Stage-2 파인튜닝 (tiny, 50k step) — G4 는 FAIL

실측 P/S + recorded 70% 로 50,000 step 완주. **파인튜닝 후 값만 유효하다.**

| 계열 | 파인튜닝 후 val trusted | val 최악 10% | test trusted |
|---|---:|---:|---:|
| machine | −0.19 | — | — |
| environment | −0.27 | — | — |
| speech | −0.34 | — | — |
| **music** | **+0.58** | **+2.64** | **+0.90** |
| **전체 trusted** | **−0.07** | +1.59 | +0.31 |

| G4 조건 | 기준 | val | test | 판정 |
|---|---:|---:|---:|---|
| Trusted 평균 | < 0 dB | −0.07 | +0.31 | val PASS / test FAIL |
| Fullband 평균 | ≤ 0 dB | +0.07 | — | FAIL |
| **최악 source family** (기능 2) | < 0 dB | **+0.58** `music` | **+0.90** `music` | **FAIL** |
| 최악 family 최악 10% | < 0 dB | +2.64 | +3.36 | FAIL |

**G4 종합 FAIL. 배포 자격 없음.**
재현: `grep -n 'G4 종합' runs/finetune_tiny/eval_recorded_{val,test}/metrics.md`

> [!CAUTION]
> **파인튜닝 "전후" 비교는 무효다 — 두 수치가 서로 다른 플랜트에서 나왔다.**
> 이전 판은 여기에 `+1.23 → −0.07 = −1.30 dB 개선` 표를 실었다. **그 표를 철회한다.**
>
> | | 기준선 `results/baseline_recorded_val/metrics.md` | 사후 `runs/finetune_tiny/eval_recorded_val/metrics.md` |
> |---|---|---|
> | 물리 상태 | `secondary_surrogate_representation_pretrain` (+ `--allow-surrogate` 경고 배너) | `measured_primary_path` |
> | Digital lead | **109** | **113** |
> | S(z) 지연 | **1342** + 256 | **1465** + 256 |
>
> 평가 플랜트 자체가 바뀌었으므로 두 수를 뺄 수 없다. 유효한 전후 비교를 하려면 **같은
> 플랜트에서 사전학습 checkpoint 를 다시 평가**해야 한다(미실시). 게다가 사후 플랜트
> (S 1465 / lead 113)조차 지금은 **폐기된 오염 아티팩트**다 — 현행은 S 1462 / lead 116.
>
> **또한 val 전체 −0.07 dB 는 0 과 통계적으로 구별되지 않는다** — cluster bootstrap 95% CI
> **[−0.456, +0.481]**. "상쇄로 돌아섰다"고 말할 근거가 없다.
>
> **"music 이 최악 계열" 이라는 판정 자체도 통계적으로 성립하지 않는다 (D3).**
> 계열 내 그룹 간 잔차 SD(pooled) **1.46 dB** → 그룹 2개 계열의 평균 SE **1.03 dB** 인데,
> 파인튜닝 후 계열 간 전체 폭은 **0.92 dB** 로 **1 SE 보다 좁다.** music val 두 그룹은
> 자기들끼리 **2.96 dB** 벌어져 있고(−0.99 vs +1.97), machine val 은 그룹이 1개라 오차
> 추정 자체가 불가능하다. **music val = 2세션 × 3클립 = 곡 6개** 이며, G4 최악계열 판정
> 전체가 곡 12개 위에 서 있다.

### 4.5 진단 — 용량 부족이 **아니었다** (철회)

`results/finetune_run1/train_curve.log` (500점) 구간 평균:

| 구간 | train nmse_t |
|---|---:|
| 100–5,000 | −0.88 |
| 10,100–15,000 | −1.93 |
| 25,100–30,000 | **−2.46** |
| 35,100–40,000 | −1.50 |
| 45,100–50,000 | −2.07 |

**학습 NMSE 가 step 10,000 이후 −2 dB 에서 정체한다.** 이후 4만 step 동안 내려가지 않고,
val 은 −0.07 dB 로 train–val 격차가 2 dB 뿐이다. 이전 판은 이것을 *"데이터가 부족하면
train 이 더 내려가면서 val 만 나빠진다(과적합). 지금은 train 자체가 막혔으니 미적합 =
용량 부족"* 으로 읽고 `tiny_wide` 사전학습(GPU 15h)을 지시했다.

> [!CAUTION]
> **그 진단은 틀렸다. 철회한다.** train 자체가 막힌 것은 맞지만, 원인은 용량이 아니라
> **① 학습 데이터에 배울 관계가 없다는 것** 과 **② 플랜트가 33~54% 틀렸다는 것** 이다.
>
> **① 데이터** ([§2.1](#21-실측-덕트-녹음-파인튜닝용-70)): coh²(`source.wav` → ERR) =
> **0.021~0.126** (같은 세션에서 coh²(REF→ERR)는 0.959~0.991). 입력 `x_ref` 와 타깃 `d` 의
> 대응이 깨진 데이터에서 회귀 손실의 하한은 데이터가 정한다 — 이 데이터의 **선형 Wiener
> 하한이 −0.23 dB** 이고, 관측된 val −0.07 dB 가 그 하한과 정확히 일치한다.
> 재정렬 후 같은 하한은 **−12.09 dB** 로 올라간다. **모델을 키워도 −2 dB 는 내려가지 않는다.**
>
> **② 플랜트** ([§2.3](#23-실측-경로-자산)): 학습에 쓴 `S(z)` 의 형상 오차가 **54.1%** 였다.
> 출하 npz 로 설계한 최적 필터를 클린 플랜트에 적용하면 **−0.54 dB** 밖에 못 낸다
> (올바른 설계는 −6.53 dB).
>
> **반증도 직접 계산했다.** 복구된 플랜트에서 FIR 길이를 늘려도 이론 상한이 거의 움직이지
> 않는다 — 512 / 1024 / 2048 / 4096 / 8192 탭 → **−3.87 / −4.08 / −4.15 / −4.16 / −4.16 dB**
> (150–1600 Hz, PSD 가중). **용량은 병목이 아니다.**
>
> → **따라서 `tiny_wide` 용량 실험은 근거를 잃었다.** 정렬을 고친 재녹음 이전에는 어떤
> 모델 크기 비교도 해석할 수 없다.

### 4.6 추론 지연 — 30W vs MAXN

RT 우선순위 `chrt -f 80`, 코어 4–7 고정, warmup 500, 3000–5000 스텝.

> [!IMPORTANT]
> **모든 조합의 max 가 40–58 ms 인 것은 커널 RT 스로틀링 때문이다** (데스크톱 잡음이 아니다).
> `kernel.sched_rt_runtime_us = 950000` / `sched_rt_period_us = 1000000` →
> SCHED_FIFO 태스크는 1초 주기마다 정확히 **50 ms** 실행을 거부당한다.
> 확인: `cat /proc/sys/kernel/sched_rt_runtime_us /proc/sys/kernel/sched_rt_period_us`.
> **벤치 명령이 쓰는 `chrt -f 80` 이 곧 원인이며, `chrt` 를 떼면 사라진다.**
> 엔진 비교에는 쓰지 않지만, **런타임도 RT 우선순위로 돌므로 같은 50 ms 정지를 겪는다**
> (hop 예산 5.33 ms 의 9배). 배포 전에 반드시 다뤄야 한다.

> [!WARNING]
> **아래 GPU 수치는 듀티 100%(연속 실행) 값이다 — 실제 ANC 조건이 아니다.**
> `scripts/bench/measure_inference_latency.py` 는 warmup 뒤 `for i in range(steps)` 로
> 쉬지 않고 추론을 돌린다(sleep 없음). 이 부하가 GPU 를 1300 MHz 에 붙잡아 둔다.
> 실제 런타임은 hop 256 = **5.33 ms 주기에 추론 ~0.3 ms → 듀티 약 6%** 라 거버너가
> **306 MHz 로 고정**하고, 그 조건에서 `tiny` TRT P50 이 **0.30 → 1.10 ms 로 3.7배**
> 나빠진다. 아래 표를 배포 판단에 쓰기 전에 **주기적 호출 조건에서 다시 재야 한다**
> (`--period-ms` 옵션 추가가 별건 과제이며 아직 저장소에 없다).
> CPU(ORT) 수치는 이 영향을 덜 받는다.

| 모델 | 블록 | 엔진 | 30W P50 | **MAXN P50** | 30W P99 | **MAXN P99** |
|---|---:|---|---:|---:|---:|---:|
| `tiny` 1.16M | 8 | ORT CPU | 1.38 | **1.24** | 1.54 | **1.44** |
| `tiny` | 8 | TRT GPU | 0.56 | **0.29** | 3.27 | 3.12 |
| `tiny_wide` 12.8M | 8 | TRT GPU | 1.13 | **0.61** | 4.63 | **3.54** |
| `tiny_wide` | 8 | ORT CPU | 9.51 | 8.26 | 10.85 | 8.82 |
| `base` 5.99M | **15** | TRT GPU | 1.33 | **0.72** | 6.00 | **3.64** |
| `base` | 15 | ORT CPU | 6.00 | 6.01 | 6.40 | 6.40 |

**세 가지가 읽힌다.**

1. **MAXN 이 GPU P50 을 절반으로 줄인다** (tiny 0.56→0.29, wide 1.13→0.61, base 1.33→0.72).
   CPU 는 클록 상한이 27% 오른 만큼(~10%) 개선된다.
2. **폭은 GPU 에 거의 공짜, 깊이는 비싸다.** `tiny_wide` 는 `tiny` 보다 11배 큰데 GPU 에서
   2배만 느리고, 파라미터가 2.1배 많은 `base`(블록 15)보다도 **빠르다**. 비용을 내는 것은
   순차 커널 런치이지 연산량이 아니다.
3. **"12.8M 모델은 GPU 에서만 실시간이 된다" 는 아직 검증되지 않았다.** 연속 실행 P99
   3.54 ms 는 예산 5.33 ms 안이지만, 이는 GPU 가 1300 MHz 에 붙잡힌 조건의 값이다.
   듀티 6% 에서 P50 이 3.7배 나빠지는 것이 확인됐으므로 P99 도 예산을 넘길 수 있다.
   **주기 호출 벤치(5.33 ms 간격, sleep 포함) 전에는 이 주장을 하지 않는다.**
   CPU 는 중앙값 8.26 ms 로 확실히 예산을 넘는다.

TrtEngine 최적화 내역(30W 기준 P50 1.32 → 0.56 ms):

| 결함 | 고친 방식 |
|---|---|
| CUDA Graph 미사용 | parity 별 그래프 캡처 (H2D→추론→D2H 한 번에) |
| pageable 호스트 메모리 | `cudaHostAlloc` 고정 버퍼 |
| 매 스텝 `set_tensor_address` 26회 | A/B 두 경우를 그래프에 굳힘 |
| 동기 대기 정책 | `cudaDeviceScheduleSpin` |

그래프 경로는 비그래프 경로와 **비트 단위로 동일**하다(최대오차 `0.000e+00`).

### 4.7 실기 ANC (사전학습 tiny + ORT CPU)

| 시나리오 | 감쇠 |
|---|---:|
| tone 300 Hz | **+6.26 dB** |
| band (trusted) | **+5.14 dB** |
| 음성 + 소음 (80–800 Hz) | **+4.39 dB** |
| 1150–1250 Hz | +0.46 dB — 상쇄도 증폭도 안 함 |

> [!CAUTION]
> **대역 밖은 크게 증폭한다 — 절대 목표 1 정면 위반이다.**
> 위 4줄은 trusted 대역 발췌였고, 같은 세션 `results/session_20260804_0939/metrics.csv` 에
> 기록된 옥타브밴드는 다음과 같다 (**음수 = 증폭**):
>
> | 시나리오 | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | tone300 | −3.56 | +6.33 | +4.96 | **−16.84** | **−15.42** | **−18.03** | **−21.56** |
> | multitone | +1.13 | +6.13 | +1.22 | −1.46 | **−16.88** | **−17.36** | **−17.96** |
> | band | +6.97 | +5.61 | +4.55 | −0.13 | −1.87 | −8.90 | −14.90 |
> | nonlinear | +5.81 | +5.13 | +4.85 | −3.41 | −4.02 | −8.61 | −12.43 |
> | hf_band | +0.49 | +0.14 | −0.95 | +0.19 | +0.10 | −5.03 | −6.22 |
> | hf_tone | +1.15 | −2.24 | −1.73 | +0.41 | +0.74 | +1.22 | −1.19 |
> | voice_in_noise¹ | +5.08 | +5.86 | +4.93 | −1.50 | −1.40 | −1.30 | −1.05 |
>
> ¹ `results/session_20260804_125538/metrics.csv` — README §1.4 그림의 세션.
>
> **손실에는 대역 밖 do-no-harm 항이 없었고 게이트에만 있었다.** voice_in_noise 만 −1 dB 대라
> §1.4 그림 하나로는 이 문제가 보이지 않는다 — **6개 중 유일하게 무해한 시나리오를 대표로
> 실었던 셈**이다. 원인은 용량이 아니라 손실 설계이며, 파인튜닝으로 해결되지 않는다.
>
> ```bash
> .venv/bin/python -c "
> import csv
> for r in csv.DictReader(open('results/session_20260804_0939/metrics.csv')):
>     print(r['scenario'], [round(float(r[f'band_{b}_att_db']),2) for b in (1000,2000,4000,8000)])"
> ```

### 4.8 검증

| 항목 | 값 |
|---|---|
| 자동 테스트 | **604개 통과** (`.venv/bin/python -m pytest -q`, 약 14분) |
| 오프라인 ↔ 스트리밍 | 최대 오차 `3e-8` |
| PyTorch ↔ ONNX Runtime | `8e-8` |
| TRT 그래프 ↔ 비그래프 | `0.000e+00` |

---

## 5. 개선방안

### 5.0 근본 원인 — 게이트 9개가 PASS 인데 전부 무의미했다

증상을 하나씩 고치면 다시 나온다. 커밋 이력 + 이번 발견 **18건**을 분류했다.

| 군집 | 건수 | **공통 발생기** |
|---|---:|---|
| **A. 두 도메인 간 시간·대역 부기** | 9 | 같은 물리량(지연 / lead / 대역 / 임계)을 **여러 곳에서 따로 유도**하고 대조하지 않는다 |
| **B. 실패해본 적 없는 게이트** | 5 | 게이트가 "통과"를 주장하는데 그 주장이 **반증된 적이 없다** |
| C. 측정 없는 성급한 결론 | 4 | TensorRT 기각 / 용량 부족 / 600 Hz 물리 한계 / 클록 드리프트 — **전부 정정됨** |

**A + B = 14/18 (78%).** 실측: 지연 산술을 독립 수행하는 파일이 **13개**
(`eval/recorded.py` 35회, `train/finetune_readiness.py` 31회,
`bench/measure_duct_transfer_map.py` 20회, `train/trainer.py` 17회 …).

**이번 사고에서 두 발생기가 정확히 어떻게 작동했는가.**

| 결함 | 발생기 | 구체적 기전 |
|---|---|---|
| `S(z)` 54% 오차 | **B** | 진짜 판별자(**P−S 상대 τ 궤적의 상수성**)가 파일 안에 이미 있었는데 게이트가 **요약 스칼라** `delay_spread_samples 32` 만 허용치 48 과 비교했다. 궤적을 읽는 코드가 없었다 |
| recorded 시간축 붕괴 | **A** | `record_duct.py` 가 출력 커서와 입력 커서를 **같은 물리 시각이라고 단언**하고 측정하지 않았다. QA 는 RMS·클리핑만 봤다 |
| 대역 밖 증폭 | **B** | do-no-harm 이 **게이트에만 있고 손실에 없었다.** 게이트가 안 보는 방향으로 학습이 최적화됐다 |
| 코퍼스 누수 | **B** | "합성 매니페스트 ∩ 실측 소스 = ∅" 게이트가 아예 없었다 |
| lead 109 / 113 / 116 혼재 | **A** | 같은 값이 `duct.yaml` · 문서 · checkpoint 메타 · 런타임 config 에 따로 적혀 있었다 |

**대응 (2026-08-06 커밋 `612152c`).**

1. ✅ **지연·대역 부기의 단일 출처.** `dsp/timing.py` 신설 — `PlantDelays.lead()` 와
   `BandPlan.resolve(...)` 로만 만들 수 있다(pydantic frozen, 손으로 생성 시 `TypeError`).
   `intersect_frequency_bands` 도 여기 한 곳에만 있다.
2. ✅ **교차 도메인 불변식 검사기.** `dsp/invariants.py` 신설 — P−S 상대 τ 상수성 /
   `coh²(재생→마이크)` / 플랜트 지문(`PlantFingerprint`) 일치 / lead 유도값 일치를
   측정·QA·게이트·런타임이 **같은 코드**로 호출한다.
3. ⚠ **실패 증명 없는 게이트 금지 — 절반만 됐다.** 게이트 열거 + FAIL fixture 메타 테스트는
   `src/deep_anc/ops/gate_registry.py` 에 있고 현재 **72개**가 등록돼 있다.
   **그러나 "정상 데이터에서 발동하지 않는가"(위양성)는 강제하지 못한다** — §5.4 참조.

### 5.1 지금 막고 있는 것 — 영향 순 (2026-08-06 갱신)

| 순위 | 병목 | 근거 | 조치 | 자원 |
|---|---|---|---|---|
| **1** | **recorded 시간축 붕괴** — 47/80 은 오프라인 복구, **33세션은 재녹음 필요** | §2.1 — coh²(source→ERR) 0.078 → 0.947 (복구분) | 실패 33세션 재녹음 | 스피커 **~38.5분** |
| **2** | **대역 밖 2–8 kHz 를 15–22 dB 증폭** (절대 목표 1 위반) | §4.7 | 손실 do-no-harm 항 **λ 교정 + 게이트 임계와 대조** | 분석 + 재학습 |
| **3** | **코퍼스 누수** — 실측 `music` 60/60 이 합성 풀에 존재 | §2.2 | held-out 게이트 배선 완료 (매니페스트 작업, 재녹음 불필요) | 분석 |
| 4 | **RT 스로틀링** — `chrt` 사용 시 1초마다 50 ms 정지 | §4.6 | 실시간 예산 5.33 ms 의 9배. 배포 전 필수 | 코드 |
| 5 | ~~`S(z)` 600 Hz 위 재현 불가~~ → **오염 반복을 게이트가 통과시킴** | §2.3 | **해결됨** — 기각 임계 강화 + 클린 18회로 재발행 | 완료 |
| 5b | **80–150 Hz 재현 한계** (클린 후에도 S 0.706~0.758) | §2.3 | 소음/상쇄 스피커 저역 개선 — **진짜 하드웨어 항목** | 하드웨어 |
| 6 | I/O 왕복 30.3 ms = 루프의 **85%** | §1.3 | 블록 256 → 128 | 재측정·재학습 |
| 7 | `machine` 그룹 8개, val/test 각 1그룹 — G4 해상도 미달 | §2.1, §4.4 D3 | 녹음 보강 (ESC-50 미사용 26종 + MIMII/DEMAND 활용) | 스피커 |
| 8 | 추론 지연 | §1.3 — 루프의 4.0% | (MAXN 전환 완료) | — |

> 이전 판은 여기에 "1. 학습 NMSE −2 dB 정체 → `tiny_wide` 사전학습 (GPU 15h)" 과
> "3. S(z) 600 Hz 위 → 상쇄 스피커 개선 (하드웨어)" 를 실었다. **둘 다 틀렸다** — 전자는
> 데이터·플랜트 결함의 결과였고(§4.5), 후자는 하드웨어가 아니라 측정 후처리 결함이었다(§2.3).
> 또 "이전 1순위였던 '실측에서 증폭'은 해결됐다(+1.23 → −0.07 dB)" 고 적었는데,
> **해결됐다고 말할 수 없다** — 사후 val −0.07 dB 는 0 과 통계적으로 구별되지 않고
> (CI [−0.456, +0.481]), 기준선 +1.23 은 다른 플랜트에서 나온 값이라 뺄 수 없다(§4.4).

### 5.2 용량 가설 — **철회한다**

이전 판은 이 절에서 *"이전까지 '용량이 병목이라는 증거가 없다' 였는데 **파인튜닝이 증거를
만들었다**"* 고 썼다. **그 판단을 철회한다.**

−2 dB 정체는 결함 2(recorded 시간축 붕괴)와 결함 1(S(z) 54% 오차)의 결과이며, 용량 가설을
지지하는 증거가 아니다. 용량이 병목인지는 **정렬을 고친 재녹음 데이터로만** 다시 물을 수 있다.

**용량이 병목이 아니라는 직접 증거 (복구된 플랜트에서 계산):**

| 근거 | 값 |
|---|---|
| FIR 길이 512 → 8192 탭 (150–1600 Hz, PSD 가중) | −3.87 → **−4.16 dB** — 4배 늘려도 0.3 dB |
| 이 데이터의 선형 Wiener 하한 (정렬 붕괴 상태) | **−0.23 dB** ≈ 관측 val −0.07 dB |
| 같은 하한 (재정렬 후) | **−12.09 dB** |
| 출하 npz 로 설계한 필터를 클린 플랜트에 적용 | **−0.54 dB** (올바른 설계 −6.53 dB) |

**과거에 큰 모델이 진 기록도 그대로 유효하다.**

| 비교 | 결과 |
|---|---|
| `base` 5.99M vs `tiny` 1.16M | 7종 소스 **전부** tiny 우세 |
| 최악 아이템 fullband | base **+13.89 dB 증폭** (do-no-harm 위반) |
| 구조 탐색 20k (4후보) | 어느 후보도 `tiny` 를 못 이김 |

`configs/train_pretrain_tiny_wide.yaml` 은 남아 있으나 **그 파일이 근거로 삼은 진단은
철회됐다** — 파일 헤더에 경고가 있다. 계속 돌릴지 중단할지는 재녹음 이후에 판단한다.

> **대가(참고).** `tiny_wide` 는 MAXN GPU 연속 실행에서 P99 3.54 ms(예산의 66%)지만,
> 이 값은 **듀티 100%** 조건이다. 실제 듀티 6% 에서는 거버너가 306 MHz 로 내려 더
> 나빠진다(§4.6).

### 5.3 순서

1. **실패 33세션 재녹음** — `record_duct.py` 의 시간축 수정본으로 다시 딴다. 저장 시점
   게이트가 `coh²(source→ERR) ≥ 0.6` (150–600 Hz) 를 강제하므로 나쁜 세션은 그 자리에서
   버려진다. **다른 모든 항목의 선결 조건이다** (§2.1).
   ⚠ 재생 전 **마이크 입력단 점검**과 **오디오 장치 점유 확인** 필수(AGENTS.md §2) —
   현재 마이크 두 채널이 풀스케일에 붙어 있어 그대로 재생하면 전량 폐기된다.
2. **손실 do-no-harm 항 완성** — 항 자체는 들어갔지만 λ 가 새 대역 구성에서 재교정되지
   않았고(실측 그래디언트 비 1333%, 목표 20~40%), **힌지 마진과 G4 임계가 서로를 모른다**
   (§5.4). 현재 2–8 kHz 를 15–22 dB 증폭한다(§4.7). 절대 목표 1 위반이며 파인튜닝으로
   해결되지 않는다.
3. **코퍼스 누수 해소** — held-out 목록은 만들어졌다(`data/manifests/recorded_holdout.json`,
   691 클립). 게이트 실데이터 양성 확인이 남았다. 재녹음 불필요.
4. **RT 스로틀링 대응** — `chrt` 사용 시 1초마다 50 ms 정지(§4.6). 실시간 예산 5.33 ms 의
   9배라 배포 전 반드시 해결해야 한다.
5. **발생기 제거** (§5.0) — 지연·대역 부기 단일 출처, 게이트 위양성 fixture.
   이걸 안 하면 같은 종류의 결함이 또 나온다.
6. 1–3 이 끝난 뒤에야 **`music` 악화 원인 규명**과 **용량 실험**을 다시 묻는다.
   ⚠ 현재 "music 이 최악" 판정은 통계적으로 성립하지 않는다(§4.4 D3).
7. 상쇄 스피커 개선 검토 — 단, 대상은 600 Hz 위가 아니라 **80–150 Hz** 다(§2.3).
8. 블록 256 → 128 로 I/O 왕복 절반 시도 (Stage-3 closed-loop 대역폭에 직접 영향).

> ✅ **2026-08-06 해소** (커밋 83c6954 · ff5de1b). 힌지 마진은 이제 G4 임계에서
> **유도된다** — `src/deep_anc/dsp/do_no_harm.py` 가 단일 출처이고
> `margin = 20·log10(10^(G/20) − 1) = −18.27 dB` 다. 설정에 값을 되쓰면 `LossConfig` 가
> 거부한다. 대역도 옥타브 경계에 정렬시켰다(가로지르면 한 옥타브에 에너지를 몰 수 있었다).
> λ_dnh 는 0.12 → **0.001** 로 재교정했다(실측 예산비 41.8 → 0.348, 목표 0.2~0.4).
> 측정은 `ANCLoss.gradient_budget` 하나이고 `tests/test_loss_gradient_budget.py` 가
> 예산을 걸어 둔다. 남은 미검증 2건은 그 파일 docstring 참조.


### 5.4 남은 발생기 — 다음 세션이 반드시 처리할 것

전부 **직접 실행으로 확인**했고, 지금 저장소에 살아 있다.

**✅ 2026-08-06 커밋 `612152c` 에서 해소된 것** (재확인 완료):

| 발생기 | 어떻게 고쳤나 |
|---|---|
| 신뢰대역 유도식이 5곳에 복붙 | `dsp/timing.py` 의 **`BandPlan.resolve(...)`** 단일 출처로 통합. 소비처 5곳이 전부 이것을 호출한다 (`trainer.py:236`, `eval/recorded.py:226`, `evaluate_offline.py:98`, `evaluate_session.py:176`, `render_anc_demo.py:158`) |
| `intersect_frequency_bands` 두 번 정의 | **`dsp/timing.py:147` 한 곳**만 남았다 |
| `configs/eval*.yaml` 의 죽은 `trusted_band_hz` | **삭제됨** (세 파일 모두 삭제 사유 주석만 남음) |
| `measured_design_ceiling_db: 6.53` | **`4.58` + `measured_design_ceiling_band_hz: [150, 1600]`** 으로 정정. 대역이 `required_path_band_hz` 를 덮는지 게이트가 검사한다. 이전 값은 요구 대역보다 **2 dB 낙관적인 fail-open** 이었고 오판정 방향이 정확히 고역 방치였다 |
| lead 가 trainer 와 게이트에서 갈라짐 (109 vs 113) | **`PlantDelays.lead()`** 로만 만들 수 있다. 손으로 쓰면 `TypeError` |
| 서로 다른 플랜트끼리 비교 | **`PlantFingerprint`** 가 막는다 |

**⚠ 아직 살아 있는 것** (전부 직접 실행으로 확인):

| # | 발생기 | 증거 | 왜 위험한가 |
|---|---|---|---|
| 1 | source→ERR 지연 궤적이 **두 벌** | `data/timeline.py:455 estimate_lag_track` (대역제한 GCC-PHAT + 품질선별 + robust) vs `dsp/invariants.py:330 measure_stream_delay_trajectory` (광대역 argmax + 원시 std) — 같은 세션에서 std **1.8 vs 1107** | **정반대 판정을 낸다.** realign-PASS 세션의 27~44% 를 QA 가 오검출로 기각 |
| 2 | do-no-harm 힌지 마진(`dnh_margin_db: 6.0`)과 G4 임계(`MAX_OUT_OF_BAND_AMPLIFICATION_DB = 1.0`)가 **서로를 모른다** | `losses/config.py:239` vs `eval/recorded.py:789`. 마진을 정확히 만족하는 모델이 게이트를 옥타브 전 대역에서 8~9 dB 차이로 FAIL (직접 실행 확인) | 힌지는 `\|S·y\|²/\|d\|²`, 게이트는 `e/d` — 물리량이 다른데 대조 코드도 테스트도 없다 |
| 3 | 게이트 메타 테스트가 **위양성을 강제하지 않는다** | 모든 게이트에 "발동시키는 fixture" 는 있지만 "정상 데이터에서 발동하지 않음" 짝이 없다 | 워치독의 반응이 전부 mute(= 상쇄 0 dB = 최악값)라 오발동이 곧 성능 0 이다 |
| 4 | `build_engine` 이 handoff 를 `duct` cfg 에서 **다시 읽는다** | `realtime/engines.py:388` `.get("handoff_extra_samples", hop)` | handoff 의 두 번째 유도 (지금은 hop 과 같은지 검사하므로 갈라지지는 않는다) |
| 5 | **λ_dnh 가 새 대역 구성에서 재교정되지 않았다** | 실측 그래디언트 비 **1333%** (목표 20~40%), sat 162% / frame 46% / mrstft 27% | 손실 항끼리 예산이 맞지 않는다 |
| 6 | **출하 `nmse_cvar_alpha: 0.7` 은 배분을 뒤집지 않는다** | 최악 4개 몫 0.17% → 1.7%. 최상 4개가 여전히 **19배** | 배분을 완전히 뒤집는 것은 `alpha=1.0` 뿐인데 어떤 출하 config 도 1.0 을 쓰지 않고, **출하 설정의 최악값 거동을 강제하는 테스트가 없다** |

**런타임 안전 — 새 워치독의 오발동 위험 2건 (실측).**

- **OUTPUT_DC 워치독이 영평균 저역 상쇄음을 DC 로 오인한다.** 이동평균 창이 10블록
  (53 ms)이라 80 Hz 를 DC 와 구분하지 못한다: 60 Hz 진폭 0.12 → 10블록에 mute,
  80 Hz 진폭 0.18 → 12블록에 mute. 80 Hz 는 `realistic_target_band_hz` **안**이고,
  실측 세션의 53 ms 이동평균 최대치가 이미 한계의 **57.5%** 다(여유 1.74배).
  **저역 상쇄를 키우는 것이 절대 목표 1 인데 키울수록 이 워치독에 걸린다.**
- **fail-closed 발산 워치독 + 베이스라인 수집 조건 완화가 결합하면, 조용한 방에서 잡은
  마이크 플로어가 유효 베이스라인으로 굳고 외부 소음원을 켜는 순간 123 ms 만에 mute 된다**
  (종단 재현 확인). 이 경로에 테스트가 하나도 없다 — 기존 테스트는 `baseline_power` 를
  전부 손으로 주입한다.

## 부록 — 재현 명령

이 문서의 수치는 **전부 아래 명령으로 저장소 안에서 재현된다.** 재현되지 않는 것은
"미검증"으로 표시했다.

```bash
# --- 하드웨어 (소리 없음) ---
nvpmodel -q ; lscpu ; free -h ; cat /proc/asound/cards
cat /proc/sys/kernel/sched_rt_runtime_us /proc/sys/kernel/sched_rt_period_us   # 950000 / 1000000

# --- 플랜트 아티팩트 §2.3 (소리 없음) ---
.venv/bin/python -c "
import numpy as np
for f in ['assets/measured/primary_path_il.npz','assets/measured/secondary_path_il.npz']:
    d=np.load(f)
    print(f, d['delay_samples'], d['consistency_band_hz'], d['excitation_band_hz'],
          np.round(d['band_consistency'],4), 'rejected', int(d['rejected_repeats']),
          'spread', int(d['delay_spread_samples']), 'anchor', int(d['anchor_repeat']))"
# lead = S + 256 - P = 116

# 오염 반복의 P-S 상대 τ 점프 (.orig 백업본)
.venv/bin/python -c "
import numpy as np
p=np.load('assets/measured/primary_path_il.npz.orig')['repeat_tau_samples']
s=np.load('assets/measured/secondary_path_il.npz.orig')['repeat_tau_samples']
print(np.round(p-s,2))"                      # 반복 11 에서 1.4 -> 32 점프

# 저장된 캡처 재분석 §4.1 (소리 없음 — 스피커를 열지 않는다)
.venv/bin/python scripts/data/reanalyse_paths_interleaved.py \
  results/calibration_interleaved/20260804_225546_f7b0fecd --dry-run

# --- 고역 증폭 §4.7 (소리 없음) ---
.venv/bin/python -c "
import csv
for r in csv.DictReader(open('results/session_20260804_0939/metrics.csv')):
    print(r['scenario'], [round(float(r[f'band_{b}_att_db']),2)
                          for b in (125,250,500,1000,2000,4000,8000)])"

# --- 클록 드리프트 근거 무효 §1.2 ---
.venv/bin/python -c "
import json; d=json.load(open('results/clock_drift/20260804_222644/clock_drift.json'))
print('ppm', d['drift_ppm'], 'residual_rms', d['residual_rms_samples']); print(d['verdict'])"

# --- G4 FAIL §4.4 ---
grep -n 'G4 종합' runs/finetune_tiny/eval_recorded_{val,test}/metrics.md
# 전후 비교가 무효인 근거 (물리상태 / lead / S 지연 세 줄을 나란히 본다)
head -16 results/baseline_recorded_val/metrics.md
head -16 runs/finetune_tiny/eval_recorded_val/metrics.md

# --- recorded 시간축 붕괴 §2.1 (소리 없음) ---
.venv/bin/python scripts/data/validate_recorded_sessions.py       # 격리 후에는 manifest 부재로 실패-폐쇄
.venv/bin/python scripts/data/realign_recorded_sessions.py --self-test
.venv/bin/python scripts/data/realign_recorded_sessions.py --root data/recorded_broken --limit 24 --dry-run

# --- 파인튜닝 진입 감사 ---
.venv/bin/python scripts/train/check_finetune.py --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured

# --- 지연 §4.6 (RT 우선순위 + 코어 고정 + 충분한 warmup) ---
taskset -c 4-7 chrt -f 80 .venv/bin/python scripts/bench/measure_inference_latency.py \
  --config configs/runtime_tiny.yaml --warmup 500 --steps 5000
# 주의: chrt -f 80 을 쓰면 RT 스로틀링으로 1초마다 50 ms 스파이크가 섞인다.
#       꼬리(max) 를 보려면 chrt 없이 한 번 더 잴 것 — 46~54 ms 스파이크가 사라진다.
#       그리고 이 벤치는 듀티 100% 다. 실제 ANC 듀티(6%)의 값이 아니다.

# --- 테스트 ---
.venv/bin/python -m pytest -q ; echo "exit=${PIPESTATUS[0]}"       # 600 passed
.venv/bin/python -m pytest --collect-only -p no:randomly 2>&1 | grep collected
```

**미검증 항목** (이 문서가 전제로 쓰되 이번 세션에서 재측정하지 않은 것):

| 항목 | 값 | 왜 미검증인가 |
|---|---|---|
| 상대 클록 드리프트 | raw A 약 +620 ppm (주기당 +3.72/6000 sample) | 보존 raw의 adjacent-cycle 재분석. 다음 물리 측정에서도 ERR/REF 공통 q gate로 재확인 필요 |
| GPU 듀티 6% 열화 | P50 0.30 → 1.10 ms @306 MHz | 주기 호출 벤치가 저장소에 없다 (`--period-ms` 미구현) |
| cluster bootstrap CI | [−0.456, +0.481] | 평가 전문가 계산. 산출물로 저장되지 않았다 |
| 150–600 Hz 이론 상한 6.53 dB | — | 게이트 값은 4.58(150–1600Hz)로 정정됐으나, 150–600Hz 값 자체는 독립 재계산이 5.21~5.41 로 불일치 |
| 저역 회복 하한 120~130 Hz | — | 저역 전용 2차 프로브 미구현·미측정 |
| ERR 마이크 단면 위치 | — | `duct.yaml positions_m.error_mic_cross_section_m` 미확정. 벽면 장착으로 판명되면 신뢰 상한을 1200 Hz 로 낮춰야 한다 |
