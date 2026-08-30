<div align="center">

# Deep ANC

**Causal deep learning for active noise control in a duct**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?logo=pytorch&logoColor=white)](requirements-train.txt)
[![Jetson](https://img.shields.io/badge/Jetson-AGX_Orin-76B900?logo=nvidia&logoColor=white)](docs/06_deployment_jetson.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## Abstract

능동소음제어(ANC)는 소음과 반대 위상의 파형을 내보내 조용한 영역을 만든다. 고전적인
FxLMS 는 적응 FIR 필터 하나로 이 문제를 풀지만, 광대역·비정상 신호에서는 수렴이 느리고
비선형 왜곡을 다루지 못한다.

이 저장소는 1.2 m 아크릴 덕트 안에 quiet zone 을 만드는 **인과(causal) 신경망 ANC** 를
구현한다. 모델은 레퍼런스와 에러 마이크를 입력받아 상쇄 파형을 **직접 예측**하고, 학습
중에는 실측된 2차경로 `S(z)` 를 미분 가능한 플랜트로 통과시켜 에러 마이크의 잔여 신호를
최소화한다. 추론 시 `S(z)` 는 쓰이지 않으므로 런타임에 적응 필터가 없다.

설계는 두 가지 절대 목표로 소급 판단한다: **(1) 저주파와 고주파를 모두 제거**하고,
**(2) 소음뿐 아니라 음성·음악까지 제거**하되 평균이 아니라 **최악값**으로 판정한다.
이 두 목표는 손실 함수와 진입 게이트 양쪽에 구조적으로 새겨져 있다.

## Features

- **End-to-end 파형 예측** — 스펙트럼 마스크나 런타임 적응 필터 없이 시간영역 상쇄 파형을 직접 낸다
- **미분 가능한 2차경로** — 실측 `S(z)` FIR 을 학습 그래프에 넣어 `e = d + S·y` 를 직접 최소화한다
- **최악값 목적함수** — dB 산술평균 대신 CVaR 집계를 써서 최악 아이템에 그래디언트를 배분한다
- **대역 밖 악화 금지** — 신뢰대역 여집합에 단측 힌지를 걸고, 그 마진을 평가 게이트 임계에서 유도한다
- **스트리밍 등가성** — 오프라인 `forward` 와 블록 단위 `streaming_step` 이 수치적으로 같다
- **ONNX / TensorRT 배포** — Jetson AGX Orin 에서 256 샘플 블록(5.33 ms 마감) 실시간 추론
- **실측 기반 검증** — 합성 시뮬레이션이 아니라 실제 덕트 녹음 82 세션으로 평가한다
- **진입 게이트 15 종** — 물리·데이터·체크포인트가 정합하지 않으면 학습이 시작되지 않는다

## Requirements

- Python 3.10+
- PyTorch 2.5 (학습 `2.5.1+cu121` / Jetson 추론은 NVIDIA JetPack 6 wheel)
- NumPy 1.26.4, SciPy, soundfile, PyYAML, tqdm
- ONNX 1.15+, ONNX Runtime **1.18.1 고정** (1.19+ 는 Tegra 에서 크래시)
- sounddevice (실기 측정·녹음)
- pytest

```bash
pip install -r requirements-train.txt     # 학습 (CUDA)
pip install -r requirements-jetson.txt    # Jetson 추론
```

Jetson 은 `torch` 를 PyPI 가 아닌 NVIDIA wheel 로 설치해야 한다. 절차는
[`scripts/jetson/setup_jetson.sh`](scripts/jetson/setup_jetson.sh) 참조.

---

## 1. Introduction

### 1.1 문제 정의

덕트 ANC 는 인과성 문제다. 소음이 레퍼런스 마이크를 지나 에러 마이크에 닿기까지의 시간
안에, 상쇄 스피커가 반대 파형을 만들어 같은 지점에 도착시켜야 한다. 이 여유가 음수면
어떤 알고리즘도 상쇄할 수 없다.

덕트 배치 (단위 m):

```
0.00  소음 스피커        ─┐
0.10  레퍼런스 마이크     │  1.2 m 아크릴 사각 덕트
1.05  상쇄 스피커         │  평면파 컷오프 1633 Hz
1.10  에러 마이크         │  축방향 공진 70 / 210 / 350 / 489 / 629 Hz
1.20  개구부             ─┘
```

### 1.2 두 가지 절대 목표

| 목표 | 통과 기준 | 측정 축 |
|---|---|---|
| **G1 — 저주파와 고주파를 모두 제거** | 한쪽 대역만 좋으면 실패 | 옥타브밴드별 감쇠, 최악 10 % 구간 |
| **G2 — 모든 소리를 제거** | 소음뿐 아니라 음성·음악도 감쇠 | 소스 종류별 감쇠의 **최악값** |

G2 가 평균이 아니라 최악값인 이유: 여섯 소스 중 다섯이 −20 dB 이고 하나가 0 dB 이면
평균은 좋아 보이지만, 그 하나가 들리는 순간 quiet zone 은 실패한 것이다.

### 1.3 기여

1. 실측 1차·2차 경로를 미분 가능한 플랜트로 학습 그래프에 넣고, 그 위에서 최악값 집계와
   대역 밖 do-no-harm 제약을 함께 최적화하는 손실을 설계했다.
2. 평가 게이트의 임계에서 손실 항의 마진을 **유도**해, 손실을 만족한 모델이 게이트를
   통과하는 것이 우연이 아니라 정리가 되게 했다.
3. 학습 데이터의 재생–캡처 시간축 정합성을 측정으로 판정하고, 그 임계를 제어 대역
   상단에서 유도하는 파이프라인을 만들었다.

---

## 2. Method

### 2.1 신호 모델과 지연 물리

레퍼런스 모드는 두 가지다.

**digital reference** (기본) — 소음원을 런타임이 직접 생성하므로 파형을 미리 안다.

```
x_ref(t) = n(t + K)            K = digital reference lead
d(t)     = P(z) · n(t − D)     P(z) = 소음 스피커 → 에러 마이크 실측 FIR
e(t)     = d(t) + S(z) · y(t)
```

`K` 는 손으로 정하는 값이 아니라 실측에서 유도된다.

```
K = S 벌크지연 + 스레드 핸드오프(256) − P 벌크지연
```

**acoustic reference** — 레퍼런스 마이크가 소음을 듣는다. `S(z)` 의 실측 지연이 그대로
예측 부담이 되므로 주기성·협대역 성분에 한정된다.

### 2.2 아키텍처 — HybridANCNet

```
입력 [B, 2, T]  (ch0 = reference, ch1 = error)
  │
  ├─ Encoder      Conv1d(win=384, stride=hop=128) → GLU → ChannelLayerNorm → 1×1
  ├─ TCN blocks   dilated causal conv, dilations 1·2·4·8(·16), repeats 2(tiny)/3(base)
  ├─ GLSTM        grouped LSTM (긴 시간 상관)
  ├─ MHSA         windowed causal multi-head self-attention (선택)
  ├─ Head         1×1 → PReLU
  └─ Decoder      ConvTranspose1d → soft limiter  y = L·tanh(u/L)
출력 [B, 1, T]
```

| 변형 | 파라미터 | 채널 | TCN repeats | dilations |
|---|---:|---:|---:|---|
| `hybrid_anc_tiny` | 1.16 M | 128 | 2 | 1·2·4·8 |
| `hybrid_anc_base` | 5.99 M | 256 | 3 | 1·2·4·8·16 |

모든 연산은 인과적이다. 스트리밍 상태(`enc_hist`, 블록별 상태, `dec_tail`)를 명시적으로
들고 다니며, 오프라인 `forward` 와 블록 단위 `streaming_step` 의 수치 등가성을 테스트가
강제한다.

### 2.3 손실 함수

```
L = A[NMSE_trusted(dB)]
  + λ_dnh   · Σ_b w_b · A[relu(대역밖 증폭_b − margin_b)]
  + λ_frame · A[프레임별 NMSE_trusted(dB)]
  + λ_mrstft· A[아이템별 multi-resolution STFT]
  + λ_sat   · 리미터 이전 활성 포화 벌점
```

`A[·]` 는 평균이 아니라 **(평균, CVaR) 혼합** 집계다. dB 산술평균의 아이템별 그래디언트는
잔차 RMS 에 반비례하므로, 이미 잘 되는 아이템이 그래디언트를 독식하고 증폭 중인 아이템이
가장 덜 배운다 — G2 와 정확히 반대 방향이다.

**신뢰대역 안과 밖이 비대칭이다.**

- **안** — 양측 목표. "줄여라". `S(z)` 의 크기와 위상을 둘 다 믿는다.
- **밖** — 단측 힌지. "키우지 마라". 판정량이 `bandpower(S·y)` 라 `∠S` 와 무관하고,
  `relu` 라 "상쇄하라"는 그래디언트를 만들지 않는다.

do-no-harm 마진은 평가 게이트의 옥타브 임계 `G` 에서 유도된다.

```
margin = 20·log10(10^(G/20) − 1)          G = 1.0 dB  →  margin = −18.27 dB
```

이 유도가 있으면 "손실을 만족한 모델은 게이트를 통과한다"가 정리가 된다. 힌지 대역은
게이트의 옥타브 경계에 정렬되어, 대역 전체 비율을 만족한 채 한 옥타브에 에너지를 몰아넣는
자유도가 없다.

### 2.4 실시간 파이프라인

```
캡처 스레드 ──► SPSC 링버퍼 ──► 추론 스레드 ──► SPSC 링버퍼 ──► 재생 스레드
              (write_pos만)              (read_pos만)

블록 256 샘플 @ 48 kHz = 5.33 ms 마감
안전장치: 출력 DC 차단기 + 워치독 7 종 (발산·포화·클록 이탈 등)
```

---

## 3. Experimental Setup

### 3.1 하드웨어

| 구성 | 사양 |
|---|---|
| 덕트 | 1.2 m 사각 아크릴, 평면파 컷오프 1633 Hz |
| 마이크 | INMP441 ×2 (I²S), 레퍼런스 X=0.10 m / 에러 X=1.10 m |
| 스피커 | 4 인치 풀레인지 ×2, PCM5102A DAC + TPA3116D2 앰프 |
| 추론 | Jetson AGX Orin (JetPack 6, CUDA 12.6) |
| 학습 | NVIDIA A100 |

### 3.2 경로 측정

1차·2차 경로는 **동시 인터리브 톤 프로브**로 잰다. 두 스피커를 같은 출력 스트림에서
교대 톤 빈으로 동시에 구동하므로, 타임베이스 워프가 두 채널에 공통으로 걸려 상대량에서
상쇄된다.

기존 순차/초기 인터리브 측정에서는 `P−S = 139~141 samples`가 반복됐지만, 그 파일들은
submitted int16 PCM, raw/analysis SHA, clock witness, fractional joint-LS 등 현행 strict
provenance가 없어 **진단 자료일 뿐 official plant가 아니다**. 새 P/S는 같은 48 kHz/256/low
스트림에서 동시에 측정하고, 모든 150–1600 Hz 부대역 일관성·xrun·clip·지연 안정성 게이트를
통과해야 한다. strict P와 S는 같은 캡처의 값끼리만 사용한다.

### 3.3 데이터셋

**합성** — 온더플라이 생성. 실측 P/S 를 지나온 `d` 를 만들고 RIR 뱅크로 도메인 랜덤화한다.
소스 혼합비는 `configs/data_sim.yaml` 이 선언하며, 선언한 태그의 manifest 가 없으면
데이터셋 생성이 **실패한다**(조용한 대체 금지).

**실측** — 덕트에서 직접 녹음한 82 세션 / 95.7 분.

| 계열 | 세션 | lineage component |
|---|---:|---:|
| machine | 30 | 19 |
| environment | 18 | 15 |
| music | 18 | 18 |
| speech | 16 | 15 |

각 세션은 저장 시점에 시간축 정합성 검사를 통과해야 한다 — 재생–캡처 지연 궤적의
robust-std, 유효창 비율, 저역·고역 코히런스. 통과하지 못한 세션은 저장되지 않는다.

---

## 4. Evaluation Protocol

평가는 4 개 게이트로 이루어지며 **G4 는 3 값 판정**이다 (PASS / FAIL / INCONCLUSIVE).
표본이 부족해 아무 말도 할 수 없는 상태를 PASS 로 흘려보내면 게이트가 없느니만 못하다.

| 게이트 | 판정 |
|---|---|
| G1 | 신뢰대역 NMSE 개선 (cluster bootstrap CI 로 0 과 구별) |
| G2 | 소스 계열별 감쇠의 **최악값** |
| G3 | 통계적 검정력 — 계열당 독립 그룹 ≥ 4 |
| G4 | **대역 밖 do-no-harm** — 옥타브 감쇠가 −1.0 dB 보다 나쁘면 실패 |

G4 를 fullband 평균으로 대신할 수 없는 이유: NMSE 는 `d` 의 에너지로 정규화되므로 `d` 에
에너지가 거의 없는 대역에서는 `e` 가 수십 dB 커져도 전체 비율이 거의 안 변한다. 실측
반증으로 8 kHz 를 21 dB 증폭하면서 fullband 기준을 통과한 사례가 있다.

### 4.1 이론 상한

아래 값은 현행 strict provenance가 없는 **legacy diagnostic P/S**에서 최적 인과 FIR
(M = 2048)을 직접 풀어 얻은 과거 추정치다. 새 strict P/S가 고정되기 전에는 official
달성 가능 상한이나 readiness 근거로 사용하지 않는다.

```
150–1600 Hz   +4.83 dB
150– 600 Hz   +5.20 dB
```

이 값은 아티팩트 sha256 을 키로 캐시되며, P/S 를 다시 측정하면 자동으로 재계산된다.
설정에 적어 둔 숫자를 게이트가 그대로 믿지 않는다.

---

## 5. Results

> **현재 상태** — 물리 계층과 데이터 파이프라인은 검증됐고, 정정된 플랜트에서의 학습은
> 아직 수행되지 않았다. 검증된 것과 미검증인 것을 구분해 적는다.

**보존된 진단·데이터 증거**

| 항목 | 값 |
|---|---|
| legacy 경로 진단 | 독립 캡처의 `P−S=139~141` 및 높은 필터 일치는 재측정 설계 근거로만 보존 |
| 실측 데이터 정합성 | 82 세션 전량이 저장 시점 시간축 게이트 통과 |
| 테스트 | 전체 pytest 0 FAIL을 코드 게이트로 강제 |

**미검증**

- 현행 raw provenance를 갖춘 strict P/S가 아직 없다. legacy P/S는 readiness를 통과하지 못한다.
- 정정된 플랜트(`[150, 1600]` 대역)로 사전학습된 체크포인트가 아직 없다. 기존 체크포인트는
  모두 `[150, 600]` 대역·폐기된 2차경로에서 학습된 것이라 파인튜닝 진입 게이트를 통과하지
  못한다.
- 합성 코퍼스 중 선언 비중 0.45 에 해당하는 원본(`dns_fullband`, `demand`, `machine`)이
  아직 확보되지 않았다.
- 따라서 **실기 상쇄 성능 수치는 아직 주장하지 않는다.**

---

## 6. Limitations

- **레퍼런스 모드** — 현재 학습·평가는 digital reference(소음원을 미리 아는 구성)를 쓴다.
  실배포에는 acoustic reference 가 필요하고, 그 모드에서는 `S(z)` 의 실측 지연이 그대로
  예측 부담이 되어 주기성·협대역 성분으로 범위가 좁아진다.
- **저역 하한** — 80–150 Hz 는 경로 일관성이 0.59~0.82 로 낮다. 목표 대역은 [80, 1600] 이지만
  검증된 신뢰대역은 [150, 1600] 이다.
- **2차경로 적응 없음** — `S(z)` 는 오프라인 측정값이고 런타임에 갱신되지 않는다. 온라인
  2차경로 추정은 보조잡음을 계속 주입해야 하므로 G2 와 충돌한다. 현재는 학습 시 플랜트
  섭동(지연 지터·이득·틸트)으로 강건성을 확보한다.
- **비선형** — 앰프·스피커 비선형은 학습 시 랜덤 비선형으로 근사하며, 실측 THD/IMD 기반
  모델은 아직 적용하지 않았다.

---

## 7. Usage

### 7.1 설치

```bash
git clone https://github.com/Roka-jsj/Deep-ANC.git
cd Deep-ANC
# Jetson: NVIDIA wheel/ORT preload 규약까지 포함한 유일한 설치 경로
bash scripts/jetson/setup_jetson.sh
# Elice: docs/05의 exact-commit/holdout bootstrap 절차 사용(학습 자동 시작 없음)
```

### 7.2 데이터 준비

```bash
# historical builder 재현 → canonical held-out. 이 단계가 manifest보다 먼저다.
# identify_pool_clips.py 결과는 진단용이며 canonical 입력으로 쓰지 않는다.
.venv/bin/python scripts/data/repair_source_pool_provenance.py \
    --repair-csv --write-active-holdout --write-regrouped-manifest --jobs 4
EXPECTED_HOLDOUT_SHA256=$(sha256sum data/manifests/recorded_holdout.json | awk '{print $1}')

# FMA tracks.csv와 public raw 6종을 확보한 뒤에만 합성 manifest를 세대 단위로 생성한다.
# 선언 원본/lineage component/holdout 중 하나라도 불완전하면 실패하는 것이 정상이다.
# 각 raw audio byte SHA도 manifest schema v2에 결속된다.
.venv/bin/python scripts/data/prepare_noise_pool.py \
    --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256"
.venv/bin/python scripts/data/validate_recorded_sessions.py \
    --manifest data/manifests/recorded_regrouped.jsonl
```

로컬에는 public raw 6종 전체가 없으므로 위 historical repair에
`--require-downstream-gates`를 붙이지 않는다. synthetic downstream이 BLOCKED인 상태에서
성공으로 위장하지 않고, Elice에서 untouched raw를 받은 뒤 `prepare_noise_pool.py`가 해당
게이트를 연다. strict P/S까지 합격한 뒤에는 `build_elice_transfer_manifest.py`로 recorded
전체, RIR, strict raw/analysis/P/S, regrouped manifest, FMA metadata와 provenance bundle을
결속하고 그 SHA를 full bootstrap에 전달한다. 자세한 명령은 [docs/05](docs/05_training_elice.md)다.

### 7.3 경로 측정 (스피커 출력 있음)

실행 직전 운영자가 실제 배선을 `ERR mic=input 0`, `REF mic=input 1`,
`noise speaker=output 0`, `cancel speaker=output 1`로 확인하고, 덕트의 스피커/마이크
기하가 설정과 같고 사용자가 입회함을 확인해야 한다. 세 confirmation flag는 그 확인을 official provenance에
기록하며, 하나라도 없으면 장치나 세션을 만들기 전에 중단한다.

레벨 미터와 실제 측정은 공용 `MeasurementLevelContract`의 probe peak **0.003**을 함께
쓴다. **코드 테스트와 무음 dry-run을 모두 통과한 뒤에만** 연속 운영 절차를 시작한다.
meter는 input-only preflight 1.5초 뒤 nominal 20.0초/hard-max 21.0초, strict P/S는
input-only preflight 총 3.0초 뒤 nominal 12.5초/hard-max 13.5초(무음 lead-in 0.5초 +
자극 12.0초)다. nominal audible 합계는 **32.5초**지만 장치 기동·명령 인계 시간을 합친
wall-clock 연결 시간은 고정값이 아니다. 각 출력 close 직후 분리하고 노브는 유지한 채 다음
명령 직전에만 재연결한다. `[스피커 출력 종료]` 안내가 뜨면 즉시 스피커/앰프를 분리한다.
raw 저장과 분석은 그 안내 이후 무음으로 진행된다. strict P/S가 합격하기 전에 장시간
재녹음을 선행하지 않는다.

현행 peak 0.003과 `-50.1 dBFS`의 대응은 보존된 paired raw가 있어야 한다. 최초 1회에는
정상 live gate가 `BLOCKED`인 것이 맞으며, 명시적 bootstrap만 이 순환을 안전하게 끊는다.
meter 명령은 submitted int16/입력 int32/telemetry를 immutable NPZ와 SHA receipt로 남긴다.
strict 명령은 그 raw의 10분 freshness, 동일 logical hardware/channel과 ALSA physical
fingerprint(`/proc/asound` PCM info, sysfs realpath/uevent/안정 속성),
recipe/status/target 및 같은 앰프 노브 확인을 검증하고, 별도 probe 없이 기존 strict raw를
두 번째 half로 사용한다. 두 raw의
상대경로·SHA-256·재계산값이 모두 맞을 때만
`assets/measured/measurement_level_evidence.json`을 원자 생성한 뒤 official 분석을 연다.

```bash
# 1) 코드 게이트(소리 없음)
.venv/bin/python -m pytest -q

# 2) 아래 두 출력은 기존 파일이 없는 새 경로여야 한다(소리 없는 dry-run).
.venv/bin/python scripts/data/measure_paths_interleaved.py --dry-run \
    --primary-out results/path_measurement_next/p.npz \
    --secondary-out results/path_measurement_next/s.npz

# 3) 최초 1회: 사용자 입회·볼륨 최소, 공용 peak 0.003(출력 20초)
.venv/bin/python scripts/data/set_amp_level.py --bootstrap-level-evidence \
    --confirm-speaker --confirm-user-present --confirm-volume-minimum

# 출력된 immutable raw 상대경로를 복사하고 앰프 노브를 바꾸지 않는다.
METER_RAW=results/calibration_interleaved/level_bootstrap/<session>/meter_raw.npz

# 4) 같은 노브에서 strict P/S(출력 스트림 12.5초, 추가 level probe 없음)
.venv/bin/python scripts/data/measure_paths_interleaved.py \
    --bootstrap-level-evidence --meter-raw "$METER_RAW" \
    --confirm-same-amplifier-setting --confirm-user-present \
    --confirm-volume-minimum \
    --confirm-routing-and-geometry \
    --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \
    --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
```

실제로는 meter PASS 출력에 포함된 **정확한 strict 명령 전체**를 그대로 복사한다. 이 명령은
충돌하지 않는 `<capture-id>` 출력명을 함께 제시한다. meter 세션에는 `meter_raw.npz`와
`meter_raw.receipt.json`, strict 세션에는 `raw_measurement.npz`, `metadata.json`,
`analysis_results.npz`, `analysis_metadata.json`이 생긴다. 최초 bootstrap PASS 때만 paired
evidence JSON이 생성되고, 모든 분석 gate PASS 뒤 위 새 이름의 P/S NPZ가 no-replace로
승격된다. 기존 `primary_path_il.npz`/`secondary_path_il.npz`는 legacy라 덮어쓰지 않는다.
canonical evidence가 이미 있는 이후 실행도 fresh meter가 필수다. 이때는
`set_amp_level.py`를 bootstrap 옵션 없이 같은 세 confirmation으로 실행하고, 출력된
`--meter-raw` strict 명령(bootstrap 옵션 없음)을 그대로 쓴다. 영구 evidence만으로 현재
앰프 노브 상태를 대신할 수 없다.

### 7.4 학습

```bash
# 사전학습
.venv/bin/python scripts/train/train.py --config configs/train_pretrain_tiny.yaml

# canonical 100k best.pt를 명시한 뒤 15/15 진입 게이트를 통과해야 시작된다.
INIT_CKPT=runs/<canonical-pretrain-contract>/ckpt/best.pt
.venv/bin/python scripts/train/check_finetune.py --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured --set init_ckpt="$INIT_CKPT"
.venv/bin/python scripts/train/run_finetune_pipeline.py \
    --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured --set init_ckpt="$INIT_CKPT"
```

### 7.5 평가와 배포

아래 export/배포 명령은 공식 test G4와 별도 natural-crest challenge가 모두 PASS한 뒤에만
실행한다. 그 전에는 closed-loop나 실제 ANC ON 평가도 시작하지 않는다.

```bash
.venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/<run>/ckpt/best.pt --n-items 64
.venv/bin/python scripts/train/export_onnx.py --ckpt runs/<run>/ckpt/best.pt --out runs/export/model.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml
```

### 7.6 테스트

```bash
.venv/bin/python -m pytest -q
```

---

## 8. Repository Structure

```
src/deep_anc/
  data/          합성·실측 데이터셋, manifest, 시간축 정합, QA
  dsp/           지연·대역 단일 출처, 2차경로, 덕트 시뮬, 불변식, 설계 상한
  losses/        ANC 손실과 경계 검증 (pydantic)
  models/        HybridANCNet, TCN·GLSTM·MHSA, 스트리밍
  eval/          지표, 실측 평가, FxLMS 베이스라인
  realtime/      링버퍼, 추론 엔진(Torch/ORT/TRT), 안전 워치독
  train/         트레이너, 파인튜닝 진입 게이트
  ops/           게이트 레지스트리, 작업 큐

configs/         덕트·데이터·모델·학습·런타임·평가 설정
scripts/         data · train · eval · bench · export · jetson
tests/           계약·회귀·공격 fixture
docs/            00 개요 · 01 지연 물리 · 02 하드웨어 · … · 12 시스템 요약
```

핵심 규약은 단일 출처를 갖는다.

| 물리량 | 단일 출처 |
|---|---|
| 지연·lead·대역 | `dsp/timing.py` — lead 는 `PlantDelays.lead()` 로만 생성 가능 |
| 대역 밖 예산 | `dsp/do_no_harm.py` — 손실 마진이 게이트 임계에서 유도 |
| 스트림 정합 임계 | `dsp/invariants.py` — 지터 상한이 제어 대역 상단에서 유도 |
| 게이트 목록 | `ops/gate_registry.py` — 모든 게이트가 음성·양성 fixture 를 갖는다 |

## 9. Safety

- 스피커 출력 스크립트는 **사용자 입회 + 볼륨 최소** 상태에서만 실행한다.
- 런타임은 항상 ANC OFF 로 시작하고, 워치독이 발산·포화·클록 이탈을 감지하면 상쇄 출력을
  0 으로 페이드한다.
- 출력에는 소프트 리미터 `y = L·tanh(u/L)` 가 항상 걸려 있다.

## Documentation

| 문서 | 내용 |
|---|---|
| [docs/01](docs/01_physics_limits.md) | 지연 물리 — 두 레퍼런스 모드의 인과성 |
| [docs/02](docs/02_hardware_setup.md) | 하드웨어 배선과 점검 절차 |
| [docs/04](docs/04_model_architecture.md) | 모델·스트리밍·ONNX 규약 |
| [docs/07](docs/07_evaluation_protocol.md) | 평가 프로토콜과 게이트 |
| [AGENTS.md](AGENTS.md) | 작업 규칙 |
| [HANDOFF.md](HANDOFF.md) | 현재 진행 상태 (이 README 는 상태를 담지 않는다) |

## License

[MIT License](LICENSE)
