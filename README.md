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
- **진입 게이트 14 종** — 물리·데이터·체크포인트가 정합하지 않으면 학습이 시작되지 않는다

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

```
P(z) 소음 → 에러      순수지연 1580 샘플     150–1600 Hz 일관성 0.9992
S(z) 상쇄 → 에러      순수지연 1440 샘플     150–1600 Hz 일관성 0.9987
P − S = 140 샘플                            유지 반복 32/32
```

**절대 지연은 재현되지 않지만 `P − S` 는 재현된다.** 독립 캡처에서 `P−S = 139~141`,
그로부터 유도되는 lead 는 115~116 이다. 그래서 `P` 와 `S` 는 반드시 **같은 캡처**의
값끼리만 함께 쓴다.

### 3.3 데이터셋

**합성** — 온더플라이 생성. 실측 P/S 를 지나온 `d` 를 만들고 RIR 뱅크로 도메인 랜덤화한다.
소스 혼합비는 `configs/data_sim.yaml` 이 선언하며, 선언한 태그의 manifest 가 없으면
데이터셋 생성이 **실패한다**(조용한 대체 금지).

**실측** — 덕트에서 직접 녹음한 82 세션 / 95.7 분.

| 계열 | 세션 | 그룹 |
|---|---:|---:|
| machine | 30 | 25 |
| environment | 18 | 17 |
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

실측 P/S 에서 최적 인과 FIR (M = 2048) 을 직접 풀어 얻은 달성 가능 상한:

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

**검증된 것**

| 항목 | 값 |
|---|---|
| 경로 측정 재현성 | 독립 캡처 2 회의 최적 필터 `−P/S` 일치 0.9976 (상대오차 7.7 %) |
| `P − S` 불변량 | 140 / 141 샘플 (독립 캡처 9 건에서 139~141) |
| 신뢰대역 일관성 | 150–1600 Hz 에서 P 0.9992 / S 0.9987 |
| 실측 데이터 정합성 | 82 세션 전량이 저장 시점 시간축 게이트 통과 |
| 테스트 | 51 개 파일 / 744 케이스 통과 |

**미검증**

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
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt      # 또는 requirements-jetson.txt
```

### 7.2 데이터 준비

```bash
# 합성 소음 풀 manifest (선언한 태그의 원본이 없으면 실패한다)
python scripts/data/prepare_noise_pool.py

# 실측 세션 → held-out → manifest → 전수 QA
python scripts/data/make_recorded_holdout.py
python scripts/data/make_recorded_manifest.py
python scripts/data/validate_recorded_sessions.py
```

### 7.3 경로 측정 (스피커 출력 있음)

```bash
python scripts/data/measure_paths_interleaved.py --confirm-volume-minimum \
    --primary-out assets/measured/primary_path_il.npz \
    --secondary-out assets/measured/secondary_path_il.npz
```

### 7.4 학습

```bash
# 사전학습
python scripts/train/train.py --config configs/train_pretrain.yaml

# 파인튜닝 — 진입 게이트를 통과해야 시작된다
python scripts/train/check_finetune.py --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured
python scripts/train/run_finetune_pipeline.py --config configs/train_finetune.yaml
```

### 7.5 평가와 배포

```bash
python scripts/eval/evaluate_offline.py --ckpt runs/<run>/ckpt/best.pt --n-items 64
python scripts/train/export_onnx.py --ckpt runs/<run>/ckpt/best.pt --out runs/export/model.onnx
python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml
```

### 7.6 테스트

```bash
python -m pytest -q
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
tests/           51 개 파일 / 744 케이스
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
