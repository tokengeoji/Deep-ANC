# 04. 모델 아키텍처 — HybridANCNet

GCRN / Transformer / WaveNet / Conv-TasNet 네 구조에서 실시간 ANC 에 필요한 요소만
취사선택해 결합한 **시간영역 인과 회귀 모델**. STFT 를 쓰지 않는다
(창 지연 0, TensorRT 의 DFT 미지원 원천 회피).

![HybridANCNet base/tiny 네트워크와 사전학습 손실 경로](../assets/diagrams/hybrid_anc_architecture.svg)

## 1. 무엇을 취하고 무엇을 버렸나

| 원천 | 취함 | 버림 | 근거 |
|---|---|---|---|
| Conv-TasNet | 학습형 1D conv 인코더/디코더, TCN 골격 | 마스크 곱 출력, global LayerNorm | ANC 출력은 입력의 마스킹이 아니라 위상반전+예측된 **새 파형** → 직접 회귀. gLN 은 비인과 |
| WaveNet | dilated **causal** depthwise conv, residual, gated activation | 샘플단위 자기회귀 | 샘플 AR 은 48kHz 실시간 불가 — dilation 을 프레임(hop 128) 단위로 |
| GCRN | GLU 게이팅, **GLSTM**(그룹 LSTM 병목) | STFT 복소 스펙트럼 매핑, 주파수축 conv2d | LSTM 의 무한 기억이 주기 잡음 예측의 핵심. STFT 는 지연·TRT 문제 |
| Transformer | **windowed causal MHSA 1층** (KV 캐시 64프레임=170ms), 상대위치 bias | 전역 attention, 대형 FFN | 회전기계 등 반복 패턴 재조회 용도로만 — FFN 은 TCN 이 대체 |

## 2. 구조 (base 기준)

```
입력 [B,2,T] (ch0=ref, ch1=err피드백, T=hop 배수)
 ├ ÷ io_scale (0.02)
 ├ 좌측 256샘플 패딩 → Encoder Conv1d(2→512, k=384, s=128) → GLU → 256ch   ← block-acquired, hop 내 +127 sample
 ├ ChannelLN + 1×1
 ├ TCN 반복 ×3 { dilation 1,2,4,8,16 }        각 블록: 1×1(256→512)+PReLU+LN
 │    ↑ 반복2 뒤: GLSTM(그룹2, 그룹당 hid256)     → dwConv k3 ×2(주경로·게이트 σ)
 │    ↑ 반복3 뒤: causal MHSA(head4×64, 윈도64)   → 1×1(512→256) + residual
 ├ Head 1×1(256→512) + PReLU
 ├ Decoder ConvTranspose1d(512→1, k=384, s=128) → 앞 T 샘플 (인과 OLA)
 └ × io_scale → 소프트 리미터 0.2·tanh(y/0.2)                     → 출력 [B,1,T]
```

| 변형 | 파라미터 | 연산 | 수용영역 | 용도 |
|---|---|---|---|---|
| tiny | 1,164,809 (1.16M) | 0.43 GMAC/s | 0.16s + LSTM | **현행 실시간 기본** (ORT CPU P99 1.5ms 실측) |
| tiny-attn | 1,231,369 (1.23M) | 측정 전 | 0.16s + LSTM + MHSA 170ms | GPU1 구조 탐색 후보 |
| tiny-long | 1,301,771 (1.30M) | 측정 전 | 0.33s + LSTM | GPU1 구조 탐색 후보 |
| tiny-long-attn | 1,368,331 (1.37M) | 측정 전 | 0.33s + LSTM + MHSA 170ms | 결합 ablation 후보 |
| base | 5,994,512 (5.99M) | 2.25 GMAC/s | 0.50s + LSTM + MHSA 170ms | TRT FP16 배포 목표 (ORT CPU 6.8ms) |
| large | (v2 옵션) | — | — | A100 teacher/distillation — 1차 릴리스 제외 |

파라미터 실측: tests/test_model_shapes.py (tiny 계열 0.9~1.5M, base 5~7M 게이트).
후보 3종은 100k LR 스케줄을 바꾸지 않고 `run_until_step=20000`에서 같은 학습곡선을
비교한다. source×band held-out와 Jetson P99를 통과하기 전에는 현행 tiny를 대체하지 않는다.

### 파라미터 분해

| 구간 | base | tiny |
|---|---:|---:|
| Encoder + ChannelLN + projection | 459,520 | 213,376 |
| TCN | 4,020,495 (268,033 × 15) | 547,848 (68,481 × 8) |
| GLSTM | 922,368 | 272,256 |
| Causal MHSA | 263,936 | — |
| Head + PReLU + Decoder | 328,193 | 131,329 |
| **합계** | **5,994,512** | **1,164,809** |

`io_scale`은 학습 파라미터가 아닌 buffer다. 현재 Elice Stage-1은 base batch 96과
tiny batch 128을 각각 100k step 학습한다. GPU0/base와 GPU1/tiny는 서로 독립된
프로세스이며, 두 모델 사이에 gradient·가중치 공유나 증류는 없다.

## 3. 인과성과 지연

- 인코더 frame은 좌측 history와 현재 hop 128 samples를 한번에 사용한다.
  따라서 첫 output phase는 같은 hop의 끝인 `x[+127]`까지 참조하며
  **sample-zero-lookahead가 아니다**. legacy Tiny Jacobian에서도 nonzero 입력 범위가
  정확히 `0..127`로 확인됐다.
- 실시간 엔진은 256-sample block 전체를 받은 뒤 출력한다. 따라서 위
  127-sample 의존은 물리적으로 구현 가능하지만, 이 256-sample handoff를 P/S·학습·
  checkpoint·runtime timing contract에 **정확히 한 번** 포함해야 한다. 프레임
  경계 인과성을 sample-level 지연 0으로 표기하지 않는다.
- digital-ref는
  자기생성 ref를 먼저 주고 noise playback을 현재 strict P/S에서 유도한
  FIFO lead만큼 지연한다. 현재 capture에서는 `1245 + 256 − 1386 = 115`
  샘플이지만, 이 숫자를 config/checkpoint/runtime에 손으로 복사하지 않는다.
  NPZ→`PlantDelays.lead()`→`TrainingTimingContract`와 그 SHA가 전 경로의
  단일 출처다.
- acoustic-ref에서는 이 확정적 선행 공급을 쓸 수 없다. 약 30ms 예측 부담은 손실 정렬과
  LSTM/MHSA의 주기 기억으로 학습하되, 광대역 랜덤음은 물리적으로 예측할 수 없다.

## 4. 스트리밍 상태 (전부 명시적 텐서, 정적 shape)

| 상태 | shape (base) | 내용 |
|---|---|---|
| `st_enc` | [1,2,256] | 인코더 입력 히스토리 (win−hop) |
| `st_i_tcn` ×15 | [1,512,2d] | dilated dwConv 좌측 히스토리 |
| `st_i_lstm_h/c` | [1,512] ×2 | GLSTM 은닉/셀 (2그룹×256 concat) |
| `st_i_attn_k/v` | [1,4,64,64] ×2 | MHSA KV 링버퍼 (concat+slice) |
| `st_i_attn_m` | [1,1,1,64] | KV 슬롯 유효성 마스크 (빈 슬롯 −1e4) — 워밍업 구간도 오프라인과 등가 |
| `st_dec` | [1,1,256] | 디코더 OLA 꼬리 |

스트리밍=오프라인 등가성: 실측 max err ~3e-8 (테스트 게이트 1e-5). 이는 256-sample
block이 이미 수신된 후의 수치 등가성이지 sample-zero-lookahead 주장이 아니다. GLSTM 은 학습 시 cuDNN nn.LSTM,
스트리밍/export 시 동일 가중치의 수동 셀 — 등가성 테스트 포함 (설계 H1).

## 5. Stage-1 학습 플랜트와 손실 (`losses/anc_loss.py`)

```
source n ─→ x_ref=n(t+K) ─→ HybridANCNet ─→ y
       └→ P_contract(n) ────────────────────────────────────→ d
y → G_nl(η=10, drive=1) → S_contract(z)+handoff → e=d+S·y

L = NMSE_150–1600Hz(dB)
    + 1.0·MR-STFT{256,512,1024,2048}×W(f)
    + 1e-3·L_pow + 1.0·L_clip(마진 0.18)
```

- `P_surrogate`는 측정 S의 장치 gain/FIR을 P에 재사용해 P/S 단위를 맞춘다. P bulk delay,
  compact FIR peak, S bulk delay, 256-sample handoff와 K는 strict NPZ의
  `TrainingTimingContract`가 구분한다. 실제 P 주파수응답은 아니므로 이 단계의 dB는 실제
  감쇠가 아니다.
- NMSE 목적함수는 S 실측 `consistency_band_hz` 150–1600Hz 와 duct 목표
  `realistic_target_band_hz` 80–1600Hz 의 교집합 = **150–1600Hz** 다
  (2026-08-05 플랜트 복구 전에는 150–600Hz).
  로그의 `nmse_t`/`nmse_trusted_db`가 최적화·best 선택 기준이고,
  `nmse_f`/`nmse_fullband_db`는 전대역 증폭 여부를 감시하는 별도 지표다.
- **W(f) 커리큘럼 A**: `duct.yaml` 목표대역 80–800Hz ×3,
  1633Hz(컷오프) 이상 ×0.25, 40Hz 미만 ×0.1. MR-STFT는 스펙트럼 균형을 담당한다.
  풀밴드 커리큘럼 B 는 **광대역 S(z) 재보정 통과 후에만** (설계 C3 게이트).
- Stage-1은 delay/gain/tilt jitter와 all-pass를 모두 끈 공칭 plant이며, 비선형도
  η=10·drive=1·hardclip=0의 사실상 선형 조건이다. 미관측 plant/비선형 랜덤화는
  실측 조건 또는 적응층을 갖춘 이후 단계에서 점진적으로 켠다.
- 극성 규약: 측정 FIR 에 극성이 포함 — **추가 부호 반전 금지** (e = d + S·y).
- 손실은 FP32 고정 (bf16 은 FFT 미지원).

### 과거 0dB 실행을 폐기한 이유

과거 실행은 단위 gain 1D `P_err` RIR과 장치 스케일 측정 S를 섞어 limiter ±0.2로는
상쇄할 수 없는 d를 만들었다. 동시에 모델이 관측하지 못하는 delay/all-pass를 batch마다
독립 랜덤화하고 fullband NMSE를 선택 기준으로 사용했다. 이때 loss≈2, NMSE≈0dB는
"긴 초기 구간"이 아니라 y≈0이 최적인 잘못된 목적의 신호다. 해당 체크포인트는 새
Stage-1에 resume하지 않는다.

현재 checkpoint에는 resolved model/data/duct 설정과 함께 lead, trusted band,
`physics_status=secondary_surrogate_representation_pretrain`이 저장된다. 측정 P로
파인튜닝한 checkpoint는 `measured_primary_path`로 구분하며, 배포 runtime은 checkpoint/ONNX
메타의 lead가 설정과 다르면 실행을 거부한다.

## 6. ONNX Export 규약 (`scripts/train/export_onnx.py`)

- opset 17, 배치 1, **모든 shape 정적**, 상태 전부 명시 입출력.
- 그래프 입력 x=[1,2,256] — 모델 hop 128 의 2프레임을 **그래프 내부 정적 언롤**
  (LSTM 2스텝 수동 셀, MHSA 2쿼리, KV concat+slice).
- 사용 op: Conv/ConvTranspose/MatMul/Sigmoid/Tanh/Softmax/PReLU/Split/Concat/Slice/
  Transpose/Reshape/LayerNormalization. 금지: LSTM/GRU op, DFT, If/Loop/Scan, 복소 dtype.
- export 후 ORT(CPU) 스트리밍 등가성 자동 검증 (실측 2.4e-8, 허용 1e-4).
