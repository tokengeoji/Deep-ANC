# 3. 모델 아키텍처 — HybridANCNet-tiny

전체 설계 근거는 [docs/04_model_architecture.md](../docs/04_model_architecture.md),
구현은 [src/deep_anc/models/hybrid_anc.py](../src/deep_anc/models/hybrid_anc.py),
설정은 [configs/model_tiny.yaml](../configs/model_tiny.yaml)이 단일 출처다.

## 3.1 왜 이 구조인가

STFT를 쓰지 않는 **시간영역 인과(causal) 회귀 모델**이다. 4가지 기존 구조에서
실시간 ANC에 필요한 요소만 취했다:

| 원천 | 취함 | 버림 | 이유 |
|---|---|---|---|
| Conv-TasNet | 학습형 1D conv 인코더/디코더, TCN 골격 | 마스크 곱 출력 | ANC 출력은 입력을 가리는(masking) 게 아니라 위상반전된 **새 파형**을 만들어야 함 |
| WaveNet | dilated causal depthwise conv, gated activation | 샘플단위 자기회귀 | 48kHz 실시간에 샘플 AR은 불가능 — dilation을 프레임(hop 128) 단위로 대체 |
| GCRN | GLU 게이팅, GLSTM(그룹 LSTM) | STFT 복소 스펙트럼 매핑 | LSTM의 장기 기억이 주기 잡음 예측의 핵심. STFT는 창 지연 + TensorRT 미지원 문제 |
| Transformer | tiny에서는 비활성(연산·TensorRT 리스크) | 전역 attention | 회전기계 등 반복 패턴 재조회용이었으나 tiny는 제거, TCN이 대체 |

## 3.2 구조도

![HybridANCNet 구조](../assets/diagrams/fig2_architecture.svg)
![TCN 블록 상세](../assets/diagrams/fig3_tcn_block.svg)
![GLSTM 병목](../assets/diagrams/fig5_glstm.svg)
![스트리밍 상태](../assets/diagrams/fig6_streaming.svg)

```
입력 [B,2,T] (ch0=레퍼런스, ch1=에러 피드백, T=hop 128의 배수)
 ├ ÷ io_scale(0.02)
 ├ 좌측 256샘플 패딩 → Encoder Conv1d(2→256, k=384, s=128) → GLU → 128ch
 ├ ChannelLayerNorm + 1×1 projection
 ├ TCN 2 반복 × dilation [1,2,4,8]   (각 블록: dilated depthwise conv + GLU 게이트)
 │    ↑ 반복 2 뒤: GLSTM(그룹 1, hidden 192) — 주기 잡음 장기 기억
 ├ Head 1×1 + PReLU
 ├ Decoder ConvTranspose1d(→1, k=384, s=128) → 앞 T 샘플만 사용(인과 OLA)
 └ × io_scale → 소프트 리미터 0.2·tanh(y/0.2) → 출력 [B,1,T]
```

## 3.3 규모 (tiny)

| 항목 | 값 |
|---|---|
| 파라미터 | 1,164,809 (≈1.16M) |
| 연산량 | 0.43 GMAC/s |
| 수용영역(receptive field) | 0.16s + LSTM 무한 기억 |
| 실측 추론 지연 | ONNX Runtime CPU, P99 1.50 ms (Jetson AGX Orin) |

`base`(5.99M 파라미터, TensorRT FP16 목표)도 존재하지만 이번 acoustic-reference
파일럿은 실시간 배포 기본값인 **tiny**로만 진행했다.

## 3.4 인과성

인코더 프레임은 좌측 history + 현재 hop 128샘플을 함께 보므로, 첫 출력 샘플도
그 hop의 마지막 입력(`x[+127]`)까지는 참조한다 — 완전한 sample-zero-lookahead는
아니다. 실시간 엔진은 256샘플 블록 전체를 받은 뒤 출력하며, 이 handoff는
학습·체크포인트·런타임 timing contract에 정확히 한 번만 반영된다(이중 정의 시
[configs/duct.yaml](../configs/duct.yaml)의 `secondary_path.handoff_extra_samples`가
단일 출처).
