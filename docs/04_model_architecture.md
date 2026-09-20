# 04. 모델 아키텍처 — HybridANCNet

HybridANCNet은 REF와 ERR를 받아 상쇄 파형을 직접 출력하는 시간영역 인과 모델이다.
추론 그래프에 STFT를 사용하지 않는다. TCN·그룹 LSTM과 선택적 causal attention을
결합하며, 구성값은 `configs/model_*.yaml`이 단일 출처다.

현재 우선 과제는 acoustic-ref다. 모델 구조가 있다는 사실과 acoustic 학습·배포가
검증됐다는 사실을 구분한다. `configs/runtime_acoustic_hybrid.yaml`의 모델은 아직
placeholder이며, 진행 상태는 [HANDOFF.md](../HANDOFF.md)에 기록한다.

## 1. 구성 요소

| 요소 | 역할과 제약 |
|---|---|
| Conv1d 인코더 / ConvTranspose1d 디코더 | 파형을 학습형 특징으로 변환하고 상쇄 파형을 직접 회귀 |
| Causal dilated TCN | 과거 프레임의 여러 시간 범위를 사용. 샘플별 자기회귀는 사용하지 않음 |
| GLU / ChannelLayerNorm | 채널 게이팅·정규화. 미래 시간축을 사용하는 global 정규화 금지 |
| 그룹 LSTM | 반복 성분의 시간 문맥 유지. 학습 경로와 export 수동 셀의 가중치 공유 |
| Windowed causal MHSA | 설정된 과거 프레임만 조회. tiny 기본형은 사용하지 않음 |
| 소프트 리미터 | 모델 출력 크기 제한. 실제 합산 출력에는 런타임 안전장치도 별도 적용 |

## 2. 구조와 변형

base 설정의 입력은 `[B,2,T]`이며 ch0=REF, ch1=ERR 피드백이다.
학습 세그먼트는 256의 배수이고 모델 내부 hop은 128이다.

```text
입력 / io_scale(0.02)
→ 좌측 256샘플 히스토리 + Conv1d(2→512, k384, stride128) + GLU
→ ChannelLayerNorm + 1×1 projection (256채널)
→ TCN 3회 반복, 각 dilation [1,2,4,8,16]
  · 반복2 뒤 GLSTM: 2그룹 × hidden256
  · 반복3 뒤 causal MHSA: 4head × 64, window64
→ Head 1×1(256→512) + PReLU
→ ConvTranspose1d(512→1, k384, stride128), causal overlap-add
→ io_scale 복원 + 0.2·tanh(y/0.2)
→ 출력 [B,1,T]
```

| 변형 | 파라미터 수 | 주요 차이 |
|---|---:|---|
| tiny | 1,164,809 | 128채널, TCN 2회, GLSTM, attention 없음 |
| tiny-attn | 1,231,369 | tiny에 attention 추가 |
| tiny-long | 1,301,771 | tiny의 시간 문맥 확장 |
| tiny-long-attn | 1,368,331 | 시간 문맥 확장과 attention 결합 |
| base | 5,994,512 | 256채널, TCN 3회, GLSTM, attention |

파라미터 수는 `tests/test_model_shapes.py`의 계약과 일치해야 한다.
acoustic 준비 설정은 tiny를 사용하지만, 이는 학습·실시간 성능 최적 모델로 확정한 결과가 아니다.
후보 교체는 같은 소스·대역·독립 분할의 성능과 실제 Jetson 처리시간으로 판단한다.
과거 ORT 지연 측정이나 Elice 실행 상태는 이 구조 문서의 현재 성능 근거로 사용하지 않는다.

`io_scale`은 학습 파라미터가 아닌 buffer다. 입출력 단위를 바꾸면 기존 체크포인트의
스케일 계약도 재검증해야 한다.

## 3. 인과성과 경로 지연

모델은 현재 처리하는 입력 블록과 과거 상태를 사용하고 미래 프레임을 참조하지 않는다.
인코더 좌측 히스토리와 디코더의 과거 overlap-add 꼬리를 보존한다.
추가 미래 프레임 lookahead가 없다는 것은 블록 수집·추론·I/O·핸드오프 지연이
없다는 뜻이 아니다. 전체 물리는 [docs/01](01_physics_limits.md)을 따른다.

- acoustic-ref는 실제 REF 입력과 lead=0을 사용한다. 현재 기하 기반 예측 부담은
  약 32.9ms이며 반복 성분의 기억만으로 예측 불가능한 소리를 상쇄할 수는 없다.
- 현재 측정 경로는 S=1465, handoff=256, P=1608샘플이다.
  digital 학습 정렬은 `113 + 1608 = 1465 + 256`이다.
- 과거 digital ONNX 호환용 `runtime_tiny.yaml`에는 lead=109가 남아 있다.
  기존 artifact의 메타·측정 조건과 현재 정렬을 혼용하지 않는다.
- mic DL/hybrid는 `reference_mode=acoustic` 메타가 있는 검증본이 필요하다.
  digital-ref 또는 모드 정보가 없는 artifact를 acoustic 모델로 사용하지 않는다.

## 4. 스트리밍 상태와 export 계약

상태는 정적 shape의 명시적 텐서 입출력이며 호출 사이에 유지한다.

| 상태 | base shape | 내용 |
|---|---|---|
| `st_enc` | [1,2,256] | 인코더 입력 히스토리 |
| TCN 상태 ×15 | [1,512,2d] | dilation별 좌측 히스토리 |
| GLSTM h/c | [1,512] ×2 | 2그룹 hidden/cell |
| MHSA k/v | [1,4,64,64] ×2 | 과거 key/value |
| MHSA mask | [1,1,1,64] | 빈 KV 슬롯의 유효성 |
| `st_dec` | [1,1,256] | 디코더 overlap-add 꼬리 |

미래 프레임 변경 불변성, 스트리밍/오프라인 등가성과 GLSTM 두 경로 등가성을 테스트한다.
스트리밍 등가성 오차 허용치는 `1e-5`다. 과거 한 번의 실측 오차를 현재 실행 결과로 대신하지 않는다.

ONNX export는 `scripts/train/export_onnx.py`를 사용한다.

- opset 17, 배치 1, 입력 `x=[1,2,256]`, 모든 shape 정적, 상태 전부 명시 입출력.
- hop128의 두 프레임을 그래프 내부에서 정적으로 펼친다. LSTM은 수동 셀로 export한다.
- LSTM/GRU op, DFT, If/Loop/Scan, 복소 dtype은 사용하지 않는다.
- export 후 ORT CPU 등가성을 확인한다. TensorRT 변환·실제 Jetson 지연은 별도 검증이다.

## 5. 학습 플랜트와 손실

모델 추론에는 STFT가 없지만 학습 손실에는 다중해상도 STFT가 있다.
플랜트는 설정된 실측 S의 FIR·gain·순수지연과 런타임 핸드오프를 사용한다.

```text
모델(REF, ERR 피드백) → y → 선택적 비선형 G → S → 잔류음 e = d + S·G(y)

L = trusted-band NMSE(dB)
    + λ_mrstft · MR-STFT{256,512,1024,2048} × W(f)
    + λ_pow · 출력 파워 + λ_clip · 클리핑 벌점
```

학습 모드마다 d와 REF의 생성 근거가 다르다.

| 경로 | 데이터·물리 해석 |
|---|---|
| acoustic 준비 | 합성 acoustic P_ref/P_err와 실제 준비 음원을 사용. REF 선행을 digital FIFO로 대체하지 않음 |
| digital measured | 실측 내부 NS→ERR의 P 사용 |
| digital secondary surrogate | S의 gain/FIR을 P 대용으로 사용. 표현 사전학습이며 실제 감쇠 증거가 아님 |

`configs/train_acoustic_prepared.yaml`은 준비 확인용 tiny/open-loop/최대 2 step 설정이다.
설정·QA 파일이 존재한다고 실제 corpus 학습이나 배포 모델 생성이 완료된 것은 아니다.
`configs/data_acoustic_prepared.yaml`은 원본·manifest·QA를 요구하고
음성·음악을 포함한다. MIMII machine은 train-only이며 validation/test에서는 제외한다.

현재 trusted NMSE는 S 검증 대역 150–600Hz와 `duct.yaml`의 기존 가중 대역
80–800Hz의 교집합이다. `nmse_t`/`nmse_trusted_db`와 전대역 증폭 감시용
`nmse_f`/`nmse_fullband_db`를 구분한다.
합성 목표 1000–1600Hz를 추가한 것은 S 신뢰대역이나 고역 손실 검증을 확대하지 않는다.

`curriculum_a`는 설정의 목표대역 ×3, 평면파 cutoff 초과 ×0.25, 40Hz 미만 ×0.1로 가중한다.
광대역 손실 승격에는 해당 S 경로 검증이 필요하다. 현재 준비 설정의 plant jitter·all-pass는
꺼져 있고 비선형은 η=10, drive=1, hardclip=0의 사실상 선형 조건이다.
실측 조건 없이 큰 랜덤 위상·비선형을 추가하면 올바른 물리 강건성 검증이 되지 않는다.

반드시 유지할 계약은 다음과 같다.

- 극성은 `e = d + S·y`. 측정 FIR의 부호를 다시 반전하지 않는다.
- 손실은 FP32로 계산한다. bf16 경로에서 FFT를 실행하지 않는다.
- closed-loop 워밍업은 전체 신호에 플랜트를 적용한 **뒤** 손실에서 잘라낸다.
- checkpoint/ONNX의 resolved 설정·reference mode·lead·물리 상태를 보존하고 런타임과 대조한다.
- surrogate·합성 acoustic 학습 결과를 실제 덕트 성능으로 해석하지 않는다.
