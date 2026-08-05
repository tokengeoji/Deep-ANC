# 00. 프로젝트 개요

## 무엇을 만드는가

덕트(사각 아크릴, 1.2m) 안의 소음을 **딥러닝 모델이 실시간으로 상쇄**하는 시스템.
소음 스피커(NS)가 좌측 폐단에서 소음을 방사하면, 레퍼런스 신호를 입력받은 모델이
상쇄 신호 y(n)을 만들어 상쇄 스피커(CS)로 출력하고, 에러 마이크(ERR)가 잔여 소음을 감시한다.

```
            X=0        X=0.1m                X=1.05m   X≈1.1m   X=1.2m
        ┌────┃──────────┃─────────────────────┃─────────┃─────────┐
  [NS]──┨    │        [REF]                 [CS]      [ERR]       ┃ → 개방단
        └────┃──────────┃─────────────────────┃─────────┃─────────┘
   소음 스피커      레퍼런스 마이크        상쇄 스피커   에러 마이크

  Jetson AGX Orin: 마이크 2ch 입력(hw:APE,1) · 스피커 2ch 출력(AB13X USB)
```

기존 FxLMS(적응 필터) 시스템과 동일 하드웨어를 쓰되, 시간영역 인과 모델로
복잡한 소리의 상쇄 파형을 직접 회귀한다. 최종 목표는 ① 저주파와 고주파를 함께
제거하고 ② 소음뿐 아니라 대화·음악까지 포함한 quiet zone을 만드는 것이다.

현재 단계는 이 최종 성능을 주장하는 단계가 아니다. noise 출력→ERR 1차경로
`P(z)`의 실측 파일이 아직 없으므로, Stage-1은 측정 `S(z)`의 FIR/gain을
`P(z)` 대용으로 재사용하는 **secondary-surrogate 표현 사전학습**이다. 이 선택은
`P/S` 단위 불일치로 학습이 영출력에 고정되는 것을 막지만, surrogate 체크포인트의
dB를 실제 덕트 감쇠 성능으로 해석해서는 안 된다.

## 시스템 전체 그림

```
[학습 — Elice Cloud 2×A100, 서로 독립된 base/tiny 프로세스]
  공개 노이즈·음성·음악 + 합성원 → 연속 source n
                    ├→ ref를 실제 playback보다 116샘플 먼저 공급
                    └→ P_surrogate=S의 FIR/gain, D_noise=1602(실측) → d
  HybridANCNet → y → 공칭 선형 S(z), 총지연 1462+256=1718 → e=d+S·y
                    └→ trusted NMSE(150–1600Hz) 최적화 + fullband NMSE 감시
  결과: physics_status=secondary_surrogate_representation_pretrain 체크포인트
[배포 — Jetson AGX Orin]
  실측 파인튜닝을 통과한 best.pt → ONNX(정적 스트리밍 그래프) → [ORT CPU | TensorRT FP16]
  3-스레드 런타임: 콜백(5.33ms) ↔ 링버퍼 ↔ 추론 스레드, 안전장치 8종
[검증]
  P/S 실측 → 덕트 녹음·파인튜닝 → OFF/ON/OFF 평가 → 밴드별 감쇠 리포트
```

## 3단계 로드맵 (Stage-1 내부 게이트 분리 — docs/01 참조)

| 단계 | 모드 | 목표 | 성능 주장 범위 |
|---|---|---|---|
| **Stage-1A (현재)** | digital-ref, secondary surrogate | `P/S` 스케일을 맞춘 공칭 선형 플랜트에서 상쇄 역매핑과 학습 건전성 확립 | **표현 사전학습만**. 실제 덕트 감쇠·FxLMS 우위 주장 금지 |
| **Stage-1B** | digital-ref, measured P/S | noise→ERR `P(z)`와 cancel→ERR `S(z)`를 같은 출력 gain/볼륨으로 실측하고 recorded 데이터로 파인튜닝 | 독립 실측 val/test를 통과한 대역만 주장 |
| **Stage-2** | digital-ref 강건화 + acoustic-ref | 실측 다중 plant·비선형 커리큘럼, 외부 주기/준정상 소음 상쇄 | THD/IMD 및 다중 조건 실측 게이트 통과 후 주장 |
| **Stage-3** | acoustic-ref 광대역 | I/O 지연 단축 후 고역 확장 | 지연과 평면파 한계를 실측으로 검증한 범위만 주장 |

같은 코드베이스를 사용하지만 단계 전환에는 config 변경만으로 충분하지 않다.
Stage-1B에는 같은 장치 조건의 `P/S` 실측이, Stage-2 이후에는 비선형·다중 plant 측정과
독립 검증이 필요하다. 지연 규약은 docs/01, 판정 기준은 docs/07 §0을 따른다.

## 저장소 지도

```
Deep_ANC/
├─ configs/          # duct(덕트 실측), model, train, runtime, eval — 모든 파라미터의 단일 출처
├─ src/deep_anc/
│  ├─ dsp/           # 미분가능 S(z), 덕트 시뮬(영상법), 비선형, 필터
│  ├─ models/        # HybridANCNet (TCN/GLSTM/MHSA), 스트리밍/Export 래퍼
│  ├─ losses/        # ANCLoss (NMSE + MR-STFT×W(f))
│  ├─ data/          # 온더플라이 합성, 노이즈풀, 실측 데이터셋, manifest
│  ├─ train/         # Trainer(open/closed-loop, DDP), 체크포인트, 재현성
│  ├─ eval/          # 지표, 플롯, FxLMS 베이스라인
│  ├─ realtime/      # 3-스레드 런타임, 엔진 4종, 링버퍼, 안전장치
│  └─ baselines/     # anc_project fxlms_core.py 사본 (출처 명기)
├─ scripts/          # data / train / eval / elice / jetson / export / bench / demo
├─ tests/            # 자동 검증 (인과성, 등가성, 물리 재현, 누수, 복구 안전성)
├─ assets/measured/  # 측정 2차경로 npz (저장소에 포함)
└─ docs/             # 이 문서들
```

## 검증 상태

- 전체 pytest 통과: 인과성(미래 무의존), 스트리밍=오프라인 등가(실측 ~3e-8, 테스트 허용 1e-5), GLSTM 이중 경로 등가,
  덕트 시뮬 공진 70/210/350Hz 재현, 데이터 분할 무누수, S(z) torch=scipy 등가
- 학습 스모크: open/closed-loop 각각 Jetson GPU에서 정상 (bf16 AMP, 손실은 FP32)
- ONNX export → ORT 등가성 max err 2.4e-8
- 추론 지연: tiny+ORT CPU P99 **1.50ms** (블록 예산 5.33ms 통과), base+ORT 6.8ms
- 현재 Stage-1 설정: 공칭 선형 plant, `D_noise=1602`, `S_total=1718`, 실제 playback
  FIFO **lead 116**, trusted NMSE **150–1600Hz** + fullband 모니터.
  ⚠ 배포 중인 ONNX 는 실측 이전 값(`lead=109`, trusted 150–600Hz)으로 사전학습된 것이라
  런타임 설정도 109 다. 두 값이 섞이지 않게 런타임이 시작 전에 거부한다 (docs/06)
- 과거 `rir_surrogate` + 미관측 plant 위상 랜덤화 + fullband NMSE로 수행한 0dB 정체
  체크포인트는 학습 목적이 잘못된 실행으로 판정했다. 새 Stage-1에 resume하지 않는다.
