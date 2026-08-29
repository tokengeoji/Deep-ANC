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

현재 단계는 최종 성능을 주장하는 단계가 아니라 파인튜닝 준비 계약을 복구하는 단계다.
기존 P/S와 checkpoint는 strict provenance와 150–1600 Hz 계약을 만족하지 않아
diagnostic-only다. 새 strict P/S, 데이터 계보, Elice public corpus와 canonical tiny 100k init을
모두 확보하기 전에는 surrogate checkpoint의 dB를 실제 덕트 감쇠로 해석하지 않는다.
단계별 PASS/FAIL/BLOCKED 의미와 절대 중단 조건은
[canonical 파인튜닝 강제 가드레일](16_canonical_finetune_guardrails.md)을 따른다.
2/4/8 kHz까지의 최종 목표와 matched FxLMS 우위·공간 검증은
[광대역 Deep-ANC 강제 가드레일](18_broadband_anc_guardrails.md)을 추가로 따른다.
125 Hz 옥타브의 하단 누락을 복구한 최종 역할은
[광대역 v3 전 옥타브 계약](27_broadband_v3_full_octave_contract.md)을 따른다. 기존
150 Hz 하단 광대역-v2는 구조·진단 역할로 보존하며 125 Hz 성능 근거로 승격하지 않는다.

## 시스템 전체 그림

```
[학습 — Elice Cloud A100 80GB 1장, tiny open-loop]
  공개 노이즈·음성·음악 + 합성원 → 연속 source n
                    ├→ strict P/S timing contract가 ref/playback 정렬을 유도
                    └→ surrogate pretrain 후 strict measured P로 fine-tune
  HybridANCNet → y → S(z)+256-sample handoff → e=d+S·y
                    └→ trusted NMSE(150–1600Hz) 최적화 + fullband NMSE 감시
  결과: 전체 계약 SHA와 completion receipt를 가진 canonical checkpoint
[배포 — Jetson AGX Orin]
  실측 파인튜닝을 통과한 best.pt → ONNX(정적 스트리밍 그래프) → [ORT CPU | TensorRT FP16]
  3-스레드 런타임: 콜백(5.33ms) ↔ 링버퍼 ↔ 추론 스레드, 안전장치 8종
[검증]
  P/S 실측 → 덕트 녹음·파인튜닝 → OFF/ON/OFF 평가 → 밴드별 감쇠 리포트
```

## 3단계 로드맵 (Stage-1 내부 게이트 분리 — docs/01 참조)

| 단계 | 모드 | 목표 | 성능 주장 범위 |
|---|---|---|---|
| **준비(현재)** | strict P/S·계보·Elice | 모든 입력 byte와 timing/checkpoint 계약 복구 | 성능 주장 금지 |
| **Stage-1A** | digital-ref, secondary surrogate | G0와 loss pilot 뒤 tiny 100k canonical pretrain | 표현 학습만 |
| **Stage-1B** | digital-ref, measured P/S | recorded 70%+synthetic 30% open-loop 50k와 one-shot G4 | 독립 G4 PASS 범위만 주장 |
| **Stage-2** | natural-crest challenge | 완전 미사용 4-family challenge | challenge PASS 뒤에만 배포 후보 |
| **광대역-v2(진단)** | digital-ref, 150–11.314kHz | 기존 구조·fixture 보존 | 125Hz 옥타브 전체 성능 주장 금지 |
| **광대역-v3(최종)** | digital-ref, 88.388Hz–11.314kHz | 125Hz–8kHz 전 옥타브 + 저역 guard + 고역 matched FxLMS 우위 + 다점 검증 | 별도 v3 P/S·데이터·G4 PASS 범위만 주장 |
| **후속** | closed-loop/acoustic-ref | 다중 plant·비선형·외부 소음 | 별도 실측 게이트 뒤 주장 |

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
- 현행 timing은 strict P/S NPZ의 bulk delay·compact FIR peak와 256-sample handoff를
  `TrainingTimingContract`가 유도한다. 수동 delay/lead 숫자는 학습 계약으로 인정하지 않는다.
- 기존 ONNX와 corrected checkpoint는 legacy diagnostic artifact이며 새 init/resume/배포에
  사용하지 않는다.
- 과거 `rir_surrogate` + 미관측 plant 위상 랜덤화 + fullband NMSE로 수행한 0dB 정체
  체크포인트는 학습 목적이 잘못된 실행으로 판정했다. 새 Stage-1에 resume하지 않는다.
