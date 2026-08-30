# docs/11 — v2 로드맵 (초안)

작성: 2026-08-03. 원 설계서(4개 전문 조사 종합)를 3개 독립 검증 렌즈(①물리·수치 검증, ②배포 제약 적대 검증, ③학습 실현성 검증)로 교차검증한 결과의 종합.
**채택 규칙: 3개 렌즈 중 하나라도 기각한 항목은 로드맵에서 제외**하고 말미의 "검토 후 제외" 절에 사유와 함께 기록한다(다음 세션 재검토 방지). 조건부 기각(시점·사양 수정으로 해소되는 것)은 수정판을 해당 단계에 조건 명시 후 편입했다.

불변 전제: 블록 256샘플=5.33ms, hop 128=2.67ms, 평면파 컷오프 1633Hz,
Jetson P99<3ms, ONNX opset17 정적그래프·상태 명시 텐서·TRT 화이트리스트
(DFT/Loop/Gather 금지). P/S bulk delay, compact FIR peak, 256-sample handoff, digital lead와
acoustic prediction horizon은 숫자로 복사하지 않고 strict NPZ의 `TrainingTimingContract`에서
유도한다. 2026-08-05 수치는 legacy 진단 기록이며 현행 official 계약이 아니다.

## 0. legacy corrected Stage-1의 교훈

기존 `rir_surrogate`+미관측 랜덤 plant 학습은 P/S 스케일 불일치와 위상 경사
상쇄로 0 출력 해에 머물렀다. corrected Stage-1은 아래 기준선으로 교체됐다.

- `secondary_surrogate`: `P(z)=S(z)`로 gain/FIR 스케일을 맞춘 표현 사전학습
- 당시 수동 digital lead와 playback FIFO를 구현했지만 현행 strict P/S 계약과 다르다
- 공칭 선형 plant: η=10, drive=1, hardclip=0, delay/gain/tilt/all-pass 섭동 OFF
- trusted NMSE로 최적화/체크포인트 선택, fullband NMSE를 do-no-harm 지표로 동시 로깅
  (**2026-08-05 플랜트 복구로 trusted 대역이 150–600Hz → 150–1600Hz 로 확대**)
- checkpoint의 resolved config·`physics_status`·lead, ONNX JSON lead 메타, 런타임 mismatch fail-fast(legacy=0)

이 체크포인트들은 **diagnostic-only**이며 현행 init/resume 자격이 없다. 새 strict P/S,
historical provenance와 Elice manifest를 결속한 계약으로 tiny 100k를 처음부터 학습하고,
학습에 쓰지 않은 recorded test를 통과하기 전에는 물리 성능을 주장하지 않는다.

---

## A. corrected Stage-1에 반영된 기반 작업·파인튜닝 선결 게이트

### A1. SEF η 물리 스케일 재보정 — Stage-2로 이연 (3/3 승인, P0-1)
`|y|≤0.2`에서 η≥10은 사실상 선형이다. corrected Stage-1은 이를 의도적으로
선택해 선형 역매핑을 먼저 학습한다. η∈{0.05,0.1,0.2}와 drive/hardclip은 A3의
THD/IMD 실측을 통과한 뒤 Stage-2 커리큘럼으로 점진 투입한다. 학습 그리드 밖
η=0.15 held-out NMSE는 그 때 병기한다.

### A2. acoustic-ref 학습 데이터 레시피 보강 (3/3 승인, P0-2)
전용 `source_mix_ratio_acoustic`과 예측 불가 성분 포함은 구현 완료했다.
현 run은 `reference_mode: digital`이다. acoustic-ref 분기의 상태는 다음과 같다.

- **반영 완료**: `configs/data_sim.yaml` acoustic-ref 모드에
  - 예측 불가 광대역 성분 명시 포함(MSE 최적해의 출력 0 수축=do-no-harm은 학습 분포에 그 성분이 있어야만 학습됨),
  - 덕트 공진(210/350/489/629Hz) 협대역 여기·고Q 색잡음 비중 확대.
- **남은 작업**: 현 W(f)는 80–800Hz 광역 가중이다. 공진별 세부 가중은
  별도 ablation으로 추가하고, 고Q 공진(Δf≤10Hz, τ≈1/(πΔf)≥30ms)에서만
  acoustic-ref 예측 이득이 있는지 검증한다.

### A3. 실측 게이트 3종 — 전체 로드맵 최우선 측정 작업 (3/3 승인, P2-1)
- ① **스피커 THD/IMD 실측**: 동기화 스윕사인 + 멀티레벨(−24~0dBFS).
  `/home/capston/anc_project/calibrate_s_path.py`는 읽기 전용 참고로만 사용하고 필요한 코드는
  이 저장소로 복사해 출처를 남긴다. THD 수% 미만이면 신경 플랜트 등 고비용 G_nl 작업을
  회피한다 — C단계 G_nl 투자 규모 결정 게이트.
- ② ~~**광대역 S(z) 재보정 150–600Hz → 80–1600Hz**~~ → **2026-08-05 해소.** 동시 인터리브 측정 + 오염 반복 기각으로 `consistency_band_hz` 가 **150–1600Hz** 가 됐다(150–1600Hz 일관성 P 0.9993 / S 0.9990). 남은 한계는 **80–150Hz** 뿐이다(클린 후에도 S 0.706~0.758 — 진짜 물리 한계). docs/02 §4.
- ③ **digital-ref P(z) 실측**: 공용 interleaved strict probe에서 같은 stream·반복으로
  noise→ERR P와 cancel→ERR S의 raw/analysis/compact NPZ를 함께 만든다. config에는 P/S NPZ
  경로만 두며 delay·FIR peak·handoff·lead는 `TrainingTimingContract`가 유도한다.
- [Novak et al., Synchronized Swept-Sine, JAES 2015]

위 실측은 모두 사용자 입회·앰프 볼륨 최저·ANC OFF 시작 상태에서만
한다. Jetson 시스템·pinmux·I²S 설정은 변경하지 않는다.

### A4. 평가 정직화 — 강튜닝 베이스라인 병기 (3/3 승인, P2-2)
고차(2048탭급) 인과 Wiener 상한 + 강튜닝 FxLMS/FsLMS를 실측 경로에서 비교표에 병기.
평가는 이미 이 저장소로 출처와 함께 복사한 `src/deep_anc/baselines/fxlms_core.py`를 사용하며,
`~/anc_project`는 수정하지 않는다. 평가 전용 — 학습 무관.
- [WaveNet-Volterra, arXiv:2504.04450, MSSP 2025의 "저차 베이스라인 비판" 선제 차단]

### A5. 지연 회계 감사 종결 기록 (3/3 승인)
revised-OLA 감사는 **불필요 확인 완료**: docs/04 기준 디코더 OLA는 과거 프레임 꼬리만 합산(알고리즘 지연 0), `test_model_shapes.py` streaming equivalence 비트단위 테스트로 검증됨. ARN의 revised OLA가 해결하는 숨은 지연(L−J)은 본 구조에 없음 — 본 기록으로 종결.

---

## B. corrected Stage-1 후 — 실측 파인튜닝 + 배포 v1.1

### B0. 파인튜닝 진입 게이트

1. 동일 stream/앰프/I/O 조건의 strict `P(z)`/`S(z)`와 immutable raw provenance를 확보한다.
2. 기존 82세션의 historical source를 재현하고 shared clip/artist/album/speaker/book의
   transitive component를 절대 나누지 않는 split을 만든다. family별 val/test component가
   각각 4개 미만이면 추가 녹음 전까지 차단한다.
3. `digital_primary_path_mode: measured`와 strict `primary_path_npz`를 선택한다. timing은
   NPZ-derived contract만 사용하고 legacy corrected checkpoint는 init/resume하지 않는다.
4. 독립 val/test에서 trusted NMSE(현 150–1600Hz)와 fullband NMSE를 항상 동시
   제시한다. 합성 offline·실기 session 평가는 S(z) 실측대역∩덕트 목표대역에서
   trusted/fullband/간극을 Markdown+NPZ에 자동 저장하며 기존 소스별·옥타브 지표도
   유지한다. 다만 Trainer val은 합성 최대 16개뿐이며 recorded val/test evaluator까지
   구현하기 전에는 진입 게이트를 완료한 것이 아니다.

### B1. 샘플 단위 delay-compensated training (3/3 승인, P1-1)
digital-ref의 연속 source K샘플 선행 슬라이스는 합성/실측 데이터셋에 구현됐다.
남은 작업은 실측 P/S로 K를 재산출하고 0∼수백 샘플을 스윕해 **K–NMSE 마진
곡선**을 만드는 것이다. acoustic-ref의 약 1458샘플 예측 타겟은 digital FIFO와
다른 연구 경로이며, 현 코드는 acoustic 모드의 nonzero digital lead를 거부한다.
- [ARN TASLP 2023: 16kHz 6ms 선행까지 무손실 실증 — digital-ref 2.3ms는 안전 구간]

### B2. digital-ref 선행 공급 — 예측을 조회로 변환 (3/3 승인, P1-2)

digital-ref 소음은 Jetson 자기생성 신호이므로 실제 재생을 FIFO로 늦춰 strict P/S가 요구한
선행량을 제공할 수 있다. 정보 창조나 녹음 미래 참조가 아니다. 합성 branch의 총 선행량과
실측 branch의 세션 정렬 잔여를 `TrainingTimingContract`가 같은 timeline으로 맞춘다.

lead는 checkpoint/ONNX 메타에 계약 SHA와 함께 저장하고 runtime mismatch를 오디오 시작 전에
거부한다. 과거 수동 lead artifact는 legacy로만 해석하며 새 학습이나 배포에 사용하지 않는다.

### B3. 비선형 커리큘럼 + worst-case 샘플링 (3/3 승인, P1-3)
η/drive 선형→강비선형 어닐링 + 배치마다 NL 파라미터 k개 추첨 후 손실 최대인 것으로 역전파(민맥스). 트레이너 수 줄, 플랜트 k회 재적용만 추가라 저비용.

### B4. NOAS 2단계 손실 파인튜닝 (3/3 승인, P1-4)
샘플별 오프라인 경사하강으로 근최적 y* 생성 → NMSE[S(G_nl(y*)), S(G_nl(y))] 파인튜닝. 추론 그래프 불변. **고정 파인튜닝셋 사전 생성 조건**(렌즈 ③ — 내부 최적화는 미분가능 FIR+SEF 플랜트라 A100 배치 처리 가능). 기대 이득은 보수적으로 1~2dB.
- [DeepASC, arXiv:2502.01185]

1~2dB는 문헌에서 가져온 실험 가설이지 현 장비의 성능 주장이 아니다. B0의 measured P/S와
독립 recorded test에서 확인되지 않으면 해당 이득을 보고하지 않는다.

### B5. S(z) 섭동 증강의 물리 기반 재구성 — **P0에서 P1로 이연** (조건부 편입)
공칭 학습에서 모델 입력으로 관측할 수 없는 delay/all-pass를 매 batch 독립 추첨하면 위상
경사가 서로 상쇄되어 다시 영출력 해로 간다. 따라서 corrected Stage-1은 모든 섭동을 0으로
고정했다. 파인튜닝에서는 다음 조건을 모두 만족할 때만 점진적으로 다시 켠다.

- **선행 조건**: A3-② 광대역 S(z), A3-③ measured P(z), 여러 세션의 실제 지연 분포 확보
- 공칭 measured plant에서 먼저 수렴시킨 뒤 작은 범위→실측 범위 순서로 curriculum 적용
- delay/all-pass를 임의 독립 난수로 주지 않고, plant ID/조건 임베딩 또는 세션별 고정 plant로
  모델이 식별할 수 있게 함
- 추가 축은 온도·마이크 위치·대역별 gain/phase처럼 측정으로 뒷받침된 범위만 사용
- trusted/fullband NMSE와 nominal/perturbed held-out를 동시에 통과하지 못하면 증강을 되돌림

### B6. 배포 v1.1 — 런타임 병렬 적응층 (3/3 승인, P2-3; 재학습 불필요, CPU, 추론 그래프 불변)
- ① 병렬 FxNLMS 잔차 브랜치: y = y_dnn + y_lin (DNN=비선형·광대역, FxLMS=협대역 잔차·정상상태). **보수적 스텝사이즈 + 발산 감시 게이트 필수(조건).** 실측 FxLMS ~0.2ms/블록으로 여유 확인됨.
- ② S-드리프트 보정 포스트 FIR C(z): 32~64탭, 항등 초기화, 블록레이트 LMS.
- ③ error-jump 검출기: 잔차 파워 점프 시 C(z)/FxNLMS 리셋 → 재캘리브레이션 트리거.
- [Luo, Shi, Gan, Hybrid SFANC-FxNLMS, IEEE SPL 2022; Kuo & Morgan]

---

## C. v2 실험 — 구조 변경 (별도 학습, v1 run과 무충돌)

### C0. 선결 조건 (모든 렌즈 공통 지적)
1. **파라미터·연산 예산 전면 재산정**: 원 설계 §2.3 표는 산술 오류로 기각됨(제외 절 참조). 인코더는 10ch 풀레이트 직결 금지 — **밴드 채널 별도 소형 투영(1×1 프리믹스 10→2ch) 또는 밴드 decimation으로 재사양** 후 재산정. 룩백 24ms 확장은 in=2ch 기준 규모(≈1.18M)로만 한정 승인.
2. **v1-base TRT FP16 실측 선행**: 현행 배포 스택(ORT CPU)에서 v1-base P99 6.8ms로 게이트 탈락 상태(tiny 1.44ms@MAXN 만 통과, docs/06). **(2026-08-06 정정)** TRT python 바인딩 10.3.0 은 venv 에 있고 FP16 plan 실측도 끝났다 — **이 사유는 해소됐다**(docs/06). 남은 제약은 GPU 지연 실측이 **듀티 100% 연속 실행** 조건에서 나온 값이라는 것이며, 듀티 6% 주기 호출 벤치 전에는 v2-base 채널 규모를 확정할 수 없다.

### C1. 승인된 그래프 요소 (전부 화이트리스트 op 조합·정적 shape·상태 명시 텐서·알고리즘 지연 0)
- **4밴드 분석 필터뱅크**: 실계수 인과 Conv1d(전대역+0–400+400–900+900–1633Hz), DFT 불사용, 상태 st_fb. 컷오프 이하 용량 집중 — 물리 정합. (C0-1 사양 조건)
- **인코더 룩백 24ms 확장**: 좌측 패딩 전용. (in=2ch 한정, C0-1 조건)
- **Volterra 2차 곱셈 분기**: x⊙delay(x) 채널 + Conv — 과거 샘플만, 인과. GLU가 이미 elementwise Mul 사용으로 TRT 경로 검증됨.
- **FiLM S-조건화**: Ŝ 임베딩 [1,16] 추가 입력 텐서 → TCN 블록 γ,β. 증강이 섭동 파라미터를 이미 생성하므로 페어 데이터 무비용. 배포 후 임베딩만 갱신하는 가중치-불변 적응.
- **이중 헤드**: (a) Deep-Filtering 시변 인과 FIR 96탭(MatMul + st_ref 버퍼) (b) 잔차 파형 디코더(기존 유지) (c) 3밴드 σ 신뢰도 게이트(do-no-harm 학습 목표).
- **MHSA 주기 comb bias**: 과거 KV 윈도(64프레임=170ms) 내 주기 지연 근방 가산 bias — 인과.

### C2. acoustic-ref 주기성 전략 (§3 — 성분 분해 원칙 유지)
- **① 자기상관 주기 특징**: ref 과거 히스토리 버퍼 + 지연 행렬 MatMul, 후보 L∈[2ms,40ms] 로그 간격 ~64개, Gather 불사용 — 인과, 승인.
- **① 소프트 comb 예측 탭 — 수정판만**: 원 명세는 인과성 위반으로 기각(제외 절). 수정판: **softmax 지연 후보를 유효 지연 mT̂≥1578샘플(=실측 P)로 제한 + ref 버퍼 ≥ P+40ms(≈3378샘플 이상, 렌즈 ② 권고 ~2×P≈2916 이상)로 확장** — 정적 shape 유지. 이 형태로만 구현.
- **② 필터계수 예측 헤드**: 사전학습 서브필터 뱅크(M≈16) + 프레임율 α 결합(MatMul) — 공진 협대역·준정상 한정, 오차 피드백 비의존이라 발산 위험 없음. [GFANC arXiv:2303.05788; GFANC-Kalman SPL 2024]
- **③ 학습·안전장치**: 30ms 타겟 시프트 커리큘럼(순주기→변조→공진→혼합), 증폭 페널티(e>d 구간 가중 벌점) 게이팅 학습, **성분별 분리 평가(주기/공진/색잡음/백색 — 백색은 0dB 무해가 성공 기준)**.

### C3. G_nl v2 — 실측 Wiener–Hammerstein (손실 그래프 전용, 추론 비용 0)
프리엠퍼시스 FIR → Chebyshev 홀수 기저 Σg_k·φ_k → 분기별 FIR H_k(z), 전부 스윕사인 실측 + 실측 중심 도메인 랜덤화. **A3-① THD 게이트 통과 조건부** — 강한 메모리성 왜곡 확인 시에만 소형 LSTM 신경 플랜트 상향. >1633Hz 고조파는 상쇄 불가(고차 모드) — y_nl의 >1633Hz 에너지 페널티(자기 왜곡 방사 억제)로만 대응.
- [Novak JAES 2015; Wright & Damskägg 2020 / arXiv:2412.01092]

### C4. 두 시간스케일 런타임 (기본층만 — 실험 3종은 기각됨, 제외 절 참조)
빠른 층(5.33ms, TRT 정적): v2 네트워크. 느린 층(CPU, 블록~초): B6 구성 + FiLM 임베딩 갱신. 추론 그래프·지연 불변.

### C5. 게이트 실패 시 비상수단 사다리 (승인)
① GLSTM→FastGRNN급 경량 셀(MatMul/Sigmoid/Tanh — export 가능) [Fast-ULCNet arXiv:2601.14925] ② 저대역 decimated 스트림 분리 ③ v2-tiny 폴백 — 단 tiny 사이징 수치는 C0-1 재산정 후 확정.

### 불가능 명시 (유지 — 정직성 조항)
acoustic-ref 광대역 랜덤(상관시간 ≪ 30ms)은 본 로드맵의 어떤 요소로도 상쇄 불가 — 주기·공진·준정상 커버리지 확대 + 백색 성분 무해화(0dB)가 정직한 상한. 1633Hz 이상은 단일 CS/ERR로 제어 불가 — 자기 왜곡 방사 억제로만 대응. 광대역 인과 상쇄는 3단계 하드웨어(I2S 직결, 전기 지연 <2.77ms) 이후에만 가능(docs/01 유지).

---

## 검토 후 제외 — 재검토 금지 목록

| # | 항목 | 기각 렌즈 | 사유 | 재상정 조건 |
|---|---|---|---|---|
| R1 | §2.3 파라미터 예산표 (v2-base ≈8.4M / v2-tiny ≈1.8M) | 3/3 | **산술 오류 확정**: Conv1d(10→512, k=1152)=10×512×1152=5.90M인데 표는 1.18M(in=2ch 값)으로 기재 — 입력 채널 ×5 누락. 실제 v2-base ≈12–13M. tiny도 인코더 단독 2.95M이라 1.8M 불가능 | 인코더 재사양(C0-1) 후 전면 재산정 |
| R2 | §2.2 인코더 10ch 풀레이트 직결 사양 (10ch × k=1152 × 512out) | 2/3 (①②는 예산표 기각에 포함, ③ 명시 기각) | 다이어그램과 예산표가 상호 모순, 인코더 GMAC만 +2.1~2.2 GMAC/s로 예산 붕괴의 원인 | 1×1 프리믹스(10→2ch) 또는 밴드 decimation 재설계 |
| R3 | §2.4 지연 예산 수치 (v2-base 3.4 GMAC/s → TRT FP16 ≤2.4ms) | 3/3 | R1에서 파생된 과소 추정(실제 ≈4.5–4.6 GMAC/s = v1의 2배). **(2026-08-06 정정)** TRT 는 설치돼 있어 실측 가능하다 — 다만 현재 수치는 전부 듀티 100% 조건이다. 현행 ORT CPU 에서 v1-base 는 여전히 P99 6.40ms 로 게이트 탈락 중 | 사전 구성된 별도 TRT 환경 + v1-base FP16 P99 실측 통과 |
| R4 | v2-tiny ≈1.8M 폴백 "안전망" 주장 | 2/3 | R1과 동일 산술 오류. 명시 사양대로면 ORT CPU 추정 P99 ≈5ms로 게이트 탈락 가능성 높음 — 안전망으로 불성립 | 인코더 재사양 시 ≈0.6 GMAC/s로 통과 가능 — 재산정·실측 후 |
| R5 | 측정-e 기반 test-time training + TRT refit | 3/3 | **(2026-08-06 정정)** TRT python API 는 설치돼 있어 이 사유는 해소됐다. 남은 기각 사유는 다음 둘이다: 공유 Orin GPU에서 배경 학습이 P99<3ms 게이트와 자원 경합 — "지연 불변" 주장 검증 불가 + 30ms 지연 플랜트 경유 온라인 가중치 갱신은 발산 시 실제 소음 증폭 사고 직결. 승인된 저위험 대체재: FiLM 임베딩 갱신(C4) | 오프라인 검증 + 경합 격리 + 사전 구성된 별도 TRT 환경 |
| R6 | Meta-AF식 학습 옵티마이저로 FxLMS 갱신 교체 | 1/3 (③) | 미분가능 폐루프 시뮬 + 메타학습 인프라가 필요한 별도 연구 과제 — 2×A100 공유·캡스톤 일정에서 비용 대비 이득 불확실. B6의 고전 FxNLMS가 같은 역할을 검증된 비용으로 수행 | (사실상 종결 — 일정 여유 생길 경우만) |
| R7 | Mamba 오프라인 티처 증류 | 1/3 (③) | 티처 사전학습에 v1급 A100 시간 추가 소요, 설계 자신이 인용한 실증(Wu & Braun — 초저지연에서 Mamba 열세)상 티처 우위 근거 약함 | (사실상 종결) |
| R8 | MAML 파인튜닝 | 1/3 (③) | 이중 최적화 비용·불안정 + few-step 적응 실행 경로가 정적 TRT 그래프에 없음(온디바이스 갱신=R5 전제와 연동). 섭동 지식은 B5·FiLM 레시피로 흡수 가능 | (사실상 종결) |
| R9 | §3① 소프트 comb 예측 탭 — **원 명세** | 1/3 (①) | **인과성 위반**: y_comb=Σ w_l·ref(t−l+P)에서 l<P=30ms인 모든 후보가 미래 샘플 인덱싱(예: l=10ms → t+20ms). 버퍼 40ms는 이를 자인하는 크기, 정수배 보정 시 ~58–60ms 룩백 필요. 학습 시 오라클 누설 → 배포 성능 붕괴 위험 | **수정판은 C2에 편입 완료** (mT̂≥1458 제한 + 버퍼 ≥P+40ms) — 원 명세는 재상정 금지 |
| R10 | 공칭 Stage-1에 미관측 random delay/all-pass 즉시 적용 | corrected 진단으로 기각 | 입력에 plant 조건이 없는데 batch마다 위상이 달라져 경사 평균이 상쇄되고 영출력 해를 강화함. 기존 `[0,512]` 필수 유지 주장은 철회 | **B5 수정판만 허용**: measured 다중 세션+조건 식별+점진 curriculum |

---

## 핵심 참고문헌 (승인 항목에 결부된 것만)

1. Zhang & Wang, "Deep ANC", *Neural Networks* 2021 — SEF η 프로토콜 원전, 비선형 커리큘럼 관행. (A1, B3)
2. Zhang, Pandey & Wang, "ARN for ANC", *IEEE/ACM TASLP* 2023 — delay-compensated training 실증(6ms 선행 무손실, NMSE −11.20dB), 시간영역·소프레임 정당화. (B1, C2-③)
3. Wu & Braun, arXiv:2409.10358 — 비대칭 창(분석 룩백 확장), filtering>mapping, 예측형의 실측 일반화 악화, 초저지연 Mamba 열세. (B2, C1)
4. Luo, Shi, Gan, "Hybrid SFANC-FxNLMS", *IEEE SPL* 2022 — DNN+적응필터 역할 분담. (B6)
5. Kuo & Morgan, *Active Noise Control Systems* — 위상 오차 ~90° 발산 한계. (B6)
6. Novak et al., "Synchronized Swept-Sine", *JAES* 2015 — THD/IMD·WH 플랜트 실측법. (A3, C3)
7. Zhang et al., "DeepASC", arXiv:2502.01185 — NOAS 2단계 손실, 밴드 분해 단조 개선. (B4, C1)
8. GFANC, ICASSP 2023, arXiv:2303.05788 + GFANC-Kalman, *IEEE SPL* 2024 — 필터계수 예측(문제 변환), 프레임 간 가중치 안정화. (C2-②)
9. Valin et al., "PercepNet", *Interspeech* 2020 — pitch-comb 주기 조건화의 우월성. (C2-①)
10. Perez et al., "FiLM", *AAAI* 2018 + Oh et al., *MSSP* 2024 — 조건화 임베딩 기반 가중치-불변 적응, 주행 중 S 갱신. (C1)
11. WaveNet-Volterra, arXiv:2504.04450, *MSSP* 2025 — Volterra 곱셈 사전, 강튜닝 베이스라인 비판. (A4, C1)
12. Wright & Damskägg 2020 / arXiv:2412.01092 — 실측 신경 플랜트 동결 보상 학습(THD 4.55%). (C3 상향 옵션)
13. Fast-ULCNet, arXiv:2601.14925 — 경량 순환 셀 교체(크기 ½·지연 −34%). (C5)
14. arXiv:2601.13849 — S 섭동 다양성(2차>1차), 감지-리셋 루프. (B5, B6-③)
15. Deep PLC Challenge, *Interspeech* 2022 — 청감용 예측의 위상 부정밀 → 상쇄 부적합(방어적 게이팅 근거). (C2-③)
16. Ghasemi et al. 2016, THF-FxLMS — tanh 대리 정당화 + "η는 실측하라". (A1)
17. Kolmogorov–Szegő 정리 — 스펙트럼 평탄도=예측 가능성(성분 분해의 이론 근거). (A2, C2)

*기각 항목 전용 문헌(Meta-AF: Casebeer et al. TASLP 2023; Feng & So arXiv:2412.19471 등)은 재검토 방지를 위해 본 목록에서 제외.*
