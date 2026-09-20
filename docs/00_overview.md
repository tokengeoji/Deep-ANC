# 00. 프로젝트 개요

## 목표와 현재 방향

기존 Jetson AGX Orin·사각 아크릴 덕트·마이크·USB DAC를 사용해
ERR 마이크 한 점의 소리를 실시간으로 상쇄하는 시스템이다. 성공 조건은
저역과 고역을 모두 감쇠하고, 소음뿐 아니라 음성·음악까지 다루는 것이다.
한 점의 감쇠를 덕트 단면 전체의 quiet zone으로 확대해 해석하지 않는다.

현재 우선순위는 외부 소리를 실제 REF 마이크로 받는 **acoustic-ref**이며,
Jetson의 우선 개선 대역은 **1000–1600Hz**다. 시스템 전체의 저역 목표도 유지한다.
자기생성 소음의 원본을 미리 아는 digital-ref는 별도의 비교·학습 경로다.

제어는 FxNLMS 기준선에서 출발해 인과 신경망과의 결합을 검증한다.
기존 하드웨어를 유지하며, 장치 교체나 시스템 설정 변경을 해결책으로 삼지 않는다.
외부 DSP와 Jetson의 출력 합산·스피커 공유 방식은 미정이다. 저장소의
`HybridEngine`은 한 Jetson 안의 DNN+FxNLMS 결합이며 외부 DSP 통합 구현이 아니다.
현재는 사용자 지시에 따라 하드웨어 연결을 보류하고 **SFANC의 필터 계산 검증과 오프라인
선택기 사전학습**을 우선한다. 구현 범위·실행 결과는 [docs/18](18_sfanc_pretraining.md)을 따른다.

## 하드웨어와 신호 경로

덕트는 길이 약 1.2m, 내측 단면 0.105×0.105m다. 좌표와 측정 경로는
[configs/duct.yaml](../configs/duct.yaml)이 단일 출처이며 ERR 위치 1.100m는 잠정값이다.

| 요소 | 역할 |
|---|---|
| REF 마이크 | 외부 소리를 먼저 관측하는 제어 입력 |
| 상쇄 스피커 CS | 제어기 출력 `y`를 재생 |
| ERR 마이크 | 잔류음 `e = d + S·y`를 관측 |
| 소음 스피커 NS | digital-ref 또는 명시적인 측정에서 내부 소음을 재생 |
| Jetson | 2채널 입력·출력과 3스레드 제어 런타임 실행 |

acoustic 기준선은 내부 소음 OFF, ANC OFF로 시작한다. 소리가 나는 측정과 실행은
사용자 입회·볼륨 최소 상태에서만 한다.

## 구현과 남은 검증

| 경로 | 저장소의 상태와 필요한 증거 |
|---|---|
| acoustic FxNLMS | `configs/runtime_acoustic.yaml`. 실측 S를 고정하고 제어 FIR만 적응하는 기준선 |
| acoustic DNN+FxNLMS | `configs/runtime_acoustic_hybrid.yaml`. 결합 API는 있지만 acoustic 모델 경로는 placeholder |
| 준비 FIR+FxNLMS | 기존 오프라인 API와 비학습 PSD bank. live 연결은 미완료 |
| SFANC 선택기 사전학습 | OMAP 16 kHz 실측 S 원본 500탭·합성 P·소스 음원으로 FP64 FIR 계산 및 REF-only CNN 선택기 학습. 계수 생성기·실측 미세조정·실시간 연결은 미완료 |
| acoustic 학습 준비 | `configs/data_acoustic_prepared.yaml`. 실제 원본·manifest·QA가 필요하며 누락 데이터를 합성원으로 대체하지 않음 |
| digital 비교 | 실측 P/S와 surrogate 모드를 별도로 지원. 기존 digital 모델을 acoustic 모델로 사용하지 않음 |

기존 **48 kHz NPZ 경로**에서 채택한 S의 검증 대역은 **150–600Hz**다. 고역 목표를 설정한 것만으로
1000–1600Hz 경로가 검증되거나 해당 대역의 학습·감쇠가 완료된 것은 아니다.
저장 S 지연과 핸드오프를 합친 35.854ms는 기하 기반 REF→ERR 선행 2.915ms보다 길다.
저장 지연의 미해결 pre-roll 문제는 [docs/01](01_physics_limits.md)을 따르며,
이 48 kHz 예산을 별도 OMAP 16 kHz 경로에 적용하지 않는다.
예측 가능한 성분의 성과를 임의의 음성·음악·광대역음 전체로 일반화하지 않는다.

하드웨어 연결을 재개한 뒤의 판단은 S/F 경로와 실제 선행 시간 확인, acoustic 기준선의 독립 녹음,
동일 조건의 하이브리드 비교 순서로 한다. 최신 환경·완료 결과·다음 작업은
[HANDOFF.md](../HANDOFF.md), 실행 위치별 조건은 [docs/14](14_pc_jetson_workplan.md)에만 유지한다.
과거 Elice 학습이나 Jetson 벤치마크를 현재 실행 상태로 간주하지 않는다.

## 저장소 지도

| 위치 | 내용 |
|---|---|
| `configs/` | 물리·모델·데이터·학습·실행·평가 설정 |
| `src/deep_anc/dsp/`, `losses/` | 측정 S, 합성 덕트, 비선형과 잔류음 손실 |
| `src/deep_anc/models/`, `train/` | HybridANCNet, 스트리밍 상태, 학습·체크포인트 |
| `src/deep_anc/data/`, `eval/` | 원본/manifest 로더, 합성·실측 데이터, 오프라인 진단 |
| `src/deep_anc/realtime/` | 엔진, SPSC 링버퍼, 녹음과 안전장치 |
| `assets/measured/` | 채택한 P/S 측정 NPZ |
| `scripts/`, `tests/` | 준비·측정·학습·분석 도구와 회귀 검사 |
| `docker/` | Docker 환경과 컨테이너 실행 안내 |

코드 읽기·수정·Python·테스트는 Docker 안에서 한다.
데이터 원본의 최종 보관소는 Google Drive이며, 임시 다운로드는 업로드 확인 후 정리한다.
자세한 규칙은 [AGENTS.md](../AGENTS.md), 실행법은 [docker/README.md](../docker/README.md)를 따른다.

설계 변경 전에는 [지연 규약](01_physics_limits.md), [모델 계약](04_model_architecture.md),
[평가 기준](07_evaluation_protocol.md), [acoustic 실행 계획](13_acoustic_hybrid.md)을 확인한다.
