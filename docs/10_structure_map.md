# 10. 코드·설정 지도

이 문서는 작업할 코드와 설정을 찾기 위한 지도다. 현재 실행 환경·시험 결과는
[HANDOFF.md](../HANDOFF.md), 우선 실행 순서는 [docs/14](14_pc_jetson_workplan.md)에 둔다.
모든 파일 작업과 Python 실행은 Docker 안에서 한다.

## 1. 현재 진입점

| 작업 | 진입점 | 해석 |
|---|---|---|
| acoustic 준비 진단 | `scripts/bench/check_acoustic_readiness.py` | 설정·S·기하선행을 읽는 무출력 검사. 기본 exit 0은 보고 성공 |
| acoustic 데이터 QA | `scripts/bench/check_acoustic_training_data.py` | 실제 manifest·RIR·QA와 train-only 정책 검사 |
| Jetson 설치 진단 | `scripts/bench/check_jetson_stack.py` | CUDA·cuDNN·ORT·TensorRT의 무오디오 검사. ANC 성능 시험 아님 |
| acoustic FxNLMS 기준선 | `deep_anc.realtime.run_realtime` + `configs/runtime_acoustic.yaml` | 외부 REF, 내부 소음 OFF, ANC OFF 시작 |
| acoustic 하이브리드 | 같은 런타임 + `configs/runtime_acoustic_hybrid.yaml` | DNN+FxNLMS API. acoustic artifact 경로는 placeholder |
| 녹음 분석 | `scripts/eval/analyze_acoustic_session.py` | OFF→ON→OFF 관측 감소량·대역·health 분석 |
| 경로 진단 | `scripts/eval/analyze_path_bands.py` | 저장된 경로 자료의 대역·위상·일관성 분석 |
| 준비 FIR 연구 | `scripts/bench/prepare_filter_bank.py`, `benchmark_prepared_fir.py` | 오프라인 후보·잔차 적응 비교. live 배선과 별개 |

출력이 가능한 측정·런타임은 사용자 입회·볼륨 최소 조건에서만 실행한다.
Docker 기본 컨테이너는 오디오 장치를 노출하지 않는다.

## 2. 설정에서 실행까지

`src/deep_anc/config.py`가 YAML과 CLI override를 읽고 참조 설정을 병합한다.
학습은 `model_config/data_config/duct_config`, 런타임은
`hardware_config/duct_config`를 병합한다. 참조 파일을 읽은 뒤 하위 override도 적용한다.
상대 경로는 현재 작업 디렉터리의 실제 경로를 먼저 확인하고 저장소 루트로 대체하므로,
재현할 때는 저장소 루트에서 실행하고 resolved 설정을 기록한다.

| 설정·키 | 주요 소비 코드 | 유지할 계약 |
|---|---|---|
| `duct.yaml: secondary_path.npz` | `dsp/secondary_path.py`, Trainer, 평가, 런타임 엔진 | S 파일을 학습·평가·실행에서 공유 |
| `secondary_path.handoff_extra_samples` | 미분가능 S, FxNLMS, readiness·평가 | NPZ 순수지연에 실제 핸드오프를 한 번만 추가 |
| `digital_reference.primary_path_npz/d_noise_delay_samples` | `data/primary_path.py` | measured P 지연과 설정이 일치해야 함 |
| `positions_m`, 덕트 기하 | `dsp/duct_sim.py`, readiness | 기하 추정을 실측 선행 시간으로 취급하지 않음 |
| `acoustics.realistic_target_band_hz` | Trainer와 손실·평가 | 기존 학습 가중 대역이며 프로젝트 전체 성공 대역과 구분 |
| `data_*.yaml: reference_mode/digital_reference_lead_samples` | 합성·실측 dataset, Trainer | acoustic은 lead=0. digital 메타와 런타임 정렬 일치 |
| `require_prepared_data/train_only_source_families` | `data/synth_dataset.py`, 준비 QA | strict 데이터 누락 거부, machine은 ANC train-only |
| `source_mix_ratio*`, `synthetic_target_band_hz` | 소스 선택·합성 생성기 | 데이터 분포와 S 검증 대역을 구분 |
| `model_*.yaml` | `models/hybrid_anc.py` | hop·상태 shape·io_scale·리미터는 artifact 계약 |
| `train_*.yaml` | `train/trainer.py` | 손실 FP32, resolved 설정·물리 상태·lead 보존 |
| `runtime_*.yaml` | `realtime/run_realtime.py`, `engines.py` | reference/controller/artifact 선택, ANC OFF 시작 |
| `hardware_jetson.yaml: audio.*` | `audio_io.py`, 런타임·측정 도구 | 장치별 소비 키를 코드에서 확인하고 S 측정 조건과 대조 |

acoustic 준비의 현재 데이터 설정은 `configs/data_acoustic_prepared.yaml`이다.
범용 `data_sim.yaml`의 합성 fallback 동작을 strict 준비 경로에 적용하지 않는다.
측정 지연·legacy digital 정렬은 [docs/01](01_physics_limits.md)에 정리되어 있다.

## 3. 학습·데이터 경로

| 모듈 | 역할 |
|---|---|
| `data/public_corpus.py` 등 준비 모듈 | 공개 음원 index·metadata·manifest·QA |
| `data/archive_*.py`, `drive_*.py` | 원본 확보·무결성·Drive 전송 기록 |
| `data/synth_dataset.py` | REF/d 합성, strict 준비 검사, split별 소스 정책 |
| `data/primary_path.py` | digital measured/secondary_surrogate/rir_surrogate 선택 |
| `data/recorded_dataset.py`, `recorded_qa.py` | 녹음 입력과 독립 분할·세션 QA |
| `dsp/duct_sim.py` | 합성 P_ref/P_err/F 경로. 실제 경로 측정을 대체하지 않음 |
| `models/hybrid_anc.py`, `models/streaming.py` | 파형 모델·상태·정적 export 래퍼 |
| `losses/anc_loss.py` | `e=d+S·G(y)`, trusted/fullband 손실·과출력 벌점 |
| `train/trainer.py` | open/closed-loop 학습, 검증·체크포인트 |

공개 원본의 최종 보관소는 Google Drive다. 준비·무결성·업로드 확인·임시 원본 정리 규약은
[docs/16](16_drive_acoustic_preparation.md)를 따른다. Drive 목록만으로
로컬 strict 학습 준비 완료를 선언하지 않는다.

acoustic 학습 설정 `train_acoustic_prepared.yaml`은 최대 2 step의 준비 확인용이다.
Elice 관련 스크립트의 존재는 현재 원격 학습이 실행 중이라는 뜻이 아니다.
학습 플랜트·상태·손실의 상세 계약은 [docs/04](04_model_architecture.md)를 따른다.

## 4. 실시간·배포 경로

`realtime/run_realtime.py`는 입력 callback, 추론 작업, UI를 분리하고
`ring_buffer.py`의 SPSC 버퍼로 블록을 전달한다.
생산자는 write 위치만, 소비자는 read 위치만 소유한다.

`realtime/engines.py`는 Torch/ORT/TensorRT/FxNLMS와 초기 HybridEngine을 제공한다.
mic DL/hybrid는 acoustic reference 메타가 있는 모델만 허용한다.
FxNLMS는 사전 측정한 S를 고정하고 제어 FIR을 적응한다.
HybridEngine은 한 Jetson의 파형 DNN+FxNLMS 결합이며 외부 DSP와의 합산 구현이 아니다.

`safety.py`와 런타임은 합산 출력 제한·페이드·발산·클리핑·출력 누락을 감시한다.
손상 구간과 지연된 ERR 이력이 적응에 섞이지 않게 보류하고,
mic 신경망 입력 손상은 초기화와 ANC OFF로 처리한다.
`recording.py`는 원시 배열과 S 해시·설정·health 메타를 저장하며 기존 파일을 덮어쓰지 않는다.

`scripts/train/export_onnx.py`는 checkpoint를 정적 ONNX와 메타로 내보낸다.
`scripts/export/build_trt.sh`는 제공된 `trtexec`를 사용하는 별도 엔진 변환 경로다.
TensorRT import 성공, 작은 진단 모델 실행, 실제 ANC 모델 export·지연·감쇠는
각각 별도로 검증한다. 도구 설치 여부를 이 문서에서 고정된 사실로 복제하지 않는다.

`baselines/prepared_fir.py`와 `filter_bank.py`의 오프라인 계수 연구는
live engine factory와 연결되지 않았다. 구조와 제한은 [docs/15](15_prepared_fir_research.md)를 따른다.

## 5. 측정·평가 경로

`scripts/data/record_duct.py`, `calibrate_wideband.py`, `measure_paths_interleaved.py`와
`scripts/bench/measure_duct_transfer_map.py`는 실제 장치를 사용하는 측정 도구다.
P/S/F, 블록·latency, 지연·클록 기준과 원시 녹음을 함께 보존한다.

`scripts/eval/evaluate_offline.py`, `evaluate_recorded.py`, `compare_fxlms.py`는
합성·녹음·기준선 비교를 맡는다. 실기 세션과 acoustic 관측 분석은
`scripts/demo/evaluate_session.py`, `eval/acoustic_session.py`로 구분된다.
독립 세션·소스 분할을 지키고 low/high·소스별·최악 구간을 함께 보고한다.

측정 NPZ의 `consistency_band_hz`와 `excitation_band_hz`는 서로 다른 증거다.
공용 S 로더는 consistency 메타를 우선 사용하고 구형 파일에는 fallback이 있으나,
acoustic 준비 진단의 고역 검증은 명시적인 신뢰대역·반복 일관성을 확인한다.
최종 판정과 관측 감소량의 한계는 [docs/07](07_evaluation_protocol.md)가 기준이다.
