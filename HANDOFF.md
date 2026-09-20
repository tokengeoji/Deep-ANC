# HANDOFF — 현재 상태와 다음 작업

최종 갱신: 2026-09-20. 이어서 작업할 때 이 문서부터 읽는다.
작업 규칙은 [AGENTS.md](AGENTS.md), 실행 절차는 [docker/README.md](docker/README.md),
현장 체크리스트는 [docs/14](docs/14_pc_jetson_workplan.md)가 단일 출처다.
이 파일에는 현재 사실만 남긴다. 과거 세션 일지·PID·실험 수치는 Git 이력에서 확인한다.

## 1. 사용자 확정 사항

- acoustic-ref 최우선. 외부 소리를 실제 REF 마이크로 받으며 소음 원본·미래 샘플을 사용하지 않는다.
- ERR 한 점의 감쇠가 기준이며 고역은 1 kHz 이상이다. **Jetson 우선 대역은 1000–1600 Hz**다.
- 시스템 전체의 저역·고역 동시 감쇠와 음성·음악까지 제거하는 quiet zone 목표를 유지한다.
- 현재 Jetson AGX Orin·덕트·마이크·USB DAC를 유지한다. 교체·구매를 해결책으로 가정하지 않는다.
- Docker 내부에서만 코드·문서·Git·Python·테스트 작업을 한다. 호스트 `.venv`는 사용하지 않는다.
- 기존 RT 커널·30W·핀·오디오·시스템 패키지는 변경하지 않는다. Docker 환경 관리만 승인됐다.
- 오디오 출력은 사용자 입회·볼륨 최소 상태에서만 한다. 이번 환경·문서 작업은 무오디오다.
- 원본의 최종 보관소는 Drive. 임시 다운로드는 검증·업로드 확인 후 해당 임시 원본만 삭제한다.
- MIMII DG fan은 **학습 보조 전용**이다. machine은 train에만 쓰고 val/test에서는 제외한다.
- 커밋·push는 승인됐다. 통합 대상은 `https://github.com/tokengeoji/Deep-ANC.git`의 `main`이다.
  `tokengeoji/DeepANC`는 가져온 별도 원본 저장소이며 두 이름을 혼동하지 않는다.
  작성자는 `SEUNG JOON JEONG <155646237+tokengeoji@users.noreply.github.com>`을 사용한다. AI 표기는 넣지 않는다.

## 2. 통합 작업 환경과 기존 Jetson 검증

이번 저장소 통합은 **x86 PC의 CPU Docker**에서 수행했다. 이 작업에서 Jetson CUDA나
실제 오디오를 실행하지 않았다. 아래는 통합 전 `Deep-ANC`에 기록된 실제 ARM64
Jetson AGX Orin / L4T R36.4.4 검증 이력이며, 현재 접속 장치는 재개할 때 다시 확인한다.

통합본 CPU Docker 검증: **1282 passed, 2 skipped, 5 subtests passed (79.50초)**.
두 skip은 현장 raw 진단 파일 및 실측 `metrics.md`가 없는 조건이다. Python 3.10.21 /
PyTorch 2.5.1+cpu에서 `pip check`, 원본 500탭 검증, 준비 스크립트의 합성 학습·checkpoint 저장을
통과했다. CUDA는 검증하지 않았고 녹음·스피커 출력도 수행하지 않았다.

### 기존 Jetson Docker 구성 기록

- `deep-anc-jetson-local:dev` 이미지 빌드 및 `deep-anc-dev` 생성 완료.
- CUDA 12.6 런타임 기반의 작은 이미지 + 전용 venv 볼륨에 의존성을 설치하는 방식이다.
  큰 venv의 이미지 export·unpack 중복을 피한다. 기본 `jetson` 이미지 정의는 별도로 유지한다.
- 호스트 cuDNN 9·TensorRT 10 라이브러리와 TensorRT Python 패키지만 읽기 전용으로 연결한다.
  호스트 전체 Python·libc 경로를 연결하지 않으며 시스템 파일은 수정하지 않는다.
- `--runtime nvidia --network host`, 비특권 ancdev 사용자, 오디오 장치 미노출 구성이다.
- 의존성 설치와 CUDA/cuDNN/ORT/TensorRT 연산 검증 완료. 이미지 ID는 `7f1fbcd3269e`다.
  `.venv`는 `deep-anc-jetson-local-venv-7f1fbcd3269e` 볼륨이며 재시작해도 유지한다.
  libgomp·tzdata 누락과 고정 cuSPARSELt wheel의 preload 경로를 수정한 최종 이미지다.

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_jetson_stack.py
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q -o addopts= -ra
```

기존 Jetson처럼 Docker 인증이 필요한 호스트에서는 위 관리 명령에 `sudo`를 붙인다.
중지 환경은 `start`로 재사용한다. 매번 이미지·컨테이너·볼륨을 삭제하지 않는다.
새 환경의 최초 설치·버전·볼륨 규약은 Docker 문서를 따른다.

### 통합 전 Jetson 검증 범위 (기존 48 kHz 경로)

- 최종 Jetson Docker 전체 회귀: **1228 passed, skip/실패 없음 (191.47초)**.
  명령은 `pytest -q -o addopts= -ra`, 로그는 `results/pytest_jetson_20260920_02.log`다.
- `pip check`·Asia/Seoul 시간대 조회·셸 문법·diff 공백 검사 통과.
- `stop`→`start` 후 venv 보존·CUDA 사용 가능·ORT/TensorRT import를 다시 확인했다.
- `runs/jetson_stack_20260920_05.json`: 최종 이미지의 PyTorch CUDA 행렬곱·cuDNN·
  ONNX Runtime 1.18.1 CPU·TensorRT 10.3 FP32 전항목 PASS. 최대 TensorRT 오차 `5.96e-8`.
  PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, cuDNN 9.3이다.
  고정 난수 Conv1d 설치 진단이지 ANC 학습·실시간 마감·감쇠 검증은 아니다.
- 기존 `runs/export/tiny_corrected.onnx`: 실제 `OrtEngine` 20블록과 독립 ORT 상태 전달,
  reset 후 재현의 출력·상태 오차 0. 상태 12개 갱신과 원본 SHA 불변을 확인했다.
  `results/jetson_project_ort_20260920T041416462142Z.json`에 기록했다.
  legacy digital lead109이고 reference_mode 키가 없으므로 acoustic 배포 모델로 인정하지 않는다.
  현재 PyTorch 모델과의 등가성·지연 성능은 이 검사에 포함되지 않는다.
- 패키지 목록은 `results/jetson_packages_20260920.txt`에 있다. 산출물은 Git 미포함이다.
- 실제 오디오 I/O 지연·고역 S·새 acoustic 감쇠는 이번 작업에서 측정하지 않았다.

## 3. 구현된 범위와 물리적 한계

- `runtime_acoustic.yaml`: REF mic·내부 소음 OFF·ANC OFF 시작의 FxNLMS 기준선.
- `runtime_acoustic_hybrid.yaml`: 한 Jetson 내부 DNN+FxNLMS 비교 구성.
  acoustic 배포 artifact는 미확정이며 기존 digital/모드미상 모델을 대신 쓰지 않는다.
- REF/ERR 오류, 클리핑·xrun·출력 누락 후 적응 보류/초기화/ANC OFF 규약과 합성 회귀가 있다.
- 녹음 schema v1은 설정·S SHA·누적 health를 기록한다. 분석 CLI는 새 경로에 진단 보고를 저장한다.
- 사전 FIR·REF-only 선택기·필터 뱅크·strict acoustic 데이터 준비는 구현했다.
  **학습된 선택기·온라인 S/F 식별·F 보상·실측 비선형 모델·사전 FIR live 연결은 미완료**다.

현재 설정/저장 S 기준의 상쇄 경로는 `1465 + handoff 256 = 1721`샘플(35.854 ms)이고,
REF 기하 선행은 약 2.915 ms다. S의 기존 반복 검증 대역은 **150–600 Hz**다.
따라서 고역 신뢰대역·광대역 인과 조건은 충족됐다고 할 수 없다.
`--require-band 1000 1600 --require-broadband` 결과를 숨기거나 게이트를 낮추지 않는다.
자세한 digital/acoustic 구분은 [docs/01](docs/01_physics_limits.md)을 따른다.

외부 DSP의 FxNLMS 10 dB 이상 감쇠는 사용자 보고다. 같은 덕트·스피커지만 마이크가 달랐고,
음원·대역·REF/ERR 역할·구동 조건은 미확인이다. 현 Jetson과 직접 비교하지 않는다.
**DSP/Jetson 출력 합산·스피커 공유·제어 스피커 수는 미정**이다.
확정 전 듀얼 제어기·크로스오버·출력 라우팅을 임의 구현하지 않는다.

## 4. 자료의 실제 위치와 보존 상태

통합 전 Jetson에서는 과거 `runs/export*/` ONNX와 `results/` 실측 디렉터리의 존재를 확인했다.
Git 통합은 이 로컬 대용량 산출물을 PC나 다른 Jetson으로 복사하지 않는다.
파일 존재만으로 현재 acoustic 모델이나 새 성능이 검증된 것은 아니다.
반면 2026-09-15 x86 PC에서 생성한 `results/drive_preparation/20260915_01/`,
`results/drive_transfer_receipts/20260915_01/`, `results/prepared_fir/pc_20260915_linear_01/`는
당시 Jetson에 회수되지 않았다. 과거 PC 경로를 현재 호스트에 있다고 가정하지 않는다.
기존 `data/manifests`에는 호스트 절대경로를 담은 legacy 자료가 있다. Docker의 strict 준비
완료로 간주하지 않는다. 합성 단위테스트는 이 로컬 자료에 의존하지 않도록 고립했다.

Drive 데이터의 마지막 확인 기록:

- 공개 원본 13개(18,599,035,802 byte)를 192조각+13 manifest로 업로드 확인한 뒤 PC 임시 원본을 정리했다.
  재개 시 **receipt와 Drive 목록부터 대조**하고 중복 다운로드·업로드하지 않는다.
- 원격 전체 복원 checksum은 미검증이다. 복원 후 다시 검증해야 한다.
- FMA 8000개 중 7개 수치 QA 실패가 있었으며 strict 학습 준비 완료로 승격하지 않았다.
- MIMII 공식 split은 메타데이터로 보존하되 ANC train-only다. section은 독립 원녹음 ID가 아니다.
- 기존 Drive 분할 백업의 part5는 당시 미확인 상태였다. 분실·삭제로 단정하거나 옛 백업을 정리하지 않는다.

복원·라이선스·QA·정리 receipt 절차는 [docs/16](docs/16_drive_acoustic_preparation.md)를 따른다.
Drive 보관 완료와 로컬 strict 데이터 준비·실제 학습 완료는 별개다.

기존 사용자 미추적 파일 3개는 이번 작업 대상이 아니며 보존한다:
`.deep_anc_live_recovery_01d8442eca8986642540c212_gainprobe_v3_raw`,
`assets/measured/primary_path_il.npz.orig`, `assets/measured/secondary_path_il.npz.orig`.

## 5. 다음 순서

1. 기존 Docker를 재사용한다. 코드 변경 시 관련 회귀와 전체 검사를 수행한다.
2. 실제 acoustic artifact의 출처·독립 학습/평가 자격을 확인한다. 현재 legacy ONNX 검사와 구분한다.
3. Drive receipt/자료 회수 상태와 strict 데이터 QA를 확인한다. 승인된 train-only 정책을 유지한다.
4. 현장 준비가 되면 [docs/14](docs/14_pc_jetson_workplan.md)에 따라 장치·REF/ERR 입력,
   고역 S·F·선행 시간과 반복 OFF/ON/OFF를 순서대로 측정한다. 현장 승인 전 오디오는 열지 않는다.
5. 외부 DSP 출력 구성이 정해져야 하는 설계는 사용자 확인 뒤 진행한다.

## 6. DeepANC에서 통합한 OMAP 경로

사용자가 성공한 OMAP-L138 FxNLMS에는 Jetson이 관여하지 않았다. 성공한 펌웨어와
`rir.txt`(16 kHz / 500탭)를 그대로 보존했다. 기존 48 kHz NPZ·설정·실시간 엔진은 변경하지 않았다.
`deepanc/`는 새 오프라인 진입점이며 `src/deep_anc/`와는 별개다.
두 경로의 샘플레이트·지연·스케일을 임의로 합치지 않는다.

OMAP 기준선에서 딥러닝을 이어서 준비할 때는 [OMAP 인수인계](docs/JETSON_HANDOFF.md)를 따른다.
Docker 안에서 준비 스크립트·전체 테스트를 수행하고, 실제 동기 ANC-OFF 녹음의 존재와 세션 분리를
확인한다. 실제 녹음과 Jetson CUDA 확인 없이는 실제 데이터 학습 준비 완료나 감쇠 성공을 선언하지 않는다.
통합 범위·검증 결과·사용법은 [통합 안내](docs/17_repository_integration.md)에 기록한다.

실측 원자료·모델·Drive 백업과 원본 `DeepANC` 저장소는 삭제하지 않는다.
