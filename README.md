# Deep ANC

통합 저장소는 **[tokengeoji/Deep-ANC](https://github.com/tokengeoji/Deep-ANC)**다.
`tokengeoji/DeepANC`의 OMAP-L138 실측 자료와 오프라인 학습 도구를 이 저장소의 `main`에 통합했다.
통합 내역과 16 kHz / 48 kHz 경로 구분은 [저장소 통합 안내](docs/17_repository_integration.md)를 따른다.

덕트 안의 ERR 마이크 한 점에 quiet zone을 만드는 인과적 능동소음제어 연구다.
Jetson AGX Orin에서 외부 소리를 REF 마이크로 받아 상쇄음을 생성한다.

## 목표와 현재 범위

- 저주파와 고주파를 **모두** 감쇠한다. 한쪽 대역만 좋아지면 전체 목표 달성이 아니다.
- 환경소음·기계음뿐 아니라 **음성·음악도** 감쇠한다. 평균만으로 실패한 소스 종류를 가리지 않는다.
- 현재 우선순위는 **acoustic-ref, Jetson 1000–1600 Hz 개선**이다. 시스템 전체 목표는 유지한다.
- 현재 Jetson·마이크·USB DAC·덕트를 유지한다. 외부 DSP와의 출력 연결은 미정이다.

현재 작업·검증 결과·미완료 사항은 **[HANDOFF.md](HANDOFF.md)** 한 곳에서 관리한다.
과거 digital-ref 감쇠나 합성 회귀를 현재 acoustic-ref 성능으로 인용하지 않는다.
판정 기준은 [평가 프로토콜](docs/07_evaluation_protocol.md)을 따른다.

## 제어 구조

```text
외부 소리 → REF → 인과 제어기 → 상쇄 스피커 → S(z) → ERR
외부 소리 ─────────────────────────────────────→ ERR
```

극성 규약은 `e = d + S·y`다. 측정 FIR에 극성이 포함되므로 부호를 다시 뒤집지 않는다.

| 구성 | 용도 | 현재 한계 |
|---|---|---|
| acoustic FxNLMS | 실제 REF/ERR 기준선 | 새 고역 S 검증·현장 감쇠 확인 필요 |
| HybridANCNet | 파형 직접 예측, 명시적 스트리밍 상태 | acoustic 배포 모델의 적합성 검증 필요 |
| HybridEngine | 한 Jetson 내부 DNN+FxNLMS 비교 | 외부 DSP 결합 구조가 아님 |
| 사전 FIR+잔차 적응 연구 API | 합성 조건의 계수 전달·선택 실험 | live 런타임 미연결 |

48 kHz, 런타임 블록 256샘플을 사용한다. ONNX는 opset 17·정적 shape·상태 명시 I/O다.
acoustic-ref에는 digital preview를 줄 수 없다. 지연·인과성 계산은
[지연 물리](docs/01_physics_limits.md), 신경망 규약은 [모델 구조](docs/04_model_architecture.md)를 따른다.
S의 가진 대역과 반복 검증된 신뢰대역을 혼동하지 않는다.

## 작업 시작

코드 읽기·편집·Git·Python·테스트는 **Docker 내부에서만** 수행한다.
호스트는 Docker 빌드·시작·접속만 담당하며 호스트 `.venv`는 사용하지 않는다.

이미 개발 컨테이너가 있으면 다음 명령을 사용한다.

```bash
bash scripts/docker/dev.sh status
# 중지 상태일 때만:
bash scripts/docker/dev.sh start
bash scripts/docker/dev.sh shell
```

최초 설치는 [Docker 안내](docker/README.md)를 따른다.
x86 PC는 `cpu`, 실제 Jetson은 `jetson` 또는 저장공간 절약형 `jetson-local`을 선택한다.
Docker 접근에 기존 sudo 인증이 필요한 경우 환경 관리 명령에만 `sudo`를 붙인다.
이것은 호스트 패키지·드라이버·전원·오디오 설정 변경 허가가 아니다.

소리 출력 없는 검사:

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --json
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --require-band 1000 1600 --require-broadband
```

마지막 명령의 실패는 요구한 고역·광대역 물리 조건 미달일 수 있다.
일반 readiness 보고의 exit 0도 실제 감쇠 성공을 의미하지 않는다.
기본 개발 컨테이너에는 오디오 장치를 노출하지 않는다.

## OMAP-L138 실측 경로에서 이어서 학습

사용자가 성공한 OMAP FxNLMS 소스는 `firmware/omap_l138/`, 원본 16 kHz·500탭 2차경로는
`rir.txt`에 보존했다. 이 경로의 학습은 `deepanc/`와 `configs/anc_train.json`을 사용한다.
기존 `src/deep_anc/`의 48 kHz Jetson 장치 경로·NPZ와는 샘플레이트, 지연 및 데이터 규약이 다르다.
OMAP 경로를 선택했다고 Jetson USB 오디오 경로가 교정되는 것은 아니다.

기존 개발 Docker를 시작한 뒤 오디오 출력 없이 준비한다.

```bash
bash scripts/docker/dev.sh exec bash tools/prepare_jetson.sh --allow-cpu  # x86 PC
# 실제 Jetson Docker에서는 --allow-cpu를 빼고 CUDA를 검증한다.
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
```

실제 16 kHz ANC-OFF 동기 녹음 준비와 학습/재개는 [OMAP 인수인계](docs/JETSON_HANDOFF.md),
[데이터 준비](docs/DATASET.md), [학습 안내](docs/TRAINING.md)에 설명되어 있다.
기존 GCRN 음성 향상 코드는 `scripts/train.py`와 `scripts/utils/`에 보존했다.
이 코드는 `scripts/train/`의 기존 Jetson 학습 도구와 목적이 다르다.

## 저장소와 문서

| 위치 | 내용 |
|---|---|
| `src/deep_anc/` | 데이터·DSP·모델·학습·평가·실시간 엔진 |
| `configs/` | 측정 경로·모델·데이터·런타임 설정 |
| `scripts/` | 데이터 준비·진단·학습·평가·Docker 관리 |
| `tests/` | 인과성·지연·안전·데이터 계약 회귀 |
| `assets/measured/` | 측정 경로, 메타데이터와 함께 해석 |
| `rir.txt`, `calibration/`, `firmware/omap_l138/` | OMAP 실측 16 kHz / 500탭 원본·근거·성공한 DSP 코드 |
| `deepanc/`, `tools/`, `configs/anc_train.json` | OMAP 경로의 저장소 루트 실행용 오프라인 학습·준비 |
| `runs/`, `results/` | 로컬 모델·실험 산출물, Git 미포함 |

읽는 순서:

1. [HANDOFF](HANDOFF.md): 현재 상태와 다음 단계
2. [AGENTS](AGENTS.md): 작업 금지사항·불변식
3. [지연 물리](docs/01_physics_limits.md): 두 reference 모드의 차이
4. [현장 실행표](docs/14_pc_jetson_workplan.md): 선행조건·측정·회수 산출물
5. [평가 프로토콜](docs/07_evaluation_protocol.md): 저역·고역·소스별 판정

분야별 상세: [전체 구조](docs/00_overview.md) · [하드웨어](docs/02_hardware_setup.md) ·
[데이터](docs/03_data_pipeline.md) · [모델](docs/04_model_architecture.md) ·
[학습](docs/05_training_elice.md) · [배포](docs/06_deployment_jetson.md) ·
[acoustic 설계](docs/13_acoustic_hybrid.md) · [사전 FIR 연구](docs/15_prepared_fir_research.md) ·
[Drive·학습 준비](docs/16_drive_acoustic_preparation.md).

## 안전과 데이터 보존

- `~/anc_project`는 읽기 전용이다. Jetson 시스템 설정은 변경하지 않는다.
- 스피커 출력은 사용자 입회·물리 볼륨 최소 상태에서만 한다. ANC는 항상 OFF로 시작한다.
- 원본 데이터의 최종 보관소는 Google Drive다. 임시 다운로드는 checksum·업로드 확인 뒤에만 정리한다.
  불완전한 백업이나 미확인 파일은 삭제하지 않는다.
- 비밀정보·원본 음원은 공개 Git에 올리지 않는다. 기존 실측·모델은 덮어쓰지 않는다.

과거 실험 일지와 삭제한 설명은 Git 이력에서 확인할 수 있다.
현재 사용법을 날짜별 기록과 혼합하지 않는다.

[MIT License](LICENSE)
