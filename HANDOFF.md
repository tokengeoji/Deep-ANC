# HANDOFF — 현재 상태와 다음 작업

최종 갱신: 2026-09-20. 이어서 작업할 때 이 문서부터 읽는다.
작업 규칙은 [AGENTS.md](AGENTS.md), 실행 절차는 [docker/README.md](docker/README.md),
현장 체크리스트는 [docs/14](docs/14_pc_jetson_workplan.md)가 단일 출처다.
이 파일에는 현재 사실만 남긴다. 과거 세션 일지·PID·실험 수치는 Git 이력에서 확인한다.

## 1. 사용자 확정 사항

- acoustic-ref 최우선. 외부 소리를 실제 REF 마이크로 받으며 소음 원본·미래 샘플을 사용하지 않는다.
- ERR 한 점의 감쇠가 기준이며 고역은 1 kHz 이상이다. **Jetson 우선 대역은 1000–1600 Hz**다.
- 최신 검토 구상은 **1 kHz를 경계로 저역은 OMAP FxNLMS, 고역은 Jetson Orin**이 맡는 구조다.
  정확한 전이대역·출력 합산·입력 공유·장치 간 전송은 미정이다. 1600 Hz를 전체 목표의 상한으로 정하지 않는다.
- 최신 작업 지시는 **연결은 나중에 하고 SFANC 필터 계산·학습부터 진행**하는 것이다.
  사전학습의 S는 사용자가 확정한 **OMAP 16 kHz 원본 실측 500탭**이다. 48 kHz 경로와 섞지 않는다.
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

## 2. 현재 Jetson 환경과 통합 후 검증

현재 checkout은 `tokengeoji/Deep-ANC/main`의 통합 커밋 `4c8d267`을 fast-forward로 반영했다.
접속 장치는 **실제 ARM64 Jetson AGX Orin / L4T R36.4.4**다. 기존 Docker·NVIDIA PyTorch를
그대로 재사용했으며 호스트 시스템·오디오 설정·측정 원본은 변경하지 않았다.

통합 자체는 x86 CPU Docker에서 수행됐고 당시 결과는 1282 passed, 2 skipped,
5 subtests passed였다. 아래 Jetson 검증은 그 이후 통합 코드를 실제 장치에서 실행한 결과다.

최종 통합 Jetson 회귀: **1640 passed, 7 subtests passed, 실패·skip 없음 (209.06초)**.
`pytest -q -o addopts= -ra`로 두 Python 경로와 SFANC·CUDA·원본 보호 회귀를 함께 실행했다.
이번 연속 FIR·지연·계산 벤치·직접 CLI/보존 검사 188개를 포함한다.
로그는 `results/pytest_sfanc_continuous_full_20260920_01.log`다. 스피커 출력은 하지 않았다.

### 통합 OMAP 16 kHz 경로

- `pip check` 및 `bash tools/prepare_jetson.sh` 통과. `--allow-cpu`를 사용하지 않았다.
- `artifacts/jetson_preflight.json`: aarch64 / RT Tegra 커널 / L4T R36.4.4,
  Orin CUDA 순전파·역전파와 유한값 확인 PASS. Python 3.10.12,
  PyTorch `2.5.0a0+872d972e41.nv24.08`, CUDA 12.6, cuDNN 9.3을 유지한다.
- `artifacts/secondary_path/report.json`: 원본 `rir.txt` 500탭·SHA와 성공한 DSP `S_hat` 일치.
  탭·gain·부호·선행 지연은 변경하지 않았다.
- `runs/smoke/run-xk0EFZzv/`: **CUDA 합성 학습 1 epoch** 및 검증, `last.pt`·`best.pt` 저장 완료.
  유한 loss/gradient/output 검사 통과. 실행 로그는 `results/omap_prepare_jetson_20260920_02.log`다.
  합성 데이터 검사이며 실제 ANC 감쇠나 실시간 지연 검증은 아니다.
- `tests/test_omap_cuda.py`에 실제 CUDA의 FFT/direct FIR 출력·입력 gradient 등가,
  청크 state와 경계를 넘는 gradient, 합성 학습 1→2 epoch 재개/Adam 상태 복원 회귀를 추가했다.
  CUDA가 없으면 이 3개는 명시적으로 skip하며 CPU 검증으로 대체하지 않는다.
- 파생 FIR 내보내기는 입력 원본과 출력 경로가 직접·심볼릭 링크·하드링크로 겹치면
  어떤 파일도 쓰기 전에 거부한다. 준비 진단의 Python 최소 버전도 저장소와 같은 3.10으로 맞췄다.
- 준비 시 여유 공간은 약 4.49 GiB로 5 GiB 미만 경고가 있다. 대용량 데이터 복원 전 공간을 확인한다.
  컨테이너에 `nvidia-jetpack` 메타패키지는 없지만 L4T 파일과 실제 CUDA 연산을 확인했다.

### SFANC 필터 계산·선택기 사전학습

- [SFANC 안내](docs/18_sfanc_pretraining.md)의 FP64 인과 FIR 계산과 REF-only CNN 학습을 추가했다.
  독립 컨볼루션·부호·지연·gradient·미래 입력 차단·분할/복원 등 신규 회귀 **149개 통과**.
- `runs/sfanc/omap_20260920_02/`: 실제 Orin CUDA에서 **60 epoch**, 검증 비용으로 선택한 epoch 52 모델.
  `bank.npz`는 128탭 FIR 8개+무제어 후보이며 S의 원본 500탭과 SHA·gain·극성·선행 탭을 보존했다.
  `selector.pt`, `report.json`, `speech_manifest.json`을 저장하고 복원 수치 등가를 확인했다.
- train/validation/test는 600/160/160창이다. LibriSpeech 음성 120/40/40창은 화자·책 연결요소로
  분리하고 나머지는 독립 합성 음원·무음이다. 원천 음성은 동기 REF/ERR 실측 녹음이 아니다.
- **P는 명시적 합성 경로**다. 추가 S 지연 0은 오프라인 조건이지 Jetson 전송 지연 실측값이 아니다.
  고역 가중 잔차·출력 에너지·한도 초과 벌점을 합친 독립 test 비용은 고정 FIR 0.574615,
  학습 선택기 0.554180이다. dB 감쇠나 실제 덕트 성공으로 해석하지 않는다.
- test 1/160창에서 선택 출력이 0.2 한도를 넘는다. 평가에는 실제 limiter를 적용하지 않았고
  초과·증폭 결과를 보고서에 보존했다. **실기 배포 불가**, 연속 필터 교체/전송도 미연결이다.
- 로그는 `results/sfanc_omap_20260920_02.log`. 첫 CE-only 학습 `_01`도 보존했다.
  두 번째 학습은 validation에서 발견한 비용 불일치를 기대 regret 손실로 보완한 것이며,
  최종 test를 본 뒤 모델/임계값을 다시 고르지 않았다. 음악 학습·실측 fine-tuning은 미실행이다.

### SFANC 연속 처리·지연 민감도와 계산시간

- 인과 FIR 이력·샘플별 계수 보간·평가용 hard clip과 연속 진단을 추가했다.
  오프라인 수치 코어이며 실제 오디오 callback·비동기 worker·OMAP 전송은 아직 없다.
- `results/sfanc_stress/omap_20260920_02/report.json`: 기존 CNN/bank와 OMAP 원본 S 동결,
  합성 P 3/8 ms와 S 외 추가 제어 지연 0/0.5/1/2/4/8 ms를 평가했다.
  새 절차 생성 음원 6개와 기존 heldout 음성 일부의 합성 접합 1개, 5개 방법으로
  **420회 실행·16,800개 지표**를 기록했다. 음악·jitter·feedback은 미검증이다.
- 같은 고역 잡음의 목표대역에서 SFANC는 P=8 ms/추가 지연 0일 때 3.556 dB,
  추가 1 ms일 때 −1.758 dB(증폭)였다. 후자 조건으로 새로 계산한 FIR은 3.393 dB였다.
  **합성 플랜트 결과**이며 동결 필터의 경로 불일치와 재설계를 구분한다.
  이 고역 사례에서 SFANC와 기존 train-best 단일 FIR의 결과는 같았다.
- FxNLMS는 같은 제한 명령을 플랜트에 적용하며 7회 clip/적응 보류를 보존했다.
  정확한 S/추가 지연과 ERR를 제공하는 cold block 진단이며 성공한 펌웨어 재현이 아니다.
  기존 test 음성 재사용은 고정 진단일 뿐 새 독립 test가 아니며 모델/임계값 재선택은 없었다.
- `results/sfanc_compute/jetson_20260920_01/report.json`: Orin에서 warmup 30회+측정 200회.
  CPU 32샘플 FIR P99 **0.0873 ms**, 매 블록 교체 **0.1299 ms**,
  특징+선택기 CPU **1.2535 ms** / CUDA **8.9363 ms**였다.
  미학습 HybridANCNet tiny PyTorch 구조는 CUDA 256샘플 P99 **11.6947 ms**였다.
  I/O·동시 실행·최악 마감 보증·TensorRT 성능·감쇠 비교가 아닌 계산시간 측정이다.
- legacy acoustic_pilot 파일은 있지만 OMAP 16 kHz 동조건 검증 HybridANCNet 학습본은
  확인하지 못했다. **HybridANCNet 대비 감쇠 우위는 미판정**이다.
- continuation에서도 `tools/prepare_jetson.sh`의 실제 CUDA 연산·역전파·합성 학습을 통과했다.
  로그 `results/omap_prepare_20260920_continuation_01.log`, smoke `runs/smoke/run-1n9s40Ap/`.
  원본·펌웨어·이전 모델·RT/30W/오디오는 보존했다. 해석은 [docs/18 §7–8](docs/18_sfanc_pretraining.md)을 따른다.

### 재사용한 Jetson Docker 구성

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

### 통합 전 별도 설치·legacy ONNX 검사 이력

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
- 사전 FIR·필터 뱅크·strict acoustic 데이터 준비와 별도 OMAP S 기반 **오프라인 학습형 REF-only 선택기**를 구현했다.
  **실측 fine-tuning·온라인 S/F 식별·F 보상·실측 비선형 모델·사전 FIR live 연결은 미완료**다.

현재 설정/저장 S 기준의 상쇄 경로는 `1465 + handoff 256 = 1721`샘플(35.854 ms)이고,
REF 기하 선행은 약 2.915 ms다. S의 기존 반복 검증 대역은 **150–600 Hz**다.
따라서 고역 신뢰대역·광대역 인과 조건은 충족됐다고 할 수 없다.
`--require-band 1000 1600 --require-broadband` 결과를 숨기거나 게이트를 낮추지 않는다.
자세한 digital/acoustic 구분은 [docs/01](docs/01_physics_limits.md)을 따른다.

**48 kHz 측정 도구의 미해결 주의점:** `measure_paths_interleaved.py`의 저장모델을
합성 순수지연 1400샘플로 재구성하면 `pre_roll=0/128/256`에서 각각 1400/1528/1656샘플이 된다.
FIR 이동량을 별도 저장 delay에서 빼지 않아 pre-roll만큼 추가 지연되는 문제를 통합본에서도 재현했다.
반복 consistency가 1이어도 시간 정렬을 보장하지 않는다. OMAP `rir.txt`와는 별도 문제이며,
기존 실측 NPZ·1465/1608·handoff256은 자동 보정하지 않았다. 원시 capture와 저장 규약을 대조해
수정·재검증하기 전에는 위 저장값 기반 예산을 확정 실측값으로 재해석하지 않는다.

외부 DSP의 FxNLMS 10 dB 이상 감쇠는 사용자 보고다. 같은 덕트·스피커지만 마이크가 달랐고,
음원·대역·REF/ERR 역할·구동 조건은 미확인이다. 현 Jetson과 직접 비교하지 않는다.
**DSP/Jetson 출력 합산·스피커 공유·제어 스피커 수는 미정**이다.
확정 전 듀얼 제어기·크로스오버·출력 라우팅을 임의 구현하지 않는다.
현재 FxNLMS 원본의 stereo LINE IN은 REF/ERR 두 채널을 사용하므로 Jetson의 AUX 출력을
그 입력에 꽂는 것만으로 기존 FxNLMS와 합산할 수 없다. 연결 구상은 [docs/13 §1.1](docs/13_acoustic_hybrid.md)을 따른다.

## 4. 자료의 실제 위치와 보존 상태

통합 후 현재 Jetson의 `data/`, `datasets/`, `assets/`, `runs/`를 확인했으나
**OMAP 16 kHz 동기 raw REF/ANC-OFF 실측 녹음은 없다.** `datasets/anc/` 자체가 없고,
OMAP 계약의 manifest·`capture.json`·`preparation.json`도 발견되지 않았다.
16 kHz 오디오 2,703개는 mono LibriSpeech FLAC이며 OMAP 녹음이 아니다.
기존 WAV 103개는 44.1/48 kHz mono 음원이다. `recorded_train.jsonl`·`recorded_regrouped.jsonl`은
각 82행의 48 kHz legacy 자료이며 현재 참조 세션 경로는 각 0/82개 존재한다.
이 자료를 리샘플링하거나 ANC-OFF라고 추정해 OMAP 학습에 사용하지 않는다.
현재 상태는 **로컬 미수집 / 기존 자료가 없으면 실측 예정**이다(사용자 확정).
실측 단계에서는 OMAP raw ADC 두 채널의 동기·손실 없는 수집 경로와 gain·배선 조건을 확인하고,
사용자 입회·볼륨 최소 상태에서 ANC OFF 녹음을 확보하는 것이다.
기존 `rir.txt`는 이미 실측된 2차경로이며, 없다고 보고한 것은 학습용 REF/ERR 동기 녹음이다.
실측 계획은 즉시 오디오 실행·펌웨어 변경을 승인한 것으로 해석하지 않는다.

수집 준비 확인(2026-09-20): 사용자는 **Windows 컴퓨터에서 CCS를 운용**하고
**TMDSEMU200-U XDS200 USB 디버그 프로브**를 사용한다고 확인했다.
프로세서는 OMAP-L138이며, 실제 보드 모델/revision·현재 CCS 버전·가용 RAM/링커 배치는 미확인이다.
저장된 프로젝트는 LCDKOMAPL138/CCS 9.3.0 설정이다. raw REF/ERR 모니터는 256샘플(16 ms)
순환 버퍼뿐이며 연속 녹음·누락 검출·WAV 회수 경로는 미구현이다.
사용자는 연결 작업을 뒤로 미루고 SFANC 학습을 먼저 진행하도록 지시했다. 보드 식별·수집 방식은
실측 단계에서 확인한다. 별도 녹음 전용 프로젝트는 아직 승인·구현하지 않았으며 현재 보류한다.
Windows에서 확보한 파일을 Jetson Docker에서 처리할 수 있지만, JTAG 접속 자체를 녹음 완료나
실시간 Jetson–DSP 통신으로 간주하지 않는다. [수집 준비 안내](docs/DATASET.md)를 따른다.

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
2. **우선 SFANC:** OMAP 원본 S를 보존한 오프라인 필터·선택기 학습을 발전시킨다.
   연속 진단에서 드러난 경로/지연 불일치를 반영한 bank 재설계·적응 결합을 검토하고,
   출력 한도·전이 구간·고역/저역 증폭을 독립 자료로 재검증한다. 합성 P 3/8 ms를 실측값으로 쓰지 않는다.
   HybridANCNet 감쇠 비교에는 동일 OMAP 경로·출력 제한·인과 지연의 별도 학습본이 필요하다.
   마지막 test를 다시 튜닝 자료로 쓰지 않는다. 연결 질문이나 녹음 부재로 가능한 오프라인 작업을 멈추지 않는다.
3. **실측 후속:** 기존 동기 녹음이 없으면 사용자 계획대로 실측한다.
   먼저 OMAP 수집 경로·gain·배선·공통 clock을 확인하고, 사용자 입회·볼륨 최소 상태에서
   [DATASET.md](docs/DATASET.md)의 16 kHz raw REF/ANC-OFF 녹음을 확보한다.
   독립 train/valid 세션과 raw 단위를 검증한 뒤 CUDA 1 epoch부터 진행한다.
   합성 결과를 실제 데이터 학습으로 대체하거나 녹음 경로를 임의로 정하지 않는다.
4. **별도 48 kHz 후속:** interleaved 저장모델의 pre-roll 회귀를 수정·재검증하고,
   실제 acoustic artifact의 출처·독립 학습/평가 자격, Drive receipt/자료 회수 상태와
   strict 데이터 QA를 확인한다. 승인된 train-only 정책을 유지한다.
5. 현장 준비가 되면 [docs/14](docs/14_pc_jetson_workplan.md)에 따라 장치·REF/ERR 입력,
   고역 S·F·선행 시간과 반복 OFF/ON/OFF를 순서대로 측정한다. 현장 승인 전 오디오는 열지 않는다.
6. 외부 DSP 출력 구성이 정해져야 하는 설계는 사용자 확인 뒤 진행한다.

## 6. DeepANC에서 통합한 OMAP 경로

사용자가 성공한 OMAP-L138 FxNLMS에는 Jetson이 관여하지 않았다. 성공한 펌웨어와
`rir.txt`(16 kHz / 500탭)를 그대로 보존했다. 기존 48 kHz NPZ·설정·실시간 엔진은 변경하지 않았다.
`deepanc/`는 새 오프라인 진입점이며 `src/deep_anc/`와는 별개다.
두 경로의 샘플레이트·지연·스케일을 임의로 합치지 않는다.

OMAP 기준선에서 딥러닝을 이어서 준비할 때는 [OMAP 인수인계](docs/JETSON_HANDOFF.md)를 따른다.
현재 CUDA 합성 준비는 통과했지만 실제 동기 ANC-OFF 녹음이 없어 실측 데이터 학습은 미실행이다.
자료 확보 후 세션 분리·단위·수집 조건을 검증한다. CUDA 준비 통과만으로 실제 데이터 학습 준비 완료나
감쇠 성공을 선언하지 않는다.
통합 범위·검증 결과·사용법은 [통합 안내](docs/17_repository_integration.md)에 기록한다.

실측 원자료·모델·Drive 백업과 원본 `DeepANC` 저장소는 삭제하지 않는다.
