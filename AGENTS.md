# AGENTS.md — AI 에이전트 작업 규칙 (Claude Code / Codex 공용)

이 저장소에서 AI 에이전트가 작업할 때 반드시 지켜야 할 규칙과 작업 방법.
**"이어서 진행해줘"라는 요청을 받으면 먼저 [HANDOFF.md](HANDOFF.md)를 읽어라** — 현재 상태와 다음 단계가 거기 있다.

## 절대 목표 2가지 (모든 결정의 판단 기준 — 사용자 명시)

1. **기능 1**: 저주파와 고주파 노이즈를 **모두** 잘 제거할 것 (한쪽 대역만 되면 실패)
2. **기능 2**: 노이즈뿐 아니라 대화(음성)·음악 등 **모든 소리**를 제거할 것 (quiet zone)

측정 매핑·판정 기준: docs/07 §0. 데이터·모델·평가 결정은 이 둘로 소급 판단한다.

## 절대 규칙 (사용자 명시 지시 — 위반 금지)

1. **`~/anc_project` 는 읽기 전용.** 기존 FxLMS 실험 환경이다. 파일 생성/수정/삭제 금지. 복사만 허용.
2. **Jetson 시스템 불가침.** 핀 설정(pinmux/I2S)·RT 커널·전원모드(nvpmodel)·jetson_clocks·
   pulseaudio/pipewire·`/etc/security/limits.d`·apt 설치 등 **sudo가 필요한 모든 시스템 변경 금지.**
   현재 구성(RT 커널, 30W)은 의도된 것이다. 작업은 저장소와 venv 등 유저 공간에서만.
3. **임의 판단 금지.** 설계에 영향을 주는 불명확한 사항은 추측하지 말고 사용자에게 질문할 것.
4. **GitHub에 비밀정보 금지.** API 키/토큰/환경변수/개인키(.pem, id_*) 커밋 금지.
   `.gitignore`의 앵커 패턴(`/data/` 등)을 비앵커로 바꾸지 말 것 (과거 사고: `data/`가 `src/deep_anc/data/`까지 무시).
5. **커밋 메시지에 AI 표기 금지.** Co-Authored-By: Claude/Codex 등 붙이지 말 것 (사용자 요청).
6. **소통은 한국어.** 문서도 한국어로 작성.
7. **Docker 안에서만 작업.** 코드 읽기·수정·Python 실행·테스트는 컨테이너 내부에서만 한다.
   호스트에서는 Docker 빌드/시작/접속 등 환경 관리만 수행한다. 호스트 `.venv`는 사용하지 않는다.
   절차는 [docker/README.md](docker/README.md). x86 CPU 검증을 Jetson GPU/오디오 실측으로 보고하지 않는다.
8. **이 PC에서 가능한 작업은 이 PC의 Docker에서 진행한다.** 코드·합성 회귀·오프라인 분석·문서화는
   Jetson 연결을 기다리며 멈추지 않는다. 실제 ARM64/CUDA/TensorRT·I/O 지연·덕트 측정만 Jetson에서 한다.
   작업별 선행조건·완료 기준·회수 산출물은 [docs/14](docs/14_pc_jetson_workplan.md)를 따른다.
9. **데이터 원본의 최종 보관소는 Google Drive다.** 최신 지시는 PC Docker의 임시 다운로드를
   허용하되 **Drive 업로드 확인 뒤 해당 PC 원본을 삭제**하는 것이다(이전 PC 다운로드 금지를 대체).
   공식 checksum·업로드 파일 ID/크기·라이선스·정리 receipt를 남긴다. 실패/미확인 파일을 지우지 않는다.
   비상업 학업 실험이며 원본은 Git에 올리지 않는다. Drive는 기존 자료 우선 재사용·중복 방지로
   정리하고, 불완전한 백업/같은 이름만으로 기존 파일을 삭제하거나 덮어쓰지 않는다.

## 환경 요약

| 위치 | 내용 |
|---|---|
| 추론 타깃 | Jetson AGX Orin (JetPack 6/R36.4.4). 현재 접속 호스트의 아키텍처와 구분할 것 |
| 개발 환경 | Docker 전용. x86 호스트는 `cpu`, 실제 ARM64 Jetson은 `jetson` 또는 저장공간 절약형 `jetson-local`. 현재 상태는 HANDOFF 참조 |
| venv | 컨테이너 `/workspace/Deep-ANC/.venv`, 이미지별 Docker 볼륨. **onnxruntime==1.18.1 고정**(1.19+는 Tegra 크래시). Jetson 이미지는 NVIDIA PyTorch wheel과 lib preload 훅을 설치하며 실기 CUDA 검증은 별도 |
| 학습 | Elice A100용 설정을 보유. 현재 원격 자원·접속·학습 실행 여부는 별도 확인하며 과거 서버/PID를 재사용하지 않는다 |
| GitHub | https://github.com/tokengeoji/Deep-ANC (공개, 이전 Roka-jsj 주소도 같은 저장소로 연결). 현재 checkout의 origin/작성자/인증은 별도 확인하고 사용자 승인 대상으로만 push한다. 과거 Jetson 키 경로를 현재 PC에 있다고 가정하지 않는다 |
| 실행 | `bash scripts/docker/dev.sh exec .venv/bin/python ...`. 테스트: `bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q` (전부 통과 유지) |

## 프로젝트 이해에 필요한 문서 (우선순위순)

1. [HANDOFF.md](HANDOFF.md) — 현재 상태·진행 중 작업·다음 단계 (**여기부터**)
2. [docs/01_physics_limits.md](docs/01_physics_limits.md) — 지연 물리. **digital-ref/acoustic-ref 두 모드의
   지연 규약이 이 프로젝트의 심장이다.** 코드 수정 전 반드시 이해할 것
3. [docs/00_overview.md](docs/00_overview.md) — 전체 구조, 3단계 로드맵, 저장소 지도
4. [docs/04_model_architecture.md](docs/04_model_architecture.md) — 모델/스트리밍/ONNX 규약
5. 나머지 docs/02~09 + [docs/appendix_legacy_fxlms.md](docs/appendix_legacy_fxlms.md)

## 건드릴 때 조심해야 하는 불변식 (테스트가 강제하지만, 의미를 알고 고칠 것)

- 지연 규약: 학습 플랜트 총지연 = 선택한 S(z) NPZ의 delay + 스레드 핸드오프(256).
  현재 interleaved S는 1465로 총 1721샘플이며, 과거 S의 1342를 현재 고정값으로 사용하지 않는다.
  digital-ref d 경로는 핸드오프 없음. **RIR에는 음향 온셋이 이미 포함 — D_noise 결합 시 t_ac(NS→ERR)를 빼는 이유** (synth_dataset.py 주석)
- 극성: `e = d + S·y` — 어디에서도 추가 부호 반전 금지 (측정 FIR에 극성 포함)
- 인과성: 모델은 미래 입력 참조 금지. 스트리밍=오프라인 수치 등가 유지
- SPSC 링버퍼: 생산자는 write_pos만, 소비자는 read_pos만 (스레드 소유권)
- 손실은 FP32 고정 (bf16은 FFT 미지원), closed-loop 워밍업 절단은 플랜트 적용 **후**
- 세그먼트 길이는 256의 배수, ONNX는 opset 17/정적 shape/상태 명시 I/O

## 안전 (실기 실행)

스피커에 소리를 내는 스크립트(record_duct, calibrate_wideband, measure_io_latency,
evaluate_session, run_realtime)는 **사용자 입회 + 볼륨 최소 상태에서만**. 런타임은 항상 ANC OFF로 시작.

## 통합된 OMAP-L138 16 kHz 오프라인 경로

현재 정식 원격은 `https://github.com/tokengeoji/Deep-ANC.git`, 통합 브랜치는 `main`이다.
별도 저장소 `tokengeoji/DeepANC`에서 성공한 OMAP FxNLMS 기준선과 학습 도구를 가져왔다.
두 저장소의 이력은 모두 보존하며 `dev`·`presentation`은 이번 통합 대상이 아니다.

- `deepanc/`는 OMAP 16 kHz 오프라인 ANC이고, `src/deep_anc/`는 기존 48 kHz Jetson 경로다.
  위 48 kHz NPZ 지연·handoff 규약을 OMAP 경로에 적용하지 않는다.
- `rir.txt`는 권위 있는 실측 2차경로(16,000 Hz, 500탭)이며 성공한
  `firmware/omap_l138/FxNLMS0/ISR.c`의 `S_hat`과 일치한다.
  `calibration/secondary_path.json`의 원본 SHA와 함께 이득·부호·전체 탭·선행 지연을 보존한다.
  정규화·자르기·피크 정렬·부호 반전·리샘플링·추측한 codec 지연 추가를 금지한다.
- OMAP 신경망 출력은 정규화된 실제 DAC 명령 `u`, 잔차는 `e=d+S*u`다.
  기존 DSP의 `u=-y`를 Python에서 다시 반전하지 않는다. 펌웨어는 변경·플래시하지 않는다.
- OMAP 성공에는 Jetson이 관여하지 않았다. 기존 Jetson 48 kHz 검증 기록과 혼합하지 않는다.
- OMAP 작업을 "이어서 해줘" 하면 `docs/JETSON_HANDOFF.md`, `docs/JETSON_SETUP.md`,
  `docs/DATASET.md`를 읽고 Git 상태·실제 플랫폼·기존 데이터부터 확인한다.
  Docker에서 `bash tools/prepare_jetson.sh`와 전체 pytest를 실행한다.
  실제 Jetson에서는 CUDA 연산·역전파 성공이 필요하며 `--allow-cpu`로 대신하지 않는다.
  `--install`은 호환 안내 옵션일 뿐 호스트 venv나 새 torch를 설치하지 않는다.
- 학습에는 공통 디지털 스케일을 유지한 동기 16 kHz raw REF와 ANC-OFF disturbance가 필요하다.
  train/valid는 녹음 세션을 분리한다. ANC-ON 잔차를 d로 사용하거나 독립 정규화·미래 정렬·
  임의 합성 P를 실측 데이터로 사용하는 것을 금지한다. 데이터가 없으면 필요한 녹음을 보고한다.
- 합성 smoke는 소프트웨어 검사다. 실제 데이터의 짧은 학습·유한 손실/gradient/output·checkpoint
  저장을 확인한 뒤 확장한다. 실시간 출력 연결은 별도 승인·실측 지연·스케일·feedback 검증이 필요하다.
- `scripts/train.py`와 `scripts/utils/`는 legacy GCRN 음성 향상이다. ANC 학습과 혼합하지 않는다.

통합 상세와 패키지별 진입점: [docs/17_repository_integration.md](docs/17_repository_integration.md).
