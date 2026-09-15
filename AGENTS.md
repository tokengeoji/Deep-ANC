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

## 환경 요약

| 위치 | 내용 |
|---|---|
| 추론 타깃 | Jetson AGX Orin (JetPack 6/R36.4.4). 현재 접속 호스트의 아키텍처와 구분할 것 |
| 개발 환경 | Docker 전용. x86 호스트는 `cpu`, 실제 ARM64 Jetson은 `jetson` 이미지 사용 |
| venv | 컨테이너 `/workspace/Deep-ANC/.venv`, 이미지별 Docker 볼륨. **onnxruntime==1.18.1 고정**(1.19+는 Tegra 크래시). Jetson 이미지는 NVIDIA PyTorch wheel과 lib preload 훅을 설치하며 실기 CUDA 검증은 별도 |
| 학습 | Elice Cloud A100 (SSH 접속 — HANDOFF.md 참조), torch 2.5.1+cu121 |
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

- 지연 규약: 학습 플랜트 총지연 = S(z) npz delay(1342) + 스레드 핸드오프(256).
  digital-ref d 경로는 핸드오프 없음. **RIR에는 음향 온셋이 이미 포함 — D_noise 결합 시 t_ac(NS→ERR)를 빼는 이유** (synth_dataset.py 주석)
- 극성: `e = d + S·y` — 어디에서도 추가 부호 반전 금지 (측정 FIR에 극성 포함)
- 인과성: 모델은 미래 입력 참조 금지. 스트리밍=오프라인 수치 등가 유지
- SPSC 링버퍼: 생산자는 write_pos만, 소비자는 read_pos만 (스레드 소유권)
- 손실은 FP32 고정 (bf16은 FFT 미지원), closed-loop 워밍업 절단은 플랜트 적용 **후**
- 세그먼트 길이는 256의 배수, ONNX는 opset 17/정적 shape/상태 명시 I/O

## 안전 (실기 실행)

스피커에 소리를 내는 스크립트(record_duct, calibrate_wideband, measure_io_latency,
evaluate_session, run_realtime)는 **사용자 입회 + 볼륨 최소 상태에서만**. 런타임은 항상 ANC OFF로 시작.
