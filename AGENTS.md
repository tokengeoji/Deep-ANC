# AGENTS.md — AI 에이전트 작업 규칙 (Claude Code / Codex 공용)

이 저장소에서 AI 에이전트가 작업할 때 반드시 지켜야 할 규칙과 작업 방법.
**"이어서 진행해줘"라는 요청을 받으면 먼저 [HANDOFF.md](HANDOFF.md)를 읽어라** — 현재 상태와 다음 단계가 거기 있다.

## 절대 목표 2가지 (모든 결정의 판단 기준 — 사용자 명시)

1. **기능 1**: 저주파와 고주파 노이즈를 **모두** 잘 제거할 것 (한쪽 대역만 되면 실패)
2. **기능 2**: 노이즈뿐 아니라 대화(음성)·음악 등 **모든 소리**를 제거할 것 (quiet zone)

측정 매핑·판정 기준: docs/07 §0. 데이터·모델·평가 결정은 이 둘로 소급 판단한다.
최종 125 Hz 옥타브부터 8 kHz 옥타브까지의 광대역 역할은
`docs/27_broadband_v3_full_octave_contract.md`를 추가 가드로 사용한다. 기존 150 Hz 하단
v2 결과를 125 Hz 옥타브 PASS로 승격하지 않는다.

## 절대 규칙 (사용자 명시 지시 — 위반 금지)

1. **`~/anc_project` 는 읽기 전용.** 기존 FxLMS 실험 환경이다. 파일 생성/수정/삭제 금지. 복사만 허용.
2. **`~/DeepANC_CRN_n_codex` 는 읽기 전용.** 참고만 하고 **절대 건드리지 말 것** (사용자 명시 지시).
   파일 생성/수정/삭제 금지, python import 금지(`python3 -B` 로 `__pycache__` 생성도 막을 것).
   **이 디렉터리에서 별도 작업이 병행될 수 있다** — 실제로 `measurements/A_P_estimate/` 가
   2026-08-05 20:40 에 갱신됐다. 따라서 스피커·마이크를 쓰기 전에 **오디오 장치 점유를 확인**하라
   (`fuser -v /dev/snd/*`, `/proc/asound/card*/pcm*/sub*/status`). 점유 중이면 측정·녹음을 하지 말 것 —
   장치 충돌은 양쪽 측정을 조용히 오염시킨다.
3. **스피커 연결 시간을 최소화하라 (2026-08-05 사용자 지시).**
   **"너무 오랫동안 연결하면 스피커를 조절하는 하드웨어가 고장난다."**
   따라서 소리를 내는 작업은 **절대 즉흥적으로 시작하지 마라.**
   - 스피커가 필요한 측정은 **전부 미리 설계·검증**하고 **한 번에 몰아서** 실행한다.
   - 실행 전에 무음 dry-run 으로 스크립트 오류를 전부 잡는다. 소리를 내며 디버깅하지 않는다.
   - 각 작업의 **예상 소요 시간을 미리 계산해 사용자에게 알린다.**
   - 끝나면 **즉시 알려서 연결을 해제**하게 한다. 방치 금지.
   - 93분짜리 전체 재녹음처럼 긴 작업은 **먼저 정말 필요한지 재검토**하고,
     쪼개서 여러 번에 나눌 수 있는지 검토한다.
4. **Jetson 시스템은 기본적으로 건드리지 않는다 — 단, 핀먹스·디바이스 트리는 예외다.**
   전원모드(nvpmodel)·jetson_clocks·`/etc/security/limits.d`·apt 설치 등 sudo가 필요한
   시스템 변경은 하지 않는다. 작업은 저장소와 venv 등 유저 공간에서 하는 것이 기본이다.

   **예외 (2026-08-06 사용자 명시 허용): 핀먹스(pinmux)와 디바이스 트리는 변경해도 된다.**
   > "하드웨어는 지금 자체에서 바꾸는건 조금 어렵고 핀먹스나 디바이스 트리 다 허용할게"

   이전 판본은 "핀 설정(pinmux/I2S) 절대 변경 금지"였다. 그 문구가 실제로 해를 끼쳤다 —
   2026-08-06 마이크 무신호를 추적할 때, **마이크가 병합 DTB의 핀먹스 오버레이에 의존한다는
   사실**(`~/FxLMS/realtime_fxlms/boot_fix/`)이 이 저장소 어디에도 없었고, 금지 규칙 때문에
   그쪽을 확인 대상에서 빼고 있었다. 규칙이 진단 경로를 막으면 그것은 안전장치가 아니다.

   ⚠ 다만 **되돌릴 방법을 먼저 마련하고** 바꿀 것(`extlinux.conf` 백업 등). ALSA 믹서처럼
   런타임에 되돌릴 수 있는 수단이 있으면 그쪽을 먼저 쓴다.
   ⚠ pulseaudio/pipewire는 여전히 **프로파일 on/off 수준**(`pactl set-card-profile`)까지만
   — 그 이상은 건드리지 않는다.
5. **임의 판단 금지.** 설계에 영향을 주는 불명확한 사항은 추측하지 말고 사용자에게 질문할 것.
6. **GitHub에 비밀정보 금지.** API 키/토큰/환경변수/개인키(.pem, id_*) 커밋 금지.
   `.gitignore`의 앵커 패턴(`/data/` 등)을 비앵커로 바꾸지 말 것 (과거 사고: `data/`가 `src/deep_anc/data/`까지 무시).
7. **커밋 메시지에 AI 표기 금지.** Co-Authored-By: Claude/Codex 등 붙이지 말 것 (사용자 요청).
8. **소통은 한국어.** 문서도 한국어로 작성.
9. **Elice 원격 전용 hotfix 금지.** Elice working tree에서만 코드를 고쳐 학습을 재개하지
   않는다. 실패 원인은 이 저장소에서 코드·회귀 테스트·운영 문서로 먼저 복구하고 GitHub의
   exact commit으로 push한 뒤, Elice는 그 40자리 SHA를 detached checkout한다. stale test
   node, stale commit/freeze/receipt, dirty tree 검사는 corpus scan·manifest 생성·GPU 학습보다
   앞에서 fail-closed해야 한다. Elice 실패를 고쳤다면 재현 fixture와 재개 지점을
   `docs/05_training_elice.md` 및 `HANDOFF.md`에 남기기 전에는 완료로 보지 않는다.

## 환경 요약

| 위치 | 내용 |
|---|---|
| 이 PC | Jetson AGX Orin (JetPack 6/R36.4.4) = **추론 타깃이자 개발 머신** |
| venv | `.venv` — torch 2.5.0a0(NVIDIA JP6.1 wheel) + CUDA 동작. **onnxruntime==1.18.1 고정**(1.19+는 Tegra 크래시). venv 재생성 시 `bash scripts/jetson/setup_jetson.sh` (lib preload 훅 포함 — 필수) |
| 학습 | Elice Cloud A100 (SSH 접속 — HANDOFF.md 참조), torch 2.5.1+cu121 |
| GitHub | https://github.com/tokengeoji/Deep-ANC (공개, 현재 `origin`). push 인증: 이 PC의 `~/.ssh/id_ed25519` |
| 실행 | 모든 프로젝트 파이썬 실행은 `.venv/bin/python`. 조기 정적 게이트: `python3 -I -B scripts/ci/check_static_contract_references.py --repo-root .`; 전체 테스트: `.venv/bin/python -m pytest -q` (둘 다 통과 유지) |

## 프로젝트 이해에 필요한 문서 (우선순위순)

1. [HANDOFF.md](HANDOFF.md) — 현재 상태·진행 중 작업·다음 단계 (**여기부터**)
2. [docs/01_physics_limits.md](docs/01_physics_limits.md) — 지연 물리. **digital-ref/acoustic-ref 두 모드의
   지연 규약이 이 프로젝트의 심장이다.** 코드 수정 전 반드시 이해할 것
3. [docs/00_overview.md](docs/00_overview.md) — 전체 구조, 3단계 로드맵, 저장소 지도
4. [docs/04_model_architecture.md](docs/04_model_architecture.md) — 모델/스트리밍/ONNX 규약
5. [docs/16_canonical_finetune_guardrails.md](docs/16_canonical_finetune_guardrails.md) — 저·고역,
   네 소리 계열, 지연, one-shot G4와 배포 차단의 강제 기준
6. 나머지 docs/02~15 + [docs/appendix_legacy_fxlms.md](docs/appendix_legacy_fxlms.md)

## 건드릴 때 조심해야 하는 불변식 (테스트가 강제하지만, 의미를 알고 고칠 것)

- 지연 규약: 학습 플랜트 총지연 = S(z) npz delay + 스레드 핸드오프(256).
  ⚠ P/S delay 숫자를 여기에 적지 마라. 과거 순차 ESS/legacy 숫자를 현행으로
  오인한 사고가 반복됐다. 값은 strict npz 가 단일 출처이고,
  lead 는 `PlantDelays.lead()` 로만 만들 수 있다(손으로 쓰면 TypeError).
  digital-ref d 경로는 핸드오프 없음. **RIR에는 음향 온셋이 이미 포함 — D_noise 결합 시 t_ac(NS→ERR)를 빼는 이유** (synth_dataset.py 주석)
- 극성: `e = d + S·y` — 어디에서도 추가 부호 반전 금지 (측정 FIR에 극성 포함)
- 인과성: 모델은 미래 입력 참조 금지. 스트리밍=오프라인 수치 등가 유지
- SPSC 링버퍼: 생산자는 write_pos만, 소비자는 read_pos만 (스레드 소유권)
- 손실은 FP32 고정 (bf16은 FFT 미지원), closed-loop 워밍업 절단은 플랜트 적용 **후**
- 세그먼트 길이는 256의 배수, ONNX는 opset 17/정적 shape/상태 명시 I/O

## 안전 (실기 실행)

스피커에 소리를 내는 스크립트(record_duct, calibrate_wideband, measure_io_latency,
evaluate_session, run_realtime)는 **사용자 입회 + 볼륨 최소 상태에서만**. 런타임은 항상 ANC OFF로 시작.
