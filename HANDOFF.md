# HANDOFF — 세션 인수인계 (다음 AI 에이전트/개발자용)

> **"이어서 진행해줘"를 받았다면**: 먼저 아래 **2026-09-15 현재 작업**을 읽어라.
> 규칙은 [AGENTS.md](AGENTS.md)가 단일 출처. 이 파일은 작업 상태가 바뀔 때마다 갱신할 것.
> 최종 갱신: 2026-09-15. 아래 과거 §0 이후의 서버/PID/실험 현황은 역사 기록이며 현재 상태가 아니다.

## 현재 작업 — 2026-09-15 acoustic 전환 및 Docker 전용 개발

### 사용자 확정 사항

- 실제 acoustic REF 마이크를 사용한다. 감쇠 기준은 ERR **한 점**, 고주파는 **1 kHz 이상**이다.
- 저역·고역을 함께 감쇠하고 음성·음악을 포함한 모든 소리를 대상으로 한다.
- 같은 덕트와 Jetson AGX Orin이 추론 타깃이다. 필요한 하드웨어 변경은 검토 가능하나 제품은 미선정이다.
- Deep ANC만 고집하지 않고 FxLMS/FxNLMS 및 경로 추정과 결합한다.
- **최신 지시: Docker 환경을 만들고 그 안에서만 작업한다.** 호스트에서는 Docker 환경 관리만 한다.
- 커밋·push는 승인됐다. **이 PC에서 가능한 작업은 이 PC에서 진행하고**, Jetson 필수 작업은 [docs/14](docs/14_pc_jetson_workplan.md)의 현장 체크리스트로 분리한다.
- 시스템 설정 변경과 무입회 스피커 출력 금지는 그대로다. 이번 작업에서는 오디오를 실행하지 않았다.

### 현재 환경과 실행

실제 접속 호스트는 x86_64다. 과거 문서의 "이 PC=Jetson"을 현재 호스트 사실로 간주하지 않는다.
CPU 이미지 `deep-anc-cpu:dev`와 개발 컨테이너 `deep-anc-dev`를 빌드·시작했다.
컨테이너 Python은 `/workspace/Deep-ANC/.venv/bin/python`, torch `2.5.1+cpu`, ORT `1.18.1`이다.
호스트 저장소를 바인드하고 `.venv`는 이미지 ID별 Docker 볼륨으로 가린다. 기존 호스트 `.venv`는 보존했지만 사용하지 않는다.
기본 환경은 비특권 사용자이며 오디오 장치를 노출하지 않는다.

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --json
bash scripts/docker/dev.sh shell
```

Jetson용 `docker/Dockerfile.jetson`도 준비했지만 **실제 ARM64 Jetson 빌드·CUDA/TensorRT·실시간 오디오 검증은 아직 하지 않았다**.
L4T R36.4.0 기반 사용자 공간은 호스트 RT 커널/드라이버를 공유한다. x86 CPU Docker가 이를 에뮬레이션하지 않는다.
명령·재시작·볼륨 규약은 [docker/README.md](docker/README.md)를 따른다.
현재 체크아웃에는 과거 `runs/`, `results/`, 학습 데이터와 acoustic 학습 artifact가 없다.
과거 Elice 접속이나 리소스 삭제는 이번 작업 범위가 아니며 자동 수행하지 않는다.

### 구현 범위와 해석

- `configs/runtime_acoustic.yaml`: acoustic FxNLMS 기준선, 내부 소음 OFF, ANC OFF 시작.
- `configs/runtime_acoustic_hybrid.yaml`: DNN 파형 출력 + FxNLMS 잔차 제어. acoustic artifact는 아직 placeholder다.
- `HybridEngine`: 공통 REF/ERR, `e=d+S*y` 부호, 합산 뒤 런타임 리미터, 양쪽 초기화.
- 학습 `reference_mode`를 ONNX 메타로 내보내고 mic DL/hybrid에서 digital/모드미상 artifact를 거부한다.
- mic 모드에서 내부 소음 OFF여도 외부 REF/ERR로 OFF 기준선을 수집하고 조건부 적응한다.
- 페이드·클리핑·xrun·출력 누락·초기화 후의 지연된 ERR/이력도 적응 허용 판정에 반영한다.
- acoustic S의 측정 block/latency 메타 누락·불일치를 런타임에서 거부한다.
- 입력 손상 후 FxNLMS는 S+제어 FIR 이력을 기다린다. mic DL/hybrid는 손상 블록 폐기+reset+ANC OFF이며 자동 ON하지 않는다. 실제 클리핑 녹음은 보존한다.
- 무출력 `check_acoustic_readiness.py`는 설정/측정 S만 읽는다. 기본 exit 0은 선택 게이트 없는 보고 성공이지 감쇠 준비 완료가 아니다.
- 이 단계는 **초기 비교용 API**다. DNN 계수 생성, 온라인 S/F 식별, F 보상, 실제 비선형 모델은 미구현이다.
- 고정 REF 전용 가짜 신경망 수렴 테스트는 ERR를 사용하는 실제 DNN과의 폐루프 안정성 검증을 대체하지 않는다.

### 이번 검증 결과

- `deep-anc-dev` 내부에서 `.venv/bin/python -m pytest -q -o addopts= -ra`: **523 passed, 2 skipped (36.71초)**.
- 건너뛴 2개는 현장 raw 진단 파일과 실측 `metrics.md` 부재 때문이다. GPU/실기 검증으로 해석하지 않는다.
- `pip check`: 의존성 충돌 없음. Docker 관리 스크립트 `bash -n` 및 tracked diff 공백 검사 통과.
- Docker의 `.venv` 전용 볼륨, `ancdev` 사용자, `Privileged=false`, 오디오 장치 미노출을 확인했다.
- 커밋·push가 승인됐다. 실제 반영 상태는 컨테이너의 `git status`, `git log`, 원격 ref로 확인한다.

### 후속 구현 — acoustic 녹음과 PC 진단 연결

- `src/deep_anc/realtime/recording.py`: 기존 5배열/fs와 함께 schema v1 scalar JSON 메타 저장.
  생성 시점 S SHA-256·주요 설정·기록 길이·전체 실행 누적 xrun/누락/fatal을 남긴다.
  기존 NPZ/깨진 symlink는 입력 사전점검 전에 거부하며 저장도 exclusive 생성이다.
  저장 실패의 부분 파일은 자동 삭제하지 않는다. 새 경로로 재시도한다.
- `src/deep_anc/eval/acoustic_session.py`와 `scripts/eval/analyze_acoustic_session.py`:
  장치/torch/ORT를 열지 않는 순수 오프라인 acoustic 녹음 분석. 명령은 [docs/14 §5](docs/14_pc_jetson_workplan.md).
  새 폴더에 JSON·metrics CSV·Markdown 및 유효 창이 있으면 windows CSV를 저장한다.
- 각 ON 사이클은 앞뒤 OFF가 필수다. 기본 1초 창, 초기 OFF 1초·ON 워밍업 2초·경계 0.5초와
  측정 S 꼬리를 제외한다. 녹음 control/gain은 출력 callback 기준이라 handoff를 중복 가산하지 않는다.
  짧은 OFF 길이에 맞춰 긴 ON 후반을 버리지 않는다. 미완료 사이클도 보고서에 남긴다.
- ERR 감소량은 앞뒤 OFF 중 작은 평균 파워 기준의 **시간이 다른 관측 비교**다.
  실제 ERR에 S를 다시 적용하지 않고 REF 변화도 정규화하지 않는다. 항상 `performance_claim_allowed=false`다.
  1 kHz 미만/이상·전체·옥타브 경계 FFT 파워·검증 대역, 중앙/p10/최악/최악 10% 평균을 따로 남긴다.
  무신호는 null, ON 신규 에너지는 별도 플래그다. FFT 옥타브는 기존 Butterworth 지표와 구분한다.
- 신뢰대역은 `consistency_band_hz`와 반복 일관성 ≥0.9로만 표시하고 **밴드 전체 경계**가 들어가야 한다.
  구형 메타 없는 녹음은 설정/S 일치 미확인·health unknown이다. 새 메타의 S/주요 설정 불일치는 거부한다.
  health는 시작/종료 포함 누적값일 뿐 시점별 적응·장애 기록이 아니며 전체 모델/설정 스냅샷도 아니다.
- 신규 테스트 80개(분석 33 + CLI 17 + 녹음 30), writer→analyzer 연결 포함. 모두 합성/fake 장치다.
  이번 변경으로 새 소리를 출력하거나 실측 자료를 수집하지 않았다. 실제 감쇠 개선의 증거로 인용하지 않는다.
- exit 0은 모든 사이클 구간 완전·최소 한 사이클 전체 대역 계산 가능, exit 2는 불완전/비교 불가(산출물 보존),
  exit 1은 입력·설정·I/O 실패다. 어느 코드도 실기 감쇠 성공 판정이 아니다.

### 현 자산 진단 — 새 감쇠 실측이 아님

컨테이너의 무출력 진단으로 다음 값을 확인했다.

| 항목 | 값 |
|---|---|
| S NPZ 순수지연 / 런타임 handoff | 1465 / 256 samples @ 48 kHz |
| 제어음 지연 합계 | 1721 samples = 35.854 ms |
| REF→ERR 기하선행 추정 | 139.94 samples = 2.915 ms |
| 필요한 예측 | 1581.06 samples = **32.939 ms** |
| S 반복 일관성 검증 대역 | **150–600 Hz**, 대역 일관성 0.9556 |

`--require-band 1000 1600 --require-broadband`는 두 조건 미달로 예상대로 exit 1이다.
약 1.6 kHz 평면파 차단을 ERR 한 점 감쇠의 절대 상한으로 해석하지 않는다.
반복 정렬 후 일관성이 높다는 사실도 실시간 클록/위상 안정성을 증명하지 않는다.
과거 clock-warp 해석을 하드웨어 원인으로 확정하지 말고 독립적인 타이밍 측정으로 확인한다.

### 다음 단계

작업 위치·명령・중단조건의 실행 문서는 [docs/14](docs/14_pc_jetson_workplan.md)다.
연구/설계 근거는 [docs/13](docs/13_acoustic_hybrid.md)를 따른다.

1. **이 PC에서 계속**: 녹음 후처리는 구현했다. 다음은 반복 측정의 **S 신뢰대역 검증 도구**,
   실측 경로 재현용 입력 규격·합성 closed-loop 시험, 느린 DNN 계수 생성/빠른 FIR의 안전한 전달 API다.
   현재 `calibrate_wideband.py` ESS 산출물은 `consistency_band_hz`가 없어 readiness에 가진 대역을 대신 넣으면 안 된다.
   설계에 영향을 주는 측정/계수 교체 선택이 불명확하면 먼저 질문한다. 실제 자료가 필요한 항목만 별도 대기시킨다.
2. **Jetson에서만**: 실제 ARM64 이미지 빌드·CUDA/TensorRT·추론 마감 검증. 호스트 시스템 변경으로 실패를 우회하지 않는다.
3. **Jetson 현장**: 장치 접근을 별도로 준비하고 입력-only 점검 후, 사용자 입회·최저 볼륨에서 광대역 S와 별도 F를 측정한다.
4. **자료 회수 후 이 PC**: 원자료 QA, 신뢰대역·지연·클록 안정성·밴드별 감쇠 분석. 설정의 S 지연이나 신뢰대역 숫자를 임의로 바꾸지 않는다.
5. **Jetson 현장**: 사전 S를 쓰는 acoustic FxNLMS의 OFF→ON→OFF 기준선 확보. 기존 digital-ref 체크포인트로 acoustic 성능을 대신 판정하지 않는다.
6. **이 PC와 학습 자원 / 이후 Jetson**: acoustic 학습·독립 closed-loop 검증 후 하이브리드 비교. 이후 느린 DNN 계수 생성 + 빠른 FIR 및 온라인 경로 추정을 발전시킨다. 외부 학습 자원 재가동은 별도 확인한다.

---

## 0. 과거 라이브 상태 (2026-08-04 기록 — 현재 상태로 간주하지 말 것)

- **Elice 인스턴스**: `elicer@central-01.tcp.tunnel.elice.io` **포트 47863**, 2×A100 80GB.
  pem = 이 Jetson의 `~/.ssh/elice.pem` (커밋 금지). 32 vCPU, 디스크 84G 여유.
- **모든 GPU 작업이 끝났다. 회수도 완료됐다 → 인스턴스를 삭제할 것.**
  두 감독자는 `drained` 상태로 살아 있으며 `recommendation: teardown`을 표시한다.
  큐에 작업을 덧붙이면 다시 일한다(감독자가 300초마다 큐 파일을 재로드).
  - PID는 재기동하면 바뀐다. 권위 있는 소유자는 `runs/.job_queue_gpu{0,1}.lock`의 owner JSON이다.
  - **`tier` 필드는 현재 메타데이터일 뿐 강제되지 않는다.** 감독자는 큐에 적힌 순서대로
    실행한다. Tier-B를 "다른 GPU의 Tier-A ETA 안에서만"으로 제한하려면 별도 구현이 필요하다.
    지금 큐는 순서 자체가 우선순위를 반영하도록 배열해 뒀다.
- **감독자 인계 실적** — 원래대로면 GPU1은 3.4시간, GPU0은 무기한 유휴였다.
  - GPU1: 01:16:32 구 watcher 종료 → 01:17:10 작업 시작 = **유휴 38초**
  - GPU0: 04:48:34 base 종료 → 04:49:10 작업 시작 = **유휴 36초**
- **완료된 GPU 작업**: `search_tiny_control` 20k → 승자 선정 → 재판정 →
  `seed_repeat_control_20k` → `seed_repeat_tiny_long_20k` → `decide_seed_repeat`,
  base best/last held-out 평가, tiny best/last 동일조건 평가, 회수 번들.
  승자가 대조군이라 100k 연장은 하지 않았다 — `pretrain_tiny_corrected` 100k 완주본이
  이미 있어 같은 학습을 반복하는 것이 낭비이기 때문이다.

### base vs tiny 최종 비교 (2026-08-04 05:03, 동일 held-out 64 아이템)

**base(5.99M)는 tiny(1.16M)보다 나은 점이 사실상 없다. 배포 후보는 tiny다.**
두 모델을 Elice에서 같은 데이터(manifest 7종 + RIR 300)로 평가한 결과다.

| 지표 (NMSE dB, 낮을수록 좋음) | base 5.99M | tiny 1.16M | 우세 |
|---|---:|---:|---|
| trusted 대역 (150–600Hz) | **−18.99** | −18.66 | base (0.33dB) |
| fullband | −15.88 | **−17.14** | **tiny (1.26dB)** |
| held-out η=0.15 trusted | **−14.78** | −14.74 | base (0.04dB) |
| held-out η=0.15 fullband | −12.97 | **−13.97** | **tiny (1.00dB)** |
| **최악 아이템 fullband** | **+13.89 (증폭)** | **+4.06** | **tiny (9.83dB)** |
| Jetson P99 (ORT CPU) | 6.8ms **게이트 미달** | **1.84ms** | **tiny** |

소스별로는 **7종 중 7종 전부 tiny가 우세**하다.

| 소스 | base | tiny | 차이 |
|---|---:|---:|---:|
| **demand (최악 소스)** | **−4.36** | **−9.24** | tiny +4.88dB |
| synthetic | −15.38 | −19.16 | tiny +3.78dB |
| esc50 | −16.32 | −18.70 | tiny +2.38dB |
| music | −22.55 | −24.07 | tiny +1.52dB |
| dns_fullband | −15.39 | −16.60 | tiny +1.21dB |
| machine | −20.70 | −21.83 | tiny +1.13dB |
| speech | −28.68 | −29.48 | tiny +0.80dB |

옥타브밴드 감쇠는 두 모델이 사실상 동일하다(125Hz~8kHz에서 차이 0.00~0.28dB).

**절대 목표 기준의 판정**
- **기능 2는 평균이 아니라 최악값 문제**다. 최악 소스 `demand`가 base −4.36dB / tiny −9.24dB로
  둘 다 나머지 소스(−15~−29dB)보다 크게 뒤진다. **현재 어느 모델도 기능 2를 충족하지 못한다.**
  주방·세탁기·사무실·지하철 같은 지속성 실환경음이 약점이다.
- **최악 아이템에서 base는 fullband를 +13.89dB 증폭한다.** 이건 do-no-harm 위반이며
  tiny(+4.06dB)보다 10dB 나쁘다. 파라미터가 5배 많다고 안전한 것이 아니다.
- base가 이기는 것은 trusted 대역 0.33dB뿐이고, 그마저 실시간 게이트를 통과하지 못한다.

**따라서 base의 TensorRT 최적화에 시간을 쓸 근거가 현재로선 없다.** 다음 단계에서 용량을
늘릴지 판단하려면 먼저 `demand` 계열 성능과 최악 아이템 증폭을 개선해야 하며, 그것은
용량 문제가 아니라 데이터 분포·손실 설계 문제로 보인다.

> 이 수치는 전부 `secondary_surrogate` 플랜트에서 나온 **표현 사전학습 지표**다.
> 실측 P/S와 recorded 세션을 통과하기 전에는 실제 덕트 감쇠 성능이 아니다.

### 구조 탐색 결론 (2026-08-04 02:56 재판정 — 이것이 권위 있는 결과)

**20k 예산에서 어떤 구조 후보도 평범한 `tiny`를 이기지 못했다.** 승자는 대조군이다.

| 후보 (primary = `eval_pilot_last`, 동일 20k) | trusted NMSE | Δ vs 대조군 | 95% CI | 판정 |
|---|---:|---:|---|---|
| `tiny_control` (기준) | −14.59 | — | — | **승자** |
| `tiny_long` | −14.812 | −0.224 | [−0.709, **+0.255**] | 실격 아님, **유의하지 않음**(CI가 0을 가로지름) |
| `tiny_long_attn` | −12.415 | +2.173 | [+1.47, +2.94] | 실격 — fullband +1.78dB, held-out +1.30dB 악화 |
| `tiny_attn` | −12.060 | +2.528 | [+1.59, +3.53] | 실격 — fullband +2.15dB, held-out +2.12dB 악화 |

확인 지표(`eval_pilot_best`)에서는 셋 다 대조군과 거의 동률이며 유의한 후보가 없다.
`last`와 `best`의 격차가 큰 것은 attention 계열의 20k 지점 분산이 크다는 뜻이다 —
동일 예산·무편향인 `last`를 1차 지표로 삼은 이유가 여기서 드러난다.

Jetson 실측 비용축과 함께 보면 결론이 더 분명하다: `tiny` P99 1.84ms vs
`tiny_long` 2.24ms(+22%). **감쇠 이득 없이 지연만 늘어난다.**

**seed 반복으로 확인 완료 (2026-08-04 05:45)** — 결론이 재현됐을 뿐 아니라,
**효과 크기보다 seed 분산이 크다**는 것이 직접 드러났다.

| seed | `tiny_long` Δ vs 대조군 (primary=last) | 95% CI | 유의 |
|---|---:|---|---|
| 20260802 | −0.224 | [−0.709, +0.255] | 아니오 |
| 20260902 | **+0.460** | [−0.655, +1.621] | 아니오 |

두 seed 사이에서 Δ가 −0.22 ↔ +0.46으로 **0.68dB 요동**한다. 판정 마진 0.30dB보다 크므로
`tiny_long`의 이득(있다면)은 run 간 잡음에 묻히는 수준이다. **구조 탐색은 종결이며
tiny를 유지한다.**

> **남은 한계.** 20k는 100k 궤적의 앞부분일 뿐이라 **후반부에 순위가 뒤집힐 가능성은
> 배제하지 못한다.** 다만 tiny의 100k 완주본이 base(5.99M)보다도 나은 상황이라
> 지금 용량·수용영역을 더 키울 근거는 없다.

### 산출물 회수 완료 (2026-08-04 05:50) — 인스턴스 삭제 가능

`runs/queue/handoff.json`의 60건 중 **46건 SHA-256 일치**, 12건은 의도적 미회수
(기각된 0dB 초기 실험 `pretrain_{base,tiny}`와 `*_aggressive` 변형), 2건은 bundle 생성
시점(04:58)에 아직 학습 중이던 `seed_repeat_tiny_long`이라 최종본과 다르다 —
그 2건은 원격과 최종 SHA를 직접 대조해 일치를 확인했다.

로컬 `runs/`에 회수된 checkpoint의 step 검증 결과:

| run | last.pt step | best_metric |
|---|---:|---:|
| `pretrain_base_corrected` | 100,000 | **−19.755** |
| `pretrain_tiny_corrected` | 100,000 | **−19.537** |
| `search_tiny_control` | 20,000 | −16.869 |
| `search_tiny_long` | 20,000 | −16.667 |
| `search_tiny_attn` | 20,000 | −16.755 |
| `search_tiny_long_attn` | 20,000 | −16.713 |
| `seed_repeat_tiny_long` | 20,000 | −16.692 |
| `seed_repeat_tiny_control` | **1,000 (손상)** | — |

마지막 항목만 §사고 기록 ④의 덮어쓰기로 무효다. **평가 metrics는 완주 시점 것이라
유효**하며 `runs/seed_repeat_tiny_control/PROVENANCE.md`에 사용 가능/불가를 명시했다.

**→ 두 GPU 모두 유휴이고 남은 GPU 작업이 없다. 사용자는 Elice 인스턴스를 삭제할 것.**
중지만 해도 스토리지는 과금되며 삭제해야 완전히 멈춘다.

> **04:49 사고 기록 ④.** GPU0 큐와 GPU1 큐에 같은 id·같은 `ckpt_dir`의 seed 반복 작업을
> 둔 탓에, GPU1이 20k를 완주한 run을 GPU0이 다시 시작해 checkpoint를 step 1000으로
> 덮어썼다. 근본 원인이 둘이다. ① `ckpt_dir` 점유 검사가 **자기 GPU 프로세스만** 봐서
> 다른 GPU 감독자를 구조적으로 놓쳤다(→ GPU 무관 검사로 교체, 두 큐가 id/ckpt_dir을
> 공유하지 못하게 하는 테스트 추가). ② **GPU0 감독자가 00:42 기동 당시의 옛 코드를
> 메모리에 들고 있었다** — 모듈만 갱신하고 재기동하지 않아 `already_done`도 `child_env`도
> 없는 버전이 돌았다. **모듈을 갱신했으면 해당 감독자를 반드시 재기동할 것.**

> **02:47 사고 기록 ①.** 첫 승자 선정에서 후보 3종이 전부 `config fingerprint 불일치`로
> 실격됐다. 지문이 `model`/`model_config`를 포함해 **구조가 다르면 항상 불일치**했기
> 때문이다 — 구조는 실험의 독립변수인데 그것으로 실격시키면 구조 탐색이 불가능해진다.
> 지문에서 구조 키를 제외하고 재판정했고, 결론(승자=대조군)은 같지만 근거가
> "실격"에서 "유의한 개선 없음"으로 정정됐다.

> **02:56 사고 기록 ②.** 감독자를 재기동했더니 이미 20k를 완주한 `search_tiny_control`을
> 처음부터 다시 돌렸다(결과를 메모리에만 뒀기 때문). checkpoint 저장은 500 step마다라
> step 500 직전에 차단해 완주본(`last.pt` step=20000, best_metric −16.869)을 지켰다.
> `restore_results()`와 디스크 완료 검사를 넣어 고쳤다. **복원은 반드시 첫 상태 기록보다
> 앞서야 한다** — 순서가 바뀌면 `jobs={}`로 덮어쓴 빈 파일을 읽어 조용히 무력화된다.

> **01:17 사고 기록 ③ — 반드시 읽을 것.** 감독자 첫 기동에서 `spawn()`이 환경을 그대로
> 물려줘 자식이 두 GPU를 다 보고 PyTorch 기본값 `cuda:0`에 올라갔다. GPU1의
> `search_tiny_control`이 **GPU0의 base 위에 겹쳐** 7.4GB를 뺏고 base를 1.84 → 1.36 it/s로
> 떨어뜨렸다. 즉시 감독자→자식 순으로 중지하고(base는 무사) `Supervisor.child_env()`로
> `CUDA_VISIBLE_DEVICES`를 고정한 뒤 재기동했다. 자식 환경이 `CUDA_VISIBLE_DEVICES=1`이고
> GPU1 UUID에 올라간 것, base가 1.79 it/s로 회복한 것을 확인했다.
> **교훈**: 이 환경변수는 `processes_using_gpu()`의 유휴 판정 근거이기도 하다. 설정하지
> 않으면 진입 게이트의 (c) 조건 자체가 무력해져 감독자끼리 서로의 자식을 못 본다.

### 상태 확인 (가장 먼저 실행)

```bash
SSH="ssh -i ~/.ssh/elice.pem -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -o ControlMaster=auto -o ControlPath=~/.ssh/cm/%r@%h-%p -o ControlPersist=600 \
  -p 47863 elicer@central-01.tcp.tunnel.elice.io"

# 큐 상태 한 번에 (표준 라이브러리만 — 학습 CPU를 뺏지 않는다)
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'

# 원 학습 로그와 GPU
$SSH 'cd ~/Deep-ANC && grep "^step " runs/train_base_corrected.log | tail -n 1; \
  tail -n 3 runs/structure_search.log; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

`idle_seconds_total`이 이 시스템의 목적함수다. 작업 전환 시 60초를 넘으면 원인을 조사한다.
감독자 로그는 `runs/queue/supervisor_gpu{0,1}.log`, 이벤트는 `runs/queue/events.jsonl`.

### 감독자가 죽었을 때

`queue_status.py`가 `[STALE]`을 표시하면 감독자 자체가 죽은 것이다. 재기동은 안전하다 —
완료된 작업은 결과로 건너뛰고, 진입 게이트를 처음부터 다시 통과한다.

```bash
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'   # 또는 0
```

### 큐에 작업을 덧붙이려면

감독자는 **작업 사이마다 큐 YAML을 다시 읽는다.** 재시작 없이 `configs/elice/queue_gpu*.yaml`에
작업을 추가하면 반영된다. 이미 결과가 있는 id는 재실행되지 않는다.

## 1. 프로젝트 한 줄 요약

덕트(사각 아크릴 1.2m) 딥러닝 능동소음제어. 학습=Elice 2×A100, 추론=이 Jetson AGX Orin.
모델 HybridANCNet(tiny 1.16M=현행 실시간 / base 5.99M=TRT 목표), digital-ref 모드 우선.
**절대 목표 2가지: ① 저주파+고주파 노이즈 제거 ② 모든 소리 제거(quiet zone)** — AGENTS.md 참조.
상세: docs/00, 물리: docs/01, 구조 지도: docs/10, 목표 측정: docs/07 §0.

## 2. 현재 상태

### 완료 ✅

- 저장소 골격·문서·자동 테스트, GitHub `Roka-jsj/Deep-ANC` 공개 운영
- 19-에이전트 리뷰(결함 15건) + 5-에이전트 구조 감사(이슈 35건) 반영
- 물리 정합 학습 목표(digital-ref lead 109, P(z) resolver, trusted-band 150–600Hz NMSE)
- 원샷 부트스트랩 `scripts/elice/bootstrap_all.sh` (환경+데이터 6종+RIR+QA+테스트+2GPU 학습)
- 데이터: DNS 16,000 / speech 8,065 / music 7,997 / MIMII 3,600 / ESC-50 2,000 / DEMAND 96
  (약 154.9시간). 손상 FMA MP3 3개는 manifest에서 제외
- recorded group-aware manifest·전수 QA·독립 evaluator·파인튜닝 fail-fast **구현 완료**
- **tiny 100k 완주·로컬 회수**: 최종 val trusted **−19.47** / full −18.27dB,
  best step89,500 trusted **−19.5372dB**. 원격 SHA-256 일치
- **tiny_long 20k 완주·로컬 회수**: best step13,500 trusted **−16.6672dB**
- **구조 탐색 종결(잠정)**: 후보 3종 중 어느 것도 20k에서 tiny를 이기지 못했다(§0 표).
  attention 계열은 fullband·held-out 일반화를 1dB 넘게 해쳐 do-no-harm 실격이다.
- **Jetson 실측 (2026-08-04)**:
  - `tiny` best.pt → ONNX, ORT 등가 `8.196e-08`, **P99 1.84ms** (게이트 <3ms 통과)
  - `tiny_long` last.pt → ONNX, ORT 등가 `7.567e-09`, **P99 2.24ms** (통과)
  - 즉 tiny_long은 수용영역 2배에 P99 +0.40ms(+22%). 구조 비교의 비용축 확보
- **GPU 작업 큐 감독자 신규 구현·배포** (`src/deep_anc/ops/job_queue.py`):
  기존 프로세스 불가침 4중 진입 게이트, 실패 격리(작업 하나 실패해도 큐 계속),
  사전 등록 승자 선정, 원자적 상태 JSON, 큐 재로드
- **파인튜닝 진입점 완성**: `--state-dir` 배선, `pipeline.lock`, advisory `status.json`,
  exit code 3/4 분리, 상대 config 경로 fail-open 수정.
  검증: `--check-only` → **exit 1 + `runs/` 미생성** (설계대로 NOT READY)
- **base 100k 완주** (04:48 KST): 최종 val trusted **−19.73** / full −17.37dB,
  best trusted **−19.75dB**. held-out 64아이템 재평가 완료
- **base vs tiny 동일조건 비교 완료** — §0 표. **배포 후보는 tiny 로 확정**

#### 2026-08-04 오후 세션에서 완료한 것

- **G2 통과 — recorded 데이터셋 수집 완료**: **80세션 / 93.3분 / 4계열 각 20개 / 64그룹**,
  분할 train 64 · val 9 · test 7, 전수 QA **80/80 PASS**.
  - `build_recording_sources.py` — 계열별 소스 WAV, tanh 소프트 클리핑으로 크레스트 10dB 제한
  - `record_session_batch.py` — 재개 가능 배치, 세션마다 즉시 QA, 일시적 xrun 1회 재시도
  - `record_duct.py` — settle 1초, **xrun 발생 세션은 저장 자체를 거부**
- **실기 ANC 시연**: 음성+80–800Hz 소음, OFF 10초 → ON 20초 → OFF 5초.
  소스대역 **+4.39dB**(보수적 기준), trusted NMSE −5.66dB. `results/session_20260804_125538/`
- **I²S 기동 트랜지언트 오염 수정** (3개 경로): 첫 0.5초가 −36.3dBFS/peak 0.062라
  실제 바닥(−67.4dBFS/peak 0.002)을 18dB 가리고 있었다. **죽은 마이크도 게이트를 통과하던
  결함**이라 회귀 테스트로 막았다. 재생 진폭도 0.15 → 0.06으로 낮췄다.
- **G4의 기능 2 인코딩 수정**: 소스별 **평균**이 아니라 **최악값**으로 판정한다.
  기존 게이트는 "음성을 6dB 증폭하지만 나머지를 잘 잡는" 모델을 통과시켰다.
- **파인튜닝 진입점 base → tiny 전환** (`train_finetune.yaml`, 테스트, 문서).
  게이트가 열리는 순간 **배포하지 않을 모델**을 학습하게 되어 있었다.
- **동시 인터리브 P/S 측정 도구 신규** — `scripts/data/measure_paths_interleaved.py`,
  `src/deep_anc/dsp/interleaved_probe.py`(자극 설계 + IR 복원 + warp 추적/역보정).
  게이트에 `interleaved_multitone` 방식을 추가하되 **ESS 보다 좁게** 검사한다
  (guard=1, 분석창 ≤2초, 톤 수, 톤 SNR, 그리고 두 파일의 `capture_id` 일치).
- **전체 회귀 테스트 336개 통과** (세션 시작 기준선 273 → +63)
- **README 전면 정리 + 그림 4종** — 덕트 도면(SVG, `duct.yaml`에서 생성), 지연 예산,
  실기 ANC 시연 파형/스펙트럼, 데이터셋 구성. 전부
  `scripts/docs/render_readme_figures.py`가 **실측 산출물에서 재생성**한다.

### 대기 ⬜ (다음 세션이 할 일 순서)

**파인튜닝 준비 상태: NOT READY. 블로커는 G1 하나뿐이다.**

```
[FAIL] finetune_readiness  (5 PASS / 4 FAIL — FAIL 4개가 전부 같은 뿌리)
  [PASS] config_fail_closed_flags / measured_primary_mode / recorded_mix_ratio
  [PASS] completed_init_checkpoint / recorded_dataset_qa
  [FAIL] official_secondary_path / official_primary_path
  [FAIL] matched_path_measurement_conditions / path_delay_and_lead   ← 위 둘에 종속
```

1. **G1 — 실측 P/S. 이것만 되면 READY다.** 2026-08-04 측정으로 원인이 확정됐다:
   재생(USB DAC)과 녹음(I²S)이 다른 클록 도메인이라 **1초 창 안에서 대응이 100–200샘플
   움직인다.** 신호 자체는 문제가 없다 — 톤 SNR 29.5dB, 대역 내 에너지 97.4%,
   반복 간 `\|H\|` 비 **1.000**. 깨진 것은 시간축 하나다(위상 직선적합 잔차 1.8–3.9 rad).

   | 접근 | 반복 일관성 (요구 0.9) |
   |---|---:|
   | 순차 ESS | 0.08–0.17 |
   | 동시 인터리브, 보정 없음 | 0.05 |
   | 동시 인터리브 + warp 역보정 (창 43ms) | **0.84 / 0.85** |
   | + 궤적 평활 | 0.54 (악화 — 요동이 실재한다는 증거) |

   다음에 시도할 것 (기대순): ① 추적 창을 43ms 아래로 내리면서 상관 첨두가 흐려지는
   지점을 찾기 ② `latency=low`로 재측정(현재 `high`는 버퍼가 커서 warp가 더 실릴 수 있다)
   ③ 재생·녹음을 **한 장치**로 모으는 방안 검토(하드웨어 변경이므로 최후의 수단).
   원자료는 `results/calibration_interleaved/20260804_132812_d1479bae/`에 있다.
   **게이트를 낮추지 않는다.** 성공한 반복만 골라 저장하는 우회도 하지 않는다.
2. G1 통과 후: `duct.yaml`의 `primary_path_npz` / `d_noise_delay_samples` 기입 → `lead` 재계산
   → `check_finetune.py` READY 확인 → Stage-2 open-loop 파인튜닝(tiny) → recorded G4.
3. 남은 감사 항목 (게이트와 무관, 언제든 가능):
   - `finetune_readiness.py`의 완료 판정이 `schedule.total_steps`를 권위로 쓰도록 —
     지금은 20k 파일럿이 "완료"로 잡힐 수 있다
   - `trainer.py`의 `freeze_encoder`를 DDP 래핑 **앞**으로 이동
   - `recorded_qa.py`의 최소 RMS(−80dBFS)가 실제 바닥(−67.4dBFS)보다 낮다 → SNR 여유로 전환
   - `make_recorded_manifest.py`가 `batch_progress.csv`의 판정을 무시한다

### 승자 연장 작업의 규약 (감독자가 자동 적용 — 참고용)

`configs/elice/queue_gpu1.yaml`의 `extension_template`에 확정돼 있다. 손으로 만들 일이
생기면 다음 세 가지를 반드시 지킨다.

- **`ckpt_dir`은 새 디렉터리** — 같은 곳에 resume 하면 pilot의 20k `best/last`를 덮어써
  구조 비교 근거가 사라진다.
- **`resume`은 `last.pt`** — `best.pt`로 되감으면 optimizer/scheduler가 후퇴해 예산을 낭비한다.
  대신 pilot의 `best.pt`를 새 ckpt 디렉터리로 **복사**해야 `trainer.py:390`의 `best_metric`
  min() 교정이 동작한다(복사하지 않으면 20k best를 넘기 전까지 `best.pt`가 아예 없다).
- **`seed`는 원 seed +100** — worker RNG는 checkpoint에 저장되지 않고 iterator 생성 시
  `seed + split_offset + worker_id*1009`로 재시드된다(`synth_dataset.py:269`). 그대로 두면
  step 20k–40k가 0–20k와 **같은 데이터를 재생**한다. (worker RNG를 checkpoint에 저장하는
  근본 수정은 별도 후속 항목이다.)
- `run_until_step`은 지정하지 않는다 → `resolve_run_until_step()`이 `total_steps` 폴백.

### 데이터/체크포인트 선택 주의

- 내장 val은 고정 16개(현재 seed에는 DEMAND 0개)라 최종 판정용이 아니다.
  `best.pt`만 맹신하지 말고 `last.pt`도 회수한다.
- 공개 데이터의 파일 단위 split은 speech 화자/책, ESC 원본, MIMII 조건, DEMAND 동시녹음
  채널 같은 상관 그룹이 split을 가로지를 수 있다.
- `secondary_surrogate` checkpoint는 표현 사전학습 전용이라 물리 성능 주장에 쓸 수 없다.
- **로컬 Jetson 오프라인 평가는 제한적이다.** `data/manifests/`와 RIR 뱅크가 없어서
  소스별 표에 `synthetic`만 남고 RIR이 즉석 32개로 대체된다. 즉 **기능 2(모든 소리)는
  로컬에서 측정할 수 없다.** 승자 선정과 소스별 평가는 반드시 Elice에서 돌린다.

### 사용자가 직접 해야 하는 것 (권한/자원 소유)

- **I²S 입력 복구**: 전원 OFF에서 공통 GND/SD/LR·pin17 접촉 확인. 이후 §3-C의 무출력
  probe 2개가 clip 0으로 반복 PASS해야 실측 재개 가능. **이것이 풀리기 전에는 실측 P/S,
  recorded 세션, 파인튜닝이 전부 막혀 있다.** Elice 사전학습 완료가 이 게이트를 대체하지 못한다.
- **Elice 인스턴스 중지/삭제**: 큐가 `drained`가 되고 회수가 끝나면 즉시. 인스턴스 켜진
  시간으로 과금되며, 중지해도 스토리지는 과금되고 삭제만 완전 중지다.
- 파인튜닝 현장 준비: AB13X·두 마이크·두 스피커 고정, 같은 출력게인의 S(cancel→ERR),
  P(noise→ERR), THD/IMD 측정. 실제 소스 독립 세션 최소 80개(1.5–2h/3–4GB),
  권장 160개(3–4h/6–8GB)와 10–15GB 여유 공간.
- 덕트 미확정값 확정 시 통보 (에러마이크 X=1.100 잠정 — 확정 시 duct.yaml + RIR 뱅크 재생성)

### 사용자가 확정한 INMP441 물리 배선 (2026-08-03)

- 두 마이크 공통: VDD 빨강→J30 pin1(3.3V), GND 검정→pin6, SCK 주황→pin12,
  WS 노랑→pin35, 두 SD 공통 갈색→pin38
- 레퍼런스 마이크 L/R 초록→pin17(3.3V, right/ch1), 에러 마이크 L/R 파랑→pin39(GND, left/ch0)
- 공식 핀표와 INMP441 L/R 규약 대조 완료. 안전 주의·근거는 docs/02 §1 참조.
- 현재 pinmux/I²S는 의도된 기존 구성이다. **sudo, Jetson-IO, pinmux, device-tree,
  오디오 데몬, 전원모드 변경은 모두 금지.**

### 현재 I²S·출력 경로·P/S 실측 상태

- APE 입력 `hw:1,1`과 AB13X 출력 `hw:2,0`은 장치로 인식되고 스트림 설정도 수락된다.
- pin17 재연결 직후 5초 probe는 ERR **−46.33dBFS**, REF **−46.64dBFS**, clip 0%로 PASS했다.
- **하지만 22:39 이후 다시 FAIL이다.** 무출력 2초 probe에서 ERR −12.84dBFS/clip 2.474%,
  REF −10.67dBFS/clip 5.029%. 10초 재검사도 FAIL. peak/raw가 정확히 ±1.0/INT32 한계까지
  도달하고 0.1초 구간별 간헐 burst가 있어 시작 과도가 아니다. 장치 점유 프로세스는 없었다.
  **스피커 출력은 전혀 시작하지 않았고 전달맵/direct FxLMS 실측은 안전 중단 상태다.**
- peak 0.005 채널 분리 실측: noise ch0와 cancel ch1 모두 ERR/REF에 도달. tone-bin 상승은
  ch0→ERR/REF +25.79/+26.15dB, ch1→ERR/REF +22.83/+28.34dB. REF 기준 ch1이 ch0보다 약
  8.3dB 강해 실제 앰프 게인/물리 매핑 차이는 별도 확인이 필요하다.
- magnitude 진단은 204–210Hz, 348–351Hz, 458–476Hz, 594–613Hz 부근 피크를 반복 검출해
  1D 예측 공진(210/350/489/629Hz)을 부분 지지한다. 이는 **공진 형상 진단**일 뿐 덕트 식별
  완료, 고정 지연, FxLMS 성능 또는 물리 좌표 확정의 근거가 아니다.
- legacy 300Hz 설정의 과거 "약 2dB 감소"는 현재 하드웨어에서 재검증한 결과가 아니다.
  2026-08-03 재현은 실제 `ON/ADAPT`까지 확인했지만 ON 로그 9개에서 중단됐고 감쇠 중앙값
  +0.11dB, 순간 최대 +0.63dB였다. **2dB 성공 증거가 아니다.**
- 19:14에 읽기 전용 감사 에이전트가 legacy 모듈을 import하면서
  `/home/capston/anc_project/__pycache__/main_realtime_anc.cpython-310.pyc` 한 개를 실수로
  생성했다. 그 경로는 이후 건드리지 않았고 **임의 삭제도 하지 말 것.**
  저장소의 안전 실행기는 `python3 -B`와 `PYTHONDONTWRITEBYTECODE=1`로 재발을 막는다.

## 3. 실행 절차

### A. GPU 작업 큐 (현행 운영 방식)

```bash
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'                    # 상태
$SSH 'cd ~/Deep-ANC && .venv/bin/python scripts/elice/job_queue.py plan \
  --queue configs/elice/queue_gpu1.yaml'                                          # 예정 순서(GPU 무접촉)
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'                     # 기동/재기동
```

> **원격 배포는 반드시 신규 경로만 scp한다.** 원격 워크트리는 HEAD가 아니라 워킹트리
> 스냅샷(dirty 37개)이라 `git pull`은 merge abort되거나, stash/checkout으로 우회하는 순간
> corrected physics 코드가 revert되어 **돌고 있는 학습이 조용히 다른 실험이 된다.**
> 또 실행 중인 bash 스크립트를 scp로 덮어쓰면 (scp가 inode를 truncate) bash가 오프셋 기준
> 지연 읽기를 하므로 **watcher가 즉사한다.** `cp -n`/`cp -rn`으로 no-clobber 설치할 것.

### B. 학습 완료 후 → Jetson 배포

```bash
mkdir -p ~/Deep_ANC/runs/pretrain_base_corrected/ckpt
scp -o ControlPath=~/.ssh/cm/%r@%h-%p -P 47863 \
  elicer@central-01.tcp.tunnel.elice.io:~/Deep-ANC/runs/pretrain_base_corrected/ckpt/best.pt \
  ~/Deep_ANC/runs/pretrain_base_corrected/ckpt/
# 회수 목록과 SHA-256 은 감독자가 runs/queue/handoff.json 에 미리 만들어 둔다
# → 이 시점에 사용자에게 "인스턴스 중지/삭제" 안내!

cd ~/Deep_ANC && .venv/bin/python scripts/train/export_onnx.py \
  --ckpt runs/pretrain_base_corrected/ckpt/best.pt --out runs/export/base.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
  --set engine.type=ort --set engine.onnx=runs/export/base.onnx
```

`secondary_surrogate` 결과는 오프라인 표현 평가까지만 한다. 실제 스피커 실행 전에는 같은
게인의 실측 P/S로 파인튜닝하고 runtime `digital_reference_lead_samples=109`를 checkpoint
메타와 맞춘다.

### C. 하드웨어 재연결 후 (사용자 입회) — docs/02 §3 + docs/08 §5

```bash
# 1) 스피커를 전혀 열지 않는 입력 게이트. 현재는 FAIL 상태이므로 여기서 중단.
cd /home/capston/Deep_ANC
.venv/bin/python scripts/bench/check_audio_input.py
.venv/bin/python scripts/bench/check_audio_input.py --require-both

# 2) 둘 다 clip 0으로 반복 PASS한 뒤 네 경로 시간-주파수 지도
.venv/bin/python scripts/bench/measure_duct_transfer_map.py --confirm-volume-minimum

# 3) 전달맵 뒤 저음량 direct FxLMS 진단. legacy S라 결과는 diagnostic-only.
.venv/bin/python scripts/demo/evaluate_fxlms_direct.py \
  --amplitude 0.005 --control-limit 0.005 --confirm-user-present-volume-minimum
```

legacy 재현이 필요하면 원본을 기본값으로 실행하지 않는다. 종료 시 weight를 저장하므로
`--weights-output`을 이 저장소의 ignored `results/`로 우회하고 `python3 -B`를 쓴다.

### D. 파인튜닝 (게이트 통과 후)

```bash
.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml --set data.digital_primary_path_mode=measured
```

현재는 설계대로 **exit 1 (NOT READY)** 이며 `runs/` 아래에 아무것도 만들지 않는다.
exit code 표와 산출물 경로는 README §6.6 참조.

## 4. 조심할 것 (세션에서 배운 것)

- Elice 터널: 로컬 타임아웃이 나도 **원격 작업은 대부분 살아있다** — 재실행 전 반드시 상태 확인.
  원격 장기작업은 `setsid nohup … < /dev/null &` 패턴만.
- Elice는 **인스턴스 켜진 시간 과금** (SSH 연결 여부 무관). 종료(중지)해도 스토리지는 과금,
  삭제만 완전 중지.
- `tail -n 1` (구식 `tail -1`은 다중 파일에서 GNU 오류)
- 원격에서 pytest를 돌릴 때는 `nice -n 19`. 32 vCPU 중 28개를 두 학습의 DataLoader가 쓴다.
- Jetson venv 재구성은 `scripts/jetson/setup_jetson.sh` (lib preload 훅 필수), ORT 1.18.1 고정
- 입력 raw가 `-1`/0 고정이면 장치가 열려도 유효 오디오가 아니다. 무출력 preflight 실패를
  `--force`로 우회하거나 스피커 출력으로 진단하지 않는다.
- S(z)/핸드오프/목표대역은 단일 출처 원칙 (duct.yaml + config.DEFAULT_HANDOFF_SAMPLES)
- 컨테이너에서 `nvidia-smi --query-compute-apps`는 비어 보일 수 있다. GPU 유휴 판정은
  memory.used와 `/proc/*/environ`의 `CUDA_VISIBLE_DEVICES`까지 함께 봐야 한다.
- `/proc/<pid>/stat`의 starttime은 **마지막 `)` 뒤부터 세어 tail[19]** 다. comm 필드에
  공백이 들어갈 수 있어 단순 split은 틀린다. PID 재사용 판별에 이 값이 필요하다.

### 읽기 전용 참고 구현

- `~/anc_project` — legacy FxLMS(Python). block FxNLMS, S(z)는 `calibrate_s_path.py`로
  오프라인 식별(256/low에서 순수지연 1342, coherence 0.40으로 낮음). 종료 시 기본
  `control_filter_last.npy`를 CWD에 저장하므로 원본에서 기본값 실행 금지.
- `~/FxLMS/realtime_fxlms` — C++ 단일 blocking `snd_pcm_readi`→계산→`writei`, block512,
  S 2048tap, W 512tap, 내부 230+370Hz digital tone reference. FxLMS 부호는 Deep_ANC의
  `e=d+S·y`와 일치하지만 REF ch1은 실제 제어에 쓰지 않는다. APE 입력/USB 출력이 독립 clock인데
  timestamp가 없고 partial write/xrun 뒤에도 계속하므로 **절대 지연·위상 근거로 쓸 수 없다.**
  `run_experiment.sh`는 PCM 100%로 바꾸므로 볼륨 최저 규칙과 충돌해 **실행 금지**다.
  기존 `fxlms_run_log.csv`의 ANC control RMS는 거의 0.299로 ±0.3 hard-limit에 포화됐고
  표시 감쇠는 −1.80~+5.69dB로 요동해 **2dB 성공 증거가 아니다.**
- 두 폴더 모두 앞으로도 읽기 전용. python import 금지(`python3 -B`).
