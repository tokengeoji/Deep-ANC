# 14. 현재 PC와 Jetson의 작업 분담·현장 실행표

> 기준일: 2026-09-15. 목표는 기존 Jetson AGX Orin·덕트에서 acoustic REF를 사용하여
> ERR 한 점의 저역과 1 kHz 이상 소리, 음성·음악까지 감쇠시키는 것이다.
> 모든 코드 작업·분석·Python 실행은 Docker 안에서 한다. 호스트 `.venv`는 사용하지 않는다.

## 1. 어디에서 무엇을 할 것인가

| 작업 | 수행 위치 | 필요한 입력·산출물 |
|---|---|---|
| 코드 수정, 설정 검증, 전체 CPU 테스트 | **현재 PC, CPU Docker** | 저장소 → 테스트 결과·코드 변경 |
| 지연 예산·S 신뢰대역 진단 | **현재 PC, CPU Docker** | 설정·기존 NPZ → 부족한 조건 보고 |
| 합성 플랜트의 FxNLMS/하이브리드 회귀 | **현재 PC, CPU Docker** | 합성 신호 → 극성·지연·적응·장애 처리 검증 |
| Jetson 이미지·CUDA·TensorRT 호환성 | **실제 Jetson, Jetson Docker** | 기존 JetPack/L4T → 라이브러리 검사 결과 |
| 실제 추론 시간·콜백 마감·클록 변동 | **실제 Jetson** | 실제 장치·실행 조건 → 지연 분포·xrun 기록 |
| REF/ERR 입력, S·F 전달경로·THD/IMD | **Jetson과 덕트 현장** | 연결된 장치 → 원시 녹음·반복 측정·메타데이터 |
| 외부 소리의 OFF→ON→OFF 감쇠 | **Jetson과 덕트 현장** | 사전 측정 S → acoustic FxNLMS 기준선 |
| 회수 자료의 품질·대역·왜곡 분석 | **자료 회수 후 현재 PC** | 현장 원자료 → 독립 분석·다음 측정 항목 |
| acoustic 데이터 처리·학습·계수 생성 개발 | **현재 PC에서 가능한 규모부터** | 검증된 자료 → 코드·소규모 검증·학습 산출물 |

Jetson 현장이 준비되지 않아도 현재 PC의 테스트·경로 진단·합성 회귀·후처리 개발은 계속한다.
현재 PC 결과를 Jetson 실시간 성능이나 덕트 감쇠 성능으로 표기하지 않는다.
설계 배경은 [docs/13](13_acoustic_hybrid.md), 환경 관리는 [docker/README](../docker/README.md)를 따른다.

## 2. 현재 PC에서 바로 수행할 작업

실행 중인 CPU 개발 컨테이너에서 다음을 확인한다. 환경이 없으면 Docker 문서의 `build cpu`와
`up cpu`, 중지 상태면 `start`를 사용한다. Python 패키지를 호스트에 설치하지 않는다.

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh exec .venv/bin/python -m pip check
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --json
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/check_acoustic_readiness.py --require-band 1000 1600 --require-broadband
```

기본 readiness는 보고용이다. exit 0을 준비 완료로 해석하지 않는다.
마지막 명령의 exit 1은 요청한 광대역·고역 조건의 미달일 수 있으며, CPU 개발을 멈추는 사유가 아니다.
S 파일이 없거나 메타데이터가 잘못되면 그 결손을 기록하고 합성 회귀는 계속한다.

관련 코드를 수정했을 때의 집중 검사 대상은 다음과 같다. 전체 검사가 이미 통과했고
추가 변경이 없다면 같은 검사를 반복할 필요는 없다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q \
  tests/test_acoustic_readiness.py tests/test_acoustic_runtime.py tests/test_hybrid_engine.py
```

이 검사는 외부 REF 기준선, 지연된 ERR에 대한 적응, 클립·xrun·입력 손상 대응 등을 다룬다.
합성 하이브리드 수렴은 실제 ERR를 입력으로 받는 DNN의 폐루프 안정성 증명이 아니다.
후속 개발은 acoustic 녹음의 후처리 규격, 실측 경로 기반 재현, 느린 계수 생성 API 순으로 진행한다.
온라인 S/F 추정, F 보상, 학습된 계수 생성기는 아직 구현·검증 완료 상태가 아니다.

## 3. Jetson 현장 작업의 공통 선행조건

- [ ] 실제 ARM64 Jetson AGX Orin이며 기존 JetPack/L4T와 Docker가 준비되어 있다.
- [ ] 호스트 RT 커널·NVIDIA 드라이버·전원 모드·핀 설정·오디오 서비스를 변경하지 않는다.
- [ ] **기본 `dev.sh up jetson`은 오디오 장치를 노출하지 않는다.** 장치 연결을 확정하고
      해당 장치에 접근하는 컨테이너 구성을 마련하기 전까지 아래 오디오 작업은 보류한다.
- [ ] `/dev/snd` 접근, ALSA 카드/PCM, ERR/REF 채널, 출력 채널, UID/GID를 실제 장치와 대조한다.
      현재 관리 스크립트에 없는 오디오 연결 옵션을 가정하지 않는다.
- [ ] 스피커를 구동하는 단계는 사용자 입회·물리 앰프 볼륨 최소 상태에서만 진행한다.
- [ ] 실험을 시작할 때 ANC는 OFF다. 소리 출력 없이 준비 상태만 확인하는 단계와 구분한다.
- [ ] 측정마다 새 결과 경로를 사용한다. 기존 NPZ·녹음·체크포인트를 덮어쓰지 않는다.
- [ ] `~/anc_project`를 실행하거나 수정하지 않는다. 저장소의 도구만 사용한다.

아래 `SESSION_ID`는 실제 실험의 고유 식별자로 바꾼다. 기존 세션 식별자를 재사용하지 않는다.
§4-B 이후의 `.venv/bin/python` 명령은 **장치 연결을 준비한 Jetson 컨테이너 내부**에서 실행하는 예시다.
현재 PC나 기본 오디오 미노출 컨테이너에서 실행할 수 있다는 뜻이 아니다.

## 4. Jetson에서만 수행할 단계

### A. 이미지·라이브러리 검증 — 오디오 불필요

```bash
bash scripts/docker/dev.sh build jetson
bash scripts/docker/dev.sh up jetson
bash scripts/docker/dev.sh exec .venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
bash scripts/docker/dev.sh exec .venv/bin/python -c 'import tensorrt; print(tensorrt.__version__)'
```

**산출물:** 이미지 ID, Jetson/L4T 정보, PyTorch·CUDA·TensorRT 검사 출력.
현재 x86 CPU 검증은 이 단계를 대체하지 않는다. 실제 Jetson 이미지·CUDA·TensorRT 검증은 미완료다.
**중단 조건:** 이미지 호환성·GPU 접근 실패 시 GPU 배포를 보류한다. 호스트 시스템 변경으로 우회하지 않는다.
라이브러리 import 성공과 실제 모델의 스트리밍 추론 검증도 별도 항목이다.

### B. 장치와 두 마이크 확인 — 스피커 출력 없음

오디오 접근 구성이 확정된 뒤 컨테이너 안에서 장치 목록과 무출력 입력을 확인한다.

```bash
.venv/bin/python -m deep_anc.realtime.run_realtime --list-devices
.venv/bin/python scripts/bench/check_audio_input.py --require-both --max-clip-ratio 0
```

**산출물:** ERR/REF별 RMS·peak·clip·raw 범위, 실제 장치와 채널 매핑.
**통과 조건:** 두 채널 모두 반복해서 유효하며 클리핑이 없다.
**중단 조건:** 무신호·포화·비정상 raw 값·채널 혼동. 입력 실패를 소리 출력으로 우회 진단하지 않는다.

### C. S(CS→ERR)를 먼저 측정

다음 예시는 현재 설정과 같은 block 256 / latency low 조건의 80–1600 Hz 측정이다.
실행 설정이 다르면 먼저 측정·런타임 조건을 일치시킨다. 측정 지연 메타만 임의로 바꾸지 않는다.

```bash
.venv/bin/python scripts/data/calibrate_wideband.py \
  --output-channel cancel --band 80 1600 --amplitude 0.005 --repeats 3 \
  --block-size 256 --latency low \
  --out results/acoustic_SESSION_ID/secondary_path.npz \
  --diagnostics-root results/acoustic_SESSION_ID/calibration \
  --confirm-volume-minimum
```

**산출물:** 품질 게이트 통과 시 새 S NPZ, 성공·실패와 관계없이 원시 출력·ERR/REF·진단 메타데이터.
**중단 조건:** 반복 일관성 미달, xrun, 클리핑, 지연 불안정, 기존 출력 파일 존재.
실패한 경로를 정식 S로 사용하거나 성공 반복만 골라 통과시키지 않는다.
현재 ESS 산출물에는 `consistency_band_hz`가 없으므로 고역 readiness가 자동 통과하지 않는다.
대역별 반복 검증을 거친 새 메타데이터 산출물을 준비해야 하며 가진 대역으로 임의 대체하지 않는다.

### D. REF 선행 시간과 F(CS→REF) 영향 확인

```bash
.venv/bin/python scripts/bench/measure_duct_transfer_map.py \
  --block-size 256 --latency low --amplitude 0.005 \
  --out-prefix results/acoustic_SESSION_ID/transfer_map --plot \
  --confirm-volume-minimum
```

이 도구는 NS→REF/ERR와 CS→REF/ERR 네 경로의 IR·주파수응답·상대 시간·콜백 시각을 기록한다.
현재 정식 전달맵 대역은 80–1600 Hz다. 1.6 kHz는 도구의 검증 범위이며 ERR 감쇠의 절대 상한이 아니다.
**산출물:** NPZ·JSON·Markdown·선택 PNG. F의 진단 근거가 생기지만 F 보상용 런타임 통합까지 완료되지는 않는다.
**중단 조건:** 전달맵 품질 미달, 선행 시간·클록 안정성 불명, REF 오염으로 안정성 판단 불가.
THD/IMD와 레벨별 비선형성은 별도 측정 설계가 필요하다. 이 명령이 자동으로 모두 측정하지 않는다.

### E. 신규 경로를 검토하고 acoustic FxNLMS 기준선 녹음

먼저 새 S에 대해 보고와 대역 조건을 검사한다. 분석은 현재 PC로 넘겨도 된다.

```bash
.venv/bin/python scripts/bench/check_acoustic_readiness.py \
  --set duct.secondary_path.npz=results/acoustic_SESSION_ID/secondary_path.npz --json
```

S의 실제 측정 조건·극성·유효대역을 확인한 후 아래 명령으로 시작한다.
**광대역 인과 여유나 고역 검증이 부족하면 전체 목표를 충족했다고 판정하지 않는다.**
제한된 대역의 주기 신호 진단을 하더라도 그 범위와 한계를 기록한다.

```bash
test ! -e results/acoustic_SESSION_ID/fxlms_trial_01.npz && \
.venv/bin/python -m deep_anc.realtime.run_realtime \
  --config configs/runtime_acoustic.yaml \
  --set duct.secondary_path.npz=results/acoustic_SESSION_ID/secondary_path.npz \
  --run-seconds 45 --record results/acoustic_SESSION_ID/fxlms_trial_01.npz
```

현장에서 10초 OFF → `A`로 ON 30초 → `A`로 OFF 5초를 기록한다. `Q`는 종료다.
설정의 `reference=mic`, digital lead 0, `noise.enabled=false`를 유지한다.
외부 소리만 평가하는 세션에서 `N`으로 내부 소음 재생을 켜지 않는다.
**산출물:** `fs`, `err`, `ref`, `source`, `control`, `anc_gain`이 담긴 녹음 NPZ.
별도로 소스 종류·위치·출력 레벨·실행 설정·S 파일 해시·터미널 통계를 남긴다.
**중단 조건:** 발산·반복 클립·xrun·출력 누락·입력 손상. 자동 OFF 후 원인 확인 없이 다시 켜지 않는다.
같은 경로에 다시 녹음하면 기존 파일이 덮어써질 수 있어 예시의 파일 존재 검사를 유지한다.

### F. 실시간 비용과 장기 하이브리드 검증

```bash
.venv/bin/python scripts/bench/measure_inference_latency.py \
  --config configs/runtime_acoustic.yaml \
  --set duct.secondary_path.npz=results/acoustic_SESSION_ID/secondary_path.npz --steps 1000
```

이 명령은 오디오 없이 엔진 `step`의 P50/P90/P99/max를 측정한다.
현재 FxNLMS 엔진은 기본 적응 OFF이므로 이 벤치마크가 실제 적응·콜백 전체 비용을 대신하지 않는다.
현장 ON 녹음에서 적응 상태·콜백 마감·xrun을 추가로 확인한다.
하이브리드는 acoustic 학습 artifact와 독립 폐루프 시험이 준비된 뒤 같은 조건으로 비교한다.
현재 placeholder 설정이나 기존 digital-ref 모델로 acoustic 하이브리드 배포를 진행하지 않는다.

## 5. 자료를 회수하면 현재 PC가 맡을 작업

1. 원시 NPZ·메타데이터·실행 설정·S 모델·터미널 로그를 세션별로 회수한다.
   원본은 보존하고 재분석 결과는 별도 새 디렉터리에 만든다. 파일 해시로 회수 일치 여부를 확인한다.
2. REF/ERR 클리핑·무음·비정상값, 출력 포화, 시간 정렬, 구간별 이상과 측정 조건을 검사한다.
3. OFF/ON 전환과 S 잔향을 제외하고 저역·1 kHz 이상·전체 대역의 감쇠와 최악 구간을 계산한다.
   외부 음원의 크기 변화가 감쇠처럼 보이지 않도록 반복 세션과 REF 변화를 함께 대조한다.
4. 전달맵의 반복별 위상·일관성·선행 시간·F 유입을 분석하고 검증 가능한 S 대역을 정한다.
   실제 시간축 불안정과 비선형 왜곡을 구분하여 다음 현장 측정 항목을 좁힌다.
5. 검증한 실측 경로로 FxNLMS·초기 하이브리드의 오프라인 재현을 추가하고 회귀 테스트를 만든다.
   후처리와 데이터 변환, acoustic 학습의 소규모 검증은 현재 PC에서 진행한다.

현재 `run_realtime --record`의 NPZ에는 `trusted_band_hz`가 없다.
따라서 이를 해당 키가 필요한 `scripts/eval/rebuild_session_artifacts.py`에 바로 넣지 않는다.
실측 S와 세션 조건을 결합하는 후처리 연결은 현재 PC에서 구현할 다음 작업이다.
기존 recorded QA/파인튜닝의 manifest 규격과 acoustic 녹음도 자동 호환된다고 가정하지 않는다.
자료가 부족하면 부족한 필드·실험을 명시하고 가능한 신호 품질 분석과 합성 검증을 계속한다.

## 6. 결과를 남기는 기준

- 저역과 1 kHz 이상, 환경소음·기계음·음성·음악의 결과를 각각 남긴다. 전체 평균만으로 성공을 선언하지 않는다.
- S의 검증 대역 밖 결과는 미검증으로 표시한다. `e = d + S·y` 극성과 실제 핸드오프를 유지한다.
- CPU 테스트, Jetson 추론 벤치마크, 실제 acoustic 감쇠를 구분하고 원자료 경로·설정·코드 버전을 연결한다.
- 과거 [docs/12](12_system_summary.md)의 수치를 이번 구조의 신규 실기 결과로 재사용하지 않는다.
- 하드웨어 변경 검토는 가능하지만 제품 구매·장치 교체·호스트 시스템 변경이 완료된 것으로 기록하지 않는다.
