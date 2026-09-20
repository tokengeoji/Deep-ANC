# 05. Elice Cloud 학습 가이드

실행 중인 학습·접속 정보·회수 산출물은 [HANDOFF](../HANDOFF.md)에서 확인한다. 이 문서는
학습 절차와 계약을 설명하며 특정 GPU 수·PID·step·ETA를 현재 상태로 고정하지 않는다.
모든 코드·Python·테스트는 승인된 **Docker 컨테이너 내부**에서 실행한다.

## 1. 환경과 데이터 준비

Elice 학습 의존성은 `requirements-train.txt`의 torch **2.5.1+cu121**을 기준으로 한다.
Jetson의 NVIDIA wheel을 학습 환경에 그대로 복사하지 않는다. 기존 학습이 실행 중인지,
선택한 GPU와 실행 경로에 충돌이 없는지 확인한 뒤 컨테이너의 전용 venv를 준비한다.

아래 명령은 Elice Docker 내부의 저장소 루트에서 실행한다. 이미 검증된 환경에 설치를 반복하지 않는다.

```bash
bash scripts/elice/setup_env.sh
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'
```

학습 데이터는 [docs/03](03_data_pipeline.md)과 [docs/16](16_drive_acoustic_preparation.md)의
Drive 보관·staging·QA 절차를 따른다. 원본을 Git으로 전송하지 않는다. PC의 임시 원본은
Drive 업로드를 확인한 뒤 정리하며, 실패·미확인 업로드와 기존 불완전 백업은 보존한다.
원본 확보와 manifest 생성, 실제 PCM 검사, 학습 준비 완료는 각각 별도 상태다.

`bootstrap_all.sh`는 과거 다운로드·환경 준비·학습 시작을 묶은 도구이며 현재 데이터 정책의
자동 진입점으로 사용하지 않는다. `--no-train`도 다운로드·데이터 변경까지 막는 옵션은 아니다.
부족한 소스를 합성으로 대체해 실제 corpus 학습으로 보고하지 않는다.

## 2. 학습 목적과 정렬

최우선은 acoustic-ref이며 준비 설정은 `train_acoustic_prepared.yaml`과
`data_acoustic_prepared.yaml`이다. 현재 이 학습 설정의 `run_until_step: 2`는 준비 확인용
한도다. 설정 파일의 존재나 2-step 실행을 완전한 corpus 학습·물리 검증으로 보고하지 않는다.
MIMII DG fan은 train-only이며 평가에서는 제외한다. Jetson 우선 목표 1000–1600 Hz를
S의 기존 신뢰대역이나 손실 검증 범위가 확대됐다는 뜻으로 해석하지 않는다.

digital 사전학습·파인튜닝 경로는 별도 비교·검증용으로 남아 있다. 현재 실행 설정은
S 지연 **1465** + handoff **256**, P 지연 **1608**, digital lead **113**이다.
과거 **1342/1489/109** surrogate artifact는 당시 resolved 설정으로 해석하며 숫자만
최신값으로 바꾸지 않는다. `secondary_surrogate`는 P의 FIR/gain을 S로 대용하는 표현 학습이고,
`measured`는 실제 P NPZ와 측정 조건을 요구한다. 파일의 존재와 경로 품질 게이트 통과도 구분한다.

정렬·데이터·물리 조건은 해결된 config와 checkpoint metadata로 대조한다. `e=d+S·y` 극성,
인과성, FP32 plant/loss, 플랜트 적용 후 워밍업 절단을 유지한다. trusted/fullband 지표를
함께 기록하고 0 dB 정체·NaN·발산이 보이면 지연·gradient·입력·데이터를 먼저 진단한다.

## 3. 실행과 재개

새 실험은 목적·GPU·설정·출력 경로·데이터 QA를 확정한 뒤 명시적으로 시작한다.
로그와 checkpoint 디렉터리를 재사용해 기존 결과를 덮어쓰지 않는다.

| 도구 | 동작·사용 조건 |
|---|---|
| `scripts/train/train.py` | 선택한 설정의 단일 실행; GPU/출력 경로를 명시적으로 관리 |
| `scripts/elice/run_parallel_models.sh` | GPU0 base, GPU1 tiny의 독립 실행; 1 GPU에서는 base만 시작 |
| `scripts/elice/run_pretrain.sh` | 여러 GPU를 감지하면 한 모델의 DDP 실행; 모델별 병렬과 다른 방식 |
| `scripts/train/run_finetune_pipeline.py` | readiness → 명시된 파인튜닝 → recorded 평가·완료 검사 |

병렬 스크립트의 과거 base/tiny 설정을 최신 acoustic 학습 설정으로 간주하지 않는다.
전체 batch·worker 수·학습 예산은 실제 GPU/CPU와 연구 설계에 맞춰 검토한다.
GPU 사용량만 보고 정상 학습을 재시작하거나 batch를 임의 변경하지 않는다.

`--resume`은 동일 목적·모델·정렬·데이터 정책의 checkpoint에서 optimizer·scheduler·step·RNG를
이어서 실행할 때 사용한다. 다른 lead/P mode/물리 상태의 checkpoint를 같은 실행의 재개로 취급하지 않는다.
사전학습 가중치를 파인튜닝의 초기값으로 쓰는 `init_ckpt`와 완전 재개는 구분한다.
실측 경로에 맞춘 초기 checkpoint의 제한된 lead 차이는 명시된 readiness 허용 범위에서만 판단한다.

## 4. 실측 파인튜닝 게이트

다음은 **digital measured 파인튜닝** 준비 검사와 실행 예시다. 최신 acoustic 준비 절차와
혼동하지 않는다. 필요한 실측 자료와 init checkpoint가 준비됐을 때 컨테이너 내부에서 실행한다.

```bash
.venv/bin/python scripts/train/check_finetune.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured

# 위 검사가 READY이고 이 실험의 실행이 정해진 경우에만 시작
.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured
```

진입 게이트는 P/S의 출처·출력 채널·반복 품질·일치하는 측정 조건, `S+handoff−P` 정렬,
완료된 init checkpoint, 실측 manifest와 파일·그룹·소스 coverage QA를 검사한다.
정확한 요구값은 `configs/train_finetune.yaml`의 `readiness`가 단일 출처다.
현재 요구 경로 대역 **150–600 Hz** 통과는 1000–1600 Hz나 전체 quiet-zone 목표 통과가 아니다.
자료 부족이나 조건 불일치를 플래그 완화·메타데이터 수정으로 우회하지 않는다.

학습 내부의 고정 합성 val만으로 파인튜닝 완료를 판정하지 않는다. 같은 checkpoint와 manifest
식별 정보로 생성한 독립 recorded val/test에서 trusted/fullband·소스별·최악 구간 지표를 확인한다.
closed-loop 단계는 추가 피드백 동역학 실험이며 데이터나 실기 안정성 검증을 대신하지 않는다.

## 5. 회수와 배포

회수 묶음에는 checkpoint, 해결된 설정, Git revision, 의존성 목록, 로그·평가 결과,
데이터·경로 식별 정보와 해시를 포함한다. 원본 데이터의 최종 보관은 Drive를 따른다.
파일 전송 성공과 독립 평가 통과를 구분하고 공개 저장소에 비밀정보·원본을 넣지 않는다.

ONNX export는 실제 사용할 checkpoint를 대상으로 수행하고 동반 JSON의 reference mode·lead·
상태 규약을 함께 보존한다. surrogate 모델 export는 추론 구현 진단용이며 실제 감쇠를 보증하지 않는다.
Jetson Docker의 라이브러리 검사, 프로젝트 모델 등가성·지연, 현장 OFF→ON→OFF 평가는
[docs/06](06_deployment_jetson.md)과 [docs/14](14_pc_jetson_workplan.md) 순서로 검증한다.
