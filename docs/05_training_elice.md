# 05. Elice Cloud 학습 가이드 (2×A100)

현재 Elice 작업은 하나의 모델을 2-GPU DDP로 돌리는 방식이 아니다. **GPU0은 base,
GPU1은 tiny를 각각 독립 프로세스로 동시에 학습**한다. 두 실행은 데이터·optimizer·checkpoint를
공유하지 않으며, 한 모델의 step 속도를 다른 GPU가 직접 높여주지는 않는다.

> **물리 유효성 경계**
>
> 현재 Stage-1은 실측 P가 없는 상태에서 `P=S`로 장치 스케일을 맞춘
> `secondary_surrogate` 표현 사전학습이다. 체크포인트의 surrogate val dB를 실제 덕트
> 감쇠로 보고하지 않는다. 과거 RIR-P/미관측 plant 위상 랜덤화/fullband 목적의 0dB
> 체크포인트는 목적함수가 잘못된 실행이며 새 학습에 resume하지 않는다.

## 1. 인스턴스 구성 (확정)

| 항목 | 값 |
|---|---|
| 인스턴스 | **G-NAHP-160** — 2× A100 80GB PCIe, 32 vCore, 384GiB RAM (₩3,980/시간) |
| 실행 환경 | **VSCode (CUDA 12.8)** — torch cu121 wheel 과 호환 (드라이버 하위호환) |
| 스토리지 | **128 GiB** (₩19.2/시간) — 온더플라이 합성 설계라 충분. 업그레이드만 가능하므로 작게 시작 |

**비용 팁**
- 스토리지·인스턴스 모두 시간당 과금 — **켜둔 채 방치 금지**.
- 코드 디버깅/소규모 실험은 **G-NAHPM-10 (MIG 1g-10GB, ₩340/시간)** 으로:
  `--set batch_size=4` 만 바꾸면 같은 코드가 그대로 돈다.
- 본 학습은 2×A100에서 base/tiny를 병렬 실행한다. 완료 시간은 로그의 실제 it/s로
  `남은 step ÷ it/s`를 계산한다. 인스턴스는 SSH 연결 여부와 무관하게 켜진 시간만큼 과금된다.

## 2. 초기 셋업 (웹 VS Code 터미널)

```bash
# 원샷 (권장): 환경+데이터+QA+테스트+2-GPU 병렬 학습까지 자동
git clone https://github.com/Roka-jsj/Deep-ANC.git && bash Deep-ANC/scripts/elice/bootstrap_all.sh
```

수동 단계 (부트스트랩 내부 동작과 동일):

```bash
git clone https://github.com/Roka-jsj/Deep-ANC.git && cd Deep-ANC
bash scripts/elice/setup_env.sh          # venv + requirements-train.txt + pip -e .
bash scripts/data/download_noise.sh 2    # DNS 2샤드(각 5.4GB) + ESC-50 — Azure 는 느리면 scripts/elice/pget.py 사용
.venv/bin/python scripts/data/prepare_noise_pool.py
.venv/bin/python scripts/data/build_rir_bank.py --n 300
.venv/bin/python scripts/data/validate_noise_pool.py   # 데이터셋 QA 리포트
.venv/bin/python -m pytest -q                         # 전체 테스트로 환경 검증
```

인터넷이 막힌 인스턴스라면: Jetson 에서 `.venv/bin/python scripts/data/pack_transfer.py` 로 만든
tar 샤드를 VS Code 탐색기로 업로드 → `for f in transfer/*.tar; do tar -xf "$f"; done`.

## 3. 학습 실행

```bash
# Stage-1 사전학습: GPU0=base, GPU1=tiny 독립 실행
# bootstrap_all.sh는 마지막 단계에서 이 스크립트를 자동 호출한다.
bash scripts/elice/run_parallel_models.sh
# 모니터링
tail -f runs/train_base_corrected.log runs/train_tiny_corrected.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
.venv/bin/tensorboard --logdir runs --port 6006  # VS Code 포트포워딩
```

`run_parallel_models.sh`는 기존 산출물이나 실행 중인 `train.py`가 있으면 덮어쓰지 않고
실패한다. 두 프로세스 중 하나가 시작 직후 실패하면 이번 호출이 시작한 둘을 함께 종료하고
산출물을 `runs/failed_start_*`에 보존한다. SSH가 끊겨도 `setsid nohup` 프로세스는 계속 돈다.

### 현재 Stage-1 실행값

| 항목 | base / GPU0 | tiny / GPU1 |
|---|---:|---:|
| 모델 파라미터 | 5,994,512 | 1,164,809 |
| batch size | **96** | **128** |
| total steps | **100,000** | **100,000** |
| warmup | **1,250 step** | **1,250 step** |
| DataLoader workers | **14** | **14** |
| prefetch factor | 4 | 4 |
| eval 간격 / 조기종료 | 500 / 끔 | 500 / 끔 |
| checkpoint | `runs/pretrain_base_corrected` | `runs/pretrain_tiny_corrected` |

공통값은 AdamW lr 1e-3, weight decay 1e-4, warmup→cosine, bf16 model forward,
FP32 plant/loss, grad clip 5.0이다. base는 96×100k=9.6M, tiny는
128×100k=12.8M sample update를 사용한다. 32 vCore 중 worker 28개를 데이터 생성에 배정해
GPU 공급 병목을 줄인다. batch를 더 크게 만드는 것은 메모리 활용률은 올릴 수 있어도
optimizer update 수와 일반화가 바뀌므로 별도 검증 없이 적용하지 않는다.

Tiny 100k 완료 뒤 GPU1의 유휴 시간을 쓰는 구조 탐색 watcher:

```bash
setsid nohup bash scripts/elice/run_structure_search.sh \
  > runs/structure_search.log 2>&1 < /dev/null &
```

watcher는 tiny `last.pt`가 실제 step 100,000인지 확인한 뒤 tiny-long/tiny-attn/
tiny-long-attn을 순서대로 실행한다. 세 후보 모두 100k cosine 스케줄은 유지하고
`run_until_step=20000`에서 멈춰 동일 초반 학습곡선을 비교하며, 기존 파일이 하나라도 있으면
시작하지 않는다. 각 후보 best/last에 독립 합성 평가까지 저장하고 오류 시 다음 후보를 막는다.

### 라이브 스냅샷 (2026-08-03 19:11 KST)

| 실행 | PID | 현재 step / 최신 val | 속도 | 단순 ETA |
|---|---:|---|---:|---|
| base / GPU0 | 22554 | 36,600 / step36,500 trusted −17.23dB, full −16.48dB | 1.8–1.84it/s | 8월4일 04:45–05:30 KST |
| tiny / GPU1 | 21433 | 79,700 / step79,500 trusted −19.48dB, full −18.25dB | 4.0–4.1it/s | 8월3일 20:35–21:00 KST |
| 구조 watcher | 24271 | tiny 완료 대기 | — | 후보 3종 완료 8월4일 02:30–04:30 KST 예상 |

두 production 로그에서 NaN/OOM/traceback은 없고 step이 계속 증가한다. 일부 FMA 손상 MP3의
mpg123 decode 경고는 데이터 로더가 다른 항목으로 재시도하는 recoverable 경고다. ETA는
`남은 step ÷ 최근 it/s`와 eval/checkpoint 여유를 합친 범위이며 인스턴스 상태에 따라 변한다.
가장 최신 상태는 `HANDOFF.md`의 SSH 명령으로 다시 확인한다. 정상 프로세스를 GPU 메모리
사용량만 보고 재시작하거나 batch를 올리지 않는다.

### 학습 목적과 정상 판정

- `digital_primary_path_mode=secondary_surrogate`: P에 S의 FIR/gain을 재사용하고
  `D_noise=1602` 적용
- `digital_reference_lead_samples=116`: 학습의 연속 source 정렬과 실제 playback FIFO가 동일
  (2026-08-05 플랜트 복구 전 사전학습된 checkpoint 는 109 다)
- S 총지연 `1462+256=1718`, delay/gain/tilt jitter 0, all-pass off
- η=10, drive=1, hardclip off의 공칭 선형 커리큘럼
- best 기준은 trusted NMSE **150–1600Hz**(CVaR 집계); fullband NMSE와 **대역 밖 최악값**도
  매 log/eval에 함께 기록

로그는 `nmse_t`(trusted)와 `nmse_f`(fullband)를 분리해 표시한다. corrected Stage-1에서도
trusted NMSE가 과적합 게이트와 초기 검증 구간 내에서 계속 0dB에 고정되면 정상으로 기다리지
말고 학습을 중단해 정렬·gradient·데이터를 점검한다. loss≈2/NMSE≈0dB였던 과거 실행은
이미 무효로 판정했다.

### 재개 규칙

동일한 corrected 설정에서 중단된 `last.pt`만 같은 GPU별 명령으로 재개한다. 과거 invalid
checkpoint나 lead/P mode가 다른 checkpoint를 재개하지 않는다. 예:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train/train.py \
  --config configs/train_pretrain.yaml \
  --resume runs/pretrain_base_corrected/ckpt/last.pt

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train/train.py \
  --config configs/train_pretrain.yaml \
  --set model_config=configs/model_tiny.yaml \
  --set batch_size=128 --set ckpt_dir=runs/pretrain_tiny_corrected \
  --resume runs/pretrain_tiny_corrected/ckpt/last.pt
```

`scripts/elice/run_pretrain.sh`는 한 모델을 여러 GPU에 DDP로 분산하는 별도 도구다.
현재 base/tiny 동시 운용의 기본 경로는 `run_parallel_models.sh`다.

### Stage-2 폐루프 파인튜닝 (선택 — 시뮬 피드백 동역학 학습)

```bash
.venv/bin/python scripts/train/train.py --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured \
    --set stage=closed_loop --set "schedule.total_steps=30000"
```

프레임 순차 unroll 이라 step 당 수 배 느리다 — 20k~50k step 권장 (설계 H1).

### 실측 파인튜닝 (덕트 녹음 후)

Jetson 에서 수집한 `data/recorded/` + manifest 를 git/zip 으로 올린 뒤:

```bash
# 준비 상태만 검사(오디오 출력 없음, 미준비 시 nonzero 종료)
.venv/bin/python scripts/train/check_finetune.py \
    --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured

# READY 뒤 원샷 실행: 자동 resume → 학습 → recorded val/test → 완료 검증
.venv/bin/python scripts/train/run_finetune_pipeline.py \
    --config configs/train_finetune.yaml \
    --set data.digital_primary_path_mode=measured
```

파이프라인과 직접 `train.py` 실행은 모두 GPU 초기화 전에 같은 readiness 게이트를 강제한다.
`duct.digital_reference.primary_path_npz`가 실제 측정 파일을 가리키는지만 보는 것이 아니라,
P/S 각각의 official 출력 채널·요구 대역(**150–1600Hz**) **모든 부대역** 일관성
`≥0.9406`·유지 반복 `≥8`·xrun 0, 동일 amplitude/block/latency,
**P−S 상대 τ 궤적의 상수성**(편차 `≤3샘플`), 그리고 P/S 지연에서 계산한 lead 까지 검사한다.
(궤적 검사가 없던 시절 게이트가 오염된 `S(z)` 를 통과시켰다 — docs/12 §2.3)
`require_measured_primary_path: true`가 surrogate 파인튜닝을,
`require_init_checkpoint: true`가 누락된 사전학습 checkpoint를,
`require_recorded_manifest: true`가 녹음 없이 합성-only로 진행되는 실수를 fail-fast한다.
manifest는 `path_base: manifest` 상대경로를 사용하므로 `data/` 묶음을 같은 구조로 전송하면
Jetson 절대경로를 다시 쓰지 않는다. readiness는 `validate_recorded_sessions.py`와 같은 파일·group
누수·family×split 전수 QA에 더해 최소 80세션/90분과 speech/music/environment/machine을 요구한다.
완료는 같은 checkpoint/manifest SHA-256으로 생성한 독립 recorded val/test가 G4 trusted/fullband를
동시에 통과해야만 인정한다. Jetson 입력은 pin17 복구 뒤 두 채널 probe를 통과했지만 아직 official
P/S와 recorded 세션이 없으므로 현재 검사는 의도대로 FAIL이며, 이 게이트를 우회하지 않는다.

## 4. 결과 회수 → Jetson

```bash
.venv/bin/python scripts/train/export_onnx.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt --out runs/export/base_corrected.onnx
# runs/export/{model.onnx, model.json} + ckpt/best.pt 를 zip 으로 다운로드
# (수십 MB — VS Code 탐색기 우클릭 Download, 또는 GitHub Release 자산으로 업로드)
```

surrogate checkpoint의 ONNX export는 추론 지연·스트리밍 통합 검증용이다. 실제 ANC 성능
배포 전에는 같은 gain/볼륨에서 noise→ERR P와 cancel→ERR S를 측정하고, measured P 및
recorded 데이터 파인튜닝과 독립 평가를 통과해야 한다. Jetson 절차는 docs/06 참조.

## 5. 자주 겪는 문제

| 증상 | 조치 |
|---|---|
| CUDA OOM | 해당 모델 batch를 한 단계 낮추고 sample budget/스케줄도 함께 재산정 |
| DataLoader 병목 (GPU util 낮음) | 두 프로세스 합 worker 수가 32 vCore를 넘지 않는지 확인; 현재 14+14 |
| corrected val trusted NMSE가 0dB 정체 | 정상으로 간주하지 말고 P mode·lead·경로 지연·gradient부터 점검 |
| torch 버전 충돌 | requirements-train.txt 는 2.5.1+cu121 고정 — Jetson(2.5.0a0)과 정렬 |
| SSH 세션 종료 | 학습은 `setsid nohup`으로 유지됨. 재접속 후 PID/step을 먼저 확인 |
| 한 GPU만 바쁨 | `train_base_corrected.pid`/`train_tiny_corrected.pid`, 각 로그와 `nvidia-smi`를 함께 확인 |
