# Deep ANC 프로젝트 현황 — 단일 운영 요약

> 상태 기준일: **2026-09-02 (KST)**  
> 저장소: `/home/capston/Deep_ANC`  
> 브랜치: `dev`  
> 이 문서는 현재 프로젝트를 파악하기 위한 운영용 기준 문서다. 매 작업마다 `runs/`와 외부 폴더를 전수 재검색하지 말고 이 문서를 먼저 읽는다.

## 1. 한 줄 결론

현재 저장소에는 **사전학습된 Deep ANC 모델이 분명히 존재한다.**
현재 Tiny 런타임은 그 모델과 ONNX를 연결해 실행할 수 있다. 다만 그 모델들은
`reference_mode=digital` 및 `secondary_surrogate`로 학습된 **legacy surrogate 모델**이다.
따라서 `reference=mic`으로 바꾸면 실제 레퍼런스 마이크 신호를 모델 입력에 넣는 실험은
가능하지만, 가중치가 acoustic-reference용으로 학습된 것은 아니다.

현재까지 `reference_mode=acoustic`으로 학습된 동일 `hybrid_anc` 체크포인트/ONNX와
현행 strict P/S에 결속된 canonical 성능 증거는 없다. 그러므로 기존 사전학습 모델을
실험하는 것과 새 acoustic-reference 모델의 공식 성능을 주장하는 것을 구분한다.

## 2. 사용자의 현재 목표

- 1.6 kHz 이상은 최적화 대상에서 제외한다.
- `[150,300]`, `[300,600]`, `[600,1000]`, `[1000,1600] Hz`를 고르게 개선한다.
- 저주파와 고주파를 모두 제거하고, 소음뿐 아니라 음성·음악도 quiet zone에서 제거한다.
- Deep ANC 평가는 `controller=dl`로 한다. `controller=fxlms`는 별도 적응형 기준선이며
  Deep ANC 사전학습 모델 평가가 아니다.

## 3. 용어와 신호 경로

| 항목 | 의미 |
|---|---|
| `controller=dl` | 사전학습 `hybrid_anc` 모델을 Torch/ONNX/TRT로 실행하는 Deep ANC 경로 |
| `controller=fxlms` | 측정 S(z)를 사용하는 FxLMS 기준선. 사전학습 DL 모델을 사용하지 않음 |
| `reference=digital` | Jetson이 알고 있는 소음원 waveform을 모델 reference로 사용 |
| `reference=mic` | 실제 reference microphone 입력 ch1을 모델 reference로 선택 |
| `digital_reference_lead_samples` | digital reference 전용 값. `reference=mic`에서는 반드시 0 |
| ERR ch0 | error microphone 입력. 스피커가 아님 |
| REF ch1 | reference microphone 입력 |
| 출력 ch0/ch1 | noise speaker / cancel speaker |
| `--start-noise` | 런타임 시작부터 생성 소음을 재생. ANC는 항상 OFF로 시작 |
| `A` 또는 Space | ANC ON/OFF 토글 |
| `N` | 생성 소음 ON/OFF |
| `Q` | 무음 처리 후 종료 |

현재 런타임에서 모델은 선택된 `ref`와 ERR feature를 `[2, hop]`로 받는다. 실제 선택
코드는 [run_realtime.py](src/deep_anc/realtime/run_realtime.py)의 inference loop에 있다.
`reference=mic`일 때 `ref_digital`이 아니라 실제 `ref_mic` ch1이 선택된다.

## 4. 현재 저장소의 모델 목록

### 4.1 실제 사전학습 Deep ANC 모델

| 경로 | 모델/학습 상태 | 입력 계약 | 사용 판단 |
|---|---|---|---|
| [`runs/pretrain_tiny_corrected/ckpt/best.pt`](runs/pretrain_tiny_corrected/ckpt/best.pt) | `hybrid_anc_tiny`, best step 89,500 | `digital`, lead 109, `secondary_surrogate_representation_pretrain` | 기존 Tiny Deep ANC 진단용 |
| [`runs/pretrain_tiny_corrected/ckpt/last.pt`](runs/pretrain_tiny_corrected/ckpt/last.pt) | 같은 모델, step 100,000 | `digital`, lead 109 | resume/비교용. 보통 best 사용 |
| [`runs/pretrain_base_corrected/ckpt/best.pt`](runs/pretrain_base_corrected/ckpt/best.pt) | `hybrid_anc_base`, best step 99,000 | `digital`, lead 109, surrogate | 기존 Base Deep ANC 진단용 |
| [`runs/pretrain_base_corrected/ckpt/last.pt`](runs/pretrain_base_corrected/ckpt/last.pt) | 같은 모델, step 100,000 | `digital`, lead 109 | resume/비교용 |

Tiny 런타임 기본 경로는 [configs/runtime_tiny.yaml](configs/runtime_tiny.yaml)의
다음 두 artifact다.

```text
checkpoint: runs/pretrain_tiny_corrected/ckpt/best.pt
ONNX:       runs/export/tiny_corrected.onnx
```

ONNX sidecar [runs/export/tiny_corrected.json](runs/export/tiny_corrected.json)은 위
Tiny checkpoint와 `hybrid_anc_tiny`, 256-sample block, lead 109를 명시한다.

### 4.2 존재하지만 공식 최종 모델로 쓰지 않는 artifact

- `runs/finetune_tiny/ckpt/{best,last}.pt`: 50k fine-tune 결과. `digital` reference,
  `digital_primary_path_mode=measured`, lead 113이다. recorded 평가에서 G4가 FAIL이므로
  현행 canonical 배포 모델이 아니다.
- `runs/search_tiny_*`, `runs/seed_repeat_tiny_*`: 20k 전후 pilot/seed 비교용이며 모두
  digital surrogate 계열이다.
- `runs/export/tiny_long.onnx`: `search_tiny_long/ckpt/last.pt`에 연결된 pilot export다.
- `runs/export_base/model.onnx`: `runs/bench_base/ckpt/init.pt` step 0 초기화 artifact다.
- `runs/export_wide/wide.onnx`: `latency_probe_random_init` probe artifact다.
- `runs/export_smoke/model.onnx`와 `runs/smoke*`: smoke test artifact다.

이 목록의 `search`, `smoke`, `init`, `probe` artifact를 사전학습 최종 모델로 부르지
않는다.

## 5. 외부 읽기 전용 폴더에서 확인된 모델

아래 파일은 존재하지만 `/home/capston/DeepANC_CRN_n_codex`의 별도 프로젝트다.
현재 `deep_anc.realtime.run_realtime`에 경로만 바꿔서 연결할 수 없다.

| 경로 | 확인된 계약 | 현재 프로젝트에서의 판단 |
|---|---|---|
| `DeepANC_CRN_n_codex/DeepANC_CRN_n_codex_v1/results/duct_source_run/best.pt` | `DeepANC-CRN-n`, 16 kHz, `reference_input_mode=source`, 150–1600 Hz | 별도 CRN/source-mode runner 필요 |
| `DeepANC_CRN_n_codex/WaveNet_VNN_Duct_Test_Bundle_20260902_080631/model/best_model.tar` | WaveNet-VNN, 16 kHz model / 48 kHz I/O, known source runner | 현재 `hybrid_anc`와 구조·입력 계약이 다름 |
| `DeepANC_CRN_n_codex/duct_cnn_anc/results/latency_model_48k_20260902_provisional/best.pt` | 48 kHz, `reference_mode=microphone`, 100–2000 Hz | `deployment_ready=false`, causality provisional. 현재 runtime과 호환되지 않음 |

외부 폴더는 읽기 전용이다. 복사·연결·실행이 필요하면 별도 작업으로 설계하고,
원본을 수정하거나 import하지 않는다.

## 6. 지금까지 확인된 실험 결과의 의미

### 6.1 기존 Tiny 사전학습 모델의 legacy 실험

`runs/pretrain_tiny_corrected`는 surrogate 사전학습 모델이다. 오프라인 demo의 큰 감쇠
수치는 실제 덕트 성능이 아니다. 과거 실제 장비 legacy raw에서 확인된 값은 다음과 같다.

| 자극 | 대역 | 보수 감쇠 | 의미 |
|---|---:|---:|---|
| 300 Hz tone | 250–350 Hz | 약 +6.48 dB | 과거 physical diagnostic에서 확인 |
| band | 80–1000 Hz | 약 +4.09 dB | legacy diagnostic 범위 |
| 1.2 kHz tone | 1150–1250 Hz | 약 +0.35 dB | 1.6 kHz 성능 근거 아님 |
| high band | 800–1600 Hz | 약 −0.05 dB | 안정적 감쇠 확인 안 됨 |

상세 근거는 [docs/66_20260831_legacy_pretrain_raw_audit.md](docs/66_20260831_legacy_pretrain_raw_audit.md)에 있다.
따라서 “기존 6 dB”는 1.6 kHz 전체 성능이 아니라 300 Hz 주변 legacy 결과다.

### 6.2 최근 FxLMS microphone 실행

최근 `controller=fxlms`, `reference=mic` 실행에서 divergence probe가
anti-noise 출력을 닫자 error power가 `3.873e-04 → 5.173e-05`로 약 8.7 dB 감소했다고
보고했다. 이는 anti-noise가 실제 ERR mic에 도달해 오히려 error를 키웠다는 physical
진단이지만, **Deep ANC 모델 평가 결과는 아니다.** 해당 실행은 deadline/fallback/xrun도
누적되어 안정적인 성능 측정으로 승격할 수 없다.

### 6.3 OFF 상태의 `저감` 표시

ANC가 OFF이거나 출력 gate가 닫힌 동안의 `저감`은 감쇠 결과가 아니다. 현재 코드는 그런
구간을 `n/a`로 표시한다. 유효한 감쇠는 같은 조건의 ANC OFF/ON error mic raw를
주파수별로 비교해야 한다.

### 6.4 2026-09-02 legacy 실행의 `저감` 오표시 원인과 수정

사용자가 제공한 최근 실행의 마지막 OFF가 `e=-44.62 dBFS`, ANC ON이
`-49.72…-52.61 dBFS`였으므로, dBFS 숫자는 **더 음수일수록 ERR 전력이 낮은 것**이다.
따라서 이 구간의 `+4.96…+7.85 dB`는 방향만 보면 OFF 기준보다 낮다는 계산과
대체로 일치했다. 예전 표시는 현재 줄과 직전 줄의 단순 차이가 아니라, ANC 출력이
닫힌 동안의 EMA 기준 전력과 현재 ERR 전력을
`10*log10(P_baseline/P_error)`로 비교한 값이었다.

그러나 해당 세션은 `xrun=120`, `fallback=29`, `deadline=2`까지 누적됐다. 즉 일부
블록은 추론 결과가 아니라 무음 fallback이었거나 callback/추론 마감이 깨진 세션이다.
이 상태의 양의 숫자를 실제 ANC 감쇠로 표시한 것은 잘못이다. 현재
`run_realtime.py`는 다음 조건을 모두 만족할 때만 `저감`을 표시한다.

- full ANC/noise gate 및 실제 output-ring 데이터
- 유효한 OFF baseline
- callback status, xrun, fallback, deadline, engine error 모두 0

그 외에는 `base=... | 저감= n/a dB | 판정=INVALID:xrun,fallback,deadline`처럼 표시한다.
이 수정은 마이크 값을 다른 값으로 바꾼 것이 아니라, **무효 세션을 성능값으로 승격하지
않는 표시·판정 게이트**다. legacy Tiny 모델과 `lead=109`를 사용한 위 실행은 여전히
diagnostic일 뿐이며 strict P/S 또는 실제 acoustic 성능 증거가 아니다.

## 7. 사전학습 Deep ANC 재현 명령

### 7.1 기존 모델의 원래 입력 계약으로 재현 — 권장

이 명령은 실제 사전학습 Tiny Deep ANC를 사용한다. 30초 동안 noise는 시작부터 나오고,
ANC는 OFF로 시작하며 `A` 키로 토글한다. 사용자가 입회하고 앰프를 최소 볼륨으로 둔
상태에서만 실행한다.

```bash
cd /home/capston/Deep_ANC

.venv/bin/python -m deep_anc.realtime.run_realtime \
  --config configs/runtime_tiny.yaml \
  --legacy-diagnostic \
  --set controller=dl \
  --set reference=digital \
  --set digital_reference_lead_samples=109 \
  --set noise.type=band \
  --set noise.band='[80,1000]' \
  --set noise.amplitude=0.05 \
  --start-noise \
  --run-seconds 30 \
  --confirm-speaker \
  --confirm-user-present \
  --confirm-volume-minimum
```

화면에 반드시 `컨트롤러: dl/ort`가 표시되어야 한다. `fxlms`가 표시되면 사전학습
Deep ANC를 실행한 것이 아니다.

### 7.2 실제 reference mic를 기존 DL 가중치에 연결하는 일회성 진단

이 명령은 모델 입력으로 실제 `reference_mic ch1`을 선택한다. 단, 기존 가중치의
학습 계약은 digital이므로 acoustic-reference 성능 증거로 해석하지 않는다.
`--legacy-diagnostic`은 native digital 모델 전용이라 이 모드에서는 넣지 않는다.

```bash
cd /home/capston/Deep_ANC

.venv/bin/python -m deep_anc.realtime.run_realtime \
  --config configs/runtime_tiny.yaml \
  --set controller=dl \
  --set reference=mic \
  --set digital_reference_lead_samples=0 \
  --set noise.type=band \
  --set noise.band='[150,600]' \
  --set noise.amplitude=0.05 \
  --start-noise \
  --run-seconds 30 \
  --confirm-speaker \
  --confirm-user-present \
  --confirm-volume-minimum
```

## 8. 현재 차단 상태와 다음 작업

### 완료/가능

- 기존 Tiny/Base 사전학습 checkpoint와 Tiny ONNX 존재 확인
- `controller=dl` native digital legacy 진단 경로
- `reference=mic`에서 실제 ch1을 model reference로 선택하는 일회성 경로
- runtime에 live `ref` meter와 OFF 구간 `n/a` 표시

### 아직 미완료/차단

- acoustic-reference로 학습된 현행 `hybrid_anc` checkpoint/ONNX
- strict P/S와 동일한 current plant에 결속된 canonical checkpoint
- 150–1600 Hz 전 대역의 raw-first physical OFF/ON 증거
- 음성·음악을 포함한 모든 source family의 quiet-zone G4 PASS
- current canonical pretrain/fine-tune 실행

현재 physical blocker는 RT5640/J511 출력 경로다. J511 jack-state가 세 번 `None`이어서
HDA breakout/harness 연결과 무음 `HP`/`HS` 확인 전에는 새 P/S 측정·canonical 학습을
열지 않는다. 상세 순서는 [HANDOFF.md](HANDOFF.md)의 최상단 current status를 따른다.

실제 acoustic-reference Deep ANC가 목표라면 순서는 다음과 같다.

1. 같은 clock의 REF→ERR timing과 current P/S를 raw-first로 확정한다.
2. `reference_mode=acoustic` 데이터 계약으로 별도 학습/검증한다.
3. 그 checkpoint를 ONNX로 export하고, `controller=dl`, `reference=mic` 전용 metadata를 만든다.
4. xrun/fallback 없이 짧은 physical OFF/ON raw를 주파수별로 분석한다.

기존 digital surrogate checkpoint에 CLI 숫자만 바꿔 acoustic 학습 모델로 승격하지 않는다.

## 9. 다음 Agent가 지킬 최소 절차

1. 저장소의 `AGENTS.md` 안전 규칙을 읽는다.
2. 이 `PROJECT_STATUS.md`를 읽고 위 표와 명령을 기준으로 작업한다.
3. `HANDOFF.md`는 physical/training/canonical 작업을 시작할 때만 상세 blocker와 최신
   인수인계 내용을 확인한다.
4. 이 문서에 이미 적힌 artifact를 다시 전수 검색하지 않는다. 실제 파일·commit·사용자
   지시가 바뀌었거나 사용자가 “전체 재확인”을 명시한 경우에만 다시 감사한다.
5. 모델·입력 계약·물리 상태가 바뀌면 이 문서와 `HANDOFF.md`를 같은 작업에서 갱신한다.
6. live audio는 사용자 입회·최소 볼륨·짧은 사전 검증 후에만 실행한다. Agent가 임의로
   스피커를 연결하거나 장시간 녹음하지 않는다.
