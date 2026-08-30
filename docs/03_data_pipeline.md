# 03. 데이터 파이프라인

## 1. 신호 모델 (Deep ANC 방식)

학습 샘플 하나는 (모델 입력 x, 1차경로 소음 d) 쌍이다. 타깃 상쇄 파형은 없다 —
손실이 미분가능 2차경로 플랜트로 에러를 직접 계산한다 (docs/04 §5).
현재 digital Stage-1의 정렬은 다음과 같다.

```
x[0] = x_ref   (digital: 연속 source n의 116샘플 선행분, acoustic: P_ref·n)
x[1] = err_in  (에러 피드백 근사: d 를 512~1024샘플 랜덤 지연 — 캡처+블록 지연 모사)
playback[t] = n[t]
x_ref[t]    = n[t+109]
d           = S_FIR·playback 을 D_noise=1602만큼 지연  # secondary surrogate
y           = model(x)
e           = d + S_total·y,  S_total delay=1462+256=1718
```

학습은 source를 `segment+109`샘플 연속으로 뽑으므로 tail zero-padding이나 파일 경계의
가짜 패턴이 없다. 런타임도 모델 입력에서 미래 샘플을 읽는 대신 실제 noise playback을
116샘플 FIFO로 늦춰 같은 관계를 만든다. 지연 물리는 docs/01 §3이 단일 근거다.
(2026-08-05 플랜트 복구 전에는 109 였다 — 배포 중인 ONNX 만 아직 109 다.)

### digital 1차경로 모드

| `digital_primary_path_mode` | d 생성 | 용도 |
|---|---|---|
| `secondary_surrogate` **(현재)** | 측정 S의 compact FIR/gain + `D_noise=1602` | P/S 스케일을 맞춘 표현 사전학습. 물리 성능 주장 금지 |
| `measured` | `duct.digital_reference.primary_path_npz`의 FIR/gain/delay | 같은 gain·볼륨으로 P/S를 실측한 뒤 실제 파인튜닝 |
| `rir_surrogate` | 1D `p_err` RIR + 음향 onset을 뺀 추가지연 | 과거 호환·진단 전용. 측정 S와 단위가 달라 본 학습에 사용 금지 |

`measured` 모드는 NPZ sample rate와 설정 delay가 다르면 즉시 실패한다. compact FIR과
순수지연은 각각 정확히 한 번만 적용하며, RIR onset과 `D_noise`를 이중 계상하지 않는다.

## 2. 소음원 구성 (`data_sim.yaml source_mix_ratio`)

| 소스 | 비율 | 이유 |
|---|---|---|
| 합성원 (`synthetic_signals.py`) | 25% | 톤+고조파, AM/FM 기계음, 협대역, 처프, 멀티톤과 덕트 공진 가중 |
| DNS-Challenge noise_fullband | 30% | 48kHz 네이티브 실환경 소음 대량 확보 |
| DNS-Challenge clean_fullband speech | 15% | 대화(음성) 제거 목표 |
| FMA-small music | 10% | 음악 제거 목표 |
| DEMAND | 8% | 48kHz 지속 환경음 |
| MIMII fan | 7% | 팬·회전 기계음(16kHz, 저역 전용) |
| ESC-50 | 5% | 비정상 이벤트음 다양성 — 강건화용 소량 |

acoustic-ref는 예측 가능한 주기성 비중을 높인 전용 비율
`synthetic/machine/dns_fullband/demand/speech/esc50 = 45/15/20/10/5/5%`를 사용한다.
전체 공개 데이터 구성은 Elice의 `scripts/elice/bootstrap_all.sh`가 준비한다. manifest가 없는
태그는 합성원으로 폴백하지만 학습 시작 전 Trainer가 누락 태그와 비율을 경고한다.

## 3. 덕트 음향 시뮬 (RIR 뱅크)

- `dsp/duct_sim.py` — 1D 영상법, closed(폐단)–open(개방단) 경계. `duct.yaml` 기하 사용.
  검증: 이론 공진 70/210/350Hz 재현 (tests/test_duct_sim.py).
- 한계: 평면파 모델 — 컷오프(1633Hz) 이상 고차 모드는 미모델링(저역통과 근사만).
- **RIR 뱅크**: 반사계수/감쇠/위치 ±1cm/저역통과 컷오프를 바꾼 변형 300개.
  RIR 변형도 train/val/test로 분리한다.
- 현재 digital `secondary_surrogate`의 d는 `p_err` RIR을 쓰지 않는다. 이 RIR은
  acoustic-ref의 `P_ref/P_err`와 legacy `rir_surrogate`에 사용된다. 따라서 Stage-1
  surrogate 결과를 덕트 RIR 도메인 랜덤화 성능으로 해석하지 않는다.

```bash
.venv/bin/python scripts/data/build_rir_bank.py --n 300     # duct.yaml 변경 시 재실행 필수
```

## 4. 현재 Stage-1 커리큘럼과 이후 증강

| 항목 | Stage-1 현재값 | 이유 / 이후 단계 |
|---|---|---|
| 레벨 | **−45 ~ −20dBFS** | limiter ±0.2 안에서 먼저 선형 역매핑 확립 |
| 스피커 비선형 | SEF η=10, drive=1, hardclip=0 | 사실상 선형. THD/IMD와 실측 P/S 확보 뒤 점진적으로 강화 |
| S 플랜트 섭동 | delay `[0,0]`, gain/tilt 0, all-pass off | 모델이 관측하지 못하는 독립 위상 랜덤화가 gradient를 상쇄하지 않도록 공칭 plant 고정 |
| 마이크 자기잡음 | SNR 5~30dB | INMP441 잡음 바닥 모사 |
| DC hum | 20% | 50/60Hz와 2차 고조파 모사 |
| 채널 dropout | ref 15% / err 15%, 동시 금지 | ref-only / err-only 운용 폴백 학습 |
| 피드백 지연 | err_in에 512~1024샘플 | open-loop 피드백 근사 및 이후 closed-loop 캡처 지연 |

과거 실행은 대규모 delay/all-pass와 비선형을 처음부터 매 batch에 독립 적용했다.
모델에 plant 조건을 주지 않은 상태에서는 상충하는 위상 gradient가 평균되어
영출력으로 수렴한다. 다중 plant·비선형 증강은 실측 조건 라벨이나 적응층을 마련한 뒤
Stage-2 커리큘럼으로 한 축씩 켠다.

## 5. 실측 수집 → 파인튜닝

```bash
# 출력 장치를 열지 않는 선행 게이트. recorded 세션은 ERR+REF 모두 PASS해야 한다.
.venv/bin/python scripts/bench/check_audio_input.py --require-both
# 세션 수집 (ANC OFF, ch1 상쇄 스피커 무음 유지)
.venv/bin/python scripts/data/record_duct.py --program tone --frequency 300 --seconds 60 \
  --source-family synthetic --group-id tone300_setup_a
.venv/bin/python scripts/data/record_duct.py --program file --file speech.wav --seconds 120 \
  --source-family speech --group-id speaker_book_001
.venv/bin/python scripts/data/record_duct.py --program file --file music.wav --seconds 120 \
  --source-family music --group-id artist_track_001
# 같은 화자·곡·원본·환경의 반복 세션은 반드시 같은 group-id를 사용한다.
# manifest 생성: group 원자성 + source_family 층화 8:1:1
.venv/bin/python scripts/data/make_recorded_manifest.py
# 전체 파일/메타/클리핑/무음/분할 누수 QA — PASS 전에는 학습 금지
.venv/bin/python scripts/data/validate_recorded_sessions.py
# 파인튜닝 (실측:합성 = 7:3 혼합, digital lead=116 — 실측 P/S 에서 유도)
.venv/bin/python scripts/train/train.py --config configs/train_finetune.yaml \
  --set data.digital_primary_path_mode=measured
# 학습에 쓰지 않은 test split의 독립 반사실 평가
.venv/bin/python scripts/eval/evaluate_recorded.py \
  --ckpt runs/finetune_tiny/ckpt/best.pt --split test
```

2026-08-03 빠져 있던 pin17(REF L/R)을 재연결한 뒤 ERR/REF는 −46dBFS대, clip 0%로
두 채널 probe를 통과했다. 유효 수집이 가능해졌지만 매 세션 직전 probe를 반복하고,
사용자 입회·볼륨 최저 조건에서만 `record_duct.py`를 실행한다. sudo/pinmux/I²S/오디오 데몬 등
Jetson 시스템 설정은 바꾸지 않는다.

세션 구조: `data/recorded/<시각_프로그램>/{mics.wav(2ch PCM_32), source.wav, session.json}`.
ANC OFF 녹음이므로 **에러 마이크 신호가 곧 d(t)** 다. digital-ref 파인튜닝은 같은 세션의
연속 `source.wav`에서 116샘플 앞선 조각을 x_ref로 쓰고, acoustic-ref는 ref 마이크 채널을
쓴다 (`recorded_dataset.py`). 실제 수집 전에는 noise→ERR P와 cancel→ERR S를 같은 출력
gain·볼륨으로 측정하고, checkpoint의 lead와 배포 runtime lead를 **같게** 유지한다.
(현행 실측 lead=116. 배포 중인 ONNX 는 109 로 학습된 것이라 그 모델에는 109가 맞다.)

## 6. manifest 스키마와 분할 규칙

JSONL 한 줄 = 파일/세션 하나:

```json
{"path":"../recorded/20260803_190000_file","path_base":"manifest","duration_s":120.0,"sample_rate":48000,"channels":2,"tag":"recorded","session_id":"20260803_190000_file","group_id":"speaker_book_001","source_family":"speech","split":"test"}
```

`path_base: manifest`가 명시된 상대경로만 manifest 파일의 부모 기준으로 해석한다. 따라서
`data/` 묶음을 Jetson에서 Elice로 그대로 옮겨도 경로를 다시 만들 필요가 없다. marker가 없는
기존 절대·상대경로는 호환성을 위해 의미를 바꾸지 않는다.

현재 분리 게이트 (tests/test_dataset.py가 검사):

1. 노이즈 풀은 **원본 파일 단위** 90/5/5 분할 (세그먼트 단위 금지)
2. RIR 뱅크는 **변형 단위** 분할
3. 실측 데이터는 **group_id 원자 단위 + source_family 층화** 80/10/10 분할

실측 manifest는 같은 `group_id`가 여러 split에 나타나면 쓰기·읽기·QA 단계에서 모두
fail-fast한다. 각 family에 충분한 그룹이 있으면 양수 비율의 train/val/test에 최소 한 그룹씩
배치한다. 구형 세션은 디렉터리명을 임시 group으로 추론하고 `metadata_inferred=true`를 남기므로,
최종 수집에서는 화자/책·곡/원본·환경/기계조건을 직접 지정해야 한다. 공개 노이즈 풀의 기존
파일 단위 split은 별도 source×band held-out 재평가 전까지 상관 누수 한계가 남는다.

`validate_recorded_sessions.py`는 저장된 파일만 블록 단위로 읽어 2채널/SR/길이/finite/RMS/
clipping, `session.json`, source 필요 여부, group 누수, family×required-split 커버리지를 검사하고
`recorded_qa.md`+`recorded_qa.json`을 남긴다. 기본 게이트를 완화하는
`--allow-incomplete-family-coverage`는 진단용이며 최종 G2 판정에는 쓰지 않는다.
`evaluate_recorded.py`도 오디오 장치를 열지 않으며 checkpoint의 resolved model/data/duct와
measured P/S/lead만 사용한다. surrogate는 `--allow-surrogate`를 명시한 진단 외에는 거부한다.

## 7. 데이터셋 적합성 — 학습 × 실시간 추론 정합 분석

배포된 모델이 보는 신호는 "덕트 마이크가 들은 소리"다. 학습 분포가 이것과 정합해야 한다.

### 소스별 적합성 매트릭스

| 소스 | 원 SR | 학습 적합성 | 추론(배포) 정합성 | 비고 |
|---|---|---|---|---|
| DNS noise_fullband | **48k 네이티브** | ◎ 실환경 소음 대량 | ◎ 팬/기계/환경 소음 = 덕트 실전 분포 | 주력 |
| DNS clean_fullband(speech) | **48k 네이티브** | ◎ 대화 제거 목표의 핵심 | ◎ digital-ref 데모(음성 재생→상쇄)와 직결 | ⚠ acoustic-ref 에선 비주기라 불가(물리) |
| ESC-50 | 44.1k→48k 리샘플 | ○ 다양성 (5%) | △ 비정상 이벤트음 — 강건화용 | 22.05k 이상 무성분(문제 없음 — 나이퀴스트 밖) |
| 합성원 | 48k 생성 | ◎ 주기성/공진 정조준 | ◎ 덕트 공진(70~629Hz) 가중 생성 | 무한 |
| FMA-small music | 44.1k→48k 리샘플 | ○ 음악 제거 목표(10%) | ○ digital-ref 데모용 | MP3 8,000개 중 soundfile 디코딩 가능한 파일을 manifest에 포함 |
| DEMAND | **48k 네이티브** | ○ 지속 환경음(8%) | ◎ 주방·세탁기·사무실·교통 환경 정합 | 6환경×16채널 |
| MIMII fan | 16k→48k 리샘플 | ○ 회전 기계음(7%) | ◎ 저역 팬/기계 소음 | 8kHz 이상 성분 없음 — 저역 전용 |

### 파이프라인의 정합 장치 (코드 근거)

| 배포 현실 | 학습 쪽 대응 |
|---|---|
| INMP441 레벨 편차 | 소스 RMS 정규화 후 Stage-1 레벨 −45~−20dBFS (`synth_dataset`) |
| 마이크 자기잡음 | SNR 5~30dB 가우시안 부가 |
| DC/저역 험 (런타임은 DCBlocker) | dc_hum 증강 20% + 손실 W(f)<40Hz ×0.1 |
| 44.1k 소스 | 로더에서 resample_poly 48k (`NoisePool`) |
| 5초 미만 클립 | 타일링 반복 (`NoisePool.sample_segment`) |
| 스피커/앰프 비선형 | Stage-1은 사실상 선형, 실측 THD/IMD 뒤 커리큘럼으로 활성화 |
| 1차/2차경로 | 현재 P=S scale surrogate + 공칭 측정 S; 실제 물리 성능은 measured P/S 파인튜닝 뒤 평가 |

### 자동 QA 게이트

다운로드 직후 `.venv/bin/python scripts/data/validate_noise_pool.py` (부트스트랩 [4/6]에 포함)가
태그별 표본 150개를 실검사한다: 샘플레이트 분포/읽기 실패율/클리핑/무음 비율/
**덕트 제어대역(<1.6kHz) 에너지 비율**. 결과는 `data/manifests/dataset_qa.md` 리포트로 남고,
치명(태그 전체 읽기 불가)이면 학습 시작 전에 중단된다.

## 8. 저장 정책

온더플라이 합성이므로 학습쌍을 디스크에 굳히지 않는다 — 원본 노이즈(30~50GB)만 저장.
Elice 128GiB 스토리지에 여유 있게 들어가며, 업로드가 필요할 땐
`.venv/bin/python scripts/data/pack_transfer.py` 로 2GB tar 샤드를 만든다.
