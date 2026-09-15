# 15. 논문 반영: 사전 FIR + FxNLMS 연구 준비와 PC 검증

> acoustic-ref 우선, ERR 한 점, 우선 개선 대역 800–1600Hz, 저역·음성·음악도 목표에 포함한다.
> Jetson·덕트·마이크·USB DAC는 그대로다. 모든 개발/시험은 Docker 안에서만 한다.
> 아래 제어기는 **오프라인 연구용**이며 실제 오디오 런타임에 연결하지 않았다.

## 1. 논문에서 반영한 부분과 다른 부분

[사용자 논문 §3–4](https://arxiv.org/html/2601.06981v1)는 프레임 단위 CNN 필터 선택과
샘플 단위 FIR 출력을 분리하고, FxLMS로 준비한 필터 집합을 사용한다. 실험은 4REF/1CS/1ERR의
선형 RIR 시뮬레이션이며 광대역 감쇠 비교는 100–700Hz다. 필터 준비 대역 20–2020Hz와
감쇠 검증 대역을 혼동하지 않는다. 현재 구성의 800–1600Hz 실기·비선형·Jetson 성능 증거가 아니다.

[Hybrid SFANC–FxNLMS §II-D](https://arxiv.org/html/2208.08082v1)는 느린 필터 선택과 빠른
FxNLMS 적응을 결합한다. 이 저장소는 두 연구의 **선택/준비 계산과 출력 계산 분리**를 적용한다.
현재 1REF 고정 하드웨어에서 방향 분류기나 다중 마이크 배열을 복제하지 않는다.
논문의 부호를 그대로 복사하지 않고 기존 `e=d+S*y`, 음의 gradient update를 유지한다.

이번 구현의 **기준 FIR + 별도 FxNLMS 잔차 유지 + 선형 crossfade**는 논문 그대로가 아닌
이 프로젝트의 실험 정책이다. CNN·온라인 선택기·생성기는 아직 없으며 준비한 후보 한 개만 시험한다.
선형 FIR만으로 비선형 왜곡이 해결됐다고 주장하지 않는다.

## 2. 구현된 연구용 API

소스는 `src/deep_anc/baselines/prepared_fir.py`다. 기존 `HybridEngine`의 신경망 파형 결합과 별도다.
`build_engine`에 새 controller를 등록하지 않았고 runtime 설정·오디오 callback은 변경하지 않았다.

| 구성 | 이번 구현 | 아직 하지 않는 것 |
|---|---|---|
| 느린 쪽 | 불변 `FilterProposal`로 기준 FIR 전달 | CNN 호출·자동 분류·학습된 생성 |
| 빠른 쪽 | 과거/현재 REF의 인과 FIR + FxNLMS 잔차 | 미래 REF·미리 아는 외부 음원 사용 |
| 교체 | 단일 소비자가 완결 블록 경계에서 검증 후 crossfade | 오디오 스레드/lock-free mailbox 연결 |
| 안전 진단 | 합산 후 단일 제한, clipping 기록, 적응 보류, 손상 시 reset | 실기 watchdog·S/F 추적·안정성 인증 |

호출 순서는 `submit(선택 사항) → generate_block(REF) → adapt_block(ERR, enabled=...)`다.
모든 호출은 **한 소비자가 직렬 소유**한다. 이 클래스 자체가 스레드 안전 전달 큐는 아니다.
fast path에는 모델 호출이나 결과 대기가 없으며 제안이 없으면 마지막 기준 FIR을 유지한다.
합성 플랜트의 y→ERR 경로와 FxNLMS filtered-x 모델에는 지정한 `secondary_delay + handoff`를
각각 정확히 한 번 반영한다. `generate_block`의 FIR 출력 자체에 추가 지연을 넣지는 않는다.
핸드오프를 0으로 둔 별도 실험을 현재 3스레드 런타임 지연으로 간주하지 않는다.

### 제안과 전환 규약

- `context_id`는 S 계수·샘플레이트·지연·handoff·제어 길이·제한·적응/전환 설정의 hash다.
  S는 호출자 배열에서 복사해 외부 수정이 context를 우회하지 못한다. 실측 경로 정확성을 인증하는 hash는 아니다.
- `generation`은 reset 때 증가한다. 이전 세대·다른 context·같거나 오래된 revision을 거부한다.
- `observed_until_sample`은 관측 구간의 **exclusive 끝**이고 이미 소비한 `sample_cursor` 이하여야 한다.
  현재 블록 전체를 관찰한 후보를 그 블록 첫 샘플에 소급 적용할 수 없다. 유효 나이도 검사한다.
- 제안은 **기준 FIR 계수만** 포함한다. 기존 잔차가 포함된 최종 계수를 다시 제안하면 이중 합산이다.
  계수 길이·유한값·L1 한도를 확인하고 전환 중 새 제안은 큐잉 없이 거부한다.
- old/new FIR은 같은 REF 이력을 한 번만 진행하며 출력 crossfade를 한다. 잔차 계수는 유지한다.
  전환 마지막 샘플 이후 `S.delay + handoff + len(S) − 1` 꼬리까지 적응을 보류하고,
  이 구간을 벗어난 첫 **완전한 블록**에서만 적응을 재개한다.
- 보류 중에도 filtered-x 이력을 진행하고 pending gradient를 소비한다. 오래된 gradient가 남지 않는다.
- 출력은 기준+잔차 합산 뒤 한 번 제한한다. clipping 블록과 지연된 영향 구간도 적응하지 않는다.
  잘못된 REF/ERR, 변환 실패, 비유한 적응 진단은 이력/계수를 비우고 새 세대로 reset한다.
- 시작 계수는 0이며 적응은 `enabled=True`로 명시할 때만 가능하다. 부호·극성 추가 반전은 없다.

이 규약은 수치/소유권 시험이다. 유한 계수·작은 출력·crossfade만으로 안정적인 음향 폐루프를 보장하지 않는다.
특히 현재 API의 clip 기록은 해당 생성 블록 기준의 보수적 guard이고 실제 장치 xrun/ADC clipping 이력은 없다.

## 3. PC에서 바로 실행할 합성 비교

저장소 루트에서 실행한다. 결과 폴더 이름은 기존에 없는 새 이름으로 바꾼다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/benchmark_prepared_fir.py \
  --out results/prepared_fir/pc_trial_01

# 선택적 비선형 스트레스: 기존 결과를 덮어쓰지 않는다.
bash scripts/docker/dev.sh exec .venv/bin/python scripts/bench/benchmark_prepared_fir.py \
  --saturation-level 0.03 --out results/prepared_fir/pc_tanh_trial_01
```

한 독립 백색소음 train에서 FxNLMS로 사전 계수를 준비하고 서로 다른 heldout seed의
300/900/1300Hz·혼합·백색소음에 동일 후보를 적용한다. 평가 ERR를 보고 후보를 선택하는 oracle는 없다.
제어 없음·cold FxNLMS·고정 FIR·사전 FIR+FxNLMS를 같은 플랜트/출력 제한으로 비교한다.
중간 1차경로 gain 변화 전후, 초기와 후기, 저역·800–1600Hz·대역 밖 결과를 모두 남긴다.
학습에 출력 clipping이 생기면 후보를 무효화하며 재시도나 성공 반복 선별을 하지 않는다.

두 시험 조건은 실제 덕트와 구분한다.

- `causal_toy`: REF 선행을 S+handoff와 같게 둔 조건이다. 현재 하드웨어의 지연을 줄였다는 뜻이 아니다.
- `long_delay_stress`: 기본 S 순수지연 1465 + handoff 256, REF 선행 140샘플을 사용한다.
  **숫자만 기존 기록을 참고**하며 S FIR 자체는 합성이다. 앞 조건에서 준비한 동일 후보를 재학습 없이
  쓰므로 후보의 1차경로 선행 조건도 불일치한다. 이를 해당 조건 최적 필터의 성능으로 해석하지 않는다.

산출물은 `report.json`, `metrics.csv`, `summary.md`다. 정상 완주 보고서에는 모든 대조군과 계산 불가/증폭 대역을 보존한다.
학습 clipping·실행 오류는 stderr와 exit 1로 종료하며 보고서가 없을 수 있으므로 실행 로그도 보존한다.
숫자 계산 불가는 `null`, 새로 발생한 ERR 에너지는 별도 플래그다. exit 0은 보고서 생성뿐이다.
항상 `synthetic_only=true`, `performance_claim_allowed=false`, `real_time_claim_allowed=false`다.
FFT 직사각 창의 누설이 있고 음성·음악 데이터, 실측 S/F, 클록 drift는 아직 시험하지 않는다.
tanh 옵션 역시 실측 비선형 모델이 아니라 스트레스 조건이다.

### 2026-09-15 CPU Docker 실행 결과 — 합성 white/fullband만의 예시

기본 seed/설정으로 선형 및 `saturation-level=0.03`을 실행했다. 아래는 백색소음 전체 대역의
서로 다른 학습/평가 신호 결과다. **800–1600Hz 실기 감쇠 결과가 아니다.**

| 합성 조건·시기 | cold FxNLMS | 고정 사전 FIR | 사전 FIR + FxNLMS |
|---|---:|---:|---:|
| 선행 여유가 있는 toy, 초기 | 0.52dB | 17.15dB | 17.15dB |
| 같은 toy, gain 변경 후 후기 | 14.83dB | 11.99dB | 20.12dB |
| 같은 toy + tanh 스트레스, gain 변경 후 후기 | 11.34dB | 9.18dB | 13.73dB |
| 긴 지연·선행 조건 불일치, 선형 후기 | 0.00dB | −1.97dB | −0.04dB |

양수는 감소·음수는 증폭이다. 준비한 계수의 초기 이득과 적응의 보정 효과를 이 toy에서는 확인했다.
긴 지연 스트레스는 거의 무감쇠이며 준비 구조가 인과성 한계를 없애지 않음을 함께 보여준다.
단일 seed·고정 합성 S·후보 하나의 결과이므로 일반적 우월성·최적성으로 확대하지 않는다.
세부 대역/소스/모든 run은 다음 PC 로컬 산출물에 있고 `results/`는 Git에 올리지 않는다.

- `results/prepared_fir/pc_20260915_linear_01/{report.json,metrics.csv,summary.md}`
- `results/prepared_fir/pc_20260915_tanh_01/{report.json,metrics.csv,summary.md}`

전체 CPU 회귀는 **632 passed, 2 skipped**다. 건너뜀은 현장 raw/실측 metrics 부재이며 실기 PASS가 아니다.

## 4. 저장 ESS의 대역별 검증 준비

Jetson 현장 측정에서 회수한 `raw_measurement.npz`가 있을 때만 실행한다.
이미 있는 S 모델 NPZ 단독으로 이 진단을 대신할 수 없다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/eval/analyze_path_bands.py \
  --raw-npz results/acoustic_SESSION_ID/calibration/RAW_SESSION/raw_measurement.npz \
  --band 800 1600 --out results/acoustic_SESSION_ID/path_band_trial_01
```

`repeat_irs`, scalar JSON 메타, 모든 반복의 저장 지연이 필요하다. cancel/ch1 ESS만 지원하며
3회 미만·반복 수 불일치·비정상 배열·지연 누락은 거부한다. 기존 측정 실패 여부와 별개로 모든 반복을 사용한다.
원시 PCM·출력·health가 없으면 해당 QA는 unknown이다. PCM에서 IR을 재추출한 도구는 아니며
저장된 반복 IR의 유도 과정은 미재검증으로 표시한다.

동일 compact FIR의 FFT와, 거기에 저장 지연의 위상을 복원한 FFT를 비교한다.
따라서 정렬 전후 차이에 추출 창 차이가 섞이지 않는다. 주파수별 반복 일관성
`|mean(H)|² / mean(|H|²)`, 최악 pair의 극성 민감 유사도, pair 크기 차이, 지연 spread를 따로 보고한다.
0.9는 **참고선**이며 기존 `coherence_median` 전체 Pearson 상관과 동일한 검증량이 아니다.
SNR·실시간 위상/클록 안정성은 이 도구로 증명하지 않는다.

경로 끝점도 확인하기 위해 이 도구의 기본 target은 **[800,1600]**이다.
하위 대역은 [800,1000), [1000,1600]으로 1kHz를 중복하지 않는다.
이는 ERR 에너지 분할 보고의 [800,1600)와 목적이 다르며 출력에 끝점 포함 여부를 표시한다.
가진 범위 밖·FFT bin 없음·무전력은 계산 불가이며 숫자나 신뢰대역으로 대체하지 않는다.

새 폴더에 JSON·CSV·Markdown만 저장하며 **S NPZ 생성·기존 자산 수정·신뢰대역 자동 승격은 하지 않는다**.
`promote_secondary_allowed=false`다. 입력/출력 오류는 exit 1, 진단 저장은 exit 0이며 품질 PASS가 아니다.

## 5. 다음 연결과 Jetson에 남은 일

이 PC에서 가능한 후속 작업은 실제 회수 자료 재현, 다양한 조건의 사전 필터 bank 준비,
과거 REF 특징만 쓰는 선택기 학습, 독립 소스/경로/레벨의 강건성 비교다.
훈련/검증/test의 녹음 그룹 분리와 후보의 S/조건 메타를 먼저 갖춘다.
현재 한 후보 시험을 완성된 SFANC/CNN으로 부르거나 데이터 없는 모델 학습을 성공으로 기록하지 않는다.

실제 Jetson에서만 확인할 것은 I/O·콜백 마감·지터, 사용자 입회 하의 S/F와 레벨별 비선형 측정,
독립 acoustic OFF→ON→OFF 세션이다. 라이브 FIR 경로 연결과 스레드 전달은 그 전에 별도 리뷰가 필요하다.
기존 USB DAC와 마이크를 유지하며 시스템 변경이나 장치 교체로 실패를 우회하지 않는다.
작업 위치·회수 목록·실기 중단 조건은 [docs/14](14_pc_jetson_workplan.md)를 따른다.
