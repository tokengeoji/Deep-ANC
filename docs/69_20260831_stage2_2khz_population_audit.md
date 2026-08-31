# Stage-2 2 kHz 모집단·계보 actual-bytes 감사

> 판정일: 2026-08-31
>
> 계약: `broadband_2khz_octave_88_2828_v1`
>
> 계약 SHA-256: `70fc33d20a43bedaa5a51f8e19aed12fff687d8fb3901501f4a49bf2746d97cf`
>
> 전체 판정: public scratch-pretrain `BLOCKED`, recorded fine-tune `BLOCKED`

이 문서는 기존 Stage-1 모집단 보고서나 결과 bytes를 수정하지 않는다. 아래 숫자는
`source_aligned.wav`, `mics.wav` ch0, source-pool WAV와 현재 manifest의 실제 bytes를
읽어 새 Stage-2 계약으로 계산한 값이다. P/S 유효성, ANC 감쇠 또는 학습 완료를
주장하는 문서가 아니다.

## 1. 증거 고정

- 전체 JSON: `results/data_audit/stage2_2khz_population_20260831_v3.json`
- JSON file SHA-256:
  `6b8d0be58868c01e3aa305510e7310d880faedfdfa4150e42c51153af7cc6ae2`
- payload 내부 evidence SHA-256:
  `a318d8c9dff68e58aaa6870ffeb569f58431d5e68d3e6c9f71f10bbb0a7bf70c`
- schema: `stage2_2khz_population_byte_audit_v3`
- 고정 분석 창: 각 세션 5--65초, 48 kHz, Welch 8192/overlap 4096
- source-density 및 ERR target-density 하한: `0.25`; source--ERR coherence 하한:
  `0.60`; 각 family×split×octave 독립 lineage group 하한: `4`

한 WAV에서 SciPy가 이해하지 못한 non-data metadata chunk를 건너뛴다는 경고가 한 번
발생했다. WAV audio data chunk는 정상 decode됐고 해당 파일의 bytes/SHA도 고정됐다.
이는 ANC 성능 증거가 아니며, 향후 transfer 때 원본 bytes 불변성을 다시 검사한다.

## 2. 가설 A — 기존 82세션이면 Stage-2 fine-tune 모집단이 충분하다

### [가설]

기존 82세션이 네 family와 125--2000 Hz objective octave에서 각각 독립 group 4개를
제공할 가능성이 있다고 가정했다.

### [근거]

기존 문서에는 95.67분, 82/82 QA PASS와 1600--2828 Hz joint-valid 3개라는 요약이
있었다. 그러나 이는 Stage-2의 다섯 objective octave와 60개
split×family×octave cell을 직접 증명하지 않는다.

### [확인 방법]

각 `source_aligned.wav`의 objective-octave density와 같은 시간창의 실제 ERR density,
source--ERR coherence를 계산했다. source density, target density, coherence를 모두
통과한 세션만 `population_joint`로 세고, 같은 lineage component의 반복 세션은 한
group으로만 셌다.

### [결과]

표의 각 cell은 `source-density group / population-joint group / 부족 group`이다.
부족은 하한 4에서 계산한다.

| Split | Family | 125 Hz | 250 Hz | 500 Hz | 1 kHz | 2 kHz |
|---|---|---:|---:|---:|---:|---:|
| train | speech | 5/3/1 | 6/6/0 | 6/6/0 | 4/1/3 | 1/0/4 |
| train | music | 10/9/0 | 10/10/0 | 10/10/0 | 8/3/1 | 2/0/4 |
| train | environment | 6/4/0 | 6/6/0 | 6/6/0 | 5/5/0 | 5/0/4 |
| train | machine | 6/6/0 | 7/7/0 | 7/7/0 | 5/3/1 | 3/0/4 |
| val | speech | 3/2/2 | 4/4/0 | 4/4/0 | 3/1/3 | 1/0/4 |
| val | music | 4/3/1 | 4/4/0 | 4/4/0 | 4/0/4 | 0/0/4 |
| val | environment | 4/2/2 | 4/4/0 | 3/3/1 | 4/3/1 | 3/1/3 |
| val | machine | 4/3/1 | 4/4/0 | 4/4/0 | 4/3/1 | 4/0/4 |
| test | speech | 3/2/2 | 4/4/0 | 4/4/0 | 4/1/3 | 0/0/4 |
| test | music | 4/4/0 | 4/4/0 | 4/4/0 | 4/1/3 | 0/0/4 |
| test | environment | 4/4/0 | 4/4/0 | 4/4/0 | 4/1/3 | 1/0/4 |
| test | machine | 4/3/1 | 4/4/0 | 4/4/0 | 4/0/4 | 0/0/4 |

60개 cell 중 31개가 부족하다. 특히 2 kHz octave
`[1414.213562, 2828.427125] Hz`에서 population-joint group은 environment val의
1개를 제외하면 모두 0개다.

2 kHz octave 평균이 1.6 kHz 부근의 near-zero를 가리지 않도록
`[1425.437949, 1795.939277) Hz` one-third-octave sentinel도 같은 raw에서 별도로
재계산했다. 표는 `source-density group / target-density+coherence group /
population-joint group / 부족 group`이다.

| Split | speech | music | environment | machine |
|---|---:|---:|---:|---:|
| train | 3/0/0/4 | 4/0/0/4 | 4/0/0/4 | 5/0/0/4 |
| val | 2/0/0/4 | 2/0/0/4 | 3/1/1/3 | 4/0/0/4 |
| test | 0/0/0/4 | 1/0/0/4 | 1/0/0/4 | 2/0/0/4 |

sentinel 12개 cell도 모두 부족하다. source density만 있는 세션을 세지 않았으며,
실제 ERR density와 source--ERR coherence를 동시에 요구했다.

과거 방식도 별도로 재계산했다. legacy 150--11313 Hz 분모에서의 physical
1600--2828 Hz target-density/coherence joint-valid는 다음 세 세션·세 독립 group으로,
역사적 요약값 3과 정확히 일치했다.

- `20260804_102951_file`
- `20260804_103501_file`
- `20260804_103737_file`

legacy 3개와 Stage-2 2 kHz objective cell을 섞지 않는다. 분모·대역·population
조건이 다르기 때문이다.

### [판정]

**Contradicted.** 기존 82세션은 Stage-2 recorded fine-tune 모집단으로 충분하지 않다.

### [다음 행동]

아래 최소 slot 계획에 새 독립 source identity를 먼저 배정하고, 짧은 실측 뒤 각
slot이 지정 octave의 source-density, 실제 ERR-density, coherence를 모두 통과했을
때만 채택한다. threshold를 낮추거나 세션 반복으로 group 수를 늘리지 않는다.

## 3. 중복 녹음을 최소화한 후보 계획

한 새 source가 여러 부족 octave를 동시에 통과한다는 최선 조건에서, 각
family×split의 최대 deficit만큼 slot을 만들고 deficit 층을 겹쳤다. 결과는 최소
**47개 새 독립 recording/component**다. 이는 달성 보장이 아니라 하한이다.
sentinel deficit을 같은 slot에 겹쳐 다시 최적화해도 하한 47은 변하지 않지만,
**47개 모든 slot이 1.6 kHz sentinel까지 동시에 통과해야 한다.**

| Split/family | 부족 `[125,250,500,1000,2000]` | 최소 slot | slot별 동시 충족 octave |
|---|---:|---:|---|
| train/speech | 1,0,0,3,4 | 4 | `125+1000+2000`; `1000+2000`×2; `2000` |
| train/music | 0,0,0,1,4 | 4 | `1000+2000`; `2000`×3 |
| train/environment | 0,0,0,0,4 | 4 | `2000`×4 |
| train/machine | 0,0,0,1,4 | 4 | `1000+2000`; `2000`×3 |
| val/speech | 2,0,0,3,4 | 4 | `125+1000+2000`×2; `1000+2000`; `2000` |
| val/music | 1,0,0,4,4 | 4 | `125+1000+2000`; `1000+2000`×3 |
| val/environment | 2,0,1,1,3 | 3 | `125+500+1000+2000`; `125+2000`; `2000` |
| val/machine | 1,0,0,1,4 | 4 | `125+1000+2000`; `2000`×3 |
| test/speech | 2,0,0,3,4 | 4 | `125+1000+2000`×2; `1000+2000`; `2000` |
| test/music | 0,0,0,3,4 | 4 | `1000+2000`×3; `2000` |
| test/environment | 0,0,0,3,4 | 4 | `1000+2000`×3; `2000` |
| test/machine | 1,0,0,4,4 | 4 | `125+1000+2000`; `1000+2000`×3 |

test 16 slots는 conditioning하지 않은 untouched natural source로 예약한다. 학습,
증강 fitting, loss/model/checkpoint 선택에 사용하지 않는다. train source를 EQ나 gain으로
conditioning해 test slot을 채우는 것은 금지한다. val도 training에 사용하지 않는다.

## 4. 가설 B — 현재 source lineage는 split leakage가 있다

### [가설]

같은 WAV/original clip 또는 연결된 speaker--book, artist--album,
machine--recording-session이 여러 split에 있을 가능성이 있다고 가정했다.

### [근거]

과거 FMA 원본 60개가 synthetic train과 겹친 이력이 있고, 파일명만 비교하면
연결 component 누수를 놓친다.

### [확인 방법]

canonical holdout validator로 source CSV를 실제 FMA `tracks.csv`, LibriSpeech
chapter metadata, ESC metadata에 다시 연결했다. component ID, composite WAV path,
composite WAV content SHA, original clip을 split별로 독립 집계했다.

### [결과]

현재 82세션의 cross-split 수는 component 0, composite WAV path 0, composite WAV
SHA 0, original clip 0이다. FMA artist/album, Libri speaker/book도 component 안에
포함됐다. 다만 현재 machine recorded source는 ESC source-file component이며,
MIMII machine/session canonical manifest는 로컬에 없다.

### [판정]

현재 82세션 lineage는 **Confirmed PASS**. MIMII machine/session public lineage는
**Inconclusive/BLOCKED**다.

### [다음 행동]

새 source를 배정할 때 기존 82개와 새 slot 전체를 함께 component화한다. MIMII는
machine ID와 recording session을 manifest의 강제 lineage key로 넣기 전에는
public pretrain admission을 열지 않는다.

### 현재 legacy public manifest와 recorded holdout의 교집합

위 PASS는 **recorded 82 내부 split**에만 해당한다. 별도 actual-bytes 감사
`results/data_audit/stage2_public_recorded_lineage_20260831.json`은 holdout에 고정된
FMA/LibriSpeech/ESC-50 metadata SHA를 현재 bytes에서 다시 확인하고, public row의
basename과 artist--album, speaker--book, ESC source key를 transitive component로
묶었다.

| Legacy public manifest | exact basename | direct identity | recorded 연결 component row |
|---|---:|---:|---:|
| speech | 8 | 784 | 819 (train 736 / val 46 / test 37) |
| music | 0 | 824 | 1005 (train 891 / val 60 / test 54) |
| ESC-50 | 82 | 163 | 163 (train 143 / val 13 / test 7) |

따라서 현 legacy public manifest는 **BLOCKED**이며 Stage-2 학습에 사용할 수 없다.
직접 identity 수 784/824/163은 transitive component 수가 아니라 하한이다. raw
public corpus에서 recorded 연결 component 전체를 제외한 뒤 manifest를 새로 만들어야
한다. 재현 명령은 다음과 같으며 output은 no-replace다.

```bash
.venv/bin/python scripts/data/audit_stage2_public_recorded_lineage.py \
  --repo-root . \
  --output results/data_audit/stage2_public_recorded_lineage_<new-id>.json
```

## 5. 가설 C — 현재 public bundle만으로 scratch pretrain을 열 수 있다

### [가설]

recorded 모집단 부족과 무관하게, 로컬 public/synthetic 자료는 Stage-2 scratch
pretrain을 시작할 정도로 bytes와 lineage가 준비됐을 가능성이 있다고 가정했다.

### [근거]

speech/music/ESC-50/DEMAND manifest는 존재하며, public scratch pretrain은 recorded
추가 녹음과 독립적으로 먼저 실행할 수 있다는 운영 방침이 확정됐다.

### [확인 방법]

manifest row마다 현재 repository에 실제 file bytes가 있는지, 선언 SHA/size가
일치하는지, 2 kHz octave 상단을 native Nyquist가 덮는지, decode와 source-density가
가능한지, split과 lineage field가 완전한지 검사했다. 없는 DNS/MIMII manifest도
묵시적으로 통과시키지 않았다.

### [결과]

| Manifest | 실제 local bytes | Lineage complete | 판정 |
|---|---:|---:|---|
| speech | 2272/2272 | 아니요 | BLOCKED |
| music | 0/7877 | 아니요 | BLOCKED |
| ESC-50 | 3/1475 | 아니요 | BLOCKED |
| DEMAND canonical_v4 | 0/96 | 예 | BLOCKED |
| DNS canonical | manifest 없음 | 아니요 | BLOCKED |
| MIMII machine canonical | manifest 없음 | 아니요 | BLOCKED |

speech actual bytes의 2 kHz source-density PASS item은 train 272, val 11, test 14로
관측됐지만, legacy manifest에 content/group/lineage field가 없어 독립 lineage group
PASS로 승격하지 않았다.

### [판정]

**Contradicted/BLOCKED.** 현재 로컬 bundle로 public scratch pretrain을 열 수 없다.
다만 이 판정은 recorded 47-slot 부족 때문이 아니다. JSON과 CLI는
`public_synthetic_scratch_pretrain`과 `recorded_measured_finetune`을 별도 status와
blocker로 낸다. 향후 public 축이 PASS면 recorded 축이 BLOCKED여도 scratch pretrain은
독립적으로 열 수 있다.

### [다음 행동]

Drive/원격 raw를 canonical 상대경로로 복구하고, DNS·MIMII를 포함한 Stage-2 manifest에
content SHA/size와 connected-component lineage를 기록한다. 실제 bytes set을 다시
감사해 public 축 PASS를 받은 뒤 scratch pretrain만 먼저 실행한다. recorded
fine-tune은 47-slot 실측 모집단과 별도 P/S·latency admission이 모두 PASS할 때까지
fail-closed한다.

## 6. 구현·재현 명령

감사 코드는 다음 파일에 있다.

- `src/deep_anc/data/stage2_2khz_population_audit.py`
- `src/deep_anc/data/stage2_public_recorded_lineage_audit.py`
- `scripts/data/audit_stage2_2khz_population.py`
- `scripts/data/audit_stage2_public_recorded_lineage.py`
- `tests/test_stage2_2khz_population_audit.py`
- `tests/test_stage2_public_recorded_lineage_audit.py`

새 output 경로만 사용한다. 기존 JSON을 덮어쓰지 않는다.

```bash
.venv/bin/python scripts/data/audit_stage2_2khz_population.py \
  --repository-root . \
  --output results/data_audit/stage2_2khz_population_<new-id>.json
```

CLI exit `0`은 두 데이터 축이 모두 PASS인 combined inventory 결과다. exit `1`은
정상적으로 blocker를 검출한 것이고, exit `2`는 입력/schema/IO 감사 실패다. 실제
scratch-pretrain admission은 JSON의 public 축만 소비하고, 전체 학습 admission으로
오인하지 않는다.
