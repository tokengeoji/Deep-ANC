# 2026-08-31 Stage-2 데이터·과거 성능·FxLMS 증거 감사

> 범위: read-only artifact 감사. 오디오/GPU/Elice를 실행하지 않았다. 현재
> Stage-2 성능 결과나 quiet-zone 성능을 발행하는 문서가 아니다.

## [가설] 현재 Stage-2 canonical 모델이 존재한다

### [근거]

working tree에는 `broadband_2khz_octave_88_2828_v1` 계약, evaluator, campaign
profile이 있다.

### [확인 방법]

local HEAD와 `origin/dev`, working tree, `runs/`의 Stage-2 checkpoint, campaign의
external contract/checkpoint binding을 직접 확인했다.

### [결과]

- local/`origin/dev`: `cc930fb43c9dfc9f1dd620d6397a066d8b37bcf8`, branch `dev`
- 감사 시작 시 working tree: modified 31 + untracked 52 = **83 entries** (이 문서 등
  후속 산출물 추가 전 snapshot)
- Stage-2 contract/evaluator/config/docs는 모두 untracked라 `origin/dev`에 없다.
- `runs/`에 Stage-2/2kHz checkpoint는 0개다.
- `configs/stage2_2khz_campaign.yaml:28-42`의 external pretrain/fine-tune contract와
  canonical checkpoint path/SHA는 모두 `null`이다.
- immutable Stage-2 contract SHA는
  `70fc33d20a43bedaa5a51f8e19aed12fff687d8fb3901501f4a49bf2746d97cf`이고,
  full-octave v3 SHA는 기존
  `53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2`로 불변이다.

### [판정]

**Contradicted.** 현재 canonical Stage-2 model의 실제 주파수 감쇠는 판단할 수 없다.

### [다음 행동]

dirty 변경을 테스트·검토·exact commit으로 고정하고, 새 Stage-2 P/S와 public bundle
admission 후 scratch pretrain을 시작한다. legacy checkpoint는 init/resume하지 않는다.

## [가설] 현재 데이터가 125 Hz--2 kHz 학습·fine-tune 모집단을 충족한다

### [근거]

82 recorded sessions와 legacy public manifests 및 Drive snapshot이 존재한다.

### [확인 방법]

실제 recorded WAV 82세션의 5개 objective octave source/ERR density와 coherence를
재계산하고, holdout에 결속된 FMA/LibriSpeech/ESC metadata bytes에서 public/recorded
transitive component를 다시 만들었다. Drive는 object path/SHA metadata만 read-only로
감사했다.

### [결과]

| 증거 | 실제 결과 | 판정 |
|---|---|---|
| `stage2_2khz_population_20260831_v3.json` SHA `6b8d0b...6ae2` | 과거 1600--2828 joint-valid 독립 group 정확히 3; objective 60 cells 중 31 부족; 1.6k sentinel 12 cells 모두 부족; 결합 최적 하한 신규 47 components | recorded fine-tune BLOCKED |
| `stage2_public_recorded_lineage_20260831.json` SHA `cfac9f...8a59` | recorded 연결 public rows: speech 819, music 1005, ESC-50 163; exact basename 8/0/82 | legacy public BLOCKED |
| 같은 lineage audit | recorded 82 내부 component/WAV path/WAV SHA/original clip cross-split 모두 0 | recorded 내부만 PASS |
| `stage2_drive_public_restore_20260831.json` SHA `d37591...4119` | FMA/Libri/ESC partial 12,819 files, 9,480,223,737 bytes 복구 가능; DNS/DEMAND/MIMII fixed archive cache 0 | public scratch pretrain BLOCKED |

speech/music/ESC legacy manifest 자체도 semantic component가 split을 가로지른다.
recorded 부족은 future public scratch-pretrain PASS를 막지 않지만, 현재 public 축도
독립적으로 BLOCKED다. final test slot은 conditioning되지 않은 natural source로
예약되며 training stimulus와 합칠 수 없다.

### [판정]

**Confirmed BLOCKED.** 양이 적어서만이 아니라 실제 source-density와 계보가 동시에
부족하다. public dataset으로 recorded 물리 coverage를 대체할 수 없다.

### [다음 행동]

public은 recorded 연결 component를 원본 raw에서 제외하고 DNS/MIMII/DEMAND까지 canonical
manifest를 재생성한다. recorded는 `docs/69_20260831_stage2_2khz_population_audit.md`의
family×split×octave 47-slot 하한을 독립 source로 채운다.

## [가설] 과거 Deep-ANC raw가 광대역 또는 unseen 성능을 입증한다

### [근거]

Jetson에 legacy physical session, recorded offline evaluation과 별도 readonly CRN 실덕트
capture가 존재한다.

### [확인 방법]

OFF/ON raw 존재, source-energy, checkpoint/plant/lead, runtime validity와 natural unseen
lineage를 분리했다. 문서 scalar보다 raw를 우선하고 current Stage-2로 승격하지 않았다.

### [결과]

| Artifact | 분류 | 실제로 말할 수 있는 것 |
|---|---|---|
| `results/session_20260804_0939/*_dl.npz`, metrics SHA `2e6933...22b42` | physical legacy diagnostic | 300 Hz +6.48 dB, band +4.09, multitone +1.43, nonlinear +5.13, 1.2k +0.35, 800--1600 −0.05; source-valid 1.6k octave −0.99~+0.36. 2/4/8 source-valid raw 없음 (`docs/66`) |
| 같은 raw의 source 없는 4/8 kHz | harmful-injection diagnostic | ERR ON +18.0/+21.6 dB, control −44.7/−40.8 dBFS. 감쇠 수치가 아니며 latency 원인은 telemetry 부재로 Inconclusive |
| `results/session_20260804_125538/raw/voice_in_noise_dl.npz` SHA `a00e8a...04c9` | invalid physical diagnostic | old metric +4.39 dB이나 underrun 87/xrun 9, source provenance 없음. unseen level 주장 불가 |
| `runs/finetune_tiny/eval_recorded_test/metrics.npz` SHA `a83c4c...3714` | offline legacy measured-session simulation | G4 FAIL: trusted/fullband NMSE +0.22/+0.32 dB; source family groups 1--2뿐. 실제 realtime OFF/ON이 아님 |
| readonly CRN `.../DIAGNOSTIC_REPORT_20260812.md` SHA `fd41f7...140c` | 다른 repository/checkpoint의 physical diagnostic | valid sessions에서 150--1600 Hz −1.71~−2.36 dB(증폭), 600--1000 −7.68~−8.51 dB. current Stage-2 결과가 아님 |

tone/band/multitone/nonlinear generator는 natural unseen Level 5가 아니다. voice mixture는
원본 계보가 없어 Level을 매길 수 없다. current model의 speech/music/environment/machine
Level-5 physical 결과는 0건이다.

### [판정]

- current broadband ANC: **Not yet demonstrated**
- current unseen natural sound generalization: **Inconclusive / no evidence**
- legacy 저역 동작: **Confirmed diagnostic-only**
- legacy 중·고역 안정 감쇠: **Contradicted**

### [다음 행동]

Stage-2 125--2k single-ERR campaign에서 lineage-clean 네 family와 untouched natural test를
분리하고, 1.6k sentinel, 2k octave, exact-zero runtime telemetry를 같은 raw receipt에 묶는다.
single-point PASS를 spatial quiet-zone으로 승격하지 않는다.

## [가설] 기존 FxLMS가 실제 덕트에서 30--35 dB를 입증했다

### [근거]

`~/anc_project`에 `secondary_path_300hz_delay70_best_35db.npz`라는 파일명이 있고,
Deep-ANC repository에는 legacy FxLMS log와 silent health NPZ가 있다.

### [확인 방법]

파일명 대신 bytes/SHA와 NPZ field, OFF/ON protocol, tail OFF, xrun, 같은 P/S/source/window의
matched Deep-ANC raw가 있는지 검사했다.

### [결과]

- `secondary_path_300hz_delay70_best_35db.npz` SHA
  `865146230237713fc77571572360593fd069d482cb5fc602c8fb56bafd90550f`는
  `secondary_path.npz`와 byte-identical한 P/S artifact다. `fit_improvement_db=0.6528`,
  `coherence_median=0.1560`; ANC OFF/ON error raw나 35 dB attenuation field가 없다.
- `results/legacy_fxlms/fxlms_300hz_amp010.log` SHA `769d05...6aff`는 ON log 9개 뒤
  비정상 종료했다. 중앙 +0.11, 말미 +0.36, 순간 +0.63 dB이나 xrun 3→9이며 tail OFF와
  weight receipt가 없다 (`docs/07_evaluation_protocol.md:271-276`).
- silent health 두 NPZ는 `anc_gain=0`, source/control=0인 무음 runtime health이며 성능 raw가
  아니다.
- HANDOFF:1112-1116의 26.10 dB/최악 17.28 dB는 strict P/S causal FIR **이론 상한**이고
  물리 FxLMS 감쇠가 아니다.
- 동일 source/P/S/gain/SPL/block/lead/window의 matched physical FxLMS vs Deep-ANC raw는 없다.

### [판정]

“FxLMS physical 30--35 dB”: **Inconclusive / not demonstrated**. 현재 artifact로
Deep-ANC 대비 우월·열등 비교도 금지한다.

### [다음 행동]

Stage-2 canonical checkpoint와 runtime exact-zero gate가 모두 준비된 뒤 같은 source와
volume에서 OFF/DL/FxLMS 순서를 교차한 matched raw를 별도 세션으로 저장한다. 30--35 dB를
기대값으로 주입하지 말고 raw `10log10(P_OFF/P_ON)`에서 다시 계산한다.
