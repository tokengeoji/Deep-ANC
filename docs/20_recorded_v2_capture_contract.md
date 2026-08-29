# 광대역 recorded-v2 실측 수집 계약

> 상태 기준일: 2026-08-28. 현재 **오디오 출력 0회**, live authority는 `None`이다.
> 이 문서는 향후 수집 경로를 고정하지만, 아직 없는 source/plant를 만들어 냈다고
> 주장하지 않는다.

## 1. 현재 판정

광대역 recorded-v2 수집은 `BLOCKED`다.

1. `canonical_training_eligible=true`인 physical fullband causal P/S evidence가 없다.
   현재 v1/v2/v3 causal 설계와 150--1600 Hz strict plant는 이 역할을 대신할 수 없다.
2. speech/music/environment/machine × train/val/test마다 독립 source 4개, 최소 48개의
   native/processed/transform/lineage receipt가 없다.
3. 따라서 exact source-plan file SHA와 plant file SHA를 고정할 수 없고,
   `RECORDED_V2_LIVE_AUTHORITY=None`을 유지한다.

현재 가능한 명령은 무음 BLOCKED 확인뿐이다.

```bash
.venv/bin/python scripts/data/record_broadband_v2.py --dry-run
```

정상 현재 결과는 exit 2와 다음 두 의미다.

```text
[BLOCKED] 외부 source-plan/plant SHA anchor 없음
[오디오] sounddevice import/open 0회, 파일 생성/수정 0개
```

실제 source와 plant가 생긴 뒤에도 먼저 다음 **dry-run**만 실행한다.

```bash
.venv/bin/python scripts/data/record_broadband_v2.py --dry-run \
  --source-plan data/source_plans/recorded_broadband_v2/<generation>.json \
  --expected-source-plan-sha256 <외부에서_고정한_64자리_SHA> \
  --plant-evidence assets/measured/<fullband-causal-plant>.json \
  --expected-plant-sha256 <외부에서_고정한_64자리_SHA> \
  --out-root data/recorded_broadband_v2/<generation>
```

구조 dry-run이 통과해도 live authority가 아니며 현재 exit 2를 유지한다. `--execute-live`는
사용자 입회·볼륨 최저·배선/geometry 확인을 모두 줘도 authority 검증 전에 sounddevice를
import하지 않는다.

## 2. 기존 수집기와 분리하는 이유

`record_duct.py`와 `record_session_batch.py`는 Stage-1의 REF-witness 정렬과 82+17 세대
수집에 사용한다. 다음 차이 때문에 그 session을 광대역-v2로 이름만 바꿀 수 없다.

| 항목 | Stage-1 기존 경로 | recorded-v2 필수 계약 |
|---|---|---|
| source 길이 | CLI 가변 | exact 15.000초, 720,000 frames |
| 제출 증거 | float `source.wav` 중심 | actual little-endian int16 mono/stereo PCM SHA |
| clock | REF 기반 저역 lag track | P-submitted/ERR/REF 공통 absolute DAC-q map |
| clock fit | 150--600 Hz 품질 게이트 | exact 152--600 Hz only, 고역 fit/repair 금지 |
| 정렬 | source를 ADC에 warp | mics를 absolute DAC-q grid로 cubic 재표본화 |
| plant | Stage-1 strict P | physical fullband causal P/S 외부 SHA anchor |
| coverage | 150--1600 Hz 중심 | persisted ERR ch0의 7대역 raw 재계산 |
| 발행 | session staging→active | immutable raw 먼저, analysis/session은 그 뒤 별도 no-replace |

## 3. exact source와 submitted PCM

구현 단일 출처는 `src/deep_anc/data/recorded_v2_capture.py`다. source-plan 한 행은 다음을
모두 묶는다.

- 사전 지정 `split`, `source_family`, `group_id`, `lineage_id`
- native lossless file path/size/SHA와 실제 native sample rate/Nyquist
- processed 48 kHz file path/size/SHA
- 48 kHz가 아니면 polyphase FIR 정확히 1회의 transform receipt와 11.314 kHz passband 증거
- exact 15초 `start_frame`, static Q15 gain, 0.1초 양끝 fade, quantizer 규약
- actual submitted mono int16 PCM SHA
- ch0=source, ch1=exact zero인 interleaved stereo int16 PCM SHA

48 kHz tensor라는 이유로 native bandwidth를 인정하지 않는다. 16/22.05 kHz 원본을
upsample한 파일은 8 kHz octave 상단 증거가 아니며, 같은 native/processed SHA 또는 같은
lineage를 독립 group으로 다시 세지 않는다.

## 4. P/ERR/REF 공통 absolute DAC-q receipt

timewarp receipt schema는
`recorded_broadband_v2_absolute_dac_q_timewarp_v1`이다.

- 입력 anchor: raw-capture evidence SHA, full submitted-output array SHA, raw mics array SHA
- witness exact 집합: `P_submitted_playback`, `ERR_ch0`, `REF_ch1`
- 세 witness가 같은 `common_map_sha256`을 사용
- fit은 even window, odd window는 leave-out holdout이며 fit/선택에 사용하지 않음
- fit band exact 152--600 Hz
- `highband_used_for_clock_fit=false`
- `highband_phase_repair_samples=0`
- fit/holdout 각각 8개 이상, score 0.995 이상
- leave-out ≤0.050 sample, cubic crosscheck ≤0.006 sample,
  combined ≤0.056 sample
- callback time은 monotonic/sample-slip witness로만 사용
- xrun/sample slip 0

ADC-frame knot와 absolute DAC-q knot는 별도 immutable NPZ의 실제 float64 array SHA까지
검사한다. 문자열 SHA나 `clock_witness=true`만으로는 통과할 수 없다.

## 5. raw-first/no-replace lifecycle

한 세션은 다음 순서를 건너뛸 수 없다.

```text
actual submitted int16 + raw ERR/REF int32 + callback witness
    ↓  renameat2(RENAME_NOREPLACE)
immutable raw directory / raw_receipt.json
    ↓  raw/xrun/clip/slip/SHA 검증
absolute DAC-q timewarp offline receipt
    ↓  inverse q map + cubic ADC resampling
source.wav + source_aligned.wav + mics.wav
    ↓  persisted WAV를 다시 읽음
9 segments × 7 bands actual ERR coverage
    ↓  all-seven-band segment ≥8
canonical session directory (RENAME_NOREPLACE)
```

raw publisher는 analysis를 실행하지 않으며 receipt에
`raw_published_before_analysis=true`, `analysis_started=false`를 저장한다. xrun/clip/slip가
있어도 raw는 `INVALID_PRESERVED_RAW`로 보존하되 aligned canonical session은 발행하지 않는다.
기존 raw/session target은 덮어쓰지 않는다. 분석 실패 시 canonical target은 비어 있고,
원 raw는 그대로 남는다.

## 6. 실제 ERR coverage 재검산

coverage 입력은 메모리상의 pre-write array가 아니라 실제 저장된 다음 파일이다.

- `source_aligned.wav`: 48 kHz, mono, exact 720,000 frames
- `mics.wav`: 48 kHz, ERR ch0/REF ch1, exact 720,000 frames

결정론적 population은 `0.25 + 1.5*k`초에서 시작하는 1.5초 구간 9개다. 각 구간에서
actual ERR ch0의 source coherence와 target-density를 일곱 point-control subband마다
재계산한다. 같은 구간에서 일곱 대역이 모두 PASS한 구간이 최소 8개여야 한 session/group이
coverage 후보가 된다.

`broadband_coverage_receipt.py`는 `require_local_files=False`인 합성 fixture를 이제
`STRUCTURAL_ONLY_NOT_CAMPAIGN_ELIGIBLE`로만 반환한다. 실제 campaign PASS에는 각 session의
`session.json`→raw receipt→timewarp receipt→persisted WAV→coverage JSON을 전부 다시 열고
SHA와 수치를 재계산해야 한다.

## 7. 향후 실기 직전 강제 항목

live authority를 별도 commit에서 고정하기 전에는 아래 항목을 실행하지 않는다.

- 전체 pytest 0 FAIL, `py_compile`, `git diff --check`
- exact source-plan/plant file SHA 외부 anchor
- 모든 source의 submitted PCM SHA 재생 직전 TOCTOU 재검증
- `/dev/snd`의 PCM owner와 `/proc/asound/.../status` 무점유 확인
- input probe 직전과 output open 직전 무점유를 각각 receipt에 저장
- 사용자 입회, 볼륨 최저, ERR/REF·NS/CS·덕트 geometry 확인
- raw target/session target freshness와 symlink 부재
- 실패 세션 자동 재출력 금지

최소 48세션이면 source audible은 총 720초(12분)다. input-only 3.5초와 silent settle 1초를
세션마다 더한 분석 제외 연결 상한은 936초(15분 36초)다. 이것은 한 번의 장시간 연결 창을
허용한다는 뜻이 아니다. source family별 짧은 batch와 즉시 분리 절차는 live authority 고정
전에 사용자에게 별도로 보고하고 승인받는다.
