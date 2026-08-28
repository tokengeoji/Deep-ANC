# Fullband causal v4 — 연속 파일럿과 조건부 canonical FIR 계약

> 상태 기준일: 2026-08-28. 이 문서는 실제 측정 합격 보고서가 아니다. v4는 현재
> signal-only이고 `LIVE_AUTHORITY=None`이다. 따라서 오디오 출력 0회,
> `canonical_training_eligible=false`이며 학습 loader에 넘길 실제 artifact도 없다.

## 1. 이번 설계가 해결하는 문제

v3는 각 PE burst 뒤 exact-zero tail을 길게 두었다. tail 안에는 acoustic pilot도 없었기
때문에, 그 구간에서 생겼다가 다음 anchor 전에 상쇄되는 DAC–ADC clock trajectory를 raw로
관측할 수 없었다. v4는 두 DAC channel에 서로 직교하는 저역 pilot을 capture 처음부터 끝까지
계속 둔다.

- sample rate/block: 48,000 Hz / 256 samples
- pilot period: 32,768 samples
- pilot band: 152–600 Hz
- P pilot actual-int16 DFT support: `k % 8 == 0`
- S pilot actual-int16 DFT support: `k % 8 == 4`
- 각 high-band PE: integer comb `1-z^-8192`를 적용하여 두 pilot 집합을 포함한
  `k % 4 == 0`에서 actual-int16 DFT가 exact zero
- P/S marker와 `fit_a`, `fit_b`, `holdout`: 서로 다른 seed와 서로 다른 PCM SHA
- 각 central payload 앞/뒤 cyclic exclusion: 16,384 samples
- 최대 후보 history: delay 4,800 + support 8,192 = 12,992 samples

따라서 고정 LTI 플랜트라면 central 32,768-sample payload의 pilot line에서 다음이 성립한다.

```text
Y_ERR,P(k) = H_ERR,P(k) X_P(k),  X_S(k)=X_PE(k)=0
Y_ERR,S(k) = H_ERR,S(k) X_S(k),  X_P(k)=X_PE(k)=0
```

REF에서도 같은 식이 독립적으로 성립한다. unknown `H(k)`의 진폭과 상수 위상은 nuisance로
profile-out할 수 있지만, DAC line의 ADC-index 방향 주파수와 시간에 따른 위상 기울기는
바꾸지 못한다. 이 때문에 **fixed-LTI 모델 클래스 안에서는** ERR/REF×P/S 네 view가 같은
DAC-q/ADC rate ratio를 식별할 수 있다. marker는 rate가 아니라 intercept와 0–4,800 sample
coarse branch를 고정한다.

## 2. signal-only 결과

무음 신호 생성 명령은 다음뿐이다.

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal_continuous.py --dry-run
```

현재 deterministic 결과:

| 항목 | 값 |
|---|---:|
| frames / duration | 688,128 / **14.336 s** |
| high-band slot duration | 10.922667 s |
| submitted peak | PCM **96** (`< 98`) |
| submitted PCM SHA-256 | `f83ca7c1e7a9193da7bcb2f0e50f198ad050c027edbd5c4c0950e798b7c53a94` |
| pilot period PCM SHA-256 | `7283fbd8976bef2e0715359d83b962e245540bd6fd2c859d1981c7580d5b18bf` |
| canonical payload SHA-256 | `0cfb77a58537c5ebebdac595a45b5b8e508b0481c1b0ddcf604761c190056fb6` |

`--execute-live`는 plan 생성, `sounddevice` import, `/dev/snd` 검사보다 먼저 exit 2다. 위 SHA는
signal review용이지 live authority가 아니다. 저장 plan을 만들 때도 기존 파일은 덮어쓰지
않는다.

### 125 Hz octave 하단 감사

docs/07의 125 Hz octave는 `[125/sqrt(2),125*sqrt(2)] =
[88.38834765,176.7766953] Hz`다. 현 shared broadband contract v2는 150 Hz에서 시작하므로
그 octave의 88.388–150 Hz를 평가하지 않는다. 따라서 **현 7대역 plan은 저역 최종 권위로
불충분하며 live/canonical authority를 열 수 없다.** 125 Hz center가 목록에 있다는 사실은
octave 전체를 측정했다는 증거가 아니다.

future contract 검토용으로 다음 signal-only 명령을 제공한다.

```bash
.venv/bin/python scripts/data/measure_paths_fullband_causal_continuous.py \
  --dry-run --excitation-lower-hz 80
```

80 Hz 후보는 duration 14.336 s 불변, payload peak 77–78, submitted peak PCM 95,
actual pilot-line null 0이며 PCM SHA는
`607a02b32b559a4e8b1c88c9c6a74d4cc94394bfd6760d5be04e97e4d06315d7`, plan payload SHA는
`fb6bb1991708e87ddd1a113b6ea221b5438e7f9c53b3bc1fd596c099960ad95a`다. 152–600 Hz
pilot은 PE의 `k%4=0` exact null 위에 있으므로 clock 설계도 바뀌지 않는다. 그러나 이 신호를
현 v2 SHA 아래 8대역 증거라고 부르지 않는다. shared v3가 `[88.388,150]` subband와
measurement excitation lower 80 Hz를 고정한 뒤 같은 builder에 contract를 주입해야 한다.
builder/scorer는 contract subband 개수를 동적으로 강제하며 현 v2 7대역 fixture는 회귀로
보존한다.

## 3. clock gate

fit에는 `lead_reference`, P/S `fit_a`, P/S `fit_b`만 쓴다. marker, P/S holdout, tail은
validation-only다. holdout 결과로 map을 다시 맞추거나 임계값을 고치지 않는다.

필수 raw gate:

1. intended float가 아니라 실제 submitted `[frames,2] int16`과 그 SHA를 분모로 쓴다.
2. actual opposite-channel pilot line은 `≤1e-8`, PE pilot line도 `≤1e-8`이다.
3. ERR/REF×P/S 네 독립 rate의 capture-end disagreement는 `≤0.050 sample`이다.
4. linear/cubic interpolation 차이는 `≤0.006 sample`이다.
5. fit 밖 marker/holdout/tail의 최대 phase residual은 `≤0.050 sample`, 합산은
   `≤0.056 sample`, 절대 hard limit은 **0.0675518903 sample**이다.
6. 모든 validation line의 complex/phase coherence는 `≥0.995`다.
7. callback frame count, ADC/DAC time_info는 monotonic이고 sample-count slip이 0이다.
8. 시간 구간별 rate/FIR/phase change-point가 하나라도 검출되면 fixed-LTI 가정 위반으로
   실패한다. 결과에 맞춰 affine 구간을 추가하지 않는다.

synthetic fixed-LTI fixture는 0 ppm과 ±413.931 ppm에서 위 임계를 통과한다. 구간별 rate
변경, callback 1-sample slip, marker alias는 실패하고, pilot line이 exact zero인 high-band
plant mutation은 clock map을 바꾸지 않는다. 이것은 실제 하드웨어 PASS가 아니다.

## 4. actual-input causal operator

각 mic에 대해 P와 S를 따로 빼서 SISO로 맞추지 않는다. 반대 경로 pilot까지 포함한 실제
stereo submitted PCM 전체를 쓰는 joint two-input linear-convolution operator를 만든다.

```text
y_m[n] = sum_k h_m,P[k] x_P[n-d_P-k]
       + sum_k h_m,S[k] x_S[n-d_S-k]
```

matrix-free `A`/`A^T`로 fit하며 dense normal matrix를 만들지 않는다. marker와 frozen clock
map에서 얻은 coarse integer onset을 먼저 고정한다. sub-sample bulk residual은
`[-0.5,0.5)`로 따로 저장하고 post-onset FIR 자체가 그 fractional phase를 표현한다.

후보 support는 raw를 보기 전에 `1024, 2048, 4096, 8192`로 고정한다. 각 support에 대해:

- actual-input exact `A^T A` extremal eigenvalue condition `≤20`
- fit_a와 fit_b 자체 residual 각각 `≤0.03`
- fit_a→fit_b와 fit_b→fit_a cross residual 각각 `≤0.05`
- P와 S 각각의 두 fit tap relative disagreement `≤0.10`
- ERR/REF 각각 P/S×contract 전 subband가 아래 energy/noise/complex/phase gate 전부 PASS
- fit_a와 fit_b FIR transfer도 P/S×contract 전 subband agreement/error/timing PASS

를 모두 통과한 **가장 짧은 support**만 freeze한다. 이 선택에는 holdout byte, scalar, summary를
전혀 쓰지 않는다. freeze 뒤 holdout residual `≤0.05`와 contract 전 subband complex agreement
`≥0.995`, relative error `≤0.10`, 위 20 dB timing gate를 terminal하게 적용한다. holdout이
실패하면 더 긴 support를 사후 선택하지 않고 새 generation을 BLOCK한다.

여기서 global residual은 진단값일 뿐 충분조건이 아니다. 각 subband는 실제 active path
int16 DFT가 nonzero이고 반대 path DFT가 `≤1e-8`인 isolated bin만 쓴다. lead와 tail의
actual submitted **두 입력 모두** `≤1e-8`인 bin에서 response power p95 noise floor를 구한다.
각 noise reference row의 exact-zero bin이 8개 이상이어야 하며 다음 predeclared gate를
하나라도 못 넘으면 그 path/mic/band가 실패한다.

| gate | 고정값 |
|---|---:|
| isolated response / phase bin | 각각 ≥8 |
| input RMS / target RMS | 각각 ≥−90 dBFS |
| target-to-exact-zero-noise | ≥20 dB |
| noise-conditioned relative residual | ≤0.10 |
| complex agreement / phase coherence | 각각 ≥0.995 |
| timing | band upper 기준 20 dB 감쇠 위상 예산 이내 |

실제로 2.828–5.657 kHz의 primary holdout만 1.12배 바꾼 synthetic 반례는 global residual
0.043872로 0.05를 통과하지만 해당 band residual 0.107143 때문에 거부된다. lead/tail에
noise만 올려 target SNR을 20 dB 아래로 만든 반례도 global residual이 거의 0이어도 거부된다.

### 4.1 actual-int16 condition 반증 — current v4는 여기서 BLOCK

CP가 `delay+support`보다 길기 때문에 central period의 linear causal convolution은 exact
circular convolution으로 계산할 수 있다. 이 연산자는 full-capture generic 연산자와 forward
relative error `7.25e-16`, adjoint error `1.35e-14`, gradient relative error `7.06e-16`로
일치했다. 이 exact 연산자의 `A^T A` extremal eigenvalue를 actual submitted int16으로 직접
계산한 결과는 다음과 같다(`d_P=1700,d_S=1300` synthetic branch; single-path 하한은 delay와
무관하다).

| role / support | condition | 판정 |
|---|---:|---|
| fit_a / 1024 | **280.374302** | FAIL (`>20`) |
| fit_b / 1024 | **297.776432** | FAIL (`>20`) |
| fit_a primary-only lower bound | 275.825697 | FAIL |
| fit_a secondary-only lower bound | 272.876663 | FAIL |

더 긴 support의 Gram은 1024 Gram을 principal submatrix로 포함한다. Cauchy interlacing에 따라
support가 늘면 `lambda_max`는 작아질 수 없고 `lambda_min`은 커질 수 없으므로 condition도
개선될 수 없다. 따라서 2048/4096/8192를 계산 결과에 맞춰 임계 완화하거나 사후 선택하지
않고 **전부 BLOCK**한다. current candidate generator는 response를 fit하기 전에 이 receipt로
fail-closed한다.

진단용 짧은 support도 해답이 아니었다.

| support | fit_a condition | canonical 여부 |
|---:|---:|---|
| 64 | 172.753798 | history 부족 + condition FAIL |
| 128 | 216.891387 | history 부족 + condition FAIL |
| 256 | 239.035777 | history 부족 + condition FAIL |
| 512 | 258.183224 | history 부족 + condition FAIL |

원인은 P/S 동시 regression이 아니다. main PE는 이미 primary/secondary slot으로 시간 분리되어
있고, fit_a single-path 하한 275.83과 joint 280.37의 차이는 작다. 100–11.4 kHz bandlimit와
`1-z^-8192`가 pilot line뿐 아니라 모든 `k%4=0` bin을 지우는 구조, sparse continuous pilot의
큰 line eigenvalue가 unconstrained 48-kHz FIR coefficient를 ill-conditioned하게 만든다.

Nyquist guard까지 넓힌 signal-only 대안도 같은 comb null을 유지하면 실패했다. 모두
actual-int16 pilot null 0, duration 14.336 s, peak≤98이며 future 8-band primary isolated bin은
`[37,89,180,239,358,734,1689,3380]`개였다.

| PE band (Hz) | peak PCM | support1024 condition |
|---|---:|---:|
| 20–23000 | 97 | 239.083241 |
| 20–23500 | 96 | 230.810945 |
| 40–23000 | 92 | 239.635226 |
| 40–23500 | 96 | **229.845945** |
| 80–23000 | 94 | 246.113415 |
| 80–23500 | 94 | 242.934447 |

payload RMS는 34.65–36.61 PCM, crest는 약 7 dB라 peak/crest 문제가 아니다. 가장 나은
40–23.5 kHz에서도 pilot을 빼면 condition 114.0701, pilot을 넣으면 229.8459다. 즉 단순
full-Nyquist 확장만으로는 20을 만족하지 못한다.

후속 후보는 별도 schema `fullband_causal_reserved_pilot_sparse_null_v5_signal_only`로 두고,
reserved 77개 line만 exact-zero로 만드는 integer subspace/coding 또는 independent electrical
clock witness로 PE와 acoustic pilot을 분리해야 한다. 새 신호도 support1024 exact condition
`≤20`을 **live 전에 먼저** 통과해야 한다. 그때까지 current v4 schema/SHA를 재해석하거나
speaker에 출력하지 않는다.

독립 signal-only crosscheck에서 같은 two-row 구조와 기존 pilot에 peak PCM 78의 near-white
actual-int16 PE를 넣었을 때 support1024 condition은 uniform int16 **3.7788**, Rademacher
**2.4454**였다. 이는 fullband PE 자체가 아니라 current bandlimit/comb-null이 blocker라는
반증이다. 단, near-white PE는 pilot line을 오염하므로 current clock estimator에 그대로 넣어
PASS할 수 없다.

더 작은 v5 후보는 main PE의 기존 P/S 시간분리를 활용한다. primary-active slot의 P pilot
line에는 반대 S pilot이 exact zero이므로 `actual P pilot+P PE` 전체를 분모로 P clock view를
fit하고, secondary-active slot에서는 반대로 S view만 fit한다. lead/tail pure pilot과 matching
marker/holdout으로 검증하고 ERR/REF×P/S가 하나의 q map에 동의해야 한다. 또는 각 line에서
P/S 두 unknown transfer를 실제 2-input design matrix로 joint profile-out한다. 두 방식 모두
새 schema(예: `fullband_causal_time_separated_joint_clock_v5_signal_only`)와 signed
±413.931ppm/piecewise/highband mutation fixture가 필요하다. 20–23.5 kHz near-Nyquist 자극은
speaker/amp의 비가청 전력과 level/SNR을 바꾸므로 signal condition PASS만으로 live를 허용하지
않고 별도 하드웨어 대역/레벨 안전 review를 먼저 거쳐야 한다.

연속 pilot 때문에 zero-tail raw 자체는 없다. 대신 최종 high-band slot 뒤 3개 pilot period 중
첫 period로 응답을 clear하고, 둘째를 clock/plant validation, 셋째를 zero-phase filter의
capture-edge guard로 둔다. frozen joint FIR로 실제 pilot response까지 예측해 제거한 뒤:

- out-of-pilot tail RMS/L1/peak가 active response의 3% 이하
- 마지막 period residual이 input-only noise floor +1 dB 이하
- support 밖 delayed echo fixture와 taps에 delay를 다시 넣는 double-delay fixture 거부

를 요구해야 한다. 이 유한 capture는 predeclared support를 반증/유지할 뿐, 무한히 늦은 echo가
없음을 증명하지 않는다. 8192도 실패하면 임계 완화가 아니라 더 긴 capture/support가 필요하다.

## 5. 조건부 canonical training artifact

향후 offline publisher가 발행할 수 있는 유일한 loader authority 이름은 다음으로 고정한다.

```text
fullband_causal_joint_fir_training_plant_v4
```

이 schema는 후속 신호가 condition gate를 통과할 때의 fail-closed interface다. current v4
신호는 §4.1에서 이미 condition FAIL이므로 이 artifact를 발행할 수 없다. loader는 fixture나
signal plan만으로 PASS하면 안 된다.
실제 artifact에는 최소한 다음을 모두 넣어야 한다.

### exact envelope와 도달 가능한 분석 원문

top-level key는 다음 집합과 **정확히** 같아야 한다. extra/missing key는 실패다.

```text
schema, authority, status, canonical_training_eligible, synthetic_fixture,
control_band_contract_sha256, sample_rate_hz, block_size, latency,
handoff_extra_samples, capture_id, operator, clock, fit, holdout,
stationarity, provenance, evidence_sha256
```

- `schema=fullband_causal_training_authority_envelope_v4`
- `authority=fullband_causal_joint_fir_training_plant_v4`
- `status=PASS`, `canonical_training_eligible=true`, `synthetic_fixture=false`
- `evidence_sha256`는 그 key만 뺀 canonical JSON의 SHA-256이다.
- `control_band_contract_sha256`가 지시한 **모든** subband가 receipt에 정확히 한 번씩
  있어야 한다. loader가 7이라는 개수를 하드코딩해서는 안 된다.

`fit` exact key는 `passed,err,ref,receipt_sha256`이다. `err`와 `ref` 각각의 exact
key는 다음과 같다.

```text
fit_a_candidate_ref, fit_b_candidate_ref,
fit_a_score, fit_b_score, fit_a_on_fit_b_score, fit_b_on_fit_a_score,
freeze_ref, selected_support_samples
```

네 score는 SHA 문자열만 두지 않고 `joint_actual_input_role_score_v4` **전체 canonical
JSON**을 inline한다. 즉 actual-zero noise-floor receipt와 P/S×계약 전 대역의 input/target
energy, SNR, noise-conditioned residual, complex/phase/timing row가 loader에 도달해야 한다.
candidate/freeze처럼 큰 JSON은 다음 exact immutable reference로 둔다.

```text
schema=immutable_json_artifact_reference_v4, path, file_sha256, internal_sha256
```

`path`는 envelope 파일 기준 lexical relative path이고 symlink component가 없어야 한다.
loader는 실제 파일 bytes의 SHA와 내부 canonical payload SHA를 둘 다 다시 계산한다.
reference의 `schema`는 항상 `immutable_json_artifact_reference_v4`이고, 실제 참조 JSON의
schema는 candidate면 `joint_actual_input_fit_candidate_v4`, freeze면
`frozen_fit_only_joint_causal_candidate_v4`여야 한다.

`holdout` exact key는 `passed,err,ref,receipt_sha256`, 각 mic exact key는
`holdout_score,terminal_receipt`다. 둘 다 full inline JSON이며 score의 noise-floor/P/S×전
대역 row가 빠지면 실패한다. `stationarity` exact key는
`passed,err,ref,change_point_receipt,change_point_receipt_sha256,receipt_sha256`다.
`err/ref`에는 full `fit_a_fit_b_transfer_stationarity_v4`를 inline한다. change-point raw
통계가 없거나 fit/holdout 뒤에 임계값을 정한 receipt는 실패다.

`provenance` exact key는 다음과 같다.

```text
repository_commit, repository_dirty,
signal_plan_path, signal_plan_file_sha256, signal_plan_payload_sha256,
submitted_pcm_path, submitted_pcm_file_sha256,
raw_path, raw_file_sha256, raw_internal_sha256,
callback_arrays_sha256,
analysis_path, analysis_file_sha256, analysis_internal_sha256,
analysis_code_sha256, environment_receipt_sha256, level_evidence_sha256,
hardware_fingerprint_sha256, xrun_count, clip_count
```

`repository_dirty=false`, xrun/clip 0이어야 하며 raw/analysis/callback/score 파일까지 실제
bytes를 따라가 재계산할 수 없으면 admission은 fail-closed다. self-asserted SHA만으로는
절대 PASS하지 않는다.

`signal_plan_payload_sha256`는 parse한 plan에서 `canonical_payload_sha256` key를 제거한 뒤
canonical JSON으로 다시 계산한 SHA이며 full parsed JSON hash가 아니다. plan 안 claimed
`canonical_payload_sha256`와 반드시 같아야 한다. `submitted_pcm_path`는 별도 no-replace
`.npy`이고 `allow_pickle=false`, C-contiguous `[frames,2] int16`만 허용한다. file SHA는 NPY
전체 bytes, submitted/source PCM SHA는 load한 array의 C-order raw bytes SHA다. plan의
frames/peak/PCM SHA와 모두 다시 대조한다.

raw NPZ internal digest domain은 `fullband_causal_v4_raw_array_archive_v1`이다. NPZ의
모든 실제 member를 key로 정렬하고 아래 operator digest와 같은 key/dtype/ndim/shape/C-bytes
규칙으로 계산한다. raw NPZ 안에 자기 자신의 internal SHA field를 넣지 않는다. callback
digest는 같은 raw NPZ의 다음 다섯 실제 array만 domain
`fullband_causal_v4_callback_arrays_v1`로 계산한다.

```text
callback_start_frames, callback_frame_counts,
input_buffer_adc_time, output_buffer_dac_time, callback_current_time
```

analysis artifact는 no-replace canonical JSON이다. `analysis_internal_sha256`는 parse한 전체
JSON을 `ensure_ascii=false, sort_keys=true, separators=(',',':'), allow_nan=false`로 다시
직렬화한 UTF-8 bytes SHA다. analysis JSON 내부에는 자기 `analysis_internal_sha256`를 넣지
않는다. file SHA는 원래 저장 bytes라서 semantic/internal SHA와 별도로 모두 맞아야 한다.

### operator reference와 NPZ exact schema

`operator` exact key는 다음과 같다.

```text
schema, npz_path, npz_file_sha256, npz_internal_sha256,
primary_fir_sha256, secondary_fir_sha256,
source_submitted_pcm_sha256, source_raw_sha256, fit_freeze_sha256,
support_samples, coarse_delay_samples, fractional_delay_samples,
bulk_delay_samples_fractional, post_onset_peak_index,
effective_delay_samples, plant_delays_payload, plant_delays_sha256
```

`schema=fullband_causal_joint_fir_operator_reference_v4`다. NPZ의 exact key는 다음이며
extra object/pickle array를 금지한다.

```text
schema, primary_post_onset_fir, secondary_post_onset_fir,
primary_coarse_delay_samples, secondary_coarse_delay_samples,
primary_fractional_delay_samples, secondary_fractional_delay_samples,
support_samples, sample_rate_hz,
source_submitted_pcm_sha256, source_raw_sha256, fit_freeze_sha256
```

`schema`와 SHA 문자열은 UTF-8 bytes의 `uint8` 1-D array, FIR은 C-contiguous float64
1-D exact support, sample/delay integer scalar는 little-endian int64, fractional scalar는
little-endian float64다. internal digest domain은
`joint_causal_operator_npz_internal_v4`로 고정한다. digest 입력은 다음을 순서대로
이어 붙인다.

1. domain UTF-8 + NUL
2. 정렬한 각 key마다 key UTF-8 + NUL
3. `dtype.str` UTF-8 + NUL
4. ndim의 little-endian int64 bytes
5. 각 shape 원소의 little-endian int64 bytes
6. C-contiguous raw bytes

file SHA, internal SHA, P/S FIR array SHA를 모두 다시 계산해야 하며 어느 하나라도 다르면
거부한다. 이 NPZ는 기존 measured-tone/broadband-source-v2 NPZ와 호환되지 않는다.

### plant와 timing

- schema/authority, `canonical_training_eligible=true`
- ERR 기준 P/S `post_onset_fir` float64 arrays와 각 array SHA
- P/S coarse integer onset, fractional residual, fractional bulk delay
- post-onset support, tap peak/effective delay, causal tail residual
- sample rate 48 kHz, block 256, latency low
- handoff 256은 S taps에 bake하지 않고 별도 필드로 한 번만 결속
- `PlantDelays.lead()`로 다시 유도 가능한 P/S timing payload와 SHA

### 원본과 clock

- exact signal plan path/file SHA/canonical payload SHA/submitted PCM SHA
- immutable actual submitted int16, raw ERR/REF, callback arrays의 path/file/internal SHA
- capture ID, hardware fingerprint, channel map, level evidence/meter SHA, xrun/clip 0
- marker P/S branch와 alias receipt
- DAC-q map knots/rate/intercept, actual pilot spectra SHA
- ERR/REF×P/S view ratios, validation residual/coherence, linear/cubic receipt
- `clock_witness_kind=continuous_acoustic_reserved_pilot_v4`
- `independent_electrical_witness_present`와, 존재하면 그 raw/SHA

독립 electrical loopback은 가장 강한 반증 자료지만 fixed-LTI 조건부 canonical에 무조건 필수는
아니다. false일 때에는 아래 stationarity 증거와 이 문서의 scope limitation을 반드시 저장한다.

### fit/holdout/stationarity

- fit_a/fit_b input/response/selected-index SHA와 각 candidate SHA
- support별 exact condition, own residual, 양방향 cross residual, tap disagreement
- shortest-pass support decision과 freeze SHA
- 네 fit/cross score 각각 P/S×계약 전 subband target-energy/SNR/noise-floor/complex/phase PASS
- untouched holdout input/response SHA, terminal residual/subband/timing full receipt와 SHA
- 시간 row별 P/S FIR, pilot phase, rate change-point 통계와 모두 PASS인 원자료
- multi-panel measured complex response와 contract 전 subband agreement receipt
- source plan과 분석 코드 exact commit, 의존성/environment receipt

criterion/trainer는 이 authority와 외부 evidence SHA를 다시 계산하고, P/S·timing·control-band
contract가 exact할 때만 causal training operator로 받아야 한다.

## 6. 정보론적 범위와 현재 판정

fixed-LTI 모델 클래스 밖에서는 다음 두 가설이 byte-identical raw를 만들 수 있다.

```text
A: fixed H, q[n] = n + delta[n]인 ADC clock
B: ideal ADC, 출력이 Hx[n + delta[n]]인 공통 time-varying plant delay
```

이 반례는 모든 acoustic system identification에 적용되는 모델 범위 한계다. 따라서 이를 이유로
electrical witness를 영구 필수로 두지는 않는다. 대신 실제 raw에서 두 독립 PE fit, untouched
holdout, ERR/REF×P/S 공통 q, 시간 구간별 FIR/phase stationarity가 모두 통과한 범위에 한해
`fixed-LTI conditional canonical`을 허용한다. 공통 time-varying delay를 독립적으로 배제해야
하는 환경에서는 electrical loopback이 추가로 필요하다.

현재는 live raw와 publisher가 없으므로 판정은 다음과 같다.

```text
fixed-LTI signal identifiability: synthetic PASS
physical clock/FIR stationarity: BLOCKED (raw 없음)
actual-int16 causal FIR condition: FAIL (fit_a 280.374 / fit_b 297.776 > 20)
conditional canonical training artifact: NOT CREATED
live output authority: None
actual audio output count: 0
```

`LIVE_AUTHORITY`는 envelope mapping이 아니라 exact Python `None`이다. future loader는 이를
추정 mapping으로 바꾸지 않고, 별도 no-replace envelope의 `authority` 문자열과 전체 evidence를
검증해야 한다.
