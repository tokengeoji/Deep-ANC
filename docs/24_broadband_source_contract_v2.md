# 광대역 source acquisition contract v2 독립 검토

> 기준일: 2026-08-28. v1을 수정하거나 PASS로 바꾸지 않는다. 외부 다운로드·Drive 변경·
> 오디오 출력은 수행하지 않았다.

## 1. 결론

immutable compressed 원본과 짧은 독립 component의 결정론적 합성은 **출처·독립성·성능
게이트를 낮추지 않고도** 별도 v2 세대로 지원할 수 있다. 단순히 `.mp3` container라는 이유만으로
탈락시키지 않는 대신, 다음 bytes chain 중 하나라도 없으면 더 엄격하게 차단한다.

```text
immutable original bytes SHA
  → codec/header receipt
  → decoder runtime fingerprint + full decoded PCM SHA
  → 48 kHz processed PCM/WAV + transform SHA
  → exact excerpt PCM SHA
  → pre-EQ native 11.314 kHz/7-band/crest receipt
  → deterministic composition + boundary receipt
  → predeclared bounded EQ receipt
  → exact 15 s submitted Q15 SHA와 실제 source 9×7 재계산
  → canonical causal P operator를 실제 적용한 predicted-ERR 9×7 재계산
```

현재 실제 상태는 여전히 **0/48, BLOCKED**다. Drive object metadata만으로 위 chain을 만들 수
없고 canonical fullband causal P도 없다. v2 schema와 fixture가 유효하다는 사실은 실제 source
후보 PASS가 아니다.

코드 단일 출처는
`src/deep_anc/data/broadband_source_contract_v2.py`의
`broadband_recorded_v2_source_contract_v2`다. v1
`broadband_recorded_v2_source_acquisition_manifest_v1`은 변경하지 않는다.

## 2. 왜 v2가 필요한가

사용자 목표는 저·고역과 speech/music/environment/machine의 실제 ANC이지 특정 container나
한 원본의 최소 길이 자체가 아니다. 기존 public music pipeline도 FMA MP3를 사용한다. 따라서
원본이 compressed라는 사실만으로 배제하는 것은 목표에서 직접 유도되지 않는다.

반대로 container 허용만 바꾸면 decoder 차이·손상·숨은 transcode를 추적할 수 없다. v2는
compressed original SHA와 decoded PCM SHA를 분리하고, decoder runtime fingerprint와 전체
duration mono decode receipt를 필수로 만든다. 원본이 stereo여도 immutable header의 channel
수를 보존하고, 고정 decoder/downmix 결과 전체를 float32 PCM SHA로 봉인한다. 같은 MP3 bytes라도
decoder 또는 decoded PCM이 달라지면 다른 artifact이며 재검증 전에는 사용할 수 없다.

현재 160개 processed pool의 source-only 9×7 진단에서는 all-seven-band segment가 8개 이상인
파일이 v2 environment 1개뿐이고 나머지는 0개였다. 이 진단은 source composition이나 container
허용만으로 48개가 채워지지 않는다는 뜻이다. density 0.25 또는 8/9 하한은 낮추지 않는다.

## 3. 두 acquisition mode

### 3.1 `single_long_form`

한 component가 processed 48 kHz에서 exact 720,000 frame을 제공한다.

우선순위는 다음과 같다.

1. lossless long-form single source
2. provenance가 완전한 immutable compressed long-form
3. multi-component sequence

rank 1/2를 선택하려면 같은 slot의 더 나은 rank 후보 수가 0임을 pre-analysis inventory와
selection receipt로 증명한다. lossless long-form을 계속 우선한다.

### 3.2 `multi_component_sequence`

15초 미만 source는 repeat/loop하지 않는다. 같은 family/split의 서로 다른 lineage component를
최소 3개 사용하고, 각 component의 processed excerpt는 최소 1.5초다.

- excerpt 길이 합은 exact 720,000 frame
- 순서와 각 excerpt raw/processed PCM SHA 고정
- 경계마다 480-frame linear Q15 fade-out/fade-in을 양쪽에 적용
- overlap 없이 길이를 보존
- coefficient 배열 SHA와 경계별 receipt 필수
- component ID, original SHA, decoded/processed/excerpt PCM SHA, lineage component 모두 후보
  내부와 전체 48후보 사이에서 중복 금지
- 각 component의 `source_family`와 `assigned_split`은 campaign slot과 exact 일치

group identity는 한 clip의 이름이 아니다. 모든 component lineage ID를 정렬한 집합의 SHA를
union identity로 사용한다. 하나의 component가 다른 candidate 또는 split에 재사용되면 관련
candidate를 모두 탈락시킨다. unrelated component를 합쳤다고 독립 group 수가 3개 늘어나는
것이 아니라 composite 전체가 group 하나다.

## 4. boundary artifact가 고역 증거가 되는 우회 방지

각 component는 합성·fade·EQ **전** exact native excerpt에서 다음을 통과해야 한다.

- native fs ≥22,628 Hz, native Nyquist ≥11,313.708 Hz
- actual native bandwidth verified
- 일곱 대역 density 각각 ≥0.25
- crest factor 0–15 dB
- spectral receipt가 decoded PCM과 native/processed excerpt SHA에 exact 결속
- `boundary_or_eq_used_for_evidence=false`

local 검증에서는 receipt 숫자만 믿지 않는다. decoded full PCM에서 native excerpt가 실제
slice인지, processed full PCM에서 processed excerpt가 실제 slice인지, processed WAV의 decoded
float32가 raw processed PCM과 동일한지 확인하고 native excerpt의 7대역 density와 crest를
다시 계산한다. 따라서 fade transient나 EQ가 원래 없던 고역의 유일한 evidence가 될 수 없다.

합성 후에는 component excerpt bytes에 계약의 Q15 boundary fade를 실제 적용해 pre-EQ PCM SHA를 다시
만든다. 현재 identity EQ 경로는 이 PCM과 final processed WAV가 sample-exact임을 확인한다.
고정 EQ 경로는 FIR coefficients를 실제 적용하는 production validator가 아직 고정되지 않아
local issuer에서 명시적으로 BLOCKED다. 임계값이나 receipt 숫자를 신뢰해 우회하지 않는다.

## 5. predeclared bounded EQ

현재 source-only 진단을 이유로 per-source adaptive EQ나 임계 완화를 허용하지 않는다. v2에서
허용하는 것은 candidate 분석 전에 별도 commit으로 고정한 다음 둘뿐이다.

- 모든 family가 같은 `global_fixed` FIR
- family 안의 모든 candidate가 같은 `family_fixed` FIR 네 개

hard ceiling은 다음과 같다.

| 항목 | 상한 |
|---|---:|
| FIR taps | 513 |
| peak boost | 12 dB |
| maximum attenuation | 12 dB |
| crest 증가 | 3 dB |
| submitted peak | 4,915 int16 (`0.15 FS`) |

filter coefficients, frequency-response receipt, policy commit, analysis commit과 git ancestry를
결속한다. candidate별 filter, 결과를 보고 바꾼 filter, 동적 압축, clipping은 금지한다. component
pre-EQ native gate를 먼저 통과하므로 고정 EQ는 bandwidth를 발명하지 못한다. 실제 FIR 적용
연산·alignment schema까지 review되어야 issuer authority를 열 수 있고, 그 전에는 구조적으로
유효한 fixed-EQ draft도 actual PASS가 아니다.

이 EQ는 `configs/data_sim.yaml`의 무작위 training augmentation과 다른 acquisition transform다.
무작위 augmentation 설정을 source authority로 재사용하지 않는다.

EQ-shaped source는 natural sound 성능 증거가 아니다. 모든 v2 final source role은
`coverage_source_not_unmodified_level5_challenge`이며, 모델 선택 뒤 별도 **unmodified Level-5
natural challenge**를 반드시 수행한다. 이 challenge 요구를 manifest에서 끌 수 없다.

## 6. 남아 있는 실제 acquisition 양

현재 verified source가 0개이므로 각 family/split에 group 4개가 그대로 필요하다.

- 모든 후보가 long-form이면 family당 12 source, 총 48 source
- 모든 후보가 short sequence이면 candidate당 component 최소 3개이므로 family당 최소 36개,
  총 최소 144개 독립 lineage component
- long/short 혼합이면 각 candidate의 union group 수는 항상 1이며 실제 component 수를 별도로
  기록

FMA MP3나 ESC-50 WAV의 파일 수를 이 숫자로 바로 세지 않는다. original SHA, full decode,
component-level spectral/crest, DSU와 기존 recorded/synthetic 교집합 0 receipt가 생긴 뒤에만
해당 slot 후보가 된다. 16 kHz와 22.05 kHz component는 v2에서도 탈락한다.

## 7. publisher skeleton과 현재 차단

`scripts/data/issue_broadband_source_manifest_v2.py`는 다음 두 경로만 제공한다.

```bash
# schema 확인: 외부 파일/오디오 접근 없음
.venv/bin/python scripts/data/issue_broadband_source_manifest_v2.py --print-contract

# DRAFT structural audit: actual PASS를 반환하지 않음
.venv/bin/python scripts/data/issue_broadband_source_manifest_v2.py \
  --campaign <campaign.json> --draft <draft-v2.json> --audit-only
```

현재 contract의 `issuer_authority`는 `None`이다. 따라서 `--issue --output <new.json>`은 입력을
발행하려 시도하기 전에 **항상 BLOCKED**이며 target을 만들지 않는다. `--audit-only`도 metadata
구조만 검사하므로 `actual_acquisition_pass=false`다.

향후 root review로 issuer를 열려면 다음을 전부 실제 bytes에서 다시 계산해야 한다.

- synthetic fixture가 아님
- 48개 모든 local original/decoded/processed/excerpt/Q15/receipt file의 size·SHA와 실제 slice 관계
- actual submitted Q15에서 고정 9 segment×7 band source density 재계산·JSON exact 일치
- submitted Q15은 recorded-v2 공용 0.1초 edge envelope와 quantizer로 WAV에서 다시 렌더링해
  저장된 raw PCM과 sample-exact 일치
- exact physical fullband causal P validator PASS와 별도 causal operator NPZ schema PASS
- operator NPZ의 FIR/delay를 submitted Q15에 실제 적용한 predicted-ERR 9×7 재계산·JSON exact 일치
- predeclared EQ commit이 candidate-analysis commit의 실제 git ancestor
- 전역 component/content/lineage 재사용 0
- 12개 split×family cell 각각 4개

causal operator NPZ는 exact key, 48 kHz, control-contract SHA, submitted-Q15 input role, nonnegative
delay, causal-history flag, finite FIR 및 FIR bytes SHA가 모두 맞아야 한다. 제한된 tone/compact
diagnostic FIR은 이 schema를 가질 수 없다. 발행 결과도
`verified_acquisition_input_not_live_source_plan`일 뿐이며 recorded-v2 live plan과 스피커 출력
authority는 계속 `None`이다. 현재는 실제 v2 source도 canonical fullband P/operator도 없으므로
상태는 **0/48, BLOCKED**다.

## 8. 반례 검토 결과

다음 우회는 negative fixture로 거부된다.

- decoder fingerprint 또는 original→decoded SHA 누락
- 16 kHz/22.05 kHz component
- component 2개뿐인 short sequence
- boundary coefficient/위치 변조
- 같은 component/original/decoded/processed/lineage를 다른 candidate에 재사용
- boundary 또는 EQ를 component native bandwidth evidence로 사용
- final predicted-ERR 9×7 중 고역 실패를 평균으로 숨김
- submitted Q15와 다른 source 9×7 JSON 숫자 제출
- causal P operator를 적용하지 않고 predicted-ERR 9×7 JSON 숫자만 제출
- lossless long-form이 있는데 낮은 rank를 선택
- candidate-adaptive EQ 또는 12 dB 초과 boost
- synthetic 48개 fixture를 actual issuer에 사용

관련 단위 테스트는 `tests/test_broadband_source_contract_v2.py`다. 구조 fixture는 실제 audio
bytes/P operator가 아니므로 언제나 structural-only다. 이 방어 조건이 유지되는 범위에서는 v2가
provenance·독립성·성능 gate를 약화시키는 반례를 발견하지 못했다. 하나라도 제거해야만 실제
후보가 채워진다면 v2를 사용하지 않고 다시 BLOCKED로 판정한다.
