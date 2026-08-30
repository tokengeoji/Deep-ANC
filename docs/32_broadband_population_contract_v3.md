# Broadband v3 population/source coverage 계약

## 1. 결론

자연 음성·음악 clip 하나에 125 Hz--8 kHz의 모든 octave가 동시에 강하게 존재할 것을
요구하면, 자연 분포를 검증하는 것이 아니라 shaped broadband 신호를 선별하게 된다. 기존
source-v2/recorded-v2의 `9구간 중 8구간 × 일곱 대역 모두 density >= 0.25` 조건에서
현재 자연 speech/music 후보가 `0/48`이 된 것은 이 문제의 실제 징후다.

v3는 목표 주파수, density 임계값, lineage 다양성 또는 batch 하한을 낮추지 않는다. 자격의
단위만 다음처럼 바로잡는다.

1. clip 전체가 아니라 `(lineage component, item, band)`별로 자격을 재계산한다.
2. 일부 대역만 가진 자연 clip도 그 대역의 유효 item으로 남긴다.
3. `split × family × band` population과 실제 family-balanced batch가 전체 목표 대역을
   강제로 덮는다.
4. 모델 고정 뒤 수행할 untouched Level-5 자연음 challenge는 population과 완전히 분리한다.

구현은
[`src/deep_anc/data/broadband_population_contract_v3.py`](../src/deep_anc/data/broadband_population_contract_v3.py)에
있다. 기존 v1/v2 manifest, issuer, serialization, digest, 학습 코드는 수정하지 않았다.

## 2. 고정된 주파수 계약

population manifest에는 `BroadbandFullOctaveContractV3.canonical()`의 payload 전체와 SHA를
동시에 넣는다. 둘 중 하나만 맞아서는 통과하지 않는다.

- control-band v3 SHA-256:
  `53579b9ff8419ac19fb2458c29a3e8a94ffbb2eeb88cc07f34b76c68033989f2`
- population v3 계약 SHA-256:
  `8f0d1e3897a2ace87059cecd584ea5da3ed0ecdb01a45a5855b2475cfe6e05c1`
- density 하한: 대역별 `>= 0.25` (변경 없음)
- physical identification: 8구간, 88.3883476483--11313.7084989848 Hz
- equal-weight objective: 125/250/500/1000/2000/4000/8000 Hz 중심의 정확한 7 octave
- source family: speech/music/environment/machine
- split: train/val/test

v2 payload를 v3로 자동 해석하거나 SHA만 바꿔 승격하는 경로는 없다.
`legacy_v2_manifest_sha256`는 반드시 `None`, `legacy_v2_automatic_promotion`과 각 candidate의
`legacy_v2_promoted`는 반드시 `false`다.

## 3. population과 batch 불변식

### 3.1 Candidate/item

각 candidate는 immutable native source, decode 결과, P-applied ERR를 각각 상대경로,
byte 수, SHA-256으로 결속한다. auditor는 로컬 실제 파일에 대해 다음을 다시 수행한다.

- symlink와 repository root 탈출 거부
- native/decoded/P-applied 파일 크기와 SHA 재검산
- decoded PCM을 little-endian mono float32로 다시 읽기
- manifest에 결속된 causal P FIR와 delay로 ERR 전체를 재계산
- 저장된 ERR와 재계산 ERR의 byte-exact 일치 확인
- item 범위가 causal valid prefix 뒤인지 확인
- physical 8구간과 objective 7 octave density 및 valid mask 재계산
- 자격을 주장한 각 band의 상단까지 native Nyquist가 실제로 닿는지 확인

adaptive EQ, band shaping, 반복 또는 loop로 자연 clip을 broadband처럼 만드는 것은
허용하지 않는다. 한 item은 적어도 한 physical band와 한 objective octave에 실제 자격이
있어야 하지만 모든 band에 자격이 있을 필요는 없다.

### 3.2 Population

각 `train/val/test × speech/music/environment/machine × physical 8구간`에 독립 lineage
component가 최소 4개 필요하다. 동일 조건을 objective 7 octave에도 별도로 적용한다.

동일 native source SHA 또는 decoded PCM SHA를 서로 다른 lineage component로 쪼갤 수 없다.
같은 component를 split 또는 family 경계 밖에 재사용할 수도 없다. 따라서 파일 복제나 이름
변경으로 독립 source 개수를 부풀리는 것은 실패한다.

### 3.3 Batch

structural batch planner는 train 또는 val pool에서 다음을 동시에 만족할 때만 계획을 만든다.

- 네 family item 수가 정확히 동일함
- physical 8구간 각각 valid item `>= 4`
- objective 7 octave 각각 valid item `>= 4`
- 동일 item 중복 없음

이 planner의 PASS는 set-cover 가능성을 보이는 구조 증거일 뿐이며 canonical 학습 허가가
아니다. 반환 schema도 `canonical_training_status=BLOCKED`, `authority=None`으로 고정한다.

## 4. 임계 완화가 아닌 이유: 반례

다음 자연 population을 생각한다.

- 각 family에 저역 중심 clip 4개: 88.388--1600 Hz 일부/전부에만 자격
- 각 family에 고역 중심 clip 4개: 1600--11313.708 Hz 일부/전부에만 자격
- 어느 clip도 physical 8구간 또는 objective 7 octave를 모두 통과하지 않음

all-seven-per-clip 규칙은 이 population의 모든 clip을 탈락시킨다. v3는 density 하한을
`0.25` 그대로 둔 채, 저역은 네 독립 저역 component가, 고역은 네 독립 고역 component가
증명하게 한다. 이어 한 batch 안에서도 모든 physical/objective band에 실제 valid item을
최소 4개 넣는다. 즉 개별 자연 clip의 spectral sparsity를 보존하면서 population과 batch의
전체 목표 대역 coverage는 더 직접적으로 강제한다.

이 반례는
[`tests/test_broadband_population_contract_v3.py`](../tests/test_broadband_population_contract_v3.py)의
actual-byte/P-recompute fixture로 고정했다. threshold를 낮추거나 EQ로 clip을 변형한 fixture가
아니다.

## 5. Untouched Level-5 challenge

Level-5 challenge source ID는 population에 하나도 들어갈 수 없다. 학습, validation, model
selection 사용도 금지한다. 모델과 experiment contract를 고정한 뒤 speech/music/
environment/machine의 새 자연음을 별도 reservation receipt에 따라 수집·평가해야 한다.

population coverage PASS는 unseen 자연음 일반화나 실제 덕트 ANC 성능 PASS가 아니다.

## 6. 현재 authority 판정

현재 모듈 상수 `POPULATION_V3_AUTHORITY`는 의도적으로 `None`이다. 따라서 현재 판정은
**BLOCKED**다.

현재 없는 authoritative evidence는 다음과 같다.

1. 실제 native/decoded/P-applied bytes를 모두 결속한 canonical v3 manifest
2. 88.388--11313.708 Hz를 덮는 실제 물리 측정 causal P authority
3. 모델/계약 고정 뒤 사용할 untouched Level-5 challenge reservation authority

테스트용 synthetic bytes와 identity FIR가 structural PASS를 만들 수는 있지만
canonical authority, 물리 성능, 학습 허가를 만들 수는 없다.

## 7. 미연결 consumer와 다음 단계

이번 변경은 기존 v2와 현행 trainer/sampler를 수정하지 않는 독립 기반이다. 따라서 다음은
아직 연결되지 않았고 fail-closed 상태다.

1. actual source decoder/publisher가 native probe와 decode transform receipt를 발급
2. physical fullband causal P issuer가 operator receipt와 FIR bytes를 발급
3. actual v3 manifest publisher가 P-applied ERR와 item density를 봉인
4. v3 audit의 external authority issuer가 실제 artifact와 exact commit을 검증
5. trainer가 authority 있는 v3 audit과 deterministic batch receipt만 소비
6. 학습/선택 고정 뒤 Level-5 challenge를 별도 one-shot 평가

이 연결 전에는 기존 v2 결과를 v3 준비 완료로 해석하거나, structural fixture PASS를 근거로
학습을 시작해서는 안 된다.
