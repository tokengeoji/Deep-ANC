# 광대역 recorded-v2 소스 inventory와 acquisition 계약

> 감사 기준일: 2026-08-28. 이 문서는 소스 확보 상태만 다룬다. 오디오 출력, Drive 파일
> 다운로드/변경, 기존 recorded 데이터 수정은 수행하지 않았다.

## 1. 결론

현재 canonical 광대역 후보는 **0/48**이며 판정은 **BLOCKED**다. 요구량은 train/val/test의
speech/music/environment/machine마다 서로 독립인 원본 4개, 즉 `3×4×4=48`개다. 어느 한
cell에도 현재 계약을 전부 증명한 후보가 없으므로 부족분은 다음과 같다.

| split | speech | music | environment | machine | 부족 합계 |
|---|---:|---:|---:|---:|---:|
| train | 0/4 | 0/4 | 0/4 | 0/4 | 16 |
| val | 0/4 | 0/4 | 0/4 | 0/4 | 16 |
| test | 0/4 | 0/4 | 0/4 | 0/4 | 16 |

실제 read-only 보고서는
`results/data_audit/broadband_source_inventory_20260828.json`이다. 파일 SHA-256은
`f572623d4330f35fdf8a5ba716fe1f7fa934d08575ab1f937d52bf4355800b03`, payload evidence
SHA-256은 `2fc2d507ac35112803c9f9a3edfc0380dcc905d2c3ece605bf82b72f4dea74f0`다.
보고서에는 개별 Drive path/hash를 노출하지 않고 전체 listing digest와 집계만 저장했다.

## 2. 실제 inventory 근거

### 2.1 Jetson 로컬

| cohort | 실제 header | 수량 | canonical 후보 | 차단 사유 |
|---|---|---:|---:|---|
| LibriSpeech FLAC | 모두 native 16 kHz, 1.445–32.645초 | 2,703 | 0 | native fs가 22,628 Hz 미만 |
| 로컬 ESC-50 WAV | 44.1 kHz, 각 5.0초 | 4 | 0 | 연속 untouched 15초 window가 없음 |
| `source_pool` WAV | 48 kHz, 각 70초 | 80 | 0 | native raw/transform/decoder/lineage/11.314 kHz 증거 없음 |
| `source_pool_v2` WAV | 48 kHz, 각 70초 | 80 | 0 | 위와 동일 |

두 `sources.csv`에는 `native_sample_rate_hz`, `native_content_sha256`,
`transform_receipt_sha256`가 없다. 따라서 48 kHz라는 가공 WAV header만 보고 native bandwidth를
주장하지 않는다.

16 kHz와 22.05 kHz는 모두 제외한다. 8 kHz octave 상단은
`8000×sqrt(2)=11,313.708... Hz`이고, 이를 native Nyquist가 포함하려면 정수 sample rate가
최소 `ceil(2×11,313.708...)=22,628 Hz`여야 한다. 업샘플링은 없는 native bin을 만들지 않는다.

### 2.2 Google Drive read-only 목록

`gdrive:DeepANC/jetson_data_backup_20260827/data`에 `rclone lsjson --recursive --files-only
--hash`만 실행했다. 외부 파일 본문을 읽거나 복사하거나 변경하지 않았다.

| cohort | Drive metadata 집계 | canonical 후보 |
|---|---:|---:|
| raw speech | 2,805 files / 360,289,905 bytes (`.flac` 2,703) | 0 |
| raw music | 8,003 files / 8,235,886,703 bytes (`.mp3` 8,000) | 0 |
| raw noise | 2,011 files / 884,047,129 bytes (`.wav` 2,000) | 0 |
| source pool v1 | 82 files / 1,075,266,633 bytes (`.wav` 80) | 0 |
| source pool v2 | 82 files / 1,075,245,766 bytes (`.wav` 80) | 0 |

Drive object metadata는 native audio header, 연속 15초, decoded PCM, lineage DSU, 실제
11.314 kHz PSD를 증명하지 못한다. MP3 8,000개는 현재 lossless-native 정책에서 탈락한다.
FLAC/WAV 확장자는 무손실 가능성을 나타낼 뿐 위 나머지 증거를 대신하지 않는다. Drive 수량을
그대로 source 후보 수로 세지 않는 이유다.

ESC-50의 5초 원본 세 개를 이어 15초로 만들거나 한 파일을 반복하는 것도 금지한다. 필요한
population은 0.25초부터 시작하는 1.5초 구간 9개 중 같은 일곱 대역을 최소 8개 구간에서
통과하는 **하나의 연속 untouched window**다. 반복은 같은 raw/lineage를 새 group으로
위장하고, concat은 원본 경계 transient와 새 가공 content를 만든다. 둘 다 독립 자연 source
coverage의 증거가 아니다.

## 3. 기존 pipeline이 실제 48-source plan을 만들 수 있는가

현재 답은 **아니다**.

- `audit_broadband_recording_campaign.py`와 `build_missing_source_campaign()`은 정확히 48개
  placeholder slot을 만든다.
- `record_broadband_v2.py`와 `validate_source_plan()`은 이미 존재하는 READY source plan을
  실제 local bytes까지 다시 검증할 수 있다.
- 그러나 기본 경로 `data/source_plans/recorded_broadband_v2/canonical_v1.json`은 존재하지
  않는다.
- verified acquisition manifest에서 READY source plan을 no-replace 발행하는 publisher도
  아직 없다.
- canonical fullband causal P가 없으므로 source→predicted ERR 고역 density 선택 자체도
  아직 열 수 없다.

즉 placeholder builder와 downstream validator 사이에 **실제 source acquisition evidence와
publisher가 비어 있다**. fixture로는 48개 완전한 evidence 행이 있으면 새 inventory 계약이
12개 cell 모두 4/4로 재집계됨을 확인했지만, fixture를 실제 후보나 live authority로
승격하지 않는다.

## 4. 최소 acquisition manifest v1

코드 단일 출처는
`src/deep_anc/data/broadband_source_inventory.py`의
`broadband_recorded_v2_source_acquisition_manifest_v1` validator다. 각 후보는 다음 증거를
별도로 가진다.

1. `origin_audio`
   - storage locator, byte size, immutable 원본 SHA-256
   - 실제 container/codec/subtype, lossless 여부
   - native sample rate/Nyquist/channels/frame count/duration과 header evidence SHA
   - 정확한 연속 untouched 15초 window의 start/frame 수
2. `decode_provenance`
   - decoder fingerprint SHA
   - decoder receipt SHA
   - decoded native PCM SHA, 실제 fs/channels/frame 수
3. `spectral_evidence`
   - decoded PCM SHA와 exact window 결속
   - control-band SHA와 canonical fullband P evidence SHA
   - 11,313.708 Hz까지 실제 분석했다는 증거
   - 9 segments×7 bands의 source density와 predicted-ERR density
4. `lineage_evidence`
   - family별 connected-component key와 DSU authority SHA
   - component ID와 실제 family value receipt SHA
5. `corpus_disjointness`
   - 기존 recorded와 모든 synthetic split에 대한 raw/processed/lineage 교집합 0 receipt
6. 기존 `candidate_metadata`
   - raw/processed/transform SHA, 1회 polyphase FIR 여부, crest, 9×7 density를 기존
     campaign validator와 byte-for-byte 동일하게 대조

`lossless`와 compressed provenance를 같은 boolean으로 뭉개지 않는다. 향후 lossy 정책을
검토하려면 immutable compressed 원본 SHA, decoder fingerprint, decoded PCM SHA를 모두
보존해야 한다. 하지만 세 값이 존재해도 현재 `lossless=false`는 계속 BLOCKED다. 결과를 보고
임계를 낮추거나 MP3를 임의 승인하지 않는다.

## 5. 다음 확보 단위

새 원본은 family마다 12개씩 확보하되 처음부터 split을 임의로 쪼개지 않는다. 먼저 원본의
speaker/book, artist/album, event/session, physical-machine/run 관계를 connected component로
묶고 component 단위로만 train/val/test에 각각 4개를 배정한다.

한 후보의 최소 물리 조건은 다음과 같다.

- native fs ≥22,628 Hz, mono lossless PCM/FLAC, 연속 untouched 길이 ≥15초
- 원본/decoded PCM/processed/transform/header/PSD/lineage SHA 전부 존재
- dynamic compression, clipping, spectral EQ, repeat, concat 금지
- 48 kHz가 아니면 검증된 passband가 11.314 kHz 이상인 polyphase FIR 정확히 1회
- canonical fullband P를 통과한 predicted-ERR 9×7 density와 crest 사전검사 PASS
- 기존 recorded 및 synthetic 전체와 content/lineage 교집합 0

이 48개와 canonical fullband causal P가 모두 생기기 전에는 READY source plan 발행, 스피커
녹음, Elice canonical 학습을 열지 않는다.
