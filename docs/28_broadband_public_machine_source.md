# 광대역 공개 machine source 후보

> 상태: 공식 metadata 검증 완료, actual audio·spectral receipt 전 `BLOCKED`
> 기준일: 2026-08-28

## 1. 기존 MIMII를 그대로 쓸 수 없는 이유

현재 계획의 MIMII 계열 원본은 16 kHz다. native Nyquist가 8 kHz이므로 중심 8 kHz
옥타브의 상단 11.314 kHz를 포함하지 못한다. 48 kHz로 업샘플해도 원래 없던 대역을
만들 수 없으므로 마지막 machine cell의 source 증거로 세지 않는다.

## 2. 선택 후보: BSD35k-CS의 `fx-m`

공식 후보는 Universitat Pompeu Fabra Music Technology Group의
`BSD35k-CS`, Zenodo DOI `10.5281/zenodo.19187100`이다.

- 공식 record: `https://zenodo.org/records/19187100`
- audio는 mono, 16-bit PCM, 44.1 kHz WAV다.
- `fx-m`은 mechanisms/engines/machines 분류이며 1,542개, 약 4.52시간이다.
- 각 row에 Freesound `sound_id`, `uploader`, 원 clip license, title/tags/description이
  있어 계보 component를 만들 수 있다.
- 전체 record는 CC BY 4.0이고 원 clip별 license가 별도다. canonical source는
  재배포·상업성 불확실성을 줄이기 위해 **CC0 clip만** 허용한다.

2026-08-28에 official Zenodo API와 metadata bytes를 읽기 전용으로 검산한 결과는 다음과
같다.

| 항목 | 실제 값 |
|---|---:|
| metadata.zip size | 4,374,871 bytes |
| metadata.zip Zenodo MD5 | `9876254ce2ed845691a9a76efe13fe5a` |
| 다운로드 bytes SHA-256 | `b595129c00e65f098bee06aaf442ed454cb30430fadd28461bcdd6628b235a51` |
| 전체 metadata row | 33,829 |
| `fx-m` row | 1,542 |
| `fx-m` CC0 row | 1,323 |
| `fx-m` CC0 unique uploader | 188 |
| audio.zip size | 35,091,942,026 bytes |
| audio.zip Zenodo MD5 | `d47968c99ad4e93a081f380b2d273acd` |

이 숫자는 source spectral PASS가 아니다. crowd-sourced label이므로 title/tag만 믿지 않고
actual WAV를 모두 decode해 finite/nonzero/channel/sample-rate/PCM SHA와 7개 정확한
옥타브의 native density를 다시 검사해야 한다.

## 3. 계보와 split 규칙

`sound_id`가 달라도 같은 uploader의 녹음은 같은 원천 장비·장소·세션일 가능성이 있다.
따라서 최소 계보 component는 `uploader`이며 다음을 강제한다.

1. uploader는 train/val/test 중 정확히 하나에만 속한다.
2. 같은 `sound_id`, decoded PCM SHA, perceptual duplicate는 split을 넘지 않는다.
3. 최종 gate의 법적 하한은 split마다 4개지만, BSD35k selection은 cluster-bootstrap
   안정성을 위해 val/test 각각 최소 **16개** 독립 uploader를 선제 예약한다.
4. 3개 이상의 서로 다른 uploader/clip을 합친 source가 필요하면 경계와 gain을
   manifest에 고정하며, 한 component를 두 source slot에서 재사용하지 않는다.
5. title/tag의 machine 종류를 보조 metadata로 보존하되 분류문구만으로 spectral
   자격을 발급하지 않는다.

## 4. Elice staging

Jetson에는 35 GB archive를 받지 않는다. 새 A100 인스턴스에서 다음 순서로 staging한다.

1. `metadata.zip`과 `README.md`를 받고 official size/MD5, 로컬 SHA-256을 봉인한다.
2. CC0 `fx-m`만 결정적으로 선택하고 uploader-disjoint split plan을 먼저 발행한다.
3. `audio.zip`을 받은 뒤 35,091,942,026 bytes와 MD5를 검증한다.
4. 선택된 `audio/<sound_id>.wav`만 풀어 per-WAV SHA를 계산한다.
5. 선택 WAV와 manifest가 모두 검증된 뒤에만 재다운로드 가능한 35 GB archive를
   삭제한다. 선택 raw WAV는 유지한다.
6. 44.1→48 kHz 변환은 검증된 polyphase FIR 정확히 1회만 허용하고, resampler 응답
   SHA와 11.314 kHz passband를 receipt에 남긴다.
7. actual Q15 source 9-crop과 v3 causal P-applied ERR의 옥타브별 8/9 gate를 통과한
   clip만 machine family source로 승격한다.

metadata-only selection 구현은 다음과 같다.

```bash
.venv/bin/python scripts/data/issue_bsd35k_machine_selection.py \
  --metadata-csv /path/to/metadata/BSD35k-CS_metadata.csv \
  --output /path/to/no-replace/bsd35k_machine_selection_v1.json
```

공식 CSV bytes에서 재계산한 selection은 train/val/test
`1017/159/147` clip, 독립 uploader `156/16/16`이다. selection plan SHA-256은
`d650728bf90cc8e90126f64198a505358eb940c739f5a13b58d868f753b83c19`다. 이 SHA는
actual audio가 없는 metadata/lineage 계획일 뿐이며 `canonical_source_eligible=false`다.

archive 전체와 선택 WAV를 동시에 장기간 보관하지 않으면 128 GiB 인스턴스에서도
staging 가능하다. 다만 public corpus 전체의 peak disk 사용량을 bootstrap이 사전에
계산해 20 GiB 안전 여유를 남기지 못하면 다운로드를 시작하지 않는다.

## 5. 현재 판정

```text
공식 metadata/라이선스/표본률 후보: PASS
actual WAV archive checksum: NOT STARTED
uploader-disjoint split plan: metadata에서 결정적 재생성 PASS (actual audio 전 BLOCKED)
actual native spectral coverage: NOT STARTED
v3 causal P-applied ERR coverage: BLOCKED (v3 P 없음)
canonical machine manifest: BLOCKED
```
