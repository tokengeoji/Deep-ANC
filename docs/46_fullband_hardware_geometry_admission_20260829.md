# Full-octave 하드웨어·덕트 geometry admission 감사 (2026-08-29)

이 감사는 Jetson의 실제 ALSA inventory, 현재 저장된 raw/config, 덕트 문서를 읽기만
수행했다. PCM을 열거나 소리를 출력하거나 외부 참고 저장소를 수정하지 않았다.

## [가설]

현재 연결 가능한 Jetson 장비만으로 ERR·REF·실제 noise/cancel 출력 전압을 같은
시간축에서 기록하여 125 Hz–8 kHz canonical P/S를 만들 수 있다.

## [근거]

- 실제 `/proc/asound/cards`에는 HDA, `APE`, `Audio`(AB13X USB)만 있다.
- AB13X `stream0` descriptor는 48 kHz/S16_LE/2채널 playback이지만 capture는
  48 kHz/S16_LE/**1채널**이다.
- APE PCM1은 ERR/REF 2채널 후보이고 PCM0은 RT5640/J511 output 후보이지만,
  J511 read-only state는 현재 `None` 세 번이며 PCM0 input-only probe는 ch0 rail clip,
  ch1 stuck이었다.
- USB AB13X와 APE는 별도 시간축이다. legacy raw의 non-affine clock 진단이 이미
  이 조합의 shared-clock authority를 부정했다.
- 현재 strict P/S는 AB13X USB/S16, 150–1600 Hz Stage-1 artifact다. RT5640/J511
  S32 fullband plant가 아니다.

## [확인 방법]

다음 후보가 `ERR + REF + actual output tap` 동시성, sample clock, raw PCM health를
모두 만족하는지 read-only inventory와 저장된 failure raw로 대조했다.

| 조합 | ERR/REF | 실제 출력 witness | 동기성 | 판정 |
|---|---:|---:|---:|---|
| APE PCM1 + AB13X | 2채널 후보 | AB13X mono capture 1채널 | APE↔USB 별도 | 불충분 |
| AB13X 단독 | 없음 | mono 1채널 | 내부만 공유 | 불가 |
| APE PCM1 + RT5640/J511 | 2채널 후보 | PCM0 input health 실패 | 미입증 | 불가 |

## [결과]

현재 구성에는 fullband P/S authority에 필요한 동시 electrical witness가 없다.
스피커를 물리적으로 연결한 사실은 실험을 무효화하지 않지만, J511 plug state조차
반대편 앰프/전원/전압/음향 출력을 증명하지 않는다. 따라서 새 S32 meter recipe가
무음 static PASS하더라도 physical P/S·학습은 계속 `BLOCKED`다.

## [판정]

**Contradicted — 현재 장비 조합만으로 canonical fullband P/S를 만들 수 없다.**

## [다음 행동]

P/S용으로는 다음 중 하나가 필요하다.

1. 권장: 같은 sample clock의 4-input 이상 ADC로 `ERR`, `REF`, noise-output safe tap,
   cancel-output safe tap을 동시에 녹음한다.
2. 디지털 INMP441를 유지한다면 APE I2S2와 output electrical ADC/tap 사이의 continuous
   hardware clock/frame witness를 별도로 만든다. 단순 USB 2채널 ADC는 충분하지 않다.

모든 tap은 앰프 전단의 고임피던스·절연·감쇠된 안전한 전압 tap이어야 하며, AGC 없는
동시 샘플링, 최소 48 kHz, raw PCM/gain/polarity/clip/SHA 보존을 요구한다. spatial
quiet-zone 최종 평가는 `REF + noise/cancel tap + ERR 5위치`를 동시에 보려면 8 input
규모가 적합하다.

---

## [가설]

덕트 단면 105×105 mm에서 계산한 1.633 kHz cutoff 때문에 2/4/8 kHz ANC를 포기해야 한다.

## [근거]

- `configs/duct.yaml`과 `docs/09_duct_structure.md`의 working geometry는 내경
  105×105 mm, 길이 1190 mm, closed→open, NS x=0/REF x=100/CS x=1050/ERR x=1100 mm다.
- 계산값 `343 / (2 × 0.105) ≈ 1633 Hz`는 첫 transverse mode onset이다.
- 그러나 ERR 최종 좌표, mic diaphragm acoustic point, flush/recess, NS seal, 실제
  내경/홀 지름은 아직 receipt로 확정되지 않았고 ERR x=1100은 CS mount 범위와 겹친다.

## [확인 방법]

plane-wave limit, higher-order mode onset, point attenuation, spatial quiet-zone claim을
서로 분리했다. v3 contract가 요구하는 8 physical subband와 7 octave 및 five-position
spatial verification도 대조했다.

## [결과]

1.633 kHz는 ANC 불가선이 아니다. 2/4/8 kHz point control은 연구 목표로 남는다.
다만 약 2/4/8 kHz에서 여러 propagating mode가 생기므로 단일 CS+ERR의 local null은
단면 전체 quiet zone을 증명하지 못한다. 8 kHz octave는 5.657–11.314 kHz 전체여서
8 kHz 단일 tone 성공으로 PASS할 수도 없다.

## [판정]

**Likely — point-control 가능성. Inconclusive — spatial quiet zone.**

## [다음 행동]

fullband P/S 전에 immutable geometry receipt를 만든다.

- diaphragm acoustic point 기준 좌표계와 실제 `a,b,L`
- NS/CS axis, speaker hole/duct seal, REF/ERR face·flush·recess·polarity
- ERR/CS overlap 해소 여부, 사진·치수·SHA

geometry는 P/S timing의 대체물이 아니라 physical cross-check다. final spatial 평가는
central + y/z ±2 mm를 포함한 최소 5 ERR 위치를 동시 기록한다. 이 작은 array의 PASS도
105 mm 단면 전체 quiet-zone claim과는 분리해 범위를 명시한다.

---

## [가설]

`/home/capston/DeepANC_CRN_n_codex/duct_cnn_anc`의 기존 raw/model이 2/4/8 kHz plant
빈칸을 메울 수 있다.

## [근거]

해당 참고 저장소는 read-only로만 조사했다. primary sweep은 100–1000 Hz, control band도
100–1000 Hz이며, fullband MLS secondary는 AB13X independent streams와 command proxy를
사용한다. geometry/positions도 null 또는 현 덕트와 다르고 status report는 repeatability
실패로 training S에 쓰지 말라고 명시한다.

## [결과]

유용한 원칙(실제 PCM/raw input/timestamp/xrun 보존)은 참고할 수 있지만 그 raw/model은
fullband causal plant, spatial quiet-zone, canonical training authority로 전용할 수 없다.

## [판정]

**Invalid experiment for canonical fullband reuse.**

## [다음 행동]

참고 저장소는 계속 읽기 전용으로 두고, 새 hardware topology와 현 덕트에서 새 raw-first
S32 P/S를 발행한다.
