# Stage-2 2 kHz 최소 3 dB 계약

> 상태: 사용자 확정 목표, 구현·실측·학습 전 `BLOCKED`
> 기준일: 2026-08-31

## 1. 역할

Stage-1의 150--1600 Hz 준비와 최종 full-octave v3의 125 Hz--8 kHz 역할 사이에
실행 가능한 Stage-2를 둔다. 이 단계의 목적은 저역을 버리고 2 kHz만 맞추는 것이
아니라, **125/250/500/1000/2000 Hz 중심 옥타브를 모두 보존하면서 실제 덕트의
2 kHz 옥타브에서 최소 3 dB 감쇠를 입증하는 것**이다.

4/8 kHz는 이 단계에서 양의 감쇠를 성공 조건으로 주장하지 않는다. 그러나 ANC ON이
그 대역에 유해한 제어 에너지를 넣은 legacy 결함은 반복할 수 없으므로 기존 G4의
do-no-harm 한도는 유지한다. 이 Stage-2 PASS를 4/8 kHz quiet-zone PASS로 승격하지
않는다.

별도 계약 ID는 `broadband_2khz_octave_88_2828_v1`로 한다. 기존
`broadband_full_octave_88_11314_v3`의 payload나 SHA를 수정·축소해 재사용하지 않는다.

## 2. 대역 정의

`2 kHz tone`과 `2 kHz 중심 octave`를 구분한다.

```text
2 kHz tone 진단:        2000 Hz 부근의 좁은 대역
2 kHz octave 본 판정:  [2000/sqrt(2), 2000*sqrt(2)]
                       = [1414.213562, 2828.427125] Hz
```

따라서 단일 2 kHz tone에서 3 dB가 나와도 speech/music/environment/machine의
2 kHz octave PASS가 아니다. P/S 식별 자극과 source coverage는 최소 2,828.427125 Hz까지
유효해야 한다. 125 Hz octave를 주장하려면 하단도 88.388348 Hz까지 실제로 덮어야 한다.

물리 식별 구간은 다음 여섯 구간을 별도로 판정한다.

```text
88.388348--150
150--300
300--600
600--1000
1000--1600
1600--2828.427125 Hz
```

## 3. 성공 기준

다음은 **합격 하한**이며, 목표 상한이나 현행 checkpoint의 성능 수치가 아니다.
3 dB에서 최적화를 멈추지 않고 검증된 감쇠가 클수록 좋다. 다만 한 대역의 큰 평균으로
다른 대역·family의 실패를 상쇄하지 않는다.

- 2 kHz octave의 각 family 평균·최악 10% 평균·독립 component cluster-bootstrap
  95% CI 하단 감쇠가 모두 `>= 3.0 dB`
- 이 2 kHz 본 판정 대역은 `[1414.214, 2828.427) Hz`이므로 1.6 kHz를 별도의
  near-zero 허용 구간으로 빼지 않는다. 1.6 kHz 부근 source-density를 실제로 가진
  segment와 독립 component가 위 평균·최악 10%·CI 하단 gate에 함께 들어가야 한다.
- 125/250/500/1000 Hz 각 octave의 family 평균·최악 10%·cluster-bootstrap CI 하단
  감쇠가 모두 `> 0 dB`; 합격 뒤에도 가능한 감쇠를 최대화
- 2 kHz 좁은 tone, 1.6--2.2 kHz band, 2 kHz full-octave natural source를 서로
  다른 진단으로 보존
- 4/8 kHz out-of-target octave 최악 10% 증폭 `< 1 dB`
- 48 kHz/256에서 deadline miss, xrun, fallback, ring drop/add, sample slip,
  허용 one-hop을 넘는 backlog 모두 `0`
- single-point 결과와 spatial quiet-zone 결과를 구분. 현 2입력 장비의 single-point
  PASS를 다점 quiet-zone PASS로 부르지 않음

checkpoint 선택은 모든 필수 셀의 gate 여유 중 최솟값을 먼저 최대화한다. 그 값이
같거나 측정 오차 안이면 2 kHz octave 평균 감쇠가 더 큰 모델을 고른다. 따라서
2 kHz만 과도하게 줄이면서 125 Hz--1 kHz, unseen family 또는 4/8 kHz DNH를 해치는
모델은 선택되지 않는다.

감쇠가 3 dB에 조금 못 미치는 것은 latency 결함을 허용하는 이유가 아니다. 성능과
runtime 안전은 독립 게이트다.

## 4. 데이터 계약

공개 데이터의 양을 늘리되 다음 조건을 동시에 지킨다.

1. speech/music/environment/machine 네 family를 균형 샘플링한다.
2. 2 kHz octave 상단 2,828.427125 Hz를 native Nyquist가 덮어야 한다. 16 kHz
   원본은 이 Stage-2에는 충분하지만 8 kHz full octave 증거로 승격하지 않는다.
3. same WAV/original clip, speaker/book, artist/album, machine/session connected
   component가 train/val/test를 가로지르지 않는다.
4. 사전학습은 새 strict Stage-2 P/S를 적용한 public/synthetic 자료를 사용한다.
5. 파인튜닝은 measured 70% + lineage-clean public/synthetic 30%를 기본값으로 한다.
6. test는 모델·손실·checkpoint 선택 뒤 정확히 한 번만 열고, 학습에 없던 독립
   component와 새 natural sound를 포함한다.
7. public frequency coverage receipt v2는 단순 count를 받지 않는다. split×family×
   125/250/500/1000/2000 Hz octave와 별도 1.6 kHz sentinel마다 qualified
   `dataset_index`, `component_id`, source 상대경로, source bytes SHA-256을 전부
   열거한다. admission은 manifest에 이를 역매핑하고 동일 source bytes를 다시 decode해
   Welch density를 재계산한다. 한 component의 여러 파일을 독립 component로 세거나,
   receipt의 count/identity만 다시 봉인해 하한 4를 만드는 것은 허용하지 않는다.

파일 수만 늘리고 2 kHz target density, plant provenance 또는 family balance가 없는
자료는 학습 admission을 열지 않는다.

## 5. checkpoint 방침

`runs/pretrain_tiny_corrected/ckpt/best.pt`와 그 export는 진단용으로만 보존한다.
해당 checkpoint는 실제 P가 아닌 `P=S` surrogate, lead 109, trusted 150--600 Hz,
고역 do-no-harm 손실 부재에 결속되어 있다. 현재 strict lead 115 및 새 2 kHz plant와
계약이 다르므로 official init/resume에 사용하지 않는다.

공식 순서는 다음과 같다.

```text
새 Stage-2 strict P/S와 source admission
-> scratch 20k surrogate + 5k measured probe
-> 선택 계약으로 scratch 100k pretrain
-> G0와 deterministic resume 검증
-> measured 70% + synthetic 30% 50k fine-tune
-> recorded val 선택
-> independent test one-shot
-> Jetson OFF/DL 및 matched FxLMS 현장 평가
```

legacy warm-start는 필요하면 별도 ablation으로만 비교한다. 그 결과가 좋아도 contract
SHA가 다른 legacy optimizer/RNG 상태를 resume하거나 canonical init으로 이름을 바꾸지
않는다.

## 6. 현재 판정

- Stage-1 strict P/S authority: 150--1600 Hz만 존재
- Stage-2 88.388--2828.427 Hz strict P/S: 없음
- Stage-2 source/recorded coverage: 미발행
- Stage-2 scratch checkpoint: 없음
- 실제 2 kHz octave 3 dB raw: 없음

125 Hz octave 하단 88.388--150 Hz는 코드만의 문제가 아니다. 기존 진단에서는 저역 S
일관성이 약했고 72 Hz 응답도 작았으므로, 현 cancel speaker로 해당 구간을 authoritative
하게 식별할 수 있는지는 새 측정에서 fail-closed로 확인해야 한다. 실패하면 임계값을
낮추거나 150 Hz 결과를 125 Hz PASS로 바꾸지 않고 물리 blocker로 기록한다.

기존 82세션의 1600--2828 Hz 진단상 joint-valid 독립 group은 전체 3개뿐이며,
environment train 2개·val 1개 외의 모든 family×split은 0개다. 따라서 기존 82세션은
저·중역 자료로 보존하되 Stage-2 실측 모집단으로 자동 승격하지 않는다. exact 재감사 뒤
family×split별 최소 4개에서 부족한 group만 짧게 추가 녹음한다.

따라서 현재 판정은 `BLOCKED`다. 다음 물리 작업은 소리를 내기 전에 현행 capture의
clock/reference 문제를 해결한 분석기, 무음 dry-run, exact 자극 시간, raw no-replace
경로를 먼저 확정한 뒤 한 번의 짧은 P/S 창으로 실행한다.

새 P/S 측정은 NS/CS 독립 aperiodic code와 capture 전체의 연속 REF clock witness를
사용하고 actual submitted int16 PCM을 분석 분모로 삼는다. fit-a/fit-b/untouched holdout을
분리하며, signal builder의 출력 상한은 25초, fresh meter 20초를 합친 audible budget은
45초 이하로 설계한다. 여섯 물리 구간 consistency `>=0.95`, 2,828.427 Hz에서 20 dB-grade
timing residual `<=0.270208 sample`, xrun/clip/status/slip 0을 낮추지 않는다.
