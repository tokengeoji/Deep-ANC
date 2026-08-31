# Stage-1 cap-aware source conditioning 무음 감사

작성일: 2026-08-31
대상: gain-probe v5가 확정한 출력 cap `5788 millionths`
역할: `coverage_training_stimulus_not_natural_unprocessed_evaluation_evidence`

## [가설]

기존 19행 selector가 고른 자연 source를 cap `0.005788`로 그대로 렌더링하면
strict P 예측의 600--1600 Hz SNR이 부족할 수 있다. 원본 bytes와 계보를 바꾸지 않은 채
활성 100 ms frame을 시간 순서대로 압축하고 작은 leveler를 적용하거나, 해당 원본의
대역 에너지가 양자화 수준이면 독립 계보 source로 교체하면 기존 임계값을 완화하지 않고
19행을 coverage-training 자극으로 만들 수 있다고 가정했다.

## [근거]

- source-plan anchor:
  `data/source_plans/recorded_additions/stage1-coverage-v2.csv`, SHA-256
  `f1b3d63fa1e455bac723a7a323aede0b602486c955bcb945281dc129cc7bc574`
- strict P anchor:
  `assets/measured/primary_path_il_strict_5dc06fdd.npz`, SHA-256
  `23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598`
- candidate override anchor:
  `configs/stage1_cap_aware_local_candidate_overrides.json`, SHA-256
  `abfd63729d661683a764574e593e00210cd17a71a0d85a65cf59f8920c2cb951`
- source lineage authority evidence SHA-256:
  `744a35adabeb34711daa1b3d2ca422111ce3ef1791ea0bb067130fe67598f649`
- 기존 DNS speech 2/5와 DEMAND 1개는 bounded conditioning만으로 부족하여
  active 82와 겹치지 않는 LibriSpeech component 2개와 source-pool environment
  component 1개로 교체했다.

이 문서가 사용하는 v2 CSV는 행 수와 family/split slot을 고정하는 진단 skeleton이다.
candidate override의 `canonical_live_eligible`은 명시적으로 `false`다. 따라서 아래 PASS를
현행 canonical 녹음 실행 권한 또는 natural/unseen 평가 증거로 해석하면 안 된다.

## [확인 방법]

새 모듈 `src/deep_anc/data/recording_source_conditioning.py`는 다음 순서로 동작한다.

1. 원본 파일 전체 peak와 시작점을 고정하고 48 kHz, 정확히 720,000 frame을 만든다.
2. 원본 peak-normalized 4대역(150--300/300--600/600--1000/1000--1600 Hz)이
   각각 PCM16 16 LSB RMS보다 큰지 먼저 검사한다. 이 문턱 아래 source는 EQ로
   증폭하지 않고 `BLOCKED_REPLACEMENT_REQUIRED`로 끝낸다.
3. identity를 먼저 검사하고, 필요할 때만 활성 100 ms frame compaction과 최대 drive 6의
   deterministic tanh leveler, 최대 6 dB의 513-tap 4-band EQ 후보를 검사한다.
4. PCM16 파생 WAV를 실제 `NoiseProgram` file renderer로 cap `5788`에 다시 렌더링한다.
5. 새 `recording_source_cap_aware_preflight/v1`은 commanded amplitude를
   `5788 millionths = 0.005788`로 exact 검증한다. legacy v1의 고정 `0.06`
   `playback_amplitude` 필드는 receipt에 직렬화하지 않는다.
6. 기존 preflight의 timeline/trusted-band metric과 strict P의 150--600/600--1600 Hz SNR
   `>= 9.54242509439325 dB`, predicted peak/RMS `<= 0.5`를 그대로 적용한다.
7. recipe/version, 원본 경로·SHA, parameter, 파생 WAV SHA와 exact audit를 receipt SHA로
   봉인한다. `--write`는 no-replace이며 기본값은 파일을 쓰지 않는 check-only다.

무음 dry-run:

```bash
.venv/bin/python scripts/data/build_cap_aware_recording_sources.py \
  --source-plan data/source_plans/recorded_additions/stage1-coverage-v2.csv \
  --source-plan-sha256 f1b3d63fa1e455bac723a7a323aede0b602486c955bcb945281dc129cc7bc574 \
  --strict-primary assets/measured/primary_path_il_strict_5dc06fdd.npz \
  --strict-primary-sha256 23fa43f1ec46d5bca6bdad53938b81bb2d2c85afc4eee35e83c555b6c4f0c598 \
  --amplitude-millionths 5788 \
  --candidate-overrides configs/stage1_cap_aware_local_candidate_overrides.json \
  --candidate-overrides-sha256 abfd63729d661683a764574e593e00210cd17a71a0d85a65cf59f8920c2cb951 \
  --out-root data/recording_source_conditioning/stage1-cap5788-candidate \
  --campaign-out results/recording_source_conditioning/stage1-cap5788-candidate.json
```

## [결과]

최종 check-only campaign SHA-256은
`6369167521cc6bf31753098e935b94660883d0e42bcfca441728e52e6132f6e7`이다.
모든 숫자는 cap `5788` exact renderer 뒤 strict P full convolution 결과다. `C`는 active-frame
compaction, `D`는 leveler drive다. 선택된 19행 모두 4-band EQ gain은 `[0,0,0,0] dB`였다.

| CSV 행 | Family / split | C | D | Timeline | SNR 150--600 | SNR 600--1600 | P peak | P RMS | Preflight / gates |
|---:|---|:---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | environment / train | N | 0.0 | 1.0000 | 17.5560 | 12.4923 | 0.03954 | 0.00415 | PASS |
| 3 | environment / val | Y | 1.0 | 0.9873 | 19.1084 | 9.8704 | 0.04905 | 0.00539 | PASS |
| 4 | environment / test | Y | 1.0 | 1.0000 | 15.8309 | 9.9737 | 0.02121 | 0.00298 | PASS |
| 5 | environment / test | N | 0.0 | 1.0000 | 13.5072 | 11.8970 | 0.03211 | 0.00485 | PASS |
| 6 | music / test | N | 0.0 | 0.9831 | 13.6423 | 12.8624 | 0.01875 | 0.00411 | PASS |
| 7 | music / test | N | 0.0 | 1.0000 | 18.5814 | 11.9409 | 0.02431 | 0.00555 | PASS |
| 8 | music / val | Y | 1.0 | 1.0000 | 19.5208 | 10.2210 | 0.03440 | 0.00644 | PASS |
| 9 | music / val | Y | 1.0 | 1.0000 | 18.4213 | 10.2068 | 0.03577 | 0.00672 | PASS |
| 10 | music / train | Y | 1.5 | 1.0000 | 20.8628 | 9.8684 | 0.03114 | 0.00744 | PASS |
| 11 | environment / test | Y | 0.0 | 0.9746 | 10.3703 | 11.0910 | 0.04199 | 0.00295 | PASS |
| 12 | speech / train | Y | 4.0 | 1.0000 | 13.7598 | 9.8799 | 0.01943 | 0.00359 | PASS |
| 13 | speech / train | Y | 2.5 | 1.0000 | 20.3695 | 10.2456 | 0.02217 | 0.00687 | PASS |
| 14 | speech / val | Y | 6.0 | 1.0000 | 21.7990 | 10.9527 | 0.03756 | 0.00700 | PASS |
| 15 | speech / test | Y | 2.5 | 1.0000 | 13.8410 | 9.6771 | 0.02085 | 0.00367 | PASS |
| 16 | speech / test | Y | 4.0 | 1.0000 | 23.2351 | 10.4166 | 0.03878 | 0.00877 | PASS |
| 17 | machine / train | Y | 2.0 | 1.0000 | 12.3863 | 10.2291 | 0.01910 | 0.00333 | PASS |
| 18 | machine / val | Y | 2.0 | 1.0000 | 15.5376 | 10.7885 | 0.01849 | 0.00435 | PASS |
| 19 | machine / test | Y | 1.5 | 1.0000 | 12.8309 | 9.7364 | 0.01443 | 0.00336 | PASS |
| 20 | machine / test | Y | 2.0 | 1.0000 | 13.2141 | 9.8443 | 0.01589 | 0.00350 | PASS |

최저 high-band SNR은 `9.6771 dB`, 최대 predicted peak는 `0.04905`, 최대 predicted RMS는
`0.00877`이다. 즉 이번 check-only에서 blocker 행은 0개이고 threshold 완화도 0개다.

Focused 회귀 검증:

```text
.venv/bin/python -m pytest -q tests/test_recording_source_conditioning.py
........ [100%]
```

## [판정]

**Confirmed (coverage-training candidate feasibility만).** cap `5788`에서 19/19 exact rendered
source가 현재 software preflight와 strict-P 2-band SNR/peak/RMS gate를 통과한다.

**Inconclusive (physical recording 및 ANC 성능).** 이 결과는 스피커나 마이크를 열지 않은
예측 감사다. 실제 ADC noise, clock, latency, coherence 또는 ANC 감쇠를 증명하지 않는다.
파생 source는 natural/unprocessed evidence가 아니며 final unseen 평가에 사용할 수 없다.

## [다음 행동]

1. 현행 canonical generation source plan에 이 candidate의 원본 선택과 conditioned output
   receipt를 별도 source kind로 명시적으로 결속한다. stale v2 SHA를 live authority로
   재사용하지 않는다.
2. 그 통합 plan/consumer의 focused test와 전체 pytest를 통과시키고 exact commit을 만든다.
3. 그 후에만 `--write`로 no-replace 파생 WAV/receipt/campaign을 발행하고, 무음 dry-run에서
   실제 녹음 명령의 19행 path/SHA/start=0 mapping이 일치하는지 검사한다.
4. 물리 녹음은 별도 안전 절차와 사용자 승인 아래 진행하고, 이 19개를 final natural/unseen
   평가 소스로 재사용하지 않는다.
