# 2026-08-31 Stage-2 raw-first clock forensic

> 권위 범위: `dev` commit
> `8c217b1cc901c07589282f5f044d05e64fe7376c`에서 실제 Jetson AGX Orin,
> AB13X USB DAC output + APE ERR/REF input으로 생성한 보존 raw만 분석한다.
> 이 문서는 P/S, ANC 감쇠, 학습 적합성을 PASS로 승격하지 않는다.

## 1. fresh meter 경계 FAIL

## [가설]

첫 meter의 `-48.1 dBFS` FAIL은 위험한 물리 과출력이 아니라 경계값과
표시 정밀도가 맞지 않아 발생했을 가능성이 있다.

## [근거]

- raw: `results/stage2_2khz_ps_v2/level_rawfirst_20260831/`
  `20260831_231125_40c1c3e2/meter_raw.npz`
- SHA-256: `011e319f7a52e42bb5931279bc778054ff0fac8083ab6a654073e402e11647b5`
- 공식 상한: `-48.100000000000 dBFS`
- raw 재계산: `-48.099607455730 dBFS`
- 상한 초과: `0.000392544270 dB`
- 마지막 8개 0.25초 구간 표준편차: `0.02038 dB`
- submitted peak `98`, CS ch1 exact zero, xrun/drop/error 0, output close confirmed

## [확인 방법]

NPZ의 `input_raw_int32`로 official band meter를 다시 계산하고, metadata의
저장값·submitted PCM·telemetry와 byte/SHA 단위로 비교했다.

## [결과]

엄밀한 `±2 dB` 계약상 FAIL은 맞지만 초과량은 실측 변동보다 52배
작다. 임계를 완화하거나 FAIL raw를 수정하지 않고, 동일 설정의 새 raw를
한 번만 재캡처했다. 두 번째 raw는 `-48.115996965291 dBFS`로 PASS했다.

- PASS raw: `results/stage2_2khz_ps_v2/level_rawfirst_20260831/`
  `20260831_231358_b549da6a/meter_raw.npz`
- SHA-256: `71212de854e5229725f259cc4a3725c2bd6d2dfa15ac6970cd8c5bd4022ba14c`

## [판정]

**Confirmed — 첫 raw는 계약상 FAIL이지만 물리적으로 위험한 과출력이라고
식별할 수 없다.**

## [다음 행동]

임계를 바꾸지 않고 UI가 관측값·상/하한·가장 가까운 경계까지의 여유를
4자리로 표시하게 한다. exact boundary PASS와 `nextafter(max,+inf)` FAIL을
회귀 테스트한다.

## 2. Stage-2 diagnostic raw FAIL

## [가설]

diagnostic의 최대 `21.51 dB` gain error와 `28.97 sample` phase error는
스피커/앰프의 심각한 비선형보다 USB DAC↔APE ADC 비동기 clock을 고정 frame
index로 비교한 분석 오류일 가능성이 있다.

## [근거]

- raw: `results/stage2_2khz_ps_v2/diagnostic_phase_raw.npz`
- SHA-256: `a68d13b52eefaf1ccf7b439fda0f5d95353d4a8bd1a91dd2778ac9918ef7805b`
- submitted/captured: 각 `557,056` frames, 48 kHz, block 256
- submitted PCM은 canonical diagnostic slice와 byte-exact
- xrun/clip/callback status 0, ADC peak `0.0553642 < 0.4`
- metadata: `clock_scope=none`, `single_stream_clock_claimed=false`,
  `hardware_sample_slip_authority=false`
- 슬롯 8개의 두 tone·두 mic 공동 q:
  `[-731, +3279, +3360, -606, +2730, +3376, -639, -603] ppm`
- tone peak의 nominal 이동은 최대 약 `+7.39 Hz`
- DAC−ADC callback timestamp 차이는 고정 latency가 아니라
  `960/976/992 samples`를 순환했다.

## [확인 방법]

1. 각 0.5초 window의 nominal FFT bin 결과를 재현했다.
2. 각 slot의 두 fundamental과 두 mic에서 공통 주파수 비율 q를 다시 적합했다.
3. q로 스케일한 주파수에서 SNR/THD/IMD와 98/(2×49) gain을 다시
   계산했다.
4. REF envelope onset으로 이전/다음 slot이 window에 섞인 것이 아닌지 확인했다.

## [결과]

- gross onset: active start 후 `1,287~1,655 samples`
- analyzer start: active start 후 `4,800 samples`; 최소 여유 `3,145 samples`
- q 보정 후 전체 row SNR: `43.75~65.13 dB`
- q 보정 후 THD: `-38.11~-58.90 dBc`
- q 보정 후 IMD: `-31.25~-51.83 dBc`
- q 보정 후 gain error: `0.16~2.10 dB`

따라서 기존 `9~21 dB` gain error와 수십 sample phase error를 물리 비선형으로
해석할 수 없다. 다만 q 보정 후에도 남는 `0.16~2.10 dB` level dependence는
새 output-master raw 없이 진짜 비선형인지 구분할 수 없다.

또한 현 analyzer의 `quiet=captured[:24000]`은 stream-start DC decay를 포함한다.
그 RMS는 ERR/REF `-40.1/-40.6 dBFS`였지만 settled
`8192:32192`는 `-70.2/-61.0 dBFS`였다. 이 선택은 distortion noise gate를
오염시킨다.

## [판정]

- **Contradicted**: 현 raw가 심각한 물리 비선형을 입증한다.
- **Confirmed**: 현 raw의 global clock stability는 canonical phase/P/S 요구를 만족하지
  못한다.
- **Invalid experiment**: 현 fixed-index diagnostic receipt를 P/S 실행·학습 권한으로
  사용한다.
- **Inconclusive**: 보정 후 남은 1~2 dB level dependence가 실제 스피커/앰프
  비선형인지 여부.

## [다음 행동]

1. APE input과 AB13X output을 한 duplex callback/cursor로 묶지 않는다.
2. InputStream을 먼저 열어 clean pre-roll을 보존하고, 별도 OutputStream
   callback만 submitted cursor와 output pacing을 소유하게 한다.
3. output 완료 후 `4,800 + 256` samples 이상 input post-roll을 보존한다.
4. 두 stream의 callback/frame/timestamp/status/raw SHA를 독립 보존하고,
   cross-stream timestamp는 coarse auxiliary로만 사용한다.
5. acoustic tone/pilot에서 하나의 global affine q를 적합한다. 각 slot의
   q가 `±1000 ppm` 경계, phase residual, endpoint disagreement을 넘으면 P/S를
   계속 차단한다. slot-local q로 강제 PASS하지 않는다.
6. split transport A/B raw가 global q를 PASS하면 combined callback pacing 오염을
   Confirmed로 승격한다. 동일 변동이 남으면 USB DAC/ALSA/hardware clock
   경로를 다음 원인으로 조사한다.
7. 현 raw의 P/S authorization은 계속 false로 보존한다. 임계 완화나
   자동 재캡처를 하지 않는다.

## 3. 학습 admission 영향

## [가설]

현 diagnostic raw만으로 Stage-2 scratch pretrain을 시작해도 된다.

## [근거]

`scripts/train/check_stage2_2khz_campaign.py`를 현 checkout에서 실제 실행하면
`canonical_scratch_pretrain_100k_ready=false`, `pretrain_smoke_ready=false`이다.
다음 실제 artifact 묶음이 없다.

- Stage-2 primary/secondary path + plant binding
- public manifest bundle + lineage + frequency coverage + transfer/bootstrap receipt
- physical/data binding을 소비한 pretrain criterion/external contract receipt

Drive read-only inventory에서 `gdrive:DeepANC/public_archive_cache`는 directory-not-found였다.
따라서 DNS3+DEMAND6+MIMII1 fixed archive 10개, `18,229,762,015 bytes`는
아직 0개다.

## [확인 방법]

실제 campaign checker의 nonzero exit/blocker list와 `rclone` remote listing을 확인했다.

## [결과]

학습·Trainer·GPU·run directory는 시작되지 않았다. 현 Stage-2 raw나
legacy P/S·checkpoint를 자동 승격하는 경로도 열리지 않았다.

## [판정]

**Confirmed — Stage-2 scratch pretrain은 아직 NOT READY다.**

## [다음 행동]

split transport 실측 PASS → Stage-2 P/S/plant binding → final clean commit →
18.23 GB cache manifest-last Drive publish/readback → Elice exact restore/decoder/lineage/
frequency manifest → criterion/external contract → 200/500-step A100 smoke 순서를 지킨다.
