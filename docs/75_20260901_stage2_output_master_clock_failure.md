# 2026-09-01 Stage-2 output-master 실제 clock 진단 실패 기록

## 결론

이번 출력은 **P/S 또는 ANC 성능 측정이 아니다.** `USB AB13X DAC → APE capture`
현재 연결에서 하나의 전역 affine clock 비율을 실제 raw가 반증했으므로, Stage-2
P/S·사전학습·파인튜닝을 열지 않는다. 임계값을 넓히거나 slot별 보정으로 PASS를
만들지 않는다.

## [가설]

USB DAC 출력과 APE 입력이 실제 덕트 signal에서 하나의 안정적인 sample-clock
관계를 공유하지 않을 수 있다.

## [근거]

- 정확한 실행 코드: `dev` / `6034fe12227b82778793a6fe6e34450b5f6442ca`
  (`origin/dev`와 일치, clean).
- fresh meter는 PASS했다. 최종 ERR 레벨은 `-48.4205 dBFS`이며, raw는
  `results/stage2_2khz_output_master_level_20260901/20260901_013819_64e83d37/meter_raw.npz`
  (`6119fe0c2ede41fbf62aaadff356691e09cf96080fa8fc4c67af254f29ba1f23`)다.
- diagnostic raw는
  `results/stage2_2khz_output_master_diagnostic/20260831T163906_937061Z_011514f5b0f3f856/diagnostic_raw.npz`
  (`e9d8c3520d8ea3bb0ade9d64fca7c26baf7170b880806ba34236c3546882ab47`)다.
- 원본 receipt는 `clock_receipt.json`
  (`97924107bc990ec2c1dc92df23facaf04315dca277a3f2e5941bf487c570d9d0`)이며,
  `FAIL_OUTPUT_MASTER_CLOCK_DIAGNOSTIC_RAW_PRESERVED`를 기록한다.
- `/proc/asound/card2/stream0`는 AB13X가 full-speed USB의 playback `ADAPTIVE`,
  capture `ASYNC`, `bSynchAddress=0` endpoint라고 보인다. 이 descriptor는 APE
  capture와 common clock임을 보장하지 않는다.

## [확인 방법]

1. raw를 재개방해 callback/clip/전송 완전성을 확인한다.
2. 8개 two-tone slot, 양쪽 microphone, 두 tone의 위상 slope를 하나의 global affine
   rate `q`로 fit한다.
3. 개별 slot 보정은 진단에만 쓰고 P/S admission에는 사용하지 않는다.

## [결과]

- 전송 자체는 정상: output `557,056` frames, input `569,344` frames, xrun `0`,
  callback status `0`, ADC clip `0`, pre/post-roll `4,608/8,192` frames.
- coarse alignment도 PASS: offset `5,632` samples, correlation `0.95912`.
- 가장 좋은 global q는 `-886.863 ppm`이지만 global coherence는 `0.460704`
  (필수 `>=0.995`)다. 8/8 slot과 32/32 view가 FAIL했다.
- raw의 slot별 apparent phase rate는 약 `-0.83k`, `+3.13k`, `+3.25k`,
  그리고 한 slot 내 change point까지 보인다. 이것은 단일 affine q로 고칠 수 없다.
- 동일 실패가 ERR/REF 양쪽에 함께 있어, 단일 microphone 잡음이나 단순 low-SNR
  문제로 설명되지 않는다. ADC clip도 없다.

## [판정]

**Confirmed — 현재 USB AB13X output + APE input 조합은 Stage-2 2 kHz strict
P/S authority에 부적합하다.** 이 raw는 immutable forensic failure artifact이며,
P/S·plant binding·학습·성능 근거로 승격할 수 없다.

## [다음 행동]

1. 이 raw와 meter raw를 그대로 보존·백업한다. 자동 재측정은 금지한다.
2. USB DAC 출력 경로를 사용하지 않고, Jetson의 common-clock 후보
   `ADMAIF1 ↔ I2S1 ↔ RT5640/J511`로 앰프 line input을 물리 연결한다.
   현재 read-only 상태에서 `I2S1 Mux=ADMAIF1`, `ADMAIF1 Mux=I2S1`이지만
   `CVB-RT Jack-state=None`이므로 실제 cable/route가 아직 증명된 것은 아니다.
3. 연결 후에는 **무음** device/route preflight부터 하고, 새 meter 20초와
   diagnostic 11.605초를 한 번만 실행한다. global-affine clock PASS 전에는 P/S
   12.395초를 열지 않는다.
4. J511 route가 실제로 동작하지 않으면, 동일 master clock을 공유하는 DAC+최소
   2-channel ADC 또는 electrical frame witness가 있는 acquisition hardware를 마련한다.
   다른 비동기 USB DAC로 교체하는 것만으로는 해결로 간주하지 않는다.
5. strict P/S와 2 kHz actuator feasibility가 PASS한 뒤에만 Elice에서 canonical
   surrogate pretrain과 measured fine-tune을 시작한다.
