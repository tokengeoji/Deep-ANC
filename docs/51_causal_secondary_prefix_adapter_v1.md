# Full-octave causal secondary-prefix adapter v1

## [가설]

미래의 125 Hz--8 kHz fullband P/S를 학습에 연결할 때, target 구간의 제어 출력
`y_target`만 S(z)에 통과시키거나 augmented digital-reference 입력으로 P(z)를 만들면
block 경계의 지연·FIR tail, preview lead, P 입력 정렬을 잃어 특히 고역 위상과
`e = P*n + S*y` 손실이 틀어질 수 있다.

## [근거]

- `SynthANCDataset`의 digital-reference 의미는 `x_ref(t)=n(t+K)`, `d(t)=P*n(t)`다.
  mic noise/hum/dropout은 controller input에만 넣는다. 따라서 augmented `x_ref`를 P에
  넣으면 lead와 augmentation 모두 physical disturbance로 오인한다.
- `CausalFIRPath`는 `integer_delay(value, coarse_delay + handoff)` 뒤 causal FIR을
  적용한다. S의 target 초기 sample은 prefix actuator output에, P의 target sample은
  prefix clean playback `n`에 의존한다.
- 기존 `BroadbandFullOctaveLossPrimitiveV3`는 이미 계산된 valid-crop `S*y`만 받고,
  controller state/prefix를 만들지 않는다.
- 현 strict P/S는 150--1600 Hz만 검증했으며, 기존 high-band raw는 synchronized
  electrical witness가 없어 full-octave P/S/학습 authority가 아니다.

## [확인 방법]

새 module은 device-agnostic tensor 계산과 provenance boundary를 다음처럼 분리한다.

1. `FullOctaveCausalPlantBindingV4`
   - canonical `BroadbandFullOctaveContractV3` 전체 digest
   - 실제 `TrainingTimingContract` schema v2 전체 payload/digest
   - P/S role, 48 kHz, 256 callback, P pre-FIR delay/FIR peak, S delay/handoff,
     동일 raw/analysis/plant authority SHA, ERR/reference channel-selection SHA
   - 88.388--11,313.708 Hz eight physical subband exact binding
   - FIR ndarray를 immutable bytes snapshot으로 복제한다.
   - **현재는 production issuer가 없다.** public constructor는 실패하며, CPU regression용
     private fixture issuer가 만든 object도 public adapter가 거부한다. 따라서 legacy P/S에
     문자열 SHA·band를 덧붙여 training descriptor로 재표기할 API가 없다.

2. `CausalSecondaryPrefixAdapterV1`
   - controller의 `forward()`가 아니라 `init_states()`와 `streaming_step()`만 사용한다.
   - input prefix/target은 모두 256 sample의 정수배이며, model hop도 256을 정확히 나눠야 한다.
   - zero-reset digital-reference surrogate segment는 clean playback timeline
     `[prefix + target + derived lead]`를 별도로 제공한다. 이 timeline은 **common
     gain/polarity/common EQ가 반영된 physical playback `n`이며 input-only mic
     noise/hum/dropout 이전**이다. 이 timeline에서 lead를 직접 유도해 clean preview
     `n(t+K)`와 physical P input `n(t)`를 분리한다.
   - caller가 제공하는 pre-augmentation reference는 이 derived clean preview와
     byte-exact tensor equality여야 한다. lead를 수동 숫자로 다시 넣거나 1 sample 밀면
     fail-closed한다.
   - controller는 augmented `x_prefix/x_target`만 보고,
     `P(clean_playback)`와 `S(concat(y_prefix, y_target))`를 함께 계산한다.
   - `d_target`을 입력으로 받지 않고 target crop의 `e=P*n+S*y`를 직접 반환한다.
   - P/S convolution과 반환 `P*x`, `S*y`, `e`는 FP32다. bf16 controller forward 출력도
     plant 직전에 FP32로 변환한다.
   - GLSTM/MHSA의 임의 외부 state를 받아 연속 runtime을 주장하는 API는 제공하지 않는다.
     v1의 보장 범위는 zero-reset training segment 내부의 causal composition이다.

검증 명령은 다음이며 ALSA, sounddevice, GPU training, run directory를 열지 않는다.

```bash
PYTHONPATH=src /home/capston/Deep_ANC/.venv/bin/python -m pytest -q \
  tests/test_causal_secondary_prefix_adapter_v1.py
```

## [결과]

테스트는 다음을 확인한다.

- FFT implementation을 재사용하지 않는 direct time-domain oracle로 P(n)와 S(y)의
  full-prefix valid crop을 각각 대조한다.
- target-only `S(y_target)`는 의도적으로 다른 결과이며 정답으로 쓸 수 없다.
- controller reference에 mic noise를 넣거나 dropout해도 P(n) target과 derived clean
  preview가 바뀌지 않는 것을 확인한다.
- `K>0` non-periodic impulse에서 `P(n(t))`와 잘못된 `P(n(t+K))`가 다름을 직접
  확인한다. common gain/polarity/causal EQ 뒤의 clean playback과 preview는 함께
  변하지만, 그 뒤 input-only augmentation은 P(n)를 바꾸지 않는 것도 대조한다.
- clean preview를 1 sample 이동한 batch는 fail-closed한다.
- 미래 target 변경이 `y_prefix`를 바꾸지 않고, `controller.forward()`는 호출되지 않는다.
- target loss gradient가 controller parameter와 prefix input까지 전달된다.
- 실제 tiny/base `HybridANCNet`(GLSTM/MHSA 포함)의 `init_states`/`streaming_step` API가
  256 block/context=256에서 통과한다.
- history 부족, 비-256 length, NaN/Inf, state-origin/segment 연속성 오류를 fail-closed 한다.
- Stage-1 contract, static electrical-witness schema, partial physical subband, P/S timing/raw mismatch를 거부한다.
- v3 admission의 `V3_TRAINER_EVALUATOR_CONSUMERS_IMPLEMENTED=False`는 그대로다.

## [판정]

**Confirmed — test-fixture에서 P/S prefix composition과 descriptor re-label 차단을 확인.**

현재 module에는 actual raw/analysis/electrical witness bytes를 검증해 production binding을
발행하는 issuer가 없다. 따라서 unit test PASS는 다음 어느 것도 증명하지 않는다.

- 실제 덕트 P/S 또는 2/4/8 kHz attenuation
- synchronized electrical witness
- full-octave population/lineage/no-leakage
- DNH gradient calibration, trainer, checkpoint, pretrain/fine-tune
- canonical training/deployment eligibility

특히 현재 Stage-1 `lead=115` NPZ, legacy checkpoint, v5/v6 fixture, v12 static witness는
public binding/adapter 경로에 들어갈 수 없으며 학습 입력으로 승격되지 않는다.

## [다음 행동]

1. 동일 acquisition clock의 ERR/REF/noise-tap/cancel-tap electrical witness(또는
   검증된 APE/external hardware-frame bridge)를 실제 hardware receipt로 확정한다.
2. 그 경로로 raw-first 48 kHz/S32/256 fullband P/S를 한 번 측정하고, eight physical
   subband와 ERR channel-selection을 immutable raw/analysis/operator bytes로 발행한다.
3. 별도 raw-bound issuer/loader에서 그 artifact bytes와 operator bytes, channel selection,
   timing-v2 payload를 다시 대조한다. 그 issuer가 non-fixture binding을 발행하기 전에는
   adapter를 Trainer에 연결하지 않는다.
4. 새 plant 뒤 canonical v3 population/manifest, family-balanced batch, DNH calibration,
   Elice bootstrap을 재발행하고 그때만 pilot → 100k pretrain → 50k fine-tune을 연다.
