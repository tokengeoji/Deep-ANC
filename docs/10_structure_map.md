# 10. 저장소 구조 전체 지도 (Structure Map)

> 근거: 2026-08-03 3종 감사(데이터흐름·문서 사실성·운영/HANDOFF) 통합 + 코드 직접 재검증.
> 이 문서는 "어떤 설정 키를 어느 코드가 소비하는가"의 단일 참조점이다.

## 1. 전체 데이터흐름

### 1.1 학습 (Elice, open-loop / Stage-1)

```
configs/train_*.yaml ──load_train_config(config.py:73)──▶ cfg{model,data,duct 병합}
                                                             │
소음원 ──────────────────────────────────────────────────────┤
  SyntheticNoise(synthetic_signals.py, 덕트공진 가중)   25%    │
  NoisePool(noise_pool.py ← data/manifests/<tag>.jsonl)      │  dns_fullband 30% + speech 15%
       ├ music(FMA-small) 10% + demand(DEMAND 48k) 8%        │  + machine(MIMII fan 16k) 7%
       └ esc50 5%; 48kHz 리샘플(resample_poly), RMS=1 정규화 │
  (acoustic-ref 선택 시 전용 비율: synthetic/machine/dns_fullband/demand/speech/esc50
   = 45/15/20/10/5/5%; 주기성 비중 상향 + 예측 불가 성분 do-no-harm 학습)
                                                             ▼
  SynthANCDataset(data/synth_dataset.py)              n(t) [T=1.5s→71,936샘플(256 배수 내림)]
  ├ RIR 뱅크: data/rir_bank/duct_rirs_v1.npz (build_rir_bank.py 300변형; 분할 5/5/90% 하드코딩)
  │   duct.yaml positions_m/reflection/duct.* ──dsp/duct_sim.py 영상법──▶ p_ref / p_err / f_fb
  ├ digital-ref: x_ref=n[t+K], K는 strict P/S와 handoff의 TrainingTimingContract에서 유도
  │   └ d=P(z)*n[t]; primary_path.resolve_digital_primary_path 선택
  │      ├ secondary_surrogate(Stage-1): P FIR/gain=S FIR/gain, P 지연은 primary NPZ에서 읽음
  │      ├ measured(파인튜닝): primary_path_npz FIR+실측 순수지연을 각 1회
  │      └ rir_surrogate(legacy/비교): p_err RIR + D_noise−t_ac(NS→ERR) [이중계상 방지]
  │   P/S bulk delay, compact FIR peak, handoff, lead와 총 선행량은 timing contract가 구분
  ├ err_in = delay(d, fb∈[512,1024]) + 마이크잡음(snr_mic_noise_db)
  │          + 전원 험(dc_hum_prob=0.2; 50/60Hz+2차) + 채널드롭아웃(0.15/0.15 하드코딩)
  ▼
x=[B,2,T] ──HybridANCNet(models/hybrid_anc.py: /io_scale(model_*.yaml에 0.02 명시) → enc(win384,hop128)
            → TCN/GLSTM/(base만 MHSA; tiny 비활성) → dec → ×io_scale
            → 0.2·tanh 소프트리미터)──▶ y=[B,1,T]
  ▼
ANCLoss(losses/anc_loss.py, FP32 강제)
  y → RandomNonlinear(현 Stage-1: drive=1, SEF η=10, hardclip=0 → 사실상 선형)
    → DifferentiableSecondaryPath(현 Stage-1: 지연 1462+핸드오프256,
       jitter/gain/tilt=0, allpass=false → 관측 가능한 공칭 plant 고정)
  e = d + S(G(y))
    ├ 최적화/체크포인트: trusted NMSE(S 실측 150–1600 ∩ 목표 80–1600 = 150–1600Hz)
    ├ 동시 관측: fullband NMSE(do-no-harm)
    └ + λ·MRSTFT×W(f)[curriculum_a: 80–800Hz ×3, >1633Hz ×0.25]
       + λ_pow·|y|² + λ_clip·relu(|y|−0.18)²
  ▼
Trainer(train/trainer.py): AdamW+cosine, bf16 autocast(손실 FP32), val trusted/fullband 동시 로깅
  → trusted val로 best.pt 선택; resolved cfg + physics_status + lead + trusted band 저장
  → 현 Elice 실행은 GPU0=base, GPU1=tiny의 독립 프로세스(가중치 공유/DDP 아님)

[Stage-2 closed-loop] 단일 GPU 전용(DDP 금지). chunk = hop×unroll_group = 512샘플
  순차 unroll, e 프리픽스를 fb_delay(≥chunk) 지연 후 err 채널로 되먹임.
  손실 절단(warmup 0.25s)은 플랜트 적용 후(anc_loss.py loss_start_sample).
```

`secondary_surrogate` Stage-1 checkpoint의 `physics_status`는
`secondary_surrogate_representation_pretrain`이다. 고정-batch overfit과 학습 안정성은
검증할 수 있지만 실제 noise→ERR 경로가 아니므로, measured `P(z)`와 독립
recorded test 전에는 물리 감쇠 주장에 쓰지 않는다.

### 1.2 배포 (Jetson)

```
best.pt(resolved cfg/physics_status/lead 포함) ──export_onnx.py(블록256, 상태 명시 I/O, ORT 등가 검증)
        ──▶ model.onnx + model.json(lead 메타; 런타임 mismatch fail-fast, legacy=0)
        ──scripts/export/build_trt.sh──▶ <name>_fp16.plan (+메타 json 복사)
          실재: runs/export/{tiny_corrected.onnx, tiny_corrected_fp16.plan, tiny_long.onnx}
runtime.yaml ──load_runtime_config(config.py:91)──▶ RealtimeANC(realtime/run_realtime.py, 3-스레드)
  [콜백]  int32 입력 → DCBlocker → in_ring / NoiseProgram(noise.*)
          생성 신호+소음 게이트 → DigitalReferenceBuffer → K샘플 늦은 ch0
          생성 신호는 즉시 ref_digital(`ref[t]=source[t+K]`, K = artifact 의 lead)
          out_ring.pop_latest → SafetySupervisor.limit(0.2) → FadeGate → ch1(int16)
  [추론]  ref(digital=소스|mic) + err → engine.step(hop=256, ==block_size 강제) → out_ring
          (1 hop 핸드오프 = 학습 handoff_extra_samples=256 과 정합 [C1])
  [엔진]  engines.py: torch(ckpt 내장 cfg 재구성)|ort|trt|fxlms(duct.secondary_path.npz+fxlms.*)
  [안전]  safety.*: 클립스트릭/발산/데드라인 워치독 → 자동 mute
```

`configs/runtime.yaml`은 배포 템플릿 호환성을 위해 lead=0이다. 109로 학습한
artifact를 사용할 때는 런타임도 109로 명시하며, Torch/ORT/TRT 엔진이
메타 불일치를 오디오 시작 전에 거부한다. 키가 없는 legacy artifact는 0이다.

### 1.3 실측·보정 루프 / 평가

```
record_duct.py(hardware.yaml) → data/recorded/<세션> → make_recorded_manifest.py
  → group_id 원자성 + source_family 층화 + manifest-relative path
  → validate_recorded_sessions.py(채널/SR/길이/finite/RMS/clip/family×split QA)
  → RecordedANCDataset(d=err mic, x_ref=digital:source.wav / acoustic:ref mic)
  → MixedIterator(recorded_ratio; 현 Trainer는 recorded train만 소비, val은 합성 최대16개)
strict interleaved probe → cancel: S(z), noise: P(z)의 raw+analysis+compact NPZ
                         → duct의 P/S NPZ 경로만 설정
                         → TrainingTimingContract가 delay/FIR peak/handoff/lead를 유도
measure_duct_transfer_map.py(단일 stream/time-division ESS 또는 multitone)
  → NS→REF/ERR + CS→REF/ERR 반복 IR·magnitude/phase/coherence/group delay
  → 같은 I2S ERR-REF TDOA / PortAudio ADC-DAC timestamp / 절대지연을 분리 저장
  → 안정 지연일 때만 acoustic/digital 인과성 예산을 유효화; NPZ+JSON+Markdown(+PNG)
run_realtime --calibrate: 3-스레드 실효지연 측정 vs (1462+256) 대조

Trainer: trusted/fullband NMSE 동시 집계, trusted로 best 선택
eval/metrics.py: S(z) excitation∩duct target 교집합 + 공용 band NMSE 규약
evaluate_offline.py: trusted/fullband/간극을 Markdown+NPZ에 저장; 소스별·옥타브·held-out 유지
evaluate_recorded.py: resolved measured P/S/lead + 독립 val/test + source family/최악10%/G4
compare_fxlms.py(동일 S(z)) / evaluate_session.py(실기 OFF→ON→OFF;
  trusted/fullband/간극 Markdown+세션 NPZ, 옥타브·miss·xrun 유지)
```

recorded 도구의 group/source 층화·이식 경로·독립 evaluator는 구현됐다. 다만 실제 measured
P/S와 독립 세션을 수집해 QA/G4를 통과하기 전에는 물리 성능 게이트가 완료된 것이 아니다.

## 2. 설정 소비 지도

범례: ✅=코드 소비, ☠=죽은 키(어떤 코드도 읽지 않음 — grep 재확인 완료), ⚠=주의.

### duct.yaml
| 키 | 상태 | 소비 지점 |
|---|---|---|
| duct.interior_length_m / cross_section_m / end_correction_factor / speed_of_sound_mps | ✅ | dsp/duct_sim.py, config.duct_distance_samples |
| duct.shape / wall_thickness_m / total_length_m / boundary, positions_m.opening, acoustics.axial_resonances_hz | ☠ | 문서용(테스트도 하드코딩값 사용) |
| positions_m.{noise_speaker,reference_mic,cancel_speaker,error_mic} | ✅ | duct_sim.duct_paths, duct_distance_samples, validate_duct |
| acoustics.plane_wave_cutoff_hz / realistic_target_band_hz | ✅ | Trainer가 trusted 교집합과 ANCLoss 주파수 가중에 주입 |
| reflection.* | ✅ | duct_sim(단, build_rir_bank 는 자체 랜덤값으로 대체) |
| secondary_path.npz | ✅ | Trainer, SynthANCDataset, offline/FxLMS 평가, realtime FxLMS 엔진이 공유 |
| secondary_path.handoff_extra_samples | ✅ | duct.yaml에 256 명시. 키 부재 시도 `DEFAULT_HANDOFF_SAMPLES=256`을 학습·평가·런타임이 공유 |
| digital_reference.primary_path_npz | ⚠ | measured와 canonical secondary-surrogate가 strict P의 bulk delay를 읽는 단일 출처. 누락/legacy provenance면 fail-fast |

`secondary_path.delay_jitter_range`와 `calibration.input_ref_rms`는 현재 duct.yaml에 없다.
지연 지터는 `data_sim.plant_perturbation.delay_jitter_range`, 입출력 스케일은
`model_*.yaml io_scale`이 각각 단일 출처다.

### data_sim.yaml
| 키 | 상태 | 소비 지점 |
|---|---|---|
| sample_rate / segment_seconds / reference_mode | ✅ | synth·recorded_dataset, trainer.fs (세그먼트는 256 배수 내림) |
| digital_primary_path_mode | ⚠ | primary_path.py→synth_dataset.py. 현 YAML은 `secondary_surrogate`; `measured`는 primary NPZ 필수, `rir_surrogate`는 legacy/비교 전용 |
| timing_contract / digital_reference_lead_samples | ⚠ | compact P 모드에서는 설정 로더가 strict P/S와 handoff로 계약과 K를 주입한다. 수동 override는 유도값과 같을 때만 검증용으로 허용. artifact 메타와 runtime은 동일해야 하고 acoustic 모드는 K>0 거부 |
| source_mix_ratio.* / source_mix_ratio_acoustic.* | ⚠ | SynthANCDataset이 reference_mode에 따라 비율표 선택, 키=manifest 태그. manifest 부재는 Trainer 배너와 dataset 메시지 후 synthetic 폴백 |
| noise_manifest_dir / rir_bank | ⚠ | SynthANCDataset이 `_resolve_path`로 해석(CWD에 실제 경로가 있으면 우선, 없으면 REPO_ROOT 폴백). RIR 부재 시 경고 후 32개 즉석 생성 |
| level_dbfs / snr_mic_noise_db | ✅ | synth_dataset. 현 Stage-1 레벨값은 −45∼−20dBFS |
| dc_hum_prob | ✅ | SynthANCDataset이 확률 0.2로 x_ref/err_in에 50/60Hz + 2차 고조파 험 추가 |
| nonlinear.* / plant_perturbation.* | ⚠ | trainer→RandomNonlinear/DifferentiableSecondaryPath. 현 Stage-1은 η=10·drive=1·hardclip=0, jitter/gain/tilt=0·allpass=false; 미관측 랜덤 plant를 공칭학습에 임의 투입하지 않음 |
| closed_loop.{feedback_delay_samples,warmup_seconds,unroll_group_frames} | ✅ | Trainer closed-loop와 synth/recorded dataset |

`closed_loop.chunk_seconds`와 `split.*`은 현재 data_sim.yaml에 없다. chunk는
hop×unroll_group=512샘플, 분할은 RIR 90/5/5,
소음 manifest 90/5/5, 실측 manifest 80/10/10으로
코드에 고정되어 있다.

### model_base/tiny.yaml
hop/win/in_channels/**io_scale**/encoder/tcn/glstm/attention/limiter.limit → HybridANCNet ✅.
`io_scale: 0.02`는 base/tiny 두 YAML에 명시되며 코드에는 키 부재 호환용
기본값 0.02가 있다. name → ONNX 메타. (`sample_rate`는 현재 model YAML에 없음.)

### train_pretrain/finetune.yaml
model/data/duct_config → config.load_train_config ✅. batch_size/num_workers/prefetch → Trainer ✅.
optimizer.{name,lr,weight_decay,betas} → Trainer ✅
(`name`은 adamw만 허용하고 다른 값은 예외 처리).
schedule/amp/grad_clip/loss.*/eval_every/early_stop/ckpt_dir/resume/seed ✅.
init_ckpt/recorded_manifest/recorded_ratio/freeze_encoder(파인튜닝) → Trainer ✅.
`require_measured_primary_path: true`는 digital 파인튜닝에서 surrogate P(z)를 시작 전에 거부한다.
`require_init_checkpoint: true`는 초기 checkpoint 누락을 거부하고, Trainer는 저장 lead와
파인튜닝 lead가 다르면 가중치를 적용한 직후 즉시 실패한다.
`loss.nmse_objective=trusted_band`는 S 실측 대역∩덕트 목표대역(현 150–1600Hz)을
최적화하고 fullband NMSE를 동시 로깅한다. `best.pt`는 trusted val 기준이다.
체크포인트 스냅샷에는 `physics_status`와 lead/trusted band alias가 남는다.

### runtime.yaml
hop(==block_size 강제), reference, digital_reference_lead_samples, controller,
engine.{type,ckpt,onnx,plan,cpu_affinity},
fxlms.*, safety.*, noise.*, record/run_seconds ✅. `start_on`은 false만 허용하며 true는
RealtimeANC 생성 즉시 안전 오류로 거부된다(항상 ANC OFF 시작).
TorchEngine은 ckpt 내장 `state["cfg"]["model"]`을 사용하며,
체크포인트 lead를 읽는다. ORT/TRT는 ONNX/plan 옆 JSON의 lead를 읽고,
세 엔진 모두 런타임 lead 불일치를 fail-fast한다. 메타 키가 없는 legacy
artifact는 0으로만 해석한다.
과거의 `engine.model_config`는 현재 YAML에 없다. 최상위 `secondary_path`도 없으며,
FxLMS 엔진과 `--calibrate`는 runtime 병합 결과의 `duct.secondary_path.npz`를
`secondary_path_npz()`로 공유한다.

### hardware_jetson.yaml
sample_rate/block_size, input/output.card+pcm, channels.*, dc_blocker_r ✅.
**audio.latency, input/output.{channels,dtype} ☠** — 런타임·record_duct 등이
("int32","int16"), (2,2), ("low","low") 하드코딩.

### eval.yaml
octave_bands_hz/scenarios/protocol/heldout_sef_eta ✅.
`trusted_band_hz` ☠ — trusted aggregate와 옥타브 신뢰 표식은 모두
S(z) `excitation_band_hz`∩duct `realistic_target_band_hz`로 계산한다.
`report_dir` → evaluate_session.py ✅: REPO_ROOT 기준 세션 NPZ 디렉토리와
`--out` 미지정 시 기본 Markdown 리포트 경로에 쓴다.
`evaluate_offline.py`와 `evaluate_session.py`는 공용 `eval.metrics`로 trusted/fullband/간극을
계산해 Markdown+NPZ에 저장한다. 오프라인의 소스별·옥타브·held-out 지표와 실기의
옥타브·miss·xrun 지표도 유지한다.

## 3. 모듈 의존 관계 (src/deep_anc)

```
config.py (REPO_ROOT, load/merge/validate)  ←─ 모든 스크립트의 진입 관문
   ▲
data/    synth_dataset ─▶ primary_path, synthetic_signals, noise_pool(─▶manifest), dsp/{duct_sim,filters,secondary_path}, config
         recorded_dataset ─▶ manifest
dsp/     duct_sim, filters, nonlinear, secondary_path (독립적 하위층)
models/  hybrid_anc ─▶ tcn_blocks, glstm, attention, streaming
losses/  anc_loss ─▶ dsp/{nonlinear,secondary_path}
train/   trainer ─▶ data/*, dsp/*, losses, models, checkpoint, reproducibility, config
realtime/ run_realtime ─▶ engines(─▶models, baselines/fxlms_core), ring_buffer, safety, noise_gen, ui, audio_io, config
eval/    metrics(대역 교집합·band NMSE), plots, fxlms_baseline  ←─ scripts/eval·demo 가 사용
baselines/ fxlms_core (레거시 FxLMS — anc_project 유산)
```
상향 의존 없음(dsp/data → train → scripts 단방향). `config._resolve_path`는 상대경로의
존재하는 CWD 후보를 먼저 쓰고, 없으면 REPO_ROOT로 폴백한다(config.py:24-30).
SynthANCDataset은 RIR·manifest·S(z) 경로에 이 함수를 적용하고 NoisePool에는 해석된 manifest
경로를 넘기므로, 과거의 원시 CWD 상대경로 사용은 해소되었다.

## 4. 실행 경로별 진입점

파인튜닝 준비의 현행 순서는 `코드/전체 테스트 → strict P/S 무음 dry-run → 사용자 입회
레벨 교정 20초 → 같은 연결 창 strict P/S 출력 12.5초 → 즉시 스피커 분리 → 오프라인
분석/데이터 계보/Elice 준비`다. strict P/S가 합격하기 전에 장시간 재녹음을 선행하지 않는다.
Elice bootstrap도 학습을 시작하지 않고 exact code, canonical holdout, 환경과 데이터만
검증한다. `build_elice_transfer_manifest.py`가 recorded 전체, RIR, strict P/S raw·analysis·NPZ,
regrouped manifest, FMA tracks, holdout, 두 source CSV와 content-addressed provenance report를
상대경로·content SHA로 결속한다. bootstrap은 commit SHA, holdout SHA와 transfer manifest
SHA를 외부 trust anchor로 모두 요구한다. readiness, experiment contract와 학습 loader는
validator가 hash한 같은 recorded/synthetic byte snapshot을 소비한다.

| 경로 | 진입점 | 소비 설정 |
|---|---|---|
| 학습(사전) | `scripts/train/train.py --config configs/train_pretrain_tiny.yaml` (공식 world size 1) | train_pretrain_tiny → model/data/duct/campaign prerequisite 병합 |
| Elice 환경·데이터 준비 | `scripts/elice/bootstrap_all.sh --expected-commit <40hex> --expected-holdout-sha256 <64hex> --expected-transfer-manifest-sha256 <64hex> --no-update` (학습 자동 시작 없음) | canonical transfer, public raw, environment receipt |
| 파인튜닝 | `train.py --config configs/train_finetune.yaml` (+recorded_manifest) | train_finetune |
| 데이터 준비 | `prepare_noise_pool.py --expected-holdout-sha256 <64hex>` | data_sim, canonical holdout/report, raw content SHA |
| strict P/S | paired raw level evidence PASS 뒤 `set_amp_level.py`(20초) → `measure_paths_interleaved.py`(12.5초) | hardware, duct, MeasurementLevelContract, measurement_level_evidence |
| 실측 데이터 | `scripts/data/{record_duct.py, make_recorded_manifest.py}` (strict P/S 합격 뒤 필요할 때만) | hardware, duct |
| 내보내기 | `scripts/train/export_onnx.py` → `scripts/export/build_trt.sh` | (ckpt 내장 cfg) |
| 실시간 | `.venv/bin/python -m deep_anc.realtime.run_realtime --config configs/runtime.yaml` (--calibrate/--list-devices) | runtime → hardware/duct 병합 |
| 벤치 | `scripts/bench/measure_{inference,io}_latency.py` | runtime, hardware |
| 평가 | `scripts/eval/{evaluate_offline,compare_fxlms}.py --ckpt …`, `scripts/demo/evaluate_session.py` | eval, duct, data_sim |

---

## 부록: 감사 이슈 반영 상태 (2026-08-03)

위 지도를 만든 3종 감사에서 확정된 이슈 35건(HIGH 2/MED 12/LOW 21)은 같은 날 커밋에서
일괄 반영되었다 — 죽은 키 정리/주석화, S(z)·핸드오프·목표대역 단일 출처화, CWD 상대경로 제거,
manifest 부재 배너, dc_hum 구현, HANDOFF 재작성 등. 상세 내역은 해당 커밋 메시지 참조.
현재 표는 반영 후 코드를 재검증한 결과다. 특히 DEMAND·MIMII 포함 소스 비율,
SEF η, YAML `io_scale`, 공유 핸드오프 기본값, 경로 해석, dc_hum, S(z) 단일 출처,
`eval.report_dir` 소비 상태는 위 본문에 현재 상태로 갱신했다. 설정을 바꿀 때는 이 지도로
소비 지점을 찾은 뒤 코드로 재확인할 것.
