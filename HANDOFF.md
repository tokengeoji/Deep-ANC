# HANDOFF — 파인튜닝 준비 복구 상태

> “이어서 진행해줘”를 받으면 이 파일과 `AGENTS.md`를 먼저 읽는다.
> 최종 갱신: 2026-08-29. 현재 실패 분석·후속 설계 브랜치: `work/v7-nonaffine-clock`.

## 0-V7. v6 실제 결과와 현재 차단 상태 (2026-08-29, 최우선)

### [가설]

v6의 시간 분리 clock checkpoint와 near-white P/S 슬롯이면 현 Jetson의 USB 출력과
I2S 입력 사이를 하나의 stationary affine rate ratio `q`로 보정할 수 있다고 가정했다.

### [근거]

- capture 실행 commit/branch: `872e59322527880330acd989a435cd31a2d16387`,
  `work/v6-clock-checkpoints`
- 20초 level meter는 중앙값 `-48.2 dBFS`로 `-50.1±2 dBFS` 계약을 PASS했다.
- v6 raw:
  `results/fullband_causal_v6/raw_capture.npz`, SHA-256
  `f153c8664106b0c341b67db940fb2fb1d76cb7e58c2fa9a6e49558e1dba50a63`
- 외부 receipt SHA-256:
  `6372cfdec4ce15013f7bdc958f47c25fa1055f1e368adaeaa1a8d5627608dbda`
- 실패 artifact:
  `results/fullband_causal_v6/failure_232a4e53a4eaa024d54b740a01c95fe1.json`,
  SHA-256 `10856999254a8dc70c3696b02aed239db1b80f217a3dfd771442cedb2aacc75d`
- capture id: `232a4e53a4eaa024d54b740a01c95fe1`
- 최종 diagnostic-only clock artifact:
  `results/fullband_causal_v6/forensics/clock_232a4e53a4eaa024d54b740a01c95fe1.json`,
  file SHA-256 `82cc750b898dbf7a2674eb6be3b03e0eb508e928545833f6694334b3b7c04eff`
- forensic clean execution commit:
  `57b3ddeebae0aa0720773b5ddbb52b7c6ad61731` (`work/v7-nonaffine-clock`)

### [확인 방법]

immutable raw의 actual submitted/captured PCM, callback/status/xrun/clip, fixed-line SNR,
global clock 목적함수의 모든 basin, path×mic 독립 최적값을 다시 계산했다. 추가로 8,192
sample Hann window와 1,024 hop으로 8개 line×두 마이크의 short-time frequency scale을
448개 시점에서 재계산했다. 이 진단은 P/S·지연·감쇠·학습 authority를 발행하지 않는다.

### [결과]

- 정확히 1,179,648 frame, 4,608×256 callback을 모두 제출·수신했고 planned/actual
  int16 PCM은 byte-exact다. valid mask 전부 true, xrun/status/clip은 0이다.
- preterminal/terminal clock line SNR 최저는 각각 24.174/26.963 dB로 20 dB gate를
  통과했다. 따라서 단선·무출력·레벨 부족 실패가 아니다.
- global objective에는 basin 20개가 있다. best는 `-354.907693 ppm`, runner는
  `-470.919359 ppm`, runner/best objective ratio는 `1.029125`로 요구값 `4.0`에
  크게 못 미친다.
- 독립 최적값은 primary ERR/REF가 `+562.050/+564.697 ppm`, secondary ERR/REF가
  `-477.045/-473.639 ppm`이다. 한 scalar `q`가 두 출력 구간에 동시에 맞지 않는다.
- short-time 중앙값은 `-4,743.581`, `-851.638`, `+3,149.531 ppm`의 세 진단 mode로
  나뉘고 여러 clock block 내부에서도 mode가 바뀐다. 448개 중 선언된 ±1,000 ppm 안은
  132개(29.46%)뿐이다. mode 간 약 3,900–4,001 ppm 간격은 `1/256=3,906.25 ppm`과
  정합하지만, 어느 소프트웨어/USB 계층이 변환하는지는 electrical loopback 없이는
  확정하지 않는다.
- 분석·operator·P/S NPZ는 하나도 발행되지 않았다.
- 위 최종 forensic artifact는 raw/receipt/failure와 분석 dependency SHA를 결속하며
  `analysis/clock/training/deployment/attenuation/plant` 권한을 모두 false로 고정한다.

### [판정]

**Invalid experiment (P/S·학습 관점).** 캡처 transport와 배선 반응은 PASS지만, 현
APE/I2S ADC와 AB13X USB DAC의 서로 다른 clock domain을 하나의 affine `q`로 설명한다는
가설은 **Contradicted**다. 최저 basin 하나를 골라 P/S로 승격하거나 q 범위·임계를 넓히는
것은 금지한다. 이 raw로 2/4/8 kHz plant, ANC 감쇠 dB 또는 파인튜닝 준비 완료를 주장하지
않는다.

### [다음 행동]

1. (완료) `work/v7-nonaffine-clock` commit `57b3dde`에서 ambiguity basin receipt와
   short-time diagnostic-only 재현 코드를 전체 pytest·clean commit·push로 봉인했다.
2. 같은 v6 acoustic capture는 반복하지 않는다. 스피커는 v7 신호·raw publisher·합성
   insert/drop/time-warp fixture와 무음 dry-run이 모두 끝날 때까지 필요하지 않다.
3. v7은 (A) 현 하드웨어의 조건부 비-affine time-map 연구 경로와 (B) ADC/DAC 공통 clock
   또는 electrical witness를 가진 canonical 경로를 분리한다. A의 결과는 B 없이 canonical
   광대역 P/S가 될 수 없다.
4. 2026-08-29 실제 Jetson read-only 감사에서 온보드 RT5640(`/sys/bus/i2c/devices/8-001c`)
   과 APE I2S1 출력 `hw:APE,0`을 확인했다. 현 I2S2 마이크 입력 `hw:APE,1`과 I2S1은
   같은 `PLL_A_OUT0` 계열을 쓰므로 **Likely shared-rate 후보**다. 하지만 J511→앰프 배선,
   mixer route, simultaneous duplex, slip 0은 아직 미검증이므로 PASS가 아니다. 다음
   브랜치에서 reversible ALSA snapshot 후 amplifier/speaker 분리 상태의 all-zero duplex를
   먼저 실행하고, 그 뒤에만 짧은 level/channel/polarity 및 새 P/S를 설계한다.
5. Elice의 마지막 endpoint `central-01.tcp.tunnel.elice.io:56230`은 key exchange 전에
   원격에서 닫혔다. 실행 중 GPU·학습은 확인되지 않았으며, valid 광대역 P/S와 새 endpoint가
   생기기 전 canonical pretrain/fine-tune은 계속 차단한다.
6. v5/v6 측정 파일과 최종 forensic JSON은 Google Drive
   `DeepANC/jetson_measurements_20260829`에 백업했다. 새 JSON은
   `rclone check --download` 결과 0 differences/1 matching file이며 로컬·원격 MD5도
   일치한다. 같은 이름의 중복 최상위 폴더는 삭제하지 않고
   `jetson_measurements_20260829_duplicate_v6_backup`으로 이름을 분리했다. 로컬 약
   50 MiB는 후속 forensic에 필요하므로 삭제하지 않는다.

## 0-A. 2026-08-29 fullband causal v6 실측 준비

기존 2026-08-27 고주파 raw는 공통 clock 유효 주기가 0개라 P/S를 발행하지 않은
`Invalid experiment`였다. 이를 진단 수치로 승격하지 않고, 시간 분리 clock checkpoint
8개와 near-white P/S slot 6개를 쓰는 v6 계약을 별도 브랜치에서 구현했다.

- 48 kHz/256, exact 24.576초, ch0 primary와 ch1 secondary를 순차 구동한다.
- 식별 gate는 88.388–11,313.708 Hz의 8개 물리 부대역을 모두 검사한다.
- v5 telemetry schema v4는 유지하고 v6만 pre-open timing이 결속된 schema v2를 쓴다.
- meter와 live/offline 모두 current clean commit·branch·실행 script SHA가 다르면 출력/분석
  전에 거부한다.
- success publisher는 immutable raw에서 분석을 독립 재실행해 caller 결과와 byte-exact일
  때만 analysis/operator를 발행한다.
- plan payload `8b37213a13131a071e10527c948580c906dfd914a1134e98a640ead259ba42f7`,
  PCM `4e8a66b983af872192624bd6759282058cfe4a845460111a24bcd684b22551a3`다.
- 세부 계약과 실행 순서는 `docs/39_fullband_causal_v6_clock_checkpoints.md`에 있다.

v6는 위 exact commit에서 20초 meter와 24.576초 캡처를 각각 한 번 실행했다. raw transport는
PASS했지만 global clock이 비-affine라 offline 분석이 fail-closed했고 P/S는 발행되지 않았다.
동일 캡처를 반복하지 않으며, 현 시점에 2/4/8 kHz 실제 ANC 감쇠 dB 또는 canonical 학습
준비 완료를 주장하지 않는다. 실제 결과와 SHA는 바로 앞 `0-V7` 절을 우선한다.

## 0. 현재 결론

파인튜닝은 아직 시작하지 않는다. 150–1600 Hz Stage-1에 쓰는 strict P/S와 level
evidence는 완료됐지만, 사용자가 요구한 2 kHz 이상 fullband 목표의 canonical plant는
아니다. 다음 증거가 모두 생긴 뒤에만 **fullband canonical 학습**을 연다.

1. (Stage-1만 완료) 기존 strict 150–1600 Hz P/S와 level evidence
2. (미완료) hardware frame identity 또는 독립 electrical witness가 있는 새 fullband P/S
3. (재발행 필요) 새 hardware/P/S·82세션·계보 자료를 결속한 Elice transfer manifest
4. Elice에서 재생성한 public corpus manifest 6종과 전체 QA
5. fullband 선택 계약으로 처음부터 완료한 tiny canonical init checkpoint

과거 `pretrain_*_corrected`, `finetune_tiny`, legacy P/S는 삭제하지 않지만 모두
diagnostic-only다. init, resume, 모델 선택, 성능 주장의 근거로 사용하지 않는다.

### 2026-08-28 runtime physical timing audit

현 Jetson의 `runtime_tiny.yaml`/`runtime.yaml`은 legacy artifact의 lead=109만
서로 맞춘 설정이다. 현재 strict P/S capture
`5ac1313488c8434bb4d672a36503df59`의 authoritative lead는
`S.delay 1245 + handoff 256 − P.delay 1386 = 115`다. 6 samples(0.125 ms)의
불일치는 2 kHz에서 약 90°, 4 kHz에서 약 180° 위상 오차가 될 수 있으므로,
legacy 숫자를 115로 고쳐 실행하는 것은 금지한다.

`src/deep_anc/realtime/plant_contract.py`는 digital-reference DL runtime이
sounddevice import·engine 생성·입력 probe보다 먼저 다음을 read-only로 대조하도록
추가됐다.

- same-capture P/S metadata, 48 kHz/256/low, 채널, xrun/repeat/consistency
- immutable raw/analysis SHA 및 paired level evidence(probe=0.003)
- `PlantDelays.lead()`가 유도한 lead=115

따라서 legacy runtime은 정상적으로 fail-closed 되어야 하며, current Tiny/Base의
실제 덕트 ANC 성능 근거로 사용하지 않는다. canonical 115 checkpoint/ONNX와 G4,
natural-crest evidence가 생긴 뒤 별도 deployment config를 만든다.

### 2026-08-28 Jetson storage cleanup

Google Drive의 `gdrive:DeepANC/jetson_data_backup_20260827/data/raw/music`는 local/remote
file count·bytes, fixed manifest SHA, `rclone check --one-way`(0 differences)까지
검증했다. 그 뒤 정확히 `data/raw/music/fma_small`만 삭제했다
(8,002 files, 7,975,472,258 bytes). `fma_metadata/tracks.csv`(260,414,445 bytes)는
Elice transfer lineage 입력이라 보존했다.

`data/raw/noise/esc50/ESC-50-master/audio`도 Drive 내용 대조가
`0 differences / 2,000 matching`인 것을 확인한 뒤에만 삭제했다
(2,000 files, 882,088,000 bytes). `meta/esc50.csv`와 repository metadata는 남겼다.
`raw/speech`는 아직 Drive 전송 중이므로 삭제하지 않는다. 두 검증 삭제 뒤 Jetson의
실제 여유 공간은 11,261,136,896 bytes (약 10.49 GiB)다. strict P/S raw/analysis·82
recorded 세션·RIR·manifest는 삭제 대상이 아니다.

### 2026-08-28 Elice v10 — 실제 cache-safe pre-init 증거

Elice `~/Deep_ANC`에서 exact clean commit
`937af1175b3818b00d54f08732d63a9ecf07907a`으로 full bootstrap과 readiness를 다시
실행했다. bootstrap은 exit 0, 전체 pytest 0 FAIL로 끝났고 receipt
`data/manifests/elice_bootstrap_receipt.json`의 SHA-256은
`63a714902401114df9c86c0d3b6604b2a1a58b313e274aa0098f4d46ee4f009c`다.

- readiness artifact는
  `results/training_prerequisites/evidence/readiness_v10_937af11/readiness.json`
  (SHA-256 `27a175bc69e0c4c54f9faf24c1f692dc8a427d974e2356fcfa4773a4ad09743e`)이다.
  **16 gate 중 15 PASS / 1 FAIL**이며, 유일한 FAIL은 의도된
  `completed_init_checkpoint: init_ckpt가 비었습니다`다.
- strict P/S·lead=115, transfer/recorded QA, lineage leakage=0, 통계 검정력과 plant
  confidence ceiling은 모두 PASS했다.
- 이전 결함인 `assets/measured/.design_ceiling_cache.json`의 tracked write는 재발하지
  않았고, readiness 종료 뒤에도 원격 `git status --porcelain`은 비어 있었다.
- 이 evidence는 데이터·환경·pre-init readiness만 증명한다. canonical init, campaign
  ledger, 실제 학습·덕트 ANC 성능은 아직 증명하지 않는다.

### campaign prerequisite schema v5

canonical 100k를 열기 전에 수기 NMSE, gradient share, pilot score/winner 또는
`passed=true`를 ledger에 적는 경로를 폐기했다. schema v5는 다음 raw artifact에서
결론을 재계산한다.

1. G0 final model state와 fixed batch의 trusted NMSE `< -6 dB`
2. loss pilot `best.pt`/`last.pt`/recorded-val `metrics.npz`/manifest의 provenance와
   raw per-segment trusted worst-10% score
3. 선택 pilot `best.pt`와 fixed batch의 strict-S DNH gradient share `0.2–0.4`
4. selected init에 결속한 measured 5k probe의 checkpoint·manifest·finite val metrics
5. 선택 loss와 같은 A100 exact-resume smoke receipt

issuer와 canonical 100k 명령은 모두 raw pilot selection으로 유도된 같은
`loss.nmse_cvar_alpha` float를 명시해야 한다. YAML 기본 alpha=0.7을 조용히 쓰지
않으므로 winner가 0.85 또는 1.0일 때도 다른 계약으로 ledger를 발행할 수 없다.

Elice의 nominal A100 80GB PCIe는 PyTorch에서 driver-reserved memory를 뺀 약
79.4GiB로 보인다. smoke runner와 receipt validator는 bootstrap과 동일하게
`79GiB` usable-memory 하한, A100 device name, exact torch/CUDA, world=1,
결정론 backend를 함께 요구한다. 따라서 40GB A100/MIG slice는 계속 거부하며,
실제 device name과 byte 값은 immutable environment receipt에 남긴다.

새 source commit으로 전환하면 bootstrap receipt도 exact commit에 결속되어 바뀐다.
따라서 위 v10 receipt를 다음 campaign의 anchor로 재사용하지 않고, 같은 raw/audit을
새 exact commit에서 다시 full bootstrap하여 새 receipt와 15/16 readiness를 만든 뒤
G0부터 시작한다.

라이브 측정은 전체 테스트, 무음 dry-run, 장치 점유·CPU gate와 사용자 입회 뒤 실행했다.
측정 종료 직후 오디오 스트림은 닫혔고 스피커 분리 안내를 출력했다.

2026-08-27 확인 결과:

- strict P/S 실제 덕트 측정은 xrun/clip 0과 ERR/REF 입력 무클리핑으로 통과했다.
  P `primary_path_il_strict_5dc06fdd.npz`는 bulk 1642/effective 1386샘플,
  S `secondary_path_il_strict_5dc06fdd.npz`는 bulk 1501/effective 1245샘플이며,
  handoff 256에서 `PlantDelays.lead()`가 lead 115샘플을 유도한다.
  150–1600Hz consistency는 P 0.9998, S 0.9997, kept repeats 19/64,
  clock drift 중앙 2.48샘플/주기, fractional joint-LS·cubic crosscheck·compact round-trip이 PASS다.
- paired level evidence `assets/measured/measurement_level_evidence.json`도 PASS다
  (meter -48.278 dBFS, interleaved ERR -48.246 dBFS,
  SHA-256 `c76ac0d3c52c20fadd761d1ed0c85e27e3599328f60ca0d164535594336e73d0`).
- strict raw/analysis와 P/S를 묶은 로컬 transfer manifest를 생성했다
  (SHA-256 `39dc271672ac2916840a9919baaf7de5bdf078d228a68457f15096d433a76b4d`,
  344 files/82 sessions).
- Elice `~/Deep_ANC`에 transfer bundle을 전송하고 exact checkout/holdout/strict P·S
  SHA를 원격에서 대조했다. 이후 full `NVIDIA A100 80GB PCIe`(world-size 1)를 확인하고
  public raw 6종을 untouched 상태로 다운로드·manifest 재생성·QA했다. 이때의 bootstrap
  전체 pytest 0 FAIL receipt SHA는
  `f56c3d1042211112627380f74315d5949f05bcf274bdcf3fefc588ea3d3caa7e`다. **이는
  decoder-audit 결속 이전 schema v3 자료의 receipt이므로 diagnostic-only이며,
  canonical v4 readiness나 학습 개시 근거가 아니다.**
- 기존 Elice loss grid 4개(각 20k surrogate pilot)는 폐기된 `lambda_dnh=0.00025`로
  실행되어 모두 diagnostic-only다. 새 `lambda_dnh=0.00075` pilot의 첫 후보도 약
  8.5k/20k에서 `y_rms≈2.2e−5`, trusted NMSE≈0 dB로 영출력 붕괴해 중단·보존했다.
  같은 strict P/S·lead=115 고정 batch에서 NMSE-only는 −12.03 dB, frame-off는
  500 step −8.54 dB, 반면 full loss와 `lambda_frame=0.2`는 각각 약 0 dB로 붕괴했다.
  따라서 signed frame-CVaR 0.5/0.2는 canonical 후보가 아니며, v1은
  `lambda_frame=0` metric-only로 alpha만 비교한다. DNH는 대역 밖 고주파 악화 금지
  장치이므로 끄지 않는다. one-sided/item-wise frame guard v2는 별도 evidence 뒤에만
  도입한다.
- 과거 strict-S gradient budget `0.264`는 `loss_start_sample=0` 측정값이었다.
  실제 Trainer의 strict S + 3549-sample 절단 fixture는 0.130이다. `gradient_budget`과
  campaign ledger는 이제 같은 loss_start_sample을 결속하며, 실제 A100 모델/배치의
  0.2–0.4 증거 없이는 canonical을 fail-closed한다.
- old exact checkout `5979491`에서 frame-off 5k surrogate control 두 개를 끝냈다.
  generic seed `20260802`는 trusted NMSE `−21.668 dB`, matched seed `20260803`/Tiny는
  loss `44.3368 → −23.2893`, trusted NMSE `−24.308 dB`, 3.40 step/s, wrapper exit 0이다.
  둘 다 `secondary_surrogate` 고정-batch 진단이며, 실제 덕트 감쇠·init 자격·model selection
  근거가 아니다. 다만 동일 P/S·lead=115에서 frame-off가 영출력 붕괴 없이 학습되는 것을
  재현한다. generic log의 구형 wrapper exit receipt는 literal `%s`라 별도 verified receipt로
  보존했고, matched control은 올바른 exit receipt를 남겼다.
- Elice music manifest 전체 6,356개 full-decode audit는 접근 방식에 따라 결과가 달라지는
  decoder 결함을 확인했다. 65,536-frame 순차 전수 검사에서는 train의 고유 38개(경고 34,
  실제 read error 3, peak>2 4; 범주는 중복 가능), 262,144-frame 검사에서는 21개가
  검출됐다. val 234개/test 357개는 두 검사에서 0건이지만, 단일 접근 방식만으로 나머지의
  적격성을 주장할 수 없다. `NoisePool`의 retry는 문제 파일을 조용히 다른 파일로
  대체하므로 적격 manifest 증거가 아니다. raw는 불변으로 보존하고, 복수 접근 방식의
  전수 decoder audit에 결속한 신규 manifest 재생성·QA가 끝날 때까지 canonical을 열지 않는다.
- 이 결함을 막는 local 구현은 완료됐다. audit은 65,536/262,144 full sequential과 seek grid,
  C/Python decoder warning·decode error·nonfinite·peak·RMS·raw SHA/size를 기록한다. bootstrap은
  `results/provenance/decoder_audit.json`에서 `data/manifests/canonical_v4/`을 새로 발행하며,
  schema v3는 diagnostic-only다. v4는 decoder runtime fingerprint/raw inventory drift/reject
  content duplicate를 fail-closed하고 runtime `NoisePool`도 fallback하지 않는다. Elice에서
  이 새 exact commit을 checkout한 뒤 실제 전수 audit·v4 QA를 아직 실행해야 한다.
- canonical `recorded_regrouped.jsonl` 전수 QA는 82/82 세션·95.67분·오류/경고 0으로 통과했다.
  불변 `session.json`의 원본 pool group과 재그룹화 manifest의 lineage group을 직접 비교하던
  QA 결함을 수정했으며, 회귀 테스트와 전체 pytest도 0 FAIL이다.
- QA 수정 커밋 `cef615ec40b18e26c1fe3e7fa53a09c715cb7a67`, strict 자산 승격 커밋
  `4c55386`, Elice 이관 상태 문서 커밋 `86c5c45`, 데이터/손실 안정화 커밋
  `2d19f14`, 예산 근거·진행 문서 커밋 `bd2a0cf`/`43563be`가 모두 원격 브랜치에
  push 완료됐다. 현재 준비 기준선은 clean 상태로 push된
  `fix/finetune-readiness-repair`의 HEAD이며, 실행에 사용할 exact SHA는
  `git rev-parse HEAD`로 확인한다. 브랜치별 범위는 `docs/08_dev_workflow.md §7`에 고정했다.
- Elice receipt가 생긴 뒤 `check_finetune.py`의 외부 입력 차단은 해소됐지만, canonical
  init checkpoint·campaign ledger가 아직 없어 readiness는 의도적으로 16/16이 아니다.
- 2026-08-27 고주파 진단 캡처는 공식 자산과 분리해 수행했지만 유효한 clock witness를
  만들지 못했다. `results/experimental_high_band/20260827_fullband/20260827_203328_1b24d0c2/`
  의 immutable raw에서 ERR/REF 공통 clock 유효 주기가 0개(최소 8, score≥0.995)로 판정되어
  `Invalid experiment`이다. xrun/clip은 0이지만 P/S NPZ를 만들지 않았고 2/4/8kHz 성능
  숫자로 사용하지 않는다. 고주파 재측정은 자극을 좁은 대역으로 재설계한 뒤 별도 dry-run과
  사용자 입회 절차를 통과해야 한다.
- 처음 듣는 소리 검증을 Level 1–5로 고정한 OOD 게이트를 `docs/07_evaluation_protocol.md
  §3.1`에 기록했고 기준선 commit `98df0b0`으로 push했다. 현재 Level 5(모델 선택 후 실제
  덕트 새 녹음) raw/session artifact는 아직 없으므로 현장 OOD 일반화는 `Not yet
  demonstrated`이다. 이 challenge는 학습·val 선택·test에 재사용하지 않는다.
- 로컬 exact commit `0a492e99d46cd0509dee9718b3f219cbb4406380`에서 전체 pytest는
  **0 FAIL**(경고는 로컬에 없는 downstream public manifest를 진단 fixture가 알리는 것),
  `bash -n`과 `git diff --check`도 통과했다. 현재 실행 중인 Elice pilot은 실행 계약을
  고정하기 위해 소스 변경이 없던 부모 commit `1aaece892ee1ab7bc5f6a224fb1b4f29171019c0`의
  detached checkout에 남겨 두었다. pilot 종료 뒤 ledger/canonical에 사용할 exact SHA를
  한 번 더 명시적으로 고정한다.

- strict-S gradient budget의 정착 절단/ledger 결속 보강 커밋
  `a37ebf53a0e2173745fff5937f6aa6768bbb2179`을 원격 브랜치에 push했고, 로컬 전체 pytest도
  `/dev/shm` basetemp에서 0 FAIL로 재확인했다. 이 commit은 frame=0 metric-only 후보의
  policy 변경 전 기준이며, 진행 중인 Elice diagnostic은 기존 `5979491` detached checkout에
  보존한다.
- 2026-08-28 Elice의 실제 raw decoder audit은 후보 37,761개 중 36,868개를 적격으로,
  893개를 부적격으로 판정했다. 적격 수는 music 7,941, DNS noise 15,553, speech 7,971,
  DEMAND 96, MIMII 3,600, ESC-50 1,707이다. 원본을 삭제하지 않고 audit SHA
  `ceac538487ffe1414d433e3a83fdee11a0d17c204427cf8e7fed92bb73c2940f`와 accepted inventory
  SHA `b665bbb7dd28fc46cdada1d9da9a0535d74ca5ae73a030fbf5612cfcc6e61955`로 보존한다.
  현재 raw 재대조와 `canonical_v4` manifest transaction이 실행 중이므로, bootstrap의
  exit 0·QA·pytest·readiness receipt가 생기기 전에는 이 수치를 readiness PASS로 승격하지
  않는다.
- 현재 실행 중인 full audit은 중단하지 않는다. 완료 뒤 새 exact source commit에서 같은
  raw·decoder 환경을 다시 사용할 때만 `bootstrap_all.sh --reuse-decoder-audit`를 명시할 수
  있다. 이 경로는 외부 전달한 report **file SHA**와 report 내부 `audit_sha256`, canonical
  full-scan recipe, 현재 decoder fingerprint, accepted/rejected를 포함한 raw 전체
  path/SHA/size를 먼저 대조하고, `prepare_noise_pool.py` transaction이 그 전수 대조를
  다시 수행한다. 어느 하나라도 불일치하면 fresh audit으로 자동 fallback하지 않고
  실패한다. 기본값은 새 full decode audit이며, local 전체 pytest 0 FAIL과 shell 문법
  검증을 통과했다.
- 현장 evaluator는 새 source-energy 계약으로 보강했다. OFF/ON은 source를 끄는 것이 아니라
  ANC control만 OFF→ON→OFF로 바꾸며, raw NPZ에는 요청 상태를 저장한다. 각 octave는 OFF
  source의 폭 정규화 PSD 비율이 충분할 때만 감쇠를 발행하고, 그렇지 않으면 `NaN/무효`와
  진단용 원계산값을 분리 저장한다. 따라서 이 필드가 없는 legacy live result는 고주파 또는
  canonical 현장 성능 근거가 아니다.
- canonical ledger가 요구하는 A100 exact-resume smoke와 canonical 계약 사이의 순환을
  `a100_pretrain_smoke` 역할로 분리했다. 이 역할은 init-eligible=false, 200–500 step,
  A100 80GiB·world=1·CUDA/bf16·결정론 조건 및 prerequisite-root 격리를 강제한다. immutable
  `stop.pt`의 실제 resume SHA, 두 arm의 환경·telemetry·model/optimizer/scheduler/RNG/
  best-metric/data-stream 동등성을 검증하고, 첫 evaluation 전 stop checkpoint의 `+inf`
  sentinel만 제한적으로 허용한다. 새 전체 pytest는 `/dev/shm`에서 exit 0이며 기존
  diagnostic manifest-missing RuntimeWarning 2건만 있다.
- smoke는 pilot 선택 **뒤** 선택된 loss와 같은 semantic target으로 실행한다. runner의
  `--loss-alpha {0.7,0.85,1.0}`는 YAML 복제가 아니라 resolved config에 float literal을
  넣어 target/contract에 결속한다(특히 `1.0`을 정수 `1`로 바꾸지 않는다). 따라서 기본
  alpha=0.7 smoke receipt를 0.85/1.0 canonical에 재사용할 수 없고, 선택된 alpha가
  달라지면 그 값으로 새 smoke를 실행한다.
- Elice의 decoder audit 뒤 raw SHA 재대조는 생략하지 않는다. 다만 다음 exact bootstrap에는
  `--raw-hash-workers 8`을 명시해 16 vCore A100 노드에서 independent same-FD SHA/size
  검증을 병렬화할 수 있다. 기본값은 1이고 1~32만 허용한다. executor는 입력 순서대로
  결과·예외를 회수하므로 manifest bytes·identity와 첫 실패 경로는 바뀌지 않으며,
  transaction 후 audit inventory와 committed manifest raw 전체를 다시 검증한다.
- public corpus lineage의 DSU는 iterative path compression과 union-by-size를 사용한다.
  이전 lexical-root 재귀 구현은 adversarial reverse merge 2,048개에서 `RecursionError`를
  재현할 수 있었지만, component의 members·identity digest·분할 의미는 root 이름이 아니라
  canonicalized data로 결정되므로 이 변경은 계보 의미를 바꾸지 않고 대형 corpus chain을
  안전하게 닫는다.
- 2026-08-28 exact SHA `70483599c3d2d8f98fe8b36cfbe3a068fa1aa43a`의 Elice v7
  audit-reuse bootstrap은 DNS official marker 대조에서 의도적으로 exit 1로 멈췄다
  (status SHA `37e2617b61f07482be7dd74bfb831861ee2491bbe8adcfd3ea173824f4bc0f85`,
  log SHA `93371453b3e30cafa8c48bb0f0a5711a6345ded3be127ec47a83c7fc23a39d69`).
  원인은 raw 손상이 아니라 구 검증이 `scan_wavs`의 accept 집합만 official archive marker와
  같아야 한다고 가정한 것이다. 실제 decoder audit은 DNS fullband accept/reject
  `15,553/447`, DNS speech `7,971/94`를 기록했고, reject member는 의도적으로 scan에서
  빠지므로 marker의 missing으로 오인됐다. 원격 raw에서 marker의 실제 canonical root가
  `data/raw/noise/dns_fullband`, `data/raw/noise/speech`임을 read-only로 재확인했다.
- 다음 commit 후보는 marker를 **audit accept ∪ audit reject의 exact partition**으로
  검증한다. `speech` tag에 함께 존재할 수 있는 `data/raw/speech/LibriSpeech`는 일반
  lineage/DSU에 계속 포함하되 DNS marker partition에서는 제외한다. transaction
  postcommit도 copied decoder audit에서 accept/reject projection과 marker evidence를
  다시 만들고 exact equality를 요구하므로, sidecar만 바꿔서 raw 누락·reject 누수를
  통과시키지 못한다.
- 같은 작업에서 decoder audit의 raw snapshot은 정상 read가 갱신할 수 있는 `atime`까지
  전체 stat로 비교해 false reject를 만들 수 있음을 재현했다. 동일 raw identity는
  device/inode/size/mtime/ctime과 regular-file mode로만 비교하도록 고쳤다. byte 변경
  탐지는 유지하며, 새 regression과 임시 audit 100회 반복이 통과했다. 이 marker/atime
  변경 후 로컬 전체 pytest도 0 FAIL(로컬 public manifest 부재를 알리는 diagnostic
  warning만 2건), shell 문법·`py_compile`·`git diff --check`도 통과했다. commit/push
  뒤에만 Elice v8 preflight와 full bootstrap을 다시 시작한다.
- Elice v8은 marker 검증을 통과해 canonical_v4를 atomic publish했다
  (build ID `893136241cded3b835daeceafedd43863b8c8a7ecfdcde7e7f29fc5495336707`,
  raw mtime 변경 0, staging 잔재 0). 그러나 [5/6] pytest에서 기존 runtime test 하나가
  실패했다. Elice의 `runs/`에는 학습 log/config snapshot만 있고 Jetson legacy runtime의
  active artifact `runs/pretrain_base_corrected/ckpt/best.pt`,
  `runs/export/tiny_corrected.onnx`는 의도적으로 없는데, test가 `runs/` directory 존재만으로
  artifact가 fetched됐다고 오인했다. v8 log SHA는
  `5b8ad3ea059af82c677946ef6180b0ac02156267b3242d55aca894310e33981c`, status SHA는
  `45d873985c93d13033163f25b882af2974c5242022445f71f27f7428e5b4c760`이다. 이는 data/P/S
  failure가 아니며 bootstrap receipt가 없으므로 published manifest도 아직 학습에 쓰지 않는다.
  다음 commit은 runtime artifact cohort(ONNX/plan 또는 legacy checkpoint)가 실제로 있는지로
  판정하게 고친다. cohort 안에 한 파일이라도 있으면 기존처럼 config path 누락을
  fail-closed하고, A100의 일반 `runs` log만 있는 상태는 unfetched로 skip한다. 해당 Elice
  상태를 재현한 회귀와 로컬 전체 pytest 0 FAIL을 확인한 뒤에만 같은 immutable raw와
  canonical_v4 transaction을 다시 검증한다.
- 2026-08-28 구 `5632c08` Elice bootstrap은 stage 4에서 약 3시간 54분 동안 user CPU만
  증가하고 raw I/O·staging·canonical_v4·로그 진행이 없었다. raw 변경 0, tracked 변경 0,
  staging 0을 재확인한 뒤 worker 하나에만 TERM을 보내 `exit_code=143` receipt를 남겼고,
  원 log/status는 `results/training_prerequisites/evidence/diagnostic/`에 같은 SHA로 보존했다.
  새 exact checkout의 preflight는 기존 기본 경로 `~/Deep-ANC`와 실제 `~/Deep_ANC`의 불일치를
  fail-closed로 잡았다. bootstrap은 이제 호출 위치/clone 이름 대신 script 위치에서 default
  repo root를 유도한다. 이 수정의 pytest·shell 검증과 새 full preflight를 통과한 SHA만 다음
  raw-audit 재사용 실행에 쓴다.
- 새 경로 수정 `ae494ad`의 Elice preflight는 PASS했고, raw audit reuse도 internal/file SHA와
  accept 36,868/reject 893을 PASS했다. 그러나 prepare의 manifest-entry 검증이 cache 없는
  absolute path index를 매 entry마다 다시 만들어 약 `36,868 × 37,761` 경로 계약 검사를 수행하는
  O(N²) 결함을 발견했다. raw hash 뒤 staging/output 0·raw 변경 0 상태에서 worker만 TERM해
  `exit_code=143` receipt를 보존했다. path index는 이제 `(repo root, ordered raw roots)`
  context별 process-local 파생 cache로 한 번만 만들며, entry별 SHA/size·accept/reject 검증과
  transaction 후 전수 raw 재검증은 그대로 유지한다. prepare stdout은 artifact 계약에 영향을
  주지 않는 flush phase log도 남긴다.

## 1. 구현된 계약

### 시간축

- `TrainingTimingContract`가 strict P/S NPZ, compact FIR peak, 256-sample handoff,
  `PlantDelays.lead()`를 구분해 합성·실측 총 선행량을 유도한다.
- config, 합성/실측 dataset, readiness, 평가가 같은 계약을 소비한다. P/S delay나 lead를
  YAML과 문서에 수동 숫자로 복사하지 않는다.
- `recorded_lead_mode: timeline`은 합성 총 선행량에서 세션별 정렬 잔여 지연을 빼서
  실측 branch lead를 만든다. 공식 1차 실행의 lead jitter와 session mixing은 0이다.

### 체크포인트와 재현성

- 체크포인트는 commit, model/stage, loss, optimizer/schedule, P/S·RIR·manifest,
  sampler/augmentation, seed와 selection metric을 포함한 `experiment_contract_sha256`를
  저장한다.
- 자동 resume은 없다. `--resume`은 같은 전체 계약의 `last.pt`에만 허용한다.
  `init_ckpt`는 완료된 canonical surrogate-pretrain의 weight-only 전이다.
- 공식 경로는 `runs/<stage>_<contract-sha>_<seed>` 형식의 신규 디렉터리이며,
  기존 결과를 덮어쓰지 않는다.
- 공식 경로는 A100 한 장(`required_world_size: 1`)이다. global sample index 기반 sampler와
  RNG 상태를 사용하며, 중단/재개 등가를 smoke에서 확인해야 한다.

### 측정

- `MeasurementLevelContract`가 meter와 strict P/S의 probe peak `0.003`을 공유한다.
- meter raw와 strict raw는 실제 제출 int16/수신 PCM, 장치·clock·recipe, SHA receipt를
  보존한다. strict 분석은 level target 대응을 raw에서 오프라인 재검증한다.
- 출력 stream이 닫힌 직후 저장·분석보다 먼저
  `[스피커 출력 종료 — 지금 스피커/앰프를 분리하세요]`를 출력한다.
- `--confirm-user-present`, `--confirm-volume-minimum`, routing/geometry와 same-amplifier
  확인이 빠지면 장치를 열기 전에 실패한다. PCM 점유, CPU idle, clock gate도 선행한다.

### 데이터와 Elice

- v1/v2 historical builder를 재현해 160개 source WAV 불변성과 CSV prefix를 검증하고,
  누락 `sources.csv`만 복구한다. `identify_pool_clips.py`는 진단용이지 권위 자료가 아니다.
- FMA artist/album, speech speaker/book, 원본 clip 공유 관계의 transitive component를
  절대 나누지 않는 `recorded_regrouped.jsonl`을 만든다.
- active 82세션만으로 canonical holdout을 만들고, synthetic manifest는 raw audio content
  SHA와 holdout SHA를 결속한다.
- content-addressed provenance report와 transfer manifest가 recorded 전체 파일, RIR,
  strict raw/analysis/P/S, regrouped manifest, FMA tracks, holdout, 두 CSV를 한 번에 결속한다.
- Elice bootstrap은 전체 40자리 commit, holdout SHA, transfer manifest SHA, `--no-update`를
  요구한다. dirty/숨김 index/graft/replace/symlink/byte mismatch는 다운로드 전에 실패한다.

## 2. 지금부터의 로컬 실행 순서

### 2.1 코드·계보 게이트

에이전트 작업이 모두 합쳐진 뒤 아래를 순서대로 실행하고 결과를 이 문서에 기록한다.

```bash
.venv/bin/python scripts/data/repair_source_pool_provenance.py \
  --repair-csv --write-active-holdout --write-regrouped-manifest --jobs 4

.venv/bin/python -m pytest -q
bash -n scripts/elice/bootstrap_all.sh scripts/elice/setup_env.sh
git diff --check
```

로컬에는 untouched public raw 6종이 없으므로 provenance 명령에
`--require-downstream-gates`를 붙이지 않는다. downstream synthetic gate가 BLOCKED인 것은
정상이며 Elice raw를 확보한 뒤에만 연다.

오디오 없는 공식 dry-run:

```bash
.venv/bin/python scripts/data/set_amp_level.py --self-test
.venv/bin/python scripts/data/measure_paths_interleaved.py \
  --dry-run --bootstrap-level-evidence \
  --primary-out assets/measured/_dry_run_primary_do_not_create.npz \
  --secondary-out assets/measured/_dry_run_secondary_do_not_create.npz \
  --diagnostics-root results/_dry_run_measurement_do_not_create
```

dry-run 뒤 위 세 경로가 생성되지 않았는지 확인한다. 그 다음 비밀정보·대용량·ignored data가
staging에 없는지 검사하고 한국어 메시지로 commit/push한다. 커밋 메시지에 AI 표기를 넣지 않는다.

### 2.2 strict P/S 라이브 측정

커밋·push 뒤에도 자동으로 소리를 내지 않는다. 먼저 read-only로 다음을 확인한다.

```bash
fuser -v /dev/snd/*
for f in /proc/asound/card*/pcm*/sub*/status; do printf '%s: ' "$f"; cat "$f"; done
```

다른 프로세스가 점유하면 중단한다. 두 마이크 입력, ERR/REF와 noise/cancel speaker 배선,
사용자 입회, 볼륨 최소, 즉시 분리 준비를 다시 확인한 뒤 다음 두 출력 창만 실행한다.

```bash
# 1) input-only 1.5초 후 nominal 20.0초, hard max 21.0초 출력
.venv/bin/python scripts/data/set_amp_level.py --bootstrap-level-evidence \
  --confirm-speaker --confirm-user-present --confirm-volume-minimum

# 출력 종료 즉시 분리. 노브를 바꾸지 않고, 10분 안에 다음 명령 직전에만 재연결한다.
# meter가 출력한 METER_RAW와 strict 명령 전체를 그대로 사용한다.
METER_RAW=results/calibration_interleaved/level_bootstrap/<session>/meter_raw.npz
.venv/bin/python scripts/data/measure_paths_interleaved.py \
  --bootstrap-level-evidence --meter-raw "$METER_RAW" \
  --confirm-same-amplifier-setting --confirm-user-present \
  --confirm-volume-minimum --confirm-routing-and-geometry \
  --primary-out assets/measured/primary_path_il_strict_<capture-id>.npz \
  --secondary-out assets/measured/secondary_path_il_strict_<capture-id>.npz
```

strict stream은 input-only preflight 3초 뒤 nominal 12.5초, hard max 13.5초다. nominal audible
합계는 32.5초다. 각 출력 close 직후 즉시 분리한다. 실패해도 재측정하지 않고 immutable raw를
먼저 분석한다. 기존 legacy NPZ는 덮어쓰지 않는다.

합격 조건은 48 kHz/256/low, 지정 channel/operator, xrun/clip 0, observed PCM과 raw/analysis
SHA, clock-q witness, fractional joint-LS+cubic crosscheck, compact round-trip,
150–1600 Hz 모든 부대역 consistency ≥0.9406, kept repeats ≥8, 안정적인 P−S 상대 지연이다.
임계값은 낮추지 않는다.

## 3. 계보와 Elice 이관

strict P/S가 합격한 뒤 canonical transfer manifest를 만든다. 실제 capture 파일을 모두
`--strict-raw`/`--strict-analysis`로 열거한다.

```bash
EXPECTED_HOLDOUT_SHA256=$(sha256sum data/manifests/recorded_holdout.json | awk '{print $1}')
.venv/bin/python scripts/data/build_elice_transfer_manifest.py \
  --rir-bank data/rir_bank/duct_rirs_v1.npz \
  --strict-raw results/<capture>/raw_measurement.npz \
  --strict-raw results/<capture>/metadata.json \
  --strict-analysis results/<capture>/analysis_results.npz \
  --strict-analysis results/<capture>/analysis_metadata.json \
  --primary-npz assets/measured/primary_path_il_strict_<capture-id>.npz \
  --secondary-npz assets/measured/secondary_path_il_strict_<capture-id>.npz \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256"

EXPECTED_TRANSFER_MANIFEST_SHA256=$(sha256sum \
  data/manifests/elice_transfer_manifest.json | awk '{print $1}')
```

manifest에 열거된 상대경로를 rsync/scp로 그대로 스트리밍한다. 로컬 tar 복제본을 만들지 않는다.
Elice에서는 exact detached checkout을 사용한다.

```bash
EXPECTED_COMMIT=<신뢰한-전체-40자리-SHA>
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update --preflight-only

# A100 80GB 1장·가용 128GiB 확인 뒤 full bootstrap
bash scripts/elice/bootstrap_all.sh \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
  --no-update
```

bootstrap은 torch `2.5.1+cu121`, CUDA 12.1 계약, A100, 저장공간, public raw 수량과 FMA
metadata를 검증하고 manifest 6종을 untouched raw에서 재생성한다. noise/recorded QA,
전체 pytest와 readiness까지 통과한 정상 사전학습 출발 상태는 init 하나만 FAIL인 15/16이다.

## 4. 공식 학습 순서

1. family→lineage component→session 균등 sampler와 공통 gain/polarity/EQ, input-only mic
   noise를 사용한다. session mixing과 lead jitter는 0이다.
2. 고정 batch G0에서 trusted NMSE < −6 dB와 lead metadata를 확인한다.
3. seed `20260803`, frame-metric-only(`lambda_frame=0`)의 `alpha∈{0.7,1.0}`을 20k
   surrogate + 5k measured probe로 recorded val만 사용해 비교한다. 0.2 dB 이내
   동률이면 alpha 0.85를 추가하고, 계속 동률이면 alpha 0.7을 택한다. alpha 1.0의
   non-finite/실행 실패는 immutable pre-forward witness를 재실행하는 failure receipt
   구현 전에는 fallback 근거로 쓰지 않고 canonical을 fail-closed한다.
   170ms frame metric은 candidate마다 기록해 비교·원인 분석에 사용한다. 고정 local
   pass threshold가 생기기 전에는 이 metric으로 성능 PASS를 주장하지 않는다. pilot checkpoint는
   init 자격이 없다.
4. 선택된 pilot `best.pt`와 fixed batch에서 strict S의 `lambda_dnh` gradient 비중 0.2–0.4를
   재계산한다. 같은 winner를 init으로 한 measured 5k probe의 completion/provenance와 finite
   recorded-val metrics를 확인한다.
5. 선택 계약의 A100 200–500 step exact-resume smoke에서 VRAM, 처리량/ETA, 중단·재개
   수치등가를 먼저 확인한다. schema v5 issuer가 G0·pilot·gradient·probe·smoke raw artifact를
   다시 검증해 canonical ledger를 no-replace 발행한 뒤에만 다음 단계로 간다.
6. 선택 계약의 tiny를 새 run에서 100k 처음부터 사전학습한다.
7. canonical init 지정 뒤 readiness 16/16을 확인하고 open-loop, recorded 70% + synthetic 30%,
   bf16 forward + FP32 loss, 50k fine-tune을 실행한다.
8. checkpoint 선택은 recorded val만 사용한다. 선택을 고정한 뒤 test를 정확히 한 번 연다.
   경계 0.3 dB 이내 또는 INCONCLUSIVE일 때만 seed `20260903`의 100k+50k를 한 번 더 한다.

공식 test G4는 trusted 150–1600 Hz 평균/모든 family 평균/최악 10%/family cluster-bootstrap
95% CI 상단이 모두 0 dB 미만, fullband 평균 ≤0 dB, 대역 밖 octave 최악 10% 증폭 <1 dB이며
판정이 PASS여야 한다.

## 5. G4 이후

G4 PASS 뒤에만 완전 미사용 natural-crest source로 speech/music/environment/machine 각 1세션
(1차 약 4분 40초 audible)을 녹음한다. 네 계열이 모두 개선되면 계열당 3개 독립 그룹을 더해
총 16세션으로 확장한다(누적 최대 약 18분 40초 audible). challenge는 학습에 쓰지 않는다.

G4와 crest challenge를 모두 통과하기 전에는 closed-loop, ONNX export/배포, 실제 ANC ON
평가로 진행하지 않는다.

## 6. 아직 남은 실행 항목

- Elice에서 새 exact commit으로 decoder 전수 audit → canonical_v4 manifest → QA/pytest/readiness를
  다시 실행하고, audit binding이 실제 raw/decoder 환경과 일치하는지 확인
- frame-metric-only alpha 2개 20k pilot을 recorded val만으로 실행·선택
- winner의 5k measured probe, 실제 A100 bf16 중단→resume 수치등가 smoke, G0·gradient
  ledger 작성 및 SHA 결속
- canonical tiny 100k surrogate-pretrain init checkpoint 생성 후 readiness 16/16 확인
- canonical measured 50k fine-tune, 고정 checkpoint의 단 한 번 G4 평가
- G4 PASS 뒤 natural-crest challenge 녹음·평가

이 항목들은 코드로 우회하거나 legacy artifact로 대체하지 않는다.

## 7. 외부 폴더 감사 및 저장소 정리(2026-08-27)

- `/home/capston/DeepANC_CRN_n_codex/duct_cnn_anc`는 읽기 전용으로 전수 감사했다.
  논문형 3,232-parameter 모델과 Primary 진단 raw는 있었지만 checkpoint/학습/ONNX가
  없고, Secondary는 전기적 speaker-input이 없는 proxy와 입력 프레임 불일치로
  `Invalid experiment`이다. 현행 P/S·lead·학습 계약은 변경하지 않는다.
- 외부 감사 상세와 재사용/차단 목록은 `docs/11_external_duct_cnn_audit.md`에 기록했다.
- 명백한 임시 readiness snapshot 두 디렉터리, 종료된 PID의 stale audio lock, 저장소
  Python/test cache는 휴지통으로 이동했다. 데이터·raw·RIR·checkpoint·legacy 결과는
  보존했다. 삭제/보존 목록과 SHA는 `docs/13_repository_cleanup_20260827.md`에 있다.
- Jetson 용량 부족을 이유로 public corpus를 중복 복제하지 않도록 Kaggle/Google Drive를
  사용할 때의 임시 staging·비밀정보·SHA·manifest 재검증 절차를 `docs/14_elice_external_data_staging.md`
  에 기록했다. 현재 Elice raw가 이미 정상적으로 있으므로 pilot 중에는 downloader를
  실행하지 않는다.
- 현재 branch HEAD는 외부 감사 기록과 후속 정리 변경을 포함한 최신 commit이다. Elice의
  진행 중 pilot은 중단하지 않고 종료 뒤 이 branch의 exact commit으로 동기화한다.
- 2026-08-27 22:03 KST read-only poll에서 Elice pilot parent PID 58467과 네 번째
  `alpha=1.0, lambda_frame=0.2` worker가 살아 있었고 로그는 약 9.3k/20k까지 진행됐다.
  A100 80GB는 GPU 56%, VRAM 6.6/81.9 GiB, Elice 디스크는 80 GiB 여유였다. 세 완료
  후보와 진행 후보 모두 `libmpg123`의 MP3 dequantization/illegal-header 경고를
  남기고 있다(완료 로그 각 159건, 진행 로그 76건 이상). 경고가 발생한 원본을 decoder
  audit로 분류하기 전에는 어떤 pilot도 canonical init이나 최종 성능 근거로 승격하지
  않는다. 실행 프로세스는 중단하지 않았다.
- 2026-08-27 22:25 KST 재확인에서 같은 worker가 약 13.8k/20k, 3.4 step/s로 진행 중이었다.
  단순 속도 외삽상 네 번째 pilot 자체는 약 30–35분 뒤 종료 예상이지만, 이는 decoder
  audit·winner 선택·measured probe·resume smoke를 포함하지 않은 추정이다. Elice에는
  `data/raw` 34 GiB, manifest 31 MiB, `runs` 110 MiB가 있고 전체 디스크 여유는 약
  80 GiB다. 현재 데이터는 이미 Elice에 있으므로 Jetson 용량 부족을 해소하기 위해
  같은 corpus를 다시 외부 저장소에서 중복 다운로드할 필요가 없다.

- Jetson `data/` 백업은 Google Drive `gdrive:DeepANC/jetson_data_backup_20260827/data`로
  진행 중이다. 13,428개/17,441,317,063바이트 고정 목록의 SHA는
  `1dd9fef8d796cc1f27fbf5d434d640c8b80554e16f04b6bfac0d3403c748bea2`이며 Drive에 올린
  manifest SHA와 일치했다. 겹치는 전체 복사 프로세스는 중단하고 `raw`, `recorded`,
  `recorded_broken`, `source_pool_v2` 전용 `rclone copy`만 유지한다. 원격 파일 수·바이트와
  `rclone check`가 고정 목록과 일치하기 전에는 원본을 삭제하지 않는다. 백업 절차와 보존/
  삭제 범위는 `docs/15_jetson_drive_backup_20260827.md`에 기록했다.
