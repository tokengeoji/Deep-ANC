# 07. 평가 프로토콜

## 0. 절대 목표 2가지와 측정 매핑

| 목표 | 측정 | 도구 |
|---|---|---|
| **기능1 — 저주파+고주파 노이즈 제거** | 옥타브밴드별 감쇠(125~8000Hz), 저역(tone300/multitone/band) + 고역(hf_tone/hf_band) 시나리오, held-out 비선형 η NMSE | evaluate_offline §기능1, evaluate_session |
| **기능2 — 모든 소리 제거 (quiet zone)** | **소스 종류별** 감쇠(합성/실환경소음/음성/음악/지속환경/기계음/이벤트음), file 시나리오(음성·음악 wav 재생→상쇄) | evaluate_offline §기능2, run_realtime `--set noise.type=file` |

한쪽 대역·한쪽 소스만 좋은 결과는 목표 미달로 판정한다. 고역(>800Hz)은 광대역 S(z)
재보정(docs/02 §4)이 선행 게이트, 1633Hz 이상은 물리 한계 명시(docs/01).
현 Stage-1의 `secondary_surrogate` 결과는 표현 사전학습 검증일 뿐이다. 실측
`P(z)`/`S(z)`와 독립 recorded test 전에는 어떤 NMSE도 덕트 물리 성능으로
주장하지 않는다.

## 1. 지표

| 지표 | 정의 | 좋은 방향 |
|---|---|---|
| trusted-band NMSE(dB) | **150–1600Hz**에서 10·log₁₀(Σ|E|²/Σ|D|²) | 음수 ↓ |
| fullband NMSE(dB) | 전 주파수에서 10·log₁₀(Σe²/Σd²) | 음수 ↓ |
| NMSE gap(dB) | trusted − fullband; 대역 집중 이득/대역 밖 행동 차이 | 0과 함께 해석 |
| 감쇠(attenuation, dB) | −NMSE = 10·log₁₀(P_d/P_e) | 양수 ↑ |
| 옥타브밴드 감쇠 | 중심 125~8000Hz, 경계 f/√2~f√2 (버터워스 4차) | — |
| 세그먼트 분포 | 1s 세그먼트 감쇠의 중앙값 / 최악 10% | — |
| 실시간 건전성 | step P99(ms), deadline miss, xrun | ↓ |

**신뢰 표기**: S(z) 보정 유효대역(현재 **150–1600Hz**, `consistency_band_hz`) 밖의 밴드 수치는 `trusted=False`(*)로
표기한다 — 광대역 재보정(docs/02 §4) 후 유효대역을 갱신할 것 (설계 L2).

**이중 판정 규칙**: corrected Trainer는 trusted NMSE와 fullband NMSE를 매 train/val
평가에서 동시에 남긴다. `best.pt`는 trusted NMSE로 선택하되, fullband NMSE가
0dB보다 나빠지면 대역 밖 소음을 증폭한 것이므로 배포 후보에서 탈락시킨다.
trusted 수치만 제시하거나 fullband 평균으로 trusted 대역 개선을 숨기지 않는다.

> [!CAUTION]
> **이 규칙이 실제로 깨진 적이 있다.** 2026-08-04 판 README/docs12 는 실기 ANC 절에
> trusted 대역 4줄만 싣고, 같은 `metrics.csv` 에 기록된 **2–8 kHz 15–22 dB 증폭**을
> 한 줄도 싣지 않았다. 6개 시나리오 중 유일하게 무해한 `voice_in_noise` 만 그림으로
> 대표해 실었다. **대역 밖 옥타브를 함께 싣지 않은 감쇠 주장은 이 프로토콜 위반이다.**

## 2. 시나리오 (configs/eval.yaml — 오프라인/실기 공통)

| 이름 | 소음 | 목적 |
|---|---|---|
| S1 `tone300` | 300Hz 톤 | FxLMS 대비 동등성 검증 |
| S2 `multitone` | 120+300+750Hz | 다중 협대역 |
| S3 `band` | 80–1000Hz 대역잡음 | digital-ref 광대역 능력 |
| S4 `nonlinear` | 210Hz+3·5차 고조파+소프트클립 | THD/IMD 게이트 후 Stage-2 비선형 일반화 검증; 현 Stage-1은 참고용 |
| S5 `file` | 실측 소음 WAV 루프 | 실전 데모 — eval.yaml 미등록: 실행 시 `--set noise.type=file --set noise.file=<wav>` 로 run_realtime 에 직접 지정 |

## 3. 오프라인 평가 (하드웨어 불필요)

```bash
# synthetic 진단 평가. official recorded test 선택에는 사용하지 않는다.
.venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/<contract-seed>/ckpt/best.pt
# 동일 시나리오·동일 S(z)의 diagnostic DL vs FxLMS 표
.venv/bin/python scripts/eval/compare_fxlms.py --ckpt runs/<contract-seed>/ckpt/best.pt
```

무학습 체크포인트 기준값 (파이프라인 검증, 2026-08-02): FxLMS 는 tone300 +88dB(이상 조건)
/ band +2.1dB / nonlinear +8.8dB, 무학습 DL 은 전부 0dB 부근 — 학습 후 이 표가 채워져야 한다.

이 합성 +88dB와 사용자가 과거 legacy 실기에서 확인했다고 전달한 **약 2dB 감소**는 서로 다른
수치다. 후자는 `secondary_path.npz`, block512/high, 300Hz, noise delay 70ms, `mu=0.001`,
control limit 0.10 조건의 역사적 baseline이며 현재 하드웨어에서 재검증되지 않았다.

### 현재 자동화 범위

- Trainer 로그·checkpoint는 trusted/fullband NMSE를 동시 출력한다. 공식 fine-tune 모델
  선택은 recorded val만 사용하고 selection bundle을 원자 고정한다.
- `eval.metrics.intersect_frequency_bands`/`band_nmse_db`가 평가 공용 규약이다.
  trusted 대역은 항상 **S(z) `excitation_band_hz` ∩ duct 목표대역**으로 산출하고,
  빈 교집·샘플레이트 불일치는 fail-fast한다.
- `evaluate_offline.py`는 합성 test의 trusted/fullband/gap, 각 아이템 분포,
  held-out 비선형 trusted/fullband를 `metrics.md`+`metrics.npz`에 저장한다.
  기존 소스별 fullband NMSE와 옥타브 감쇠/`trusted` 표식도 유지한다.
- `evaluate_session.py`는 매 시나리오×컨트롤러의 trusted/fullband/gap을
  Markdown과 세션 NPZ에 남기고 기존 옥타브·miss·xrun 리포트를 유지한다.
- 이 스크립트는 모델 구조는 checkpoint에서 읽지만 데이터/duct는
  `--data-config`/`--duct-config`에서 다시 읽는다. measured 파인튜닝 평가에서는
  checkpoint의 resolved 스냅샷과 동일한 measured P/S/lead 설정 파일을 명시해야 한다.
- `evaluate_recorded.py`는 checkpoint의 **resolved** model/data/duct만 사용하고 기본적으로
  `measured_primary_path` artifact만 허용한다. 이식 가능한 manifest의 group 누수를 다시
  검사하고, 세션 가장자리 0.25초를 제외한 결정적 segment에서 `e=d+S·y`를 먼저 계산한 뒤
  warmup 0.25초를 절단한다. trusted/fullband/gap, source family, 옥타브, 최악 10%와 G4
  PASS/FAIL을 `metrics.md`+`metrics.npz`에 저장한다. surrogate는 명시적
  `--allow-surrogate` 진단만 가능하며 물리 성능으로 해석하지 않는다.

공식 test는 이 스크립트를 임의로 직접 호출하지 않는다. `run_finetune_pipeline.py`가 val
selection을 재검증해 발급한 campaign capability를 정확히 한 번 소비하고, staging 디렉터리에서
완성한 결과를 no-replace로 원자 출판한다. 1시드 clear PASS 또는 검증된 2시드 final selection이
아니면 capability가 발급되지 않는다.

## 4. 실기 평가 (덕트, 사용자 입회)

```bash
# 출력 장치를 열지 않는 선행 입력 게이트
.venv/bin/python scripts/bench/check_audio_input.py
.venv/bin/python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 multitone band nonlinear
```

이 live ANC ON 프로토콜은 공식 recorded G4와 natural-crest challenge PASS 뒤에만 연다.
시나리오마다 **OFF 10s(베이스라인) → ON 30s → OFF 5s**, 게이트 램프 ±1~2s 는
분석에서 제외. 산출: `results/eval_report_<시각>.md` (전대역/밴드별 감쇠, miss/xrun) +
세션 원시 npz. FxLMS 와 DL 은 **같은 세션 묶음에서 연속 측정**해 조건을 통일한다.

### 4.1 덕트 전달경로의 시간-주파수 지도

사용자가 말하는 “덕트 구조 파악”은 기하 치수만 추정하는 것이 아니라, 다음 네 경로를 같은
실험 규약으로 식별하는 것을 뜻한다.

| 구동 출력 | 수음 입력 | 전달경로 | ANC에서의 의미 |
|---|---|---|---|
| Noise Speaker(NS) | REF | NS→REF | acoustic reference가 소음을 먼저 관측하는 경로 |
| Noise Speaker(NS) | ERR | NS→ERR | 1차 경로 `P(z)`의 관측 |
| Cancelling Speaker(CS) | REF | CS→REF | 제어음의 reference 누설·피드백 경로 |
| Cancelling Speaker(CS) | ERR | CS→ERR | 2차 경로 `S(z)`의 관측 |

각 경로는 80–1600Hz의 크기·위상·coherence·group delay로 기록한다. 시간값은 반드시
두 종류로 분리한다. ERR/REF가 같은 I²S 입력 시계를 공유해 얻는 **마이크 간 상대 TDOA**는
덕트 내 전파 순서를 나타낸다. 반면 USB 출력→I²S 입력의 **절대 지연 상태**에는 장치 버퍼와
서로 다른 시계가 포함되므로, 반복 안정성 검증 전에는 고정 음향 지연으로 해석하지 않는다.

2026-08-03 현재 저레벨 300Hz 채널 분리 측정에서 NS와 CS 두 출력 경로가 모두 ERR/REF에
도달함을 확인했다. 같은 입력 스트림에서 계산한 상대 TDOA는 다음과 같이 반복 안정적이었다.

| 구동 | ERR−REF 상대 지연 | 도달 순서 | 판정 |
|---|---:|---|---|
| NS | +135~+138 samples = +2.79~+2.88ms | REF가 ERR보다 먼저 수음 | 상대 시간지도에 사용 가능 |
| CS | −142 samples = −2.958ms | ERR가 REF보다 먼저 수음 | 상대 시간지도에 사용 가능 |

장시간 ESS의 주파수 크기 형상은 반복 상관 평균이 P **0.962**, S **0.966**이었지만,
출력→마이크 dominant peak는 P **37.79–64.88ms**, S **37.19–80.31ms**로 이동했다.
따라서 이는 전달경로의 **진단용 magnitude 지도**일 뿐이며, 공식 `P(z)`/`S(z)` 산출물은
생성되지 않았다. 연속 재생 시간을 늘리거나 좋은 반복만 고르는 방식으로 G1을 통과시키지 않는다.

이를 한 세션에서 다시 측정하는 도구는 `measure_duct_transfer_map.py`다. NS 반복→무음→CS 반복을
같은 full-duplex callback 안에서 시간분할하며, 네 경로의 반복 IR·복소 전달함수·크기·위상·
coherence·group delay와 raw/corrected 절대지연, ERR−REF TDOA, PortAudio ADC/DAC timestamp를
NPZ·JSON·Markdown(선택 PNG)에 저장한다. 출력 반응이 무음 구간보다 충분히 크지 않거나 timestamp,
상대/차분 지연, topology, xrun/clip 중 하나라도 실패하면 `duct_identification_complete=false`다.

```bash
.venv/bin/python scripts/bench/measure_duct_transfer_map.py \
  --confirm-volume-minimum --confirm-speaker --confirm-user-present
```

### 4.2 단일 callback FxLMS 진단

`scripts/demo/evaluate_fxlms_direct.py`는 기존 3-thread ring의 miss와 FxLMS 자체를 분리하기
위해 digital reference 생성, NS 재생, FxLMS 제어, ERR/REF 수음을 한 PortAudio callback에서
수행한다. 기본 조건은 peak 0.005의 300Hz, digital-reference lead 70ms,
`mu=0.001`, control limit 0.10, block512/high, **OFF 10s → ON 30s → OFF 5s**다. 저음량 첫
실측은 `--control-limit 0.005`로 별도 제한한다.
출력 전 ERR/REF raw preflight를 통과해야 하며, xrun·입출력 clip·ON duty 95% 미만·미완주·
실제 적응 duty 95% 미만·제어 비활성·hard-limit 발생·초기/후행 OFF 미복귀·마지막 양 채널
zero flush 실패 중 하나라도 있으면 측정을 무효화한다. 원시 입력·실제 출력·
제어신호·최종 weight와 판정 JSON/NPZ를 새 결과 디렉터리에 저장하고 기존 결과를 덮어쓰지 않는다.

측정 자체가 유효해도 사용한 `S(z)`가 공식 ESS 반복 일관성·지연 안정성 게이트를 통과하지
않았다면 `performance_claim_allowed=false`다. 양의 matched ERR RMS 감쇠는 이 조건까지 통과해야
`performance_success=true`가 된다. 현재 legacy `S(z)`는 진단용이므로, direct 평가에서 감소가
관측되더라도 공식 성능 성공으로 승격할 수 없다. 이 평가기의 실제 저레벨 세션 결과도 아직 없으며,
앞선 짧은 legacy ON/ADAPT 로그와 3-thread 실험은 **유효 FxLMS 성능 성공이 아니다**.

실기 전에 필수 입력이 무신호가 아니고 클리핑이 없는지 먼저 검사한다. digital-reference/FxLMS는
ERR ch0가 필수이고, acoustic-reference와 recorded 수집은
`.venv/bin/python scripts/bench/check_audio_input.py --require-both`로 ERR/REF 모두를 요구한다.
2026-08-03 빠져 있던 pin17을 복구한 뒤 한때 ERR/REF가 −46dBFS대, clip 0%로 통과했다.
그러나 22:39 재검사에서 ERR/REF clip 2.474%/5.029%, 10초 재검사도 0.887%/3.381%로 다시
FAIL했다. 간헐 burst라 시작 과도가 아니며 이때 스피커 출력은 시작하지 않았다. 이 최신 FAIL이
현재 권위 상태다. 모든 출력 세션 직전에 재검사하고 clip 0으로 반복 PASS하지 않으면 진행하지 않는다.
`secondary_surrogate` 체크포인트는 실기 성능 평가 대상이 아니다. 실측
`P(z)` 파인튜닝과 G1–G4를 통과한 artifact에 한해, 사용자 입회·앰프 볼륨
최저·ANC OFF 시작 상태에서만 실행한다.

legacy 원본의 다음 명령은 과거 약 2dB baseline의 파라미터 기록이다.

```bash
cd /home/capston/anc_project
python3 -B main_realtime_anc.py \
  --noise-type tone --frequency 300 --noise-amplitude 0.05 \
  --noise-delay-ms 70 --mu 0.001 --control-limit 0.10 \
  --block-size 512 --latency high
```

원본 스크립트는 정상 종료 시 `control_filter_last.npy`를 저장하므로 읽기전용
`~/anc_project`에서 그대로 재실행하지 않는다. 입력 정상화 뒤 재현이 필요하면
`--weights-output /home/capston/Deep_ANC/results/legacy_fxlms/control_filter_last.npy`를 추가한다.
`-B`는 원본 디렉터리의 `__pycache__` 생성·갱신을 막으므로 제거하지 않는다.
논문/보고서에는 legacy UI의 순간값 대신 위 `evaluate_session.py`의 동일 조건
OFF→ON→OFF 리포트를 사용한다. 입력 문제를 해결하기 위한 sudo/pinmux/I²S/RT 커널 변경은 금지다.

현재 재검증 상태는 다음과 같다. audible legacy 진단은 실제 ON/ADAPT와 filter norm 증가를
확인했지만 ON 로그 9개에서 비정상 종료했고, 중앙 감쇠 +0.11dB(말미 +0.36dB), 순간 최대
+0.63dB, xrun 3→9였으며 OFF tail과 weight 저장이 없다. `evaluate_session.py` 재측정은
ANC ON duty 0%였고, 무음 3-thread 진단도 256/low에서 miss 261·xrun 311,
512/high에서 xrun 0이나 miss 307이었다. 이 수치들은 알고리즘 경로 또는 런타임 결함 진단이며
과거 약 2dB 성공의 재현이나 현재 성능 결과로 인용하지 않는다.

## 5. 비교 공정성 원칙

1. 동일 소음 프로그램·레벨·볼륨 (시나리오 설정 공유)
2. 동일 S(z) 자산 (duct.yaml 의 secondary_path.npz 를 양쪽이 공유)
3. FxLMS 수렴 시간을 인정 — ON 구간 후반부로 평가 (오프라인은 후반 1/3)
4. 실기에서는 마이크 캘리브레이션이 없어도 감쇠(비율)는 유효 — 절대 SPL 주장은 하지 않는다
5. **강튜닝 베이스라인 병기** [로드맵 A4]: 기본 설정 FxLMS 만이 아니라 스텝사이즈/탭수를
   튜닝한 FxLMS 와 고차 인과 Wiener 상한을 함께 표에 실어 "약한 베이스라인 비판"을 선제 차단한다

## 6. 파인튜닝 준비·물리 성능 주장 게이트

| 게이트 | 통과 조건 | 미통과 시 의미 |
|---|---|---|
| G0 표현 학습 | 고정 batch overfit에서 trusted NMSE < −6dB, **lead 메타 정합**(설정 lead == 실측 `S+handoff−P`) | 모델/경사/정렬 파이프라인 결함 |
| G1 경로 실측 | 동일 캡처의 `P(z)`/`S(z)` **동시 인터리브** 측정. 요구 대역(150–1600Hz) **모든 부대역** 일관성 ≥0.9406, 유지 반복 ≥8, **P−S 상대 τ 궤적이 상수**(편차 ≤3샘플), 타임베이스 드리프트 ≤2샘플/주기, 정렬 신뢰도 ≥0.95 | surrogate 이외 물리 주장 불가 |
| G2 데이터 | 소스 family×대역 커버리지, 그룹 단위 8:1:1, 독립 recorded val/test, **재생→캡처 결맞음 coh²(source→ERR) ≥0.6 (150–600Hz)**, **합성 매니페스트 ∩ 실측 소스 = ∅** | 누수·환경 암기·**시간축 붕괴** 가능성 |
| G3 파인튜닝 | `digital_primary_path_mode: measured`, measured 70% + synth 30%, P/S/lead 스냅샷 보존 | 표현 사전학습 상태 |
| G4 독립 평가 | trusted **150–1600Hz** < 0dB와 fullband ≤0dB를 동시 통과, 소스별 **최악값** < 0dB, **대역 밖 do-no-harm**, 검정력·그룹 부트스트랩 CI. 판정은 **3값**(PASS/FAIL/**INCONCLUSIVE**) | 국소 개선 또는 대역 밖 증폭 |
| G5 Jetson/실기 | artifact lead fail-fast 통과, P99 <3ms, watchdog/xrun 기록, FxLMS와 동일 세션 비교 | 배포 성능 주장 불가 |

G0의 고정-batch 수치는 의도적 과적합 진단이지 일반화 성능이 아니다.
`make_recorded_manifest.py`는 group 원자성+source-family 층화 8:1:1을 제공하고,
`validate_recorded_sessions.py`는 family×split 커버리지와 파일 QA를 기본 치명 게이트로 검사한다.
도구 구현은 완료됐지만 실제 독립 세션을 수집해 PASS하기 전에는 G2/G4가 통과한 것이 아니다.

> [!CAUTION]
> **2026-08-04 에는 이 표의 G1–G3 가 전부 PASS 였고, 전부 무의미했다.** 당시 게이트는
> ① `S(z)` 형상이 54% 틀린 아티팩트를 통과시켰고(요약 스칼라만 보고 궤적을 안 봤다),
> ② 재생↔녹음 시간축이 붕괴한 데이터셋을 "전수 QA 80/80 PASS" 로 통과시켰다(정렬을
> 검사하지 않았다). **게이트가 초록불이라는 사실은 검증이 아니라 게이트의 시야에 대한
> 진술이다.** 위 G1/G2 조건이 길어진 것은 그 사고의 결과이며, 새 게이트는 전부
> **실패 fixture 와 짝**으로 선언돼 있다(`src/deep_anc/ops/gate_registry.py`, 메타 테스트가
> 1:1 대응을 강제). 자세한 근본 원인은 [docs/12 §5.0](12_system_summary.md#50-근본-원인--게이트-9개가-pass-인데-전부-무의미했다).
>
> **아직 남은 구멍**: 메타 테스트는 "발동시킬 수 있는가" 만 강제하고 "정상 데이터에서
> 발동하지 않는가"(위양성)는 강제하지 못한다.

## 7. 리포트 양식

`results/eval_report_*.md` 표 + 다음 플롯(오프라인 도구 재사용):
ANC OFF/ON 스펙트로그램, PSD 오버레이(off/FxLMS/DL), 옥타브밴드 막대(신뢰 회색 표기).
캡스톤 보고서에는 시나리오 표 + 밴드 막대 + 물리 한계 요약(docs/01 §5)을 함께 실을 것.
