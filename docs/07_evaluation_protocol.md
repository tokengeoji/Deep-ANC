# 07. 평가 프로토콜

## 0. 절대 목표 2가지와 측정 매핑

| 목표 | 측정 | 도구 |
|---|---|---|
| **기능1 — 저주파+고주파 노이즈 제거** | 옥타브밴드별 감쇠(125~8000Hz), 저역(tone300/multitone/band) + 고역(hf_tone/hf_band) 시나리오, held-out 비선형 η NMSE | evaluate_offline §기능1, evaluate_session |
| **기능2 — 모든 소리 제거 (quiet zone)** | **소스 종류별** 감쇠(합성/실환경소음/음성/음악/지속환경/기계음/이벤트음), file 시나리오(음성·음악 wav 재생→상쇄) | evaluate_offline §기능2, run_realtime `--set noise.type=file` |

한쪽 대역·한쪽 소스만 좋은 결과는 목표 미달로 판정한다. **2026-09-15 사용자 확정 기준은
acoustic REF, ERR 한 점, 고역 1 kHz 이상**이다. 고역은 광대역 S(z) 반복 검증이 선행 게이트다.
2026-09-17 확정한 **Jetson 우선 개선 대역은 1000–1600Hz**이며 시스템 전체 저역도 함께 평가한다.
800–1000Hz 경계와 기존 800–1600Hz 연구 지표는 별도로 보존하며 새 우선대역의 결과로 바꿔 부르지 않는다.
현재 마이크·USB DAC·Jetson·덕트는 모두 유지한다. 하드웨어 교체를 성공의 전제로 삼지 않는다.
약 1633Hz의 평면파 모드 차단을 ERR 한 점 감쇠의 절대 상한으로 해석하지 않는다(docs/01).
위 표의 기존 digital 시나리오와 아래 과거 실적은 acoustic 실기 검증을 대신하지 않는다.
현재 acoustic 녹음의 무출력 분석은 **§8**을 따르며, 외부 소리 세션의 내부 소음은 OFF로 유지한다.
기존 digital Stage-1의 `secondary_surrogate` 결과는 표현 사전학습 검증일 뿐이다. 실측
`P(z)`/`S(z)`와 독립 recorded test 전에는 어떤 NMSE도 덕트 물리 성능으로
주장하지 않는다.

아래 Python 명령은 Docker 컨테이너 내부에서 실행한다. 실행 호스트에 맞는 환경 선택은
[docker/README](../docker/README.md), 최신 완료 상태와 산출물 위치는 [HANDOFF](../HANDOFF.md)를 따른다.

## 1. 지표

| 지표 | 정의 | 좋은 방향 |
|---|---|---|
| trusted-band NMSE(dB) | 지정한 신뢰대역에서 10·log₁₀(Σ|E|²/Σ|D|²); 기존 측정 S 기준 150–600Hz | 음수 ↓ |
| fullband NMSE(dB) | 전 주파수에서 10·log₁₀(Σe²/Σd²) | 음수 ↓ |
| NMSE gap(dB) | trusted − fullband; 대역 집중 이득/대역 밖 행동 차이 | 0과 함께 해석 |
| 감쇠(attenuation, dB) | −NMSE = 10·log₁₀(P_d/P_e) | 양수 ↑ |
| 옥타브밴드 감쇠 | 중심 125~8000Hz, 경계 f/√2~f√2 (버터워스 4차) | — |
| 세그먼트 분포 | 1s 세그먼트 감쇠의 중앙값 / 최악 10% | — |
| 실시간 건전성 | step P99(ms), deadline miss, xrun | ↓ |

**신뢰 표기**: S(z) 보정 유효대역(기존 측정 자산은 150–600Hz) 밖의 밴드 수치는 `trusted=False`(*)로
표기한다 — 광대역 재보정(docs/02 §4) 후 유효대역을 갱신할 것 (설계 L2).

**이중 판정 규칙**: corrected Trainer는 trusted NMSE와 fullband NMSE를 매 train/val
평가에서 동시에 남긴다. `best.pt`는 trusted NMSE로 선택하되, fullband NMSE가
0dB보다 나빠지면 대역 밖 소음을 증폭한 것이므로 배포 후보에서 탈락시킨다.
trusted 수치만 제시하거나 fullband 평균으로 150–600Hz 개선을 숨기지 않는다.

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
# 테스트 split 종합 평가 → runs/<exp>/eval/{metrics.md, psd.png, spec.png, band.png}
.venv/bin/python scripts/eval/evaluate_offline.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt
# 동일 시나리오·동일 S(z) 에서 DL vs FxLMS 표
.venv/bin/python scripts/eval/compare_fxlms.py --ckpt runs/pretrain_base_corrected/ckpt/best.pt
```

무학습 체크포인트 기준값 (파이프라인 검증, 2026-08-02): FxLMS 는 tone300 +88dB(이상 조건)
/ band +2.1dB / nonlinear +8.8dB, 무학습 DL 은 전부 0dB 부근 — 학습 후 이 표가 채워져야 한다.

이 합성 +88dB와 사용자가 과거 legacy 실기에서 확인했다고 전달한 **약 2dB 감소**는 서로 다른
수치다. 후자는 `secondary_path.npz`, block512/high, 300Hz, noise delay 70ms, `mu=0.001`,
control limit 0.10 조건의 역사적 baseline이며 현재 하드웨어에서 재검증되지 않았다.

### 현재 자동화 범위

- Trainer 로그·TensorBoard·checkpoint 선택은 trusted/fullband NMSE를 동시 출력한다.
  단, 현 val은 고정 합성 배치 최대 16개이며 recorded val/test를 소비하지 않는다.
- `eval.metrics.intersect_frequency_bands`/`band_nmse_db`가 평가 공용 규약이다.
  기존 offline/session 평가의 trusted 대역은 **S(z) `trusted_band_hz()` ∩ duct 목표대역 ∩ Nyquist**로
  산출하고, 빈 교집·샘플레이트 불일치는 fail-fast한다. `trusted_band_hz()`는 `consistency_band_hz`를
  우선하며 이 검증 대역 메타가 없는 legacy 자산에만 `excitation_band_hz`로 폴백한다. 이 호환 동작은 가진대역의
  반복 검증을 뜻하지 않는다. acoustic readiness와 §8 녹음 진단은 폴백 없이 검증 대역 메타와
  반복 일관성 기준을 요구한다.
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

```bash
.venv/bin/python scripts/eval/evaluate_recorded.py \
  --ckpt runs/finetune_tiny/ckpt/best.pt \
  --manifest data/manifests/recorded_train.jsonl --split test
```

## 4. 실기 평가 (덕트, 사용자 입회)

```bash
# 출력 장치를 열지 않는 선행 입력 게이트
.venv/bin/python scripts/bench/check_audio_input.py
.venv/bin/python scripts/demo/evaluate_session.py --controllers fxlms dl --scenarios tone300 multitone band nonlinear
```

프로토콜: 시나리오마다 **OFF 10s(베이스라인) → ON 30s → OFF 5s**, 게이트 램프 ±1~2s 는
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
  --confirm-volume-minimum
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
| G0 표현 학습 | 고정 batch overfit에서 trusted NMSE < −6dB, lead=109 메타 정합 | 모델/경사/정렬 파이프라인 결함 |
| G1 경로 실측 | 동일 하드웨어 게인에서 `P(z)`와 `S(z)` ESS 반복 일관성 ≥0.9; S 목표 80–1600Hz | surrogate 이외 물리 주장 불가 |
| G2 데이터 | 소스 family×대역 커버리지, 그룹 단위 8:1:1, 독립 recorded val/test | 누수·환경 암기 가능성 |
| G3 파인튜닝 | `digital_primary_path_mode: measured`, measured 70% + synth 30%, P/S/lead 스냅샷 보존 | 표현 사전학습 상태 |
| G4 독립 평가 | trusted 150–600Hz < 0dB와 fullband ≤0dB를 동시 통과, 소스·대역·최악 10% 병기 | 국소 개선 또는 대역 밖 증폭 |
| G5 Jetson/실기 | artifact lead fail-fast 통과, P99 <3ms, watchdog/xrun 기록, FxLMS와 동일 세션 비교 | 배포 성능 주장 불가 |

G0의 고정-batch 수치는 의도적 과적합 진단이지 일반화 성능이 아니다.
`make_recorded_manifest.py`는 group 원자성+source-family 층화 8:1:1을 제공하고,
`validate_recorded_sessions.py`는 family×split 커버리지와 파일 QA를 기본 치명 게이트로 검사한다.
도구 구현은 완료됐지만 실제 독립 세션을 수집해 PASS하기 전에는 G2/G4가 통과한 것이 아니다.

## 7. 리포트 양식

`results/eval_report_*.md` 표 + 다음 플롯(오프라인 도구 재사용):
ANC OFF/ON 스펙트로그램, PSD 오버레이(off/FxLMS/DL), 옥타브밴드 막대(신뢰 회색 표기).
캡스톤 보고서에는 시나리오 표 + 밴드 막대 + 물리 한계 요약(docs/01 §5)을 함께 실을 것.

## 8. Acoustic 런타임 녹음 진단 — Docker 오프라인 분석, 무출력

`scripts/eval/analyze_acoustic_session.py`는 장치·신경망을 열지 않고 런타임 NPZ를 분석한다.
실행 명령, 현장 OFF→ON→OFF 확보와 회수 목록은 [docs/14 §4-E·§5](14_pc_jetson_workplan.md)를 따른다.
이 도구의 양수 dB는 **서로 다른 시각에 관측한 ERR 감소량**이지 ANC의 인과적 효과 인증이 아니다.
외부 음원 변화·S/F 변동·입력 SNR은 별도 통제와 실측이 필요하므로 항상
`diagnostic_only=true`, `performance_claim_allowed=false`를 남긴다.

### 8.1 구간·대역·통계

- `anc_gain <= 0.001`은 OFF, `>= 0.999`는 ON이며 사이 값은 전환이다.
  각 ON마다 바로 앞뒤 OFF가 있어야 한다. 중간 OFF나 마지막 미완료 사이클을 합치거나 삭제하지 않는다.
- 기본 창은 1초, 초기 OFF guard 1초, ON 워밍업 2초, 양쪽 경계 guard 0.5초다.
  구간 앞 guard는 S의 `delay_samples + FIR 길이 − 1`보다 짧을 수 없다.
  저장 control/gain은 출력 callback 시각이므로 handoff를 중복 가산하지 않는다.
  창 미만의 말미 샘플 수는 보존해 보고하고, OFF 길이에 맞춰 긴 ON을 잘라내지 않는다.
- ERR/REF를 고정 길이 창별 **한쪽 FFT Parseval 에너지**로 계산한다. 저역은 `[0,1000)`,
  고역은 `[1000,Nyquist]`라 1 kHz 성분은 고역에만 속한다. DC는 저역/전체 대역에 포함한다.
  별도로 `target_800_1600=[800,1600)`, `target_800_1000=[800,1000)`,
  `target_1000_1600=[1000,1600)`을 보고한다. 1600Hz는 target에서 항상 제외한다.
  우선 개선 판정에는 `target_1000_1600`을 사용하되 저역·고역 전체와 나머지 두 지표도 함께 남긴다.
  `fs<3200`은 전체 우선대역 관측 불가로 거부하고, fs=3200의 기존 full/high Nyquist 포함은 유지한다.
  옥타브 경계 `f/√2 ~ f√2`도 FFT bin 적분이며, §1의 기존 Butterworth 4차 출력과 동일 지표가 아니다.
  유한 창의 스펙트럼 누설·주파수 분해능 한계가 있고 실제 마이크 SNR을 측정한 것은 아니다.
- ON과 비교할 기준 파워는 `min(mean(앞 OFF 창 파워), mean(뒤 OFF 창 파워))`다.
  관측 감소량은 `10 log10(기준 파워 / ON 평균 파워)`이며 양수 감소·음수 증가다.
  앞·뒤 OFF 각각과 비교한 값, OFF 간 변화량, REF 변화량도 따로 남긴다.
  실제 ERR에 S를 재적용하거나 `ERR − S*control`을 정답 d로 재구성하지 않는다.
  ON REF에는 F(CS→REF)가 섞일 수 있어 REF 정규화로 감쇠값을 만들지 않는다.
- 중앙값·p10·최악 창·**최악 10% 평균**은 별도 값이다. 최악 10% 평균은 ON 창 N개 중
  관측 감소량이 작은 `ceil(N*0.1)`개 평균이다. 무효 창을 버리고 분포를 좋게 만들지 않는다.
- `power_floor=1e-12`는 **수치 계산 바닥**이며 실측 마이크 noise floor가 아니다.
  분모·기준이 바닥 이하이면 감소량은 `null`이지 0dB·무한 감쇠가 아니다.
  기준이 바닥 이하인데 ON 창에 에너지가 생기면 `emergent_on_energy=true`와 파워 차이를 남긴다.
- `consistency_band_hz`와 반복 일관성 `>=0.9`가 있어야 검증 대역을 표시한다.
  **밴드 전체 경계가 검증 대역 안에 있어야** `trusted=true`이며 중심 주파수만으로 판정하지 않는다.
  예를 들어 150–600Hz 검증에서 500Hz 옥타브(약 354–707Hz)는 미검증이다.
  가진 대역으로 대체하거나 미검증 고역 행을 삭제하지 않는다.

### 8.2 기록 추적성과 보고 한계

새 runtime 녹음은 scalar `recording_schema_version=1`, JSON 문자열 `recording_meta_json`을 저장한다.
S 해시, 모드·샘플레이트·블록·handoff·채널·출력 제한 등의 주요 조건이 분석 설정과 다르면 거부한다.
녹음 당시 S 원본과 전체 실행 설정도 별도로 회수한다. 메타는 모든 DNN/FxNLMS 파라미터나
모델 artifact의 전체 스냅샷은 아니며 `recording_context_verified`는 저장된 필드의 일치만 뜻한다.
메타 없는 구형 파일은 사용자 제공 설정을 사용하되 `legacy_recording_context_unverified`를 표시한다.

`runtime_health`는 시작·종료 처리를 포함한 **전체 실행 누적치**다. ring drops는 폐기 샘플 수,
xrun·underrun은 카운터, fatal_error는 bool이다. 구간별 장애·적응/리미터 이력은 아직 없으므로
0 카운터만으로 선택한 ON 창이 건전하다고 인증하지 않는다. 구형 녹음의 누락값은 `null`이다.
저장 입력의 `abs(x)>=0.98` 비율은 가공된 녹음의 clipping **대용 지표**이며 raw ADC clip을 대체하지 않는다.
비선형성·위상/클록 안정성·저역/고역 전체 목표·quiet zone 성공은 이 보고서만으로 판정하지 않는다.

산출물은 전체 JSON·대역별 CSV·요약 Markdown이며 유효 창이 있을 때만 창 파워 CSV도 만든다.
계산 불가 사이클/대역은 CSV·요약에 `null` 행으로 남긴다. 기존 출력은 덮어쓰지 않는다.
exit 0은 모든 사이클의 구간 완전성과 **최소 한 사이클**의 전체 대역 계산 가능, exit 2는 불완전/계산 불가,
exit 1은 입력·설정·I/O 실패다. exit 0이어도 신뢰대역 미달·음원 변화·증폭이 있을 수 있다.

## 9. 사전 FIR 연구와 ESS 대역 진단의 별도 해석

[docs/15](15_prepared_fir_research.md)의 사전 FIR 벤치마크는 합성 train/heldout을 분리한 구조 시험이다.
저역·우선대역·대역 밖, 초기/조건 변화/후기와 모든 대조군을 보존하되 실측 감쇠·실시간성은 주장하지 않는다.
S 모델 반복 진단의 `|mean(H)|²/mean(|H|²)`는 기존 전체 IR Pearson 일관성과 다른 값이다.
그 값이 0.9를 넘더라도 `consistency_band_hz` 승격·클록 안정성·SNR·비선형 모델 검증으로 쓰지 않는다.
경로 진단 CLI의 기본 target은 끝점을 포함한 [1000,1600]이며, §8 ERR 에너지 분할의
[1000,1600)와 구분한다. Python API의 호환 기본 [800,1600]과 과거 보고서를 새 대역 결과로 고쳐 쓰지 않는다.
