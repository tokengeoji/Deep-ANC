# HANDOFF — 세션 인수인계 (다음 AI 에이전트/개발자용)

> **"이어서 진행해줘"를 받았다면**: §0 라이브 상태 → §0.5 다음 단계 순으로 실행하라.
> 규칙은 [AGENTS.md](AGENTS.md)가 단일 출처. 이 파일은 작업 상태가 바뀔 때마다 갱신할 것.
> 최종 갱신: **2026-08-06**

## 0. 라이브 상태

**현재 브랜치: `fix/finetune-readiness-repair`** (main 에서 분기, 아직 merge 안 됨).
테스트 **600개 전부 통과** (`.venv/bin/python -m pytest -q`, 약 14분).
Elice 인스턴스는 **사용자가 삭제**했고 산출물은 전부 로컬 회수됨.

### ⛔ 지금 배포·학습을 재개하면 안 되는 이유 (한 줄 요약)

**파인튜닝은 G4 FAIL 했고(`music` val +0.58 / test +0.90), 그 학습에 쓴 데이터 80세션은
시간축이 붕괴해 전량 격리됐다. 재녹음이 유일한 블로커다.**

### ⚠ 2026-08-05 세션에서 밝혀진 것 — 앞선 결론 다수가 틀렸다

이전 HANDOFF/README/docs 의 다음 서술은 **전부 틀렸고**, 저장소 문서에 아직 남아 있다(§0.5 참조):

| 틀린 서술 | 실제 (검증됨) | 문서 반영 |
|---|---|:---:|
| "600 Hz 위는 S(z) 가 재현 안 됨 = 덕트 물리 한계" | **측정 후처리 결함.** 오염 반복을 버리니 신뢰대역이 **150–1600 Hz** | ✅ 2026-08-06 |
| "−2 dB 정체 = 용량 부족" | **아니다.** 학습 데이터 시간축 붕괴 + 플랜트 오차 | ✅ |
| "1 kHz 위는 변하지 않는다" | **2–8 kHz 를 15–22 dB 증폭한다** (`metrics.csv` 실측) | ✅ |
| "파인튜닝으로 1.30 dB 개선" | **무효.** 전후가 서로 다른 플랜트 (S 1342/lead 109 vs 1465/113) | ✅ |
| "재생·녹음이 비동기 클록이라 드리프트" | 상대 드리프트 **+0.4 ppm**(10분 12샘플). 실제는 출력 버퍼 **프레임 슬립** | ✅ |
| "max 46–54 ms 는 데스크톱 잡음" | `kernel.sched_rt_runtime_us=950000` **RT 스로틀링** (1초 주기 정확히 50 ms) | ✅ |
| GPU 지연 벤치 수치 | **연속 실행(듀티 100%)** 값이다. 실제 듀티 6% 에서 거버너가 306 MHz 고정 → P50 0.30→**1.10 ms** | ✅ |
| "TensorRT 는 python 바인딩 미설치 → 사용 불가" | `import tensorrt` **10.3.0 정상**, FP16 plan 빌드·실측 완료. `trtexec` 만 PATH 에 없다 | ✅ |
| "전수 QA 80/80 PASS" | 형식만 봤다. 새 QA 로는 **0/80 PASS** — 전량 격리됨 | ✅ |
| "진입 게이트 9개 전부 PASS" | **초록불 9개가 아무것도 보증하지 못했다** | ✅ |
| README 의 "skip+residual 분기" | 코드에 **skip 분기가 없다** (`return x + project(u·σ(g))`) | ✅ 그림도 재생성 |
| `runtime.yaml` 의 `model.onnx` / `model_fp16.plan` | **존재하지 않는 파일**이었다 (현재 실재 경로로 수정됨) | ✅ |
| "테스트 352개" | **600개** | ✅ |

또한 **파인튜닝은 G4 FAIL 했다** (`music` 계열, val +0.58 / test +0.90). 어느 문서에도 없었다.
지금은 README §1.3 · §7.6, docs/12 §4.4 에 명시돼 있다.

> **`configs/eval*.yaml` 3곳의 `trusted_band_hz: [150, 600]` 은 아직 남아 있다.**
> 어떤 코드도 읽지 않는 **죽은 설정**이라 동작에는 영향이 없지만(전수 grep 확인),
> 권위 있어 보여서 다음 사람을 속인다. 제거하거나 읽히게 하거나 둘 중 하나로 정리할 것.

### 이번 세션(2026-08-05~06)에서 완료한 것

1. **플랜트 복구 완료** — 저장된 원시 캡처 11건을 오프라인 재분석(스피커 0회).
   신규 `scripts/data/reanalyse_paths_interleaved.py`. 기존 npz 는 `.orig` 로 백업.
   채택 캡처 `225546_f7b0fecd` (48반복 중 30기각·18유지·앵커 13).
2. **측정 게이트 신설** — **P−S 상대 τ 연속성**(두 채널이 같은 출력 스트림이므로 상수여야 함)과
   타임베이스 드리프트 검사. 기존 게이트는 `delay_spread 32` 를 허용치 48 과 비교해 **통과시켰다**.
   그 외 15개 게이트 신설, 전부 **negative fixture 와 짝** (`ops/gate_registry.py`, 총 72개).
3. **파레토 분석** — 결함 18건을 분류하니 **14건(78%)이 발생기 2개**에서 나왔다(§0.7).
4. **신규 결함 발견** — 실측 music 60트랙 중 **55개(92%)가 합성 학습 스트림에 중복**(§0.6 D1).
5. **recorded 시간축 결함의 발생기 수정** — `record_duct.py` 가 출력/입력 커서를 "같은 물리
   시각"이라고 **단언**하던 것을 **측정**으로 바꿨다(신규 `src/deep_anc/data/timeline.py` 단일
   출처). 저장 시점 게이트 2종 신설. 80세션은 되돌릴 수 있게 **격리**
   (`data/recorded_broken/`, `quarantine_ledger.json`).
6. **손실 재설계** — 대역 밖 do-no-harm 힌지(신뢰대역 여집합에서 **자동 유도**, 리터럴 목록 없음),
   집계를 산술평균 → **CVaR**, 죽은 clip 항을 포화 페널티로 교체, `loss_start_sample` 을
   학습·평가 **단일 출처**(`PlantSettle.derive`)로. ⚠ λ 미교정 — §0.6 참조.
7. **런타임 안전** — 출력 DC 차단기 + 워치독 7종을 열거 가능하게 만들고 **각각 실제로
   발동시키는 테스트**를 붙였다. 스트릭 카운터 → 슬라이딩 윈도(교대 미스를 영원히 못 잡던 결함).
   ⚠ 오발동 위험 2건 미해결 — §0.6 참조.
8. **문서 전면 정정** (2026-08-06) — README / docs 00~12 / HANDOFF 의 틀린 서술 교체,
   그림 재생성(`fig1_system`, `fig3_tcn_block`, `duct_layout`, `fig0_duct_3d` 가 설정에서
   자동 갱신됨).
9. `pydantic 2.13.4` 도입(검증 1.4 µs/건 실측), `AGENTS.md` 에 `~/DeepANC_CRN_n_codex`
   읽기 전용 + **오디오 장치 점유 확인 의무** 추가.

### 플랜트 복구 결과 (확정 수치 — 이 값을 쓸 것)

```
lead = 116 샘플  (= S 1462 + handoff 256 − P 1602)   ← 이전 113
P 벌크지연 1602 / S 벌크지연 1462 / P−S = 140
d_noise_delay_samples = 1602
신뢰대역 = 150–1600 Hz   ← 이전 150–600 (2.67배 확대)
```

**독립 캡처 9건 전부에서 P−S = 139~141, lead = 115~117 (중앙 116).**
⚠ **절대 지연은 재현되지 않는다** (low-latency 1565~1659 / high 2858~2888, 드리프트 364~729 ppm).
**P−S 만이 물리 불변량이다.** S 와 **같은 캡처·같은 앵커** 값만 함께 써야 한다
(아티팩트에 `anchor_repeat`, `kept_repeat_indices` 기록됨).

재생성 전후 일관성 (같은 캡처로 파이프라인 효과만 분리):

| 대역 Hz | P 전 → 후 | S 전 → 후 |
|---|---|---|
| 150–300 | 0.996 → 0.999 | 0.964 → **0.998** |
| 300–600 | 0.961 → 1.000 | 0.970 → **1.000** |
| 600–1000 | 0.895 → 0.999 | 0.837 → **0.999** |
| 1000–1600 | 0.752 → 0.999 | 0.737 → **0.999** |
| 80–150 | 0.868 → 0.910 | 0.748 → 0.706 ← **진짜 물리 한계** |

**출하본 S 의 형상 오차 54.1%** (P 는 17.0%). 채택 캡처는 `225546_f7b0fecd`
(leave-one-out 독립 캡처 정확도 최선 S 2.54%, 유지 반복 18개. 이전 출하 캡처 03f4c088 은
7건 중 **최악** 8.21%였다). 되돌리려면
`reanalyse_paths_interleaved.py <세션> --write --overwrite`.

**이론 상쇄 상한 (이 플랜트, lead=116)**: 150–600 Hz **−6.53 dB** / 150–1600 Hz −4.77 dB.
플랜트 불확실성 비용은 0.09~0.19 dB 로 **플랜트는 더 이상 병목이 아니다**.
**출하 npz 로 설계한 필터를 클린 플랜트에 적용 = −0.54 dB** (올바른 설계 −6.53) —
고역 증폭과 −2 dB 정체의 플랜트 측 원인이 이것으로 확증됐다.

## 0.5 다음 단계 ("이어서 진행해줘" 는 여기부터 위에서 아래로)

**파인튜닝을 지금 재개하면 또 낭비다.** S(z) 는 고쳤지만 **학습 데이터가 그대로**다.

---

### ⛔ 1. 사용자가 먼저 해야 한다 — 마이크 입력단이 죽어 있다 (하드웨어)

**녹음 전 마이크 상태 실측 결과 두 채널 모두 int32 풀스케일에 붙어 있다.**

```
ch0  레일 비율 0.0475  RMS 0.3403      ← 정상 세션은 peak 0.0063 / RMS 0.00107
ch1  레일 비율 0.1033  RMS 0.4020      ← 정상 세션은 peak 0.0390 / RMS 0.00830
raw int32 min/max = -2147483648 / 2147483647  (양쪽 레일 도달)
지배 성분 6/12/18/23 Hz, 12초 내내 정상상태 — 기동 트랜지언트 아님
```

장치 점유는 없었다(`fuser`: pulseaudio 컨트롤 노드만). 즉 **물리적 무전원/미연결**이다.
이 상태로 재생하면 QA `max_clip_ratio=0.005` 에 10~20배로 걸려 전량 폐기되므로
AGENTS 규칙 3("소리를 내며 디버깅하지 않는다")에 따라 **재생하지 않았다.**

**이 결함 자체가 게이트 공백이었다** — 기존 자가진단은 하한(`ref_db < -80`)만 봤고 상한이
없어 "레일에 붙은 마이크"를 오히려 아주 살아 있는 마이크로 통과시킨다. 상한 게이트를
추가했고 실제로 막는다(`[중단] 마이크 입력이 풀스케일에 붙어 있습니다 … EXIT=1`, 재생 전).

**사용자 조치**: 마이크 프리앰프/입력단 전원과 배선을 연결한 뒤 아래를 순서대로 실행.

```bash
# (a) 배선 확인 — 소리 안 남. 여기서 EXIT=0 이 나와야 (b)로 간다
fuser -v /dev/snd/*                       # 장치 점유 확인 (필수)
.venv/bin/python scripts/data/record_duct.py --program silence --seconds 3 \
    --out-root /tmp/rec_check
# 기대: '레일 비율 0.0000/0.0000', ch1(ref) 가 -80 dBFS 보다 큼

# (b) 검증 녹음 25초 — 스피커 우는 시간 25초
.venv/bin/python scripts/data/record_duct.py --program band --band 80 1600 \
    --amplitude 0.06 --seconds 25 --source-family environment \
    --group-id environment-verify --out-root data/recorded_verify
# 기대: coh²(source→ERR,150-600Hz) 0.0xx → 0.9xx   ← 임무 완료 기준
#       [중단] 이 뜨면 저장되지 않는다 (실패-폐쇄)
```

---

### 2. recorded 80세션 재녹음 — **유일한 실질 블로커** (§0.6 결함 2)

`record_duct.py` 의 시간축 결함은 **고쳤다**(측정으로 대체 + 저장 시점 게이트).
오프라인 재정렬은 24세션 표본에서 **16/24 (66.7%)** 만 복구되므로 80세션 환산 약 53세션
= 게이트(`≥80세션 그리고 ≥90분`) 미달이다. **게이트를 낮추지 않으므로 재녹음이 정답이다.**

```bash
.venv/bin/python scripts/data/record_session_batch.py --confirm-speaker --amplitude 0.06
# 계열별 분할도 가능: --families environment machine  /  --families music speech
.venv/bin/python scripts/data/make_recorded_manifest.py
.venv/bin/python scripts/data/validate_recorded_sessions.py     # 기대: 80/80 PASS
```

**예상 소요 약 118분** (스피커 실제 발음 93.3분 + 세션마다 재정렬 ~11초).
세션마다 즉시 판정하므로 2시간 뒤에 전량 폐기를 발견하는 일은 없다.
⚠ 재생 전 **오디오 장치 점유 확인 필수** (`~/DeepANC_CRN_n_codex` 병행 작업 중).

---

### 3. 손실 λ 교정과 게이트 임계 대조 — **재학습 전에 반드시** (§0.6 결함 3)

do-no-harm 항 자체는 들어갔지만 **세 가지가 남아 있다.**

- **λ_dnh 가 새 대역 구성에서 재교정되지 않았다.** 실측 그래디언트 비 **1333%**(목표 20~40%).
  `sat` 162% / `frame` 46% / `mrstft` 27%. 20k step ablation 전에 한산할 때 다시 재라.
- **힌지 마진(`+6.0 dB`)과 G4 임계(`−1.0 dB`)가 서로를 모른다.** 마진을 정확히 만족하는
  모델이 게이트를 옥타브 전 대역에서 **8~9 dB 차이로 FAIL** 한다(직접 실행 확인).
  힌지는 `|S·y|²/|d|²`, 게이트는 `e/d` — 물리량이 다른데 대조 코드도 테스트도 없다.
- **출하 `alpha=0.7` 은 배분을 뒤집지 않는다.** 최악 4개 몫 0.17% → 1.7% 로 10배 늘 뿐,
  최상 4개가 여전히 **19배**를 가져간다. 초기 5k step 분산을 보고 0.85~1.0 으로 올릴지 판단.
- `configs/train_finetune.yaml` 이 `band_weight: curriculum_a`(대역 밖 **양측** 압력) 인 채로
  단측 힌지를 함께 걸고 있다 — 설계가 "서로 상쇄한다"고 못박은 구성이다. `trusted_only` 로
  바꾸고 폐기 키(`lambda_pow` / `lambda_clip` / `clip_margin`)를 지울 것.

---

### 4. 코퍼스 누수 마무리 (§0.6 D1) — 재녹음 불필요, 매니페스트 작업만

held-out 목록은 생성됐다(`data/manifests/recorded_holdout.json`, 691 클립)고
`prepare_noise_pool.py --holdout` 이 구성 단계에서 차단하는 것까지 확인했다.
**남은 것**: manifest 격리 때문에 실데이터 양성 확인이 미완. 그리고 `music` 의 `group_id` 를
FMA 트랙 ID 버킷 → **아티스트/앨범**으로 바꾸려면 `fma_metadata`(tracks.csv, 342MB)가 필요하다.

---

### 5. 남은 발생기 제거 (§0.7) — 이걸 안 하면 같은 결함이 또 나온다

전부 **직접 실행으로 확인**했고 지금 저장소에 살아 있다. 상세는
[docs/12 §5.4](docs/12_system_summary.md#54-남은-발생기--다음-세션이-반드시-처리할-것).

| # | 발생기 | 왜 급한가 |
|---|---|---|
| 1 | 신뢰대역 유도식이 **5곳에 복붙** (`trainer.py:205`, `eval/recorded.py:216`, `evaluate_offline.py:97`, `evaluate_session.py:175`, `render_anc_demo.py:159`) | `duct.yaml` 을 바꿔도 5곳이 따라온다는 보장이 없다 |
| 2 | `intersect_frequency_bands` **두 번 정의** (`losses/anc_loss.py:28`, `eval/metrics.py:31`) | 같은 이름이 다르게 동작 |
| 3 | `configs/eval*.yaml` 의 `trusted_band_hz: [150,600]` **죽은 설정** | 권위 있어 보이는데 아무 코드도 안 읽는다 |
| 4 | `measured_design_ceiling_db: 6.53` 이 **대역 표시 없는 손 float** | 요구 대역 [150,1600] 상한은 4.77, 독립 재계산은 5.21~5.41 — **셋이 다르다.** 키를 지우면 `inf` 로 fail-open |
| 5 | source→ERR 지연 궤적이 **두 벌** (`data/timeline.py` vs `dsp/invariants.py`) | 같은 세션에 std 1.8 vs 1107 로 **정반대 판정**. realign-PASS 세션의 27~44% 를 QA 가 오검출 기각 |
| 6 | 게이트 메타 테스트가 **위양성을 강제하지 않는다** | 워치독 반응이 전부 mute(= 상쇄 0 dB = 최악값)라 오발동이 곧 성능 0 |

**해야 할 것**: `Lead.derive(...)` / `BandPlan.resolve(...)` 로만 만들 수 있게 하고(pydantic
frozen, 손으로 생성 시 `TypeError`) 나머지는 읽기만. **손실 대역(optimize)과 보고 대역
(measure)을 타입으로 분리** — 보고 대역이 좁으면 절대목표 1 을 검증할 방법 자체가 없다.
그리고 `duct.yaml` 값을 바꿨을 때 5곳이 전부 따라오는지 **테스트로 강제**하라.

---

### 6. 런타임 안전 — 오발동 2건 (실기 전에 반드시)

- **OUTPUT_DC 워치독이 영평균 저역 상쇄음을 DC 로 오인한다.** 이동평균 창이 10블록(53 ms)
  이라 80 Hz 를 DC 와 구분 못 한다: 60 Hz 진폭 0.12 → 10블록 mute, 80 Hz 진폭 0.18 →
  12블록 mute. 80 Hz 는 `realistic_target_band_hz` **안**이고, 실측 세션의 53 ms 이동평균
  최대치가 이미 한계의 **57.5%** 다. **저역 상쇄를 키우는 것이 절대목표 1 인데 키울수록
  이 워치독에 걸린다.** 오기각 방지 테스트가 375 Hz(블록당 정확히 2주기) 하나뿐이다.
- **fail-closed 발산 워치독**: 조용한 방에서 잡은 마이크 플로어가 유효 베이스라인으로 굳고,
  외부 소음원을 켜는 순간 **123 ms 만에 mute** 된다(종단 재현 확인). 이 경로에 테스트가
  하나도 없다 — 기존 테스트는 `baseline_power` 를 전부 손으로 주입한다.
- `configs/runtime*.yaml` 의 폐기 키(`clip_streak_mute` / `deadline_miss_mute`)가 남아 있다.
  런타임이 시작 시 안내하고 무시한다. 새 임계값은 전부 코드 기본값을 쓰는 상태다.

---

### 7. 무료 기하 개선 (0 샘플 비용, 13–17 dB 이득)

ERR 마이크를 CS 에서 **100 mm 이상** 떨어뜨려 **측벽 중앙높이**에 달아라.
`H = D_e − t(REF→CS)` 항등식상 **ERR 위치는 H 에 정확히 0 의 영향**이고, CS 근접장/(1,0)
모드 오염만 사라진다. ⚠ **미검증**: `configs/duct.yaml` 의
`positions_m.error_mic_cross_section_m` 이 아직 없다. 벽면 장착으로 판명되면 S 신뢰 상한을
1200 Hz 로 낮춰야 한다(실측 `|S|` 가 1448 Hz −10.1 dB → 1640 Hz −25.0 dB 로 하강하는 것이
벽면 강결합을 반증하는 정황이지만 직접 확인은 아니다).

---

### 8. 하지 말 것

- **`tiny_wide` 용량 실험** — 근거가 철회됐다(§0.6). 재녹음 이후에 다시 판단한다.
- **`runs/finetune_tiny` 결과를 현재 플랜트 성능으로 인용** — 그 학습은 S 1465 / lead 113
  (폐기된 오염 아티팩트)에서 나왔다.
- **게이트 임계 완화** — 절대 금지. 재녹음이 게이트를 못 채워도 게이트를 낮추지 않는다.

## 0.6 결함 대장 (전부 검증됨 — 상태 표기 2026-08-06)

| # | 결함 | 상태 |
|---|---|---|
| **1** | 학습에 쓴 `S(z)` 형상이 **54% 틀림** (오염 반복 5개를 게이트가 통과) | ✅ **수정** — 재발행 + 게이트 신설 |
| **2** | recorded 80세션 **시간축 붕괴** | ⚠ **발생기는 고침, 데이터는 재녹음 필요** |
| **3** | 대역 밖 **2–8 kHz 를 15–22 dB 증폭** | ⚠ **부분** — 손실 항 추가, λ 미교정 |
| **D1** | 코퍼스 누수 — 실측 `music` 60/60 이 합성 풀에 존재 | ⚠ **부분** — held-out 생성, 게이트 양성 확인 미완 |
| **D2** | 실측 지연이 설정과 ~250 샘플 어긋남 | ⚠ 교차검증 게이트 신설됨. 재녹음 후 재판정 |
| **D3** | G4 판정 자체가 통계 해상도 미달 | ⚠ 미해결 — 녹음 보강 필요 |
| **D4** | `music` `group_id` 가 FMA 트랙 ID 버킷 | ⚠ 미해결 — `fma_metadata` 필요 |
| **D5** | 크레스트 제한 10 dB 가 계열 차이를 지움 | ⚠ 미해결 — 권고안 아래 |
| **RT** | 런타임 안전 워치독 | ✅ 7종 수정 / ⚠ **오발동 2건 신규** |

---

**결함 1 — `S(z)` 가 33~54% 틀려 있었다 (✅ 수정).**
`alignment_scores` 반복 11–15 가 0.750–0.758 로 별도 무리인데 기각 임계 0.5 라
`rejected_repeats: 0`. 결정적 증거는 **P−S 상대 τ** — 같은 DAC·같은 출력 스트림 인터리브라
상수여야 하는데 반복 11 에서 **1.4 → 32 샘플 점프**(출력 버퍼 프레임 슬립). 게이트는 요약
스칼라 `delay_spread 32` 를 허용치 48 과 비교해 **통과시켰다.**
재발행 후 전대역 S 0.781 → **0.998**, 형상 오차 **P 17.0% / S 54.1%** 였음이 확인됐다.
**출하 npz 로 설계한 필터를 클린 플랜트에 적용 = −0.54 dB**(올바른 설계 −6.53 dB).

**결함 2 — 실측 녹음 80세션의 시간축 붕괴 (⚠ 재녹음 필요).**
```
coh²(source.wav → ERR mic) = 0.021 ~ 0.126   ← 학습이 배워야 할 관계
coh²(REF mic  → ERR mic)   = 0.959 ~ 0.991   ← 음향 자체는 멀쩡 (같은 I²S 클록)
source→ERR 지연 표준편차    = 248 ~ 4813 샘플
```
창별 최적정렬 후 coh²: 1.5s 0.430 / 0.5s 0.541 / **0.1s 0.745** / 25ms 0.518 —
창을 줄여도 0.75 에서 멈춘다. **느린 드리프트가 아니라 빠른 위상 점프**이므로
**일정 ppm 리샘플 dewarp 는 듣지 않는다.**
원인: `record_duct.py` 가 `sd.Stream(device=(in_dev, out_dev))` 로 서로 다른 두 장치를
duplex 로 묶고 콜백에서 출력커서와 입력커서를 **인덱스로만** 정렬 — "같은 물리 시각"이라고
**단언**하고 측정하지 않았다.

**✅ 발생기는 고쳤다**: 정렬 수치를 신규 `src/deep_anc/data/timeline.py` 에서만 유도하고
`record_duct` / `recorded_qa` / 재정렬 스크립트가 전부 그것을 호출한다. 저장 시점 게이트 신설.
**⚠ 데이터는 재녹음해야 한다**: 오프라인 재정렬은 24세션 표본에서 **16/24 (66.7%)** 복구
(coh² 0.012~0.138 → 0.817~0.971, 선형 Wiener 하한 −0.23 → **−12.09 dB**) — 80세션 환산
약 53세션이라 게이트 미달. 80세션은 `data/recorded_broken/` 으로 **격리**(되돌릴 수 있음).

> **D7 판정 (전문가 충돌 해소).** 데이터 전문가의 "일정 ppm 리샘플 dewarp" 가설은
> **기각**한다. 전역 기울기를 빼도 오차의 **99~102% 가 남고**, 잔차 파워의 83~98% 가
> 0.1–0.5 Hz 대역에 있다(주기 3.95~5.33 s). **결정론적 주기 헌팅**이며 연속 시변 워프가
> 이것을 처리한다.

**결함 3 — 고역 증폭 (절대목표 1 정면 위반, ⚠ 부분 수정).**
`results/session_20260804_0939/metrics.csv`: tone300 이 1k **−16.84** / 2k −15.42 /
4k −18.03 / 8k **−21.56** dB (음수 = 증폭). 손실에 대역 밖 항이 **없고** 게이트에만 있었다.
✅ 대역 밖 do-no-harm 힌지 추가(신뢰대역 여집합에서 자동 유도, 단측·위상 무관).
⚠ **남은 것은 §0.5-3 참조** — λ 미교정(비 1333%), 힌지 마진과 게이트 임계 불일치(8~9 dB),
출하 `alpha=0.7` 이 배분을 뒤집지 못함.

**D1 — 코퍼스 누수 (⚠ 부분).** 실측 music 60트랙 **전부(100%)** 가 합성 풀 원본에 있고,
결정적 split 재현 결과 **55개(92%)가 합성 train** 에 있다. speech/machine/environment 는 **0%**.
같은 오디오에 **상충하는 정답**이 간다(합성 −18 dB 가능 / 실측 −0.4 dB 천장).
✅ `corpus_disjoint` 게이트 + held-out 목록(691 클립) + `prepare_noise_pool.py --holdout`.
⚠ manifest 격리로 실데이터 양성 확인 미완.

**D2 — 실측 지연이 설정과 ~250 샘플 어긋난다.** 독립 세 방법 일치(포락선 1672 / 반송파 1663±73 /
스윕 ~1670) vs 설정 유도 ~1950. 비용 계열별 **+0.71 ~ +2.39 dB**.
✅ `measured_source_delay_agreement` / `invariant_measured_delay_agreement` 게이트 신설
(실제 아티팩트에서 1672 vs 1849, 차이 177 > 허용 64 를 검출하는 것 확인).
⚠ 재녹음 후 재판정 필요.

**D3 — G4 판정 자체가 해상도 미달.** 계열 내 그룹 간 잔차 SD(pooled) **1.46 dB** →
그룹 2개 계열의 평균 SE **1.03 dB** > 계열 간 폭 **0.92 dB**.
music val 두 그룹이 자기들끼리 **2.96 dB** 벌어져 있고(−0.99 vs +1.97), machine val 은
그룹 1개라 SE 추정 불가. **music val = 2세션 × 3클립 = 곡 6개.**
**"music 이 최악" 은 통계적 근거가 없다.** ✅ G4 가 3값(PASS/FAIL/**INCONCLUSIVE**) 판정 +
그룹 부트스트랩 CI 를 내도록 바뀜 — 판정 불가는 통과가 아니다.

**D4 — music `group_id` 가 FMA 트랙 ID 버킷**이라 누수 방지 기능을 못 한다(아티스트/앨범 아님).
실측: 세션/그룹 비율이 music **1.00**, speech **1.00**(감사 미언급), machine 2.50,
environment 1.25 — **music 과 speech 는 그룹화 기능이 아예 없다.**
`data/raw/music/fma_small` 에 README.txt 뿐이고 `fma_metadata`(tracks.csv, 342MB)가 없다.

**D5 — 크레스트 제한 10 dB** (`build_recording_sources.py`) 가 네 계열을 **9.61–9.98 dB
(폭 0.37 dB)** 로 균질화해 **music 을 측정 가능한 모든 축에서 가장 쉬운 신호로** 만들었다.
현장 소음은 15–25 dB.
⚠ **감사의 "3~5배 올려도 안전(22 dB 여유)" 은 틀렸다** — 클리핑하는 것은 소스가 아니라
REF 마이크이고 실측 12세션 최악 peak 0.2123 = **13.4 dB** 여유다.
```
레벨 5배(+14dB) — 감사안   REF peak 1.064  여유 -0.6 dB ← 클리핑
크레스트 15dB + 레벨 +3dB  REF peak 0.533  여유 +5.4 dB ← 권고
```

**D6 — machine 이 8그룹인 것은 물리 한계가 아니라 선택이다.** `ESC50_MACHINE` 이 ESC-50
50개 중 24개만 쓰고 **26개를 버렸다**(siren, car_horn, clock_alarm, can_opening,
keyboard_typing, mouse_click, door_wood_knock/creaks, glass_breaking 등 9종이 기계·도시음).
게다가 **MIMII(팬/펌프/밸브/슬라이더)와 DEMAND 가 이미 합성 풀에 있는데 실측 machine
수집에 전혀 쓰이지 않았다.** 재녹음 시 여기서 그룹 수를 늘릴 수 있다.

**런타임 안전 (✅ 7종 수정 / ⚠ 오발동 2건 신규).**
수정된 것: S1 클립 워치독이 DL 경로에서 죽은 코드였던 것(모델이 `0.2*tanh` 라 `|y|>0.2`
불가능 → 포화율 + RMS 상한으로 교체) / S2 데드라인 워치독이 교대 미스를 영원히 못 잡던 것
(스트릭 → 슬라이딩 윈도, 교대 미스 47블록에 mute) / S3 발산 워치독이 `baseline_power==0`
이면 조용히 비활성이던 것(**fail-closed** 로 전환) / S6 NaN 을 조용히 삼키던 것 /
S7 **출력 DC 보호 신설**(DCBlocker r=0.9995, 150 Hz −0.0006 dB) / 입력 백로그 8 hop(42.7 ms)
→ 1 hop 대칭화 / `runtime*.yaml` 의 존재하지 않는 plan·onnx 경로.
**⚠ 신규 오발동 2건은 §0.5-6 참조** — 실기 전에 반드시 처리할 것.

## 0.7 파레토 — 결함은 같은 곳에서 반복된다 (사용자 지시)

커밋 이력 + 이번 발견 **18건**을 분류한 결과:

| 군집 | 건수 | **공통 발생기** |
|---|---:|---|
| **A. 두 도메인 간 시간 정렬 부기** | 9 | 같은 물리량(지연/lead/대역)을 **여러 곳에서 따로 유도**하고 대조 안 함 |
| **B. 실패해본 적 없는 게이트** | 5 | 게이트가 "통과"를 주장하는데 그 주장이 **반증된 적이 없음** |
| C. 측정 없는 성급한 결론 | 4 | TensorRT 기각 / 용량 부족 / 600Hz 한계 / 클록 드리프트 — 전부 정정됨 |

**A+B = 14/18 (78%)**. 실측: 지연 산술을 독립 수행하는 파일이 **13개**
(`eval/recorded.py` 35회, `train/finetune_readiness.py` 31, `bench/measure_duct_transfer_map.py` 20,
`train/trainer.py` 17, …).

**해야 할 것 (증상 수정보다 우선):**
1. **지연·정렬 부기의 단일 출처** — 한 곳에서만 유도하고 나머지는 읽기만.
   `Lead.derive(s_delay, handoff, p_delay)` 로만 만들 수 있게 해 손으로 못 쓰게 하라(pydantic).
2. **교차 도메인 불변식 검사기** — P−S 상대 τ 상수성 / `coh²(재생→마이크)` / 플랜트 지문 일치 /
   lead 유도값 일치. 측정·QA·게이트·런타임이 **같은 코드**를 호출해야 한다.
3. **실패 증명 없는 게이트 금지** — 게이트를 열거해 **FAIL 시키는 fixture 가 없으면 실패하는
   메타 테스트**. 게이트 9개가 전부 PASS 인데 전부 무용지물이었던 사고의 유일한 구조적 방어다.

## 0.8 acoustic-reference 실배포 가능성 (2026-08-05 분석, 부분 완료)

사용자 질문 *"예측지평 33 ms 때문에 상한이 0 dB 라는데 샘플을 늘리면 되지 않나?"* 에 대한 분석.

**핵심 항등식**: `H = D_e − t(REF→CS)` — **ERR 마이크 위치는 H 에 영향이 0.**
레버는 오직 **REF→CS 거리**와 **전기 지연 D_e** 다.

```
루프지연 L 대 광대역 상한 (150–600 Hz, 실측 Wiener)
L=140:  −37.83 dB    L=256: −4.15    L=512: −0.21    L=757: −0.02    L=1718: −0.00
        └── 절벽이 L=256~512 사이 ──┘
```

| 시나리오 | D_e (샘플) | 필요 REF→CS (H≤128) |
|---|---:|---|
| 현재 (USB 256/low, 3스레드) | 1714 | 11 m 이상 (비현실적) |
| 핸드오프 0 만 | 1470 | 9.6 m |
| **APE I²S 128 + 핸드오프 0** | **750** | **4.4 m** |

- **코덱/앰프/마이크 고정분 = 479 샘플(9.98 ms)** — 버퍼를 24 ms 흔들어도 잔차 9.0~11.7 ms 로
  일정하다(5개 설정 교차검증) = **소프트웨어로 접근 불가, 하드웨어 교체만**.
- 실측 소스풀 80개(64–1600 Hz)에서 H=1581 일 때 중앙 −0.22 dB / **최악 −0.02 dB**.
  H=140 이면 중앙 −4.75 / 최악 −1.41.
- **대역폭이 결정한다**: H=1609 에서도 BW 1 Hz **−17.09 dB** / 5 Hz **−33.65** / 20 Hz −6.41 /
  450 Hz −0.02. **팬·모터 같은 준주기 소음은 지금 지연으로도 된다. 음성·음악·광대역이 죽는다.**
- **feedback ANC 는 답이 아니다** — 루프지연이 그대로 예측지평이 되어 feedforward 보다
  음향선행 140 샘플만큼 **더 나쁘다**(실측 차이 0.01 dB). 게다가 Bode 적분이 대가를
  보호대역 안쪽에 **+4.8~+18.8 dB** 로 되돌려 절대목표 2 와 구조적으로 양립 불가.
- ⚠ **미완**: 신호예측가능성·비선형상한 두 각도와 적대적 검증이 중단됨.
  스크립트 `.claude/.../workflows/scripts/acoustic-ref-feasibility-*.js`, run `wf_ff3b17f5-6b4`.
  **`digital-ref lead 를 올려 얻은 수치는 acoustic-ref 로 전이되지 않는다**
  (lead 는 digital-ref 전용 자유변수).

### base vs tiny 최종 비교 (2026-08-04 05:03, 동일 held-out 64 아이템)

**base(5.99M)는 tiny(1.16M)보다 나은 점이 사실상 없다. 배포 후보는 tiny다.**
두 모델을 Elice에서 같은 데이터(manifest 7종 + RIR 300)로 평가한 결과다.

| 지표 (NMSE dB, 낮을수록 좋음) | base 5.99M | tiny 1.16M | 우세 |
|---|---:|---:|---|
| trusted 대역 (150–600Hz) | **−18.99** | −18.66 | base (0.33dB) |
| fullband | −15.88 | **−17.14** | **tiny (1.26dB)** |
| held-out η=0.15 trusted | **−14.78** | −14.74 | base (0.04dB) |
| held-out η=0.15 fullband | −12.97 | **−13.97** | **tiny (1.00dB)** |
| **최악 아이템 fullband** | **+13.89 (증폭)** | **+4.06** | **tiny (9.83dB)** |
| Jetson P99 (ORT CPU) | 6.8ms **게이트 미달** | **1.84ms** | **tiny** |

> 이 절과 아래의 `1.84ms` / `6.8ms` 는 **2026-08-04 30W 시절 값**이다. 전원모드 표기가 없어
> 대조가 안 되므로 인용하지 말 것 — **지연 수치의 단일 출처는
> [docs/12 §4.6](docs/12_system_summary.md#46-추론-지연--30w-vs-maxn)** 이다
> (MAXN: tiny ORT P99 **1.44ms**, base 6.40ms). 그리고 모든 GPU 수치는 **듀티 100%** 벤치다.

소스별로는 **7종 중 7종 전부 tiny가 우세**하다.

| 소스 | base | tiny | 차이 |
|---|---:|---:|---:|
| **demand (최악 소스)** | **−4.36** | **−9.24** | tiny +4.88dB |
| synthetic | −15.38 | −19.16 | tiny +3.78dB |
| esc50 | −16.32 | −18.70 | tiny +2.38dB |
| music | −22.55 | −24.07 | tiny +1.52dB |
| dns_fullband | −15.39 | −16.60 | tiny +1.21dB |
| machine | −20.70 | −21.83 | tiny +1.13dB |
| speech | −28.68 | −29.48 | tiny +0.80dB |

옥타브밴드 감쇠는 두 모델이 사실상 동일하다(125Hz~8kHz에서 차이 0.00~0.28dB).

**절대 목표 기준의 판정**
- **기능 2는 평균이 아니라 최악값 문제**다. 최악 소스 `demand`가 base −4.36dB / tiny −9.24dB로
  둘 다 나머지 소스(−15~−29dB)보다 크게 뒤진다. **현재 어느 모델도 기능 2를 충족하지 못한다.**
  주방·세탁기·사무실·지하철 같은 지속성 실환경음이 약점이다.
- **최악 아이템에서 base는 fullband를 +13.89dB 증폭한다.** 이건 do-no-harm 위반이며
  tiny(+4.06dB)보다 10dB 나쁘다. 파라미터가 5배 많다고 안전한 것이 아니다.
- base가 이기는 것은 trusted 대역 0.33dB뿐이고, 그마저 실시간 게이트를 통과하지 못한다.

**따라서 base의 TensorRT 최적화에 시간을 쓸 근거가 현재로선 없다.** 다음 단계에서 용량을
늘릴지 판단하려면 먼저 `demand` 계열 성능과 최악 아이템 증폭을 개선해야 하며, 그것은
용량 문제가 아니라 데이터 분포·손실 설계 문제로 보인다.

> 이 수치는 전부 `secondary_surrogate` 플랜트에서 나온 **표현 사전학습 지표**다.
> 실측 P/S와 recorded 세션을 통과하기 전에는 실제 덕트 감쇠 성능이 아니다.

### 구조 탐색 결론 (2026-08-04 02:56 재판정 — 이것이 권위 있는 결과)

**20k 예산에서 어떤 구조 후보도 평범한 `tiny`를 이기지 못했다.** 승자는 대조군이다.

| 후보 (primary = `eval_pilot_last`, 동일 20k) | trusted NMSE | Δ vs 대조군 | 95% CI | 판정 |
|---|---:|---:|---|---|
| `tiny_control` (기준) | −14.59 | — | — | **승자** |
| `tiny_long` | −14.812 | −0.224 | [−0.709, **+0.255**] | 실격 아님, **유의하지 않음**(CI가 0을 가로지름) |
| `tiny_long_attn` | −12.415 | +2.173 | [+1.47, +2.94] | 실격 — fullband +1.78dB, held-out +1.30dB 악화 |
| `tiny_attn` | −12.060 | +2.528 | [+1.59, +3.53] | 실격 — fullband +2.15dB, held-out +2.12dB 악화 |

확인 지표(`eval_pilot_best`)에서는 셋 다 대조군과 거의 동률이며 유의한 후보가 없다.
`last`와 `best`의 격차가 큰 것은 attention 계열의 20k 지점 분산이 크다는 뜻이다 —
동일 예산·무편향인 `last`를 1차 지표로 삼은 이유가 여기서 드러난다.

Jetson 실측 비용축과 함께 보면 결론이 더 분명하다: `tiny` P99 1.84ms vs
`tiny_long` 2.24ms(+22%). **감쇠 이득 없이 지연만 늘어난다.**

**seed 반복으로 확인 완료 (2026-08-04 05:45)** — 결론이 재현됐을 뿐 아니라,
**효과 크기보다 seed 분산이 크다**는 것이 직접 드러났다.

| seed | `tiny_long` Δ vs 대조군 (primary=last) | 95% CI | 유의 |
|---|---:|---|---|
| 20260802 | −0.224 | [−0.709, +0.255] | 아니오 |
| 20260902 | **+0.460** | [−0.655, +1.621] | 아니오 |

두 seed 사이에서 Δ가 −0.22 ↔ +0.46으로 **0.68dB 요동**한다. 판정 마진 0.30dB보다 크므로
`tiny_long`의 이득(있다면)은 run 간 잡음에 묻히는 수준이다. **구조 탐색은 종결이며
tiny를 유지한다.**

> **남은 한계.** 20k는 100k 궤적의 앞부분일 뿐이라 **후반부에 순위가 뒤집힐 가능성은
> 배제하지 못한다.** 다만 tiny의 100k 완주본이 base(5.99M)보다도 나은 상황이라
> 지금 용량·수용영역을 더 키울 근거는 없다.

### 산출물 회수 완료 (2026-08-04 05:50) — 인스턴스 삭제 가능

`runs/queue/handoff.json`의 60건 중 **46건 SHA-256 일치**, 12건은 의도적 미회수
(기각된 0dB 초기 실험 `pretrain_{base,tiny}`와 `*_aggressive` 변형), 2건은 bundle 생성
시점(04:58)에 아직 학습 중이던 `seed_repeat_tiny_long`이라 최종본과 다르다 —
그 2건은 원격과 최종 SHA를 직접 대조해 일치를 확인했다.

로컬 `runs/`에 회수된 checkpoint의 step 검증 결과:

| run | last.pt step | best_metric |
|---|---:|---:|
| `pretrain_base_corrected` | 100,000 | **−19.755** |
| `pretrain_tiny_corrected` | 100,000 | **−19.537** |
| `search_tiny_control` | 20,000 | −16.869 |
| `search_tiny_long` | 20,000 | −16.667 |
| `search_tiny_attn` | 20,000 | −16.755 |
| `search_tiny_long_attn` | 20,000 | −16.713 |
| `seed_repeat_tiny_long` | 20,000 | −16.692 |
| `seed_repeat_tiny_control` | **1,000 (손상)** | — |

마지막 항목만 §사고 기록 ④의 덮어쓰기로 무효다. **평가 metrics는 완주 시점 것이라
유효**하며 `runs/seed_repeat_tiny_control/PROVENANCE.md`에 사용 가능/불가를 명시했다.

**→ 두 GPU 모두 유휴이고 남은 GPU 작업이 없다. 사용자는 Elice 인스턴스를 삭제할 것.**
중지만 해도 스토리지는 과금되며 삭제해야 완전히 멈춘다.

> **04:49 사고 기록 ④.** GPU0 큐와 GPU1 큐에 같은 id·같은 `ckpt_dir`의 seed 반복 작업을
> 둔 탓에, GPU1이 20k를 완주한 run을 GPU0이 다시 시작해 checkpoint를 step 1000으로
> 덮어썼다. 근본 원인이 둘이다. ① `ckpt_dir` 점유 검사가 **자기 GPU 프로세스만** 봐서
> 다른 GPU 감독자를 구조적으로 놓쳤다(→ GPU 무관 검사로 교체, 두 큐가 id/ckpt_dir을
> 공유하지 못하게 하는 테스트 추가). ② **GPU0 감독자가 00:42 기동 당시의 옛 코드를
> 메모리에 들고 있었다** — 모듈만 갱신하고 재기동하지 않아 `already_done`도 `child_env`도
> 없는 버전이 돌았다. **모듈을 갱신했으면 해당 감독자를 반드시 재기동할 것.**

> **02:47 사고 기록 ①.** 첫 승자 선정에서 후보 3종이 전부 `config fingerprint 불일치`로
> 실격됐다. 지문이 `model`/`model_config`를 포함해 **구조가 다르면 항상 불일치**했기
> 때문이다 — 구조는 실험의 독립변수인데 그것으로 실격시키면 구조 탐색이 불가능해진다.
> 지문에서 구조 키를 제외하고 재판정했고, 결론(승자=대조군)은 같지만 근거가
> "실격"에서 "유의한 개선 없음"으로 정정됐다.

> **02:56 사고 기록 ②.** 감독자를 재기동했더니 이미 20k를 완주한 `search_tiny_control`을
> 처음부터 다시 돌렸다(결과를 메모리에만 뒀기 때문). checkpoint 저장은 500 step마다라
> step 500 직전에 차단해 완주본(`last.pt` step=20000, best_metric −16.869)을 지켰다.
> `restore_results()`와 디스크 완료 검사를 넣어 고쳤다. **복원은 반드시 첫 상태 기록보다
> 앞서야 한다** — 순서가 바뀌면 `jobs={}`로 덮어쓴 빈 파일을 읽어 조용히 무력화된다.

> **01:17 사고 기록 ③ — 반드시 읽을 것.** 감독자 첫 기동에서 `spawn()`이 환경을 그대로
> 물려줘 자식이 두 GPU를 다 보고 PyTorch 기본값 `cuda:0`에 올라갔다. GPU1의
> `search_tiny_control`이 **GPU0의 base 위에 겹쳐** 7.4GB를 뺏고 base를 1.84 → 1.36 it/s로
> 떨어뜨렸다. 즉시 감독자→자식 순으로 중지하고(base는 무사) `Supervisor.child_env()`로
> `CUDA_VISIBLE_DEVICES`를 고정한 뒤 재기동했다. 자식 환경이 `CUDA_VISIBLE_DEVICES=1`이고
> GPU1 UUID에 올라간 것, base가 1.79 it/s로 회복한 것을 확인했다.
> **교훈**: 이 환경변수는 `processes_using_gpu()`의 유휴 판정 근거이기도 하다. 설정하지
> 않으면 진입 게이트의 (c) 조건 자체가 무력해져 감독자끼리 서로의 자식을 못 본다.

### 상태 확인 (가장 먼저 실행)

```bash
SSH="ssh -i ~/.ssh/elice.pem -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -o ControlMaster=auto -o ControlPath=~/.ssh/cm/%r@%h-%p -o ControlPersist=600 \
  -p 47863 elicer@central-01.tcp.tunnel.elice.io"

# 큐 상태 한 번에 (표준 라이브러리만 — 학습 CPU를 뺏지 않는다)
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'

# 원 학습 로그와 GPU
$SSH 'cd ~/Deep-ANC && grep "^step " runs/train_base_corrected.log | tail -n 1; \
  tail -n 3 runs/structure_search.log; \
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

`idle_seconds_total`이 이 시스템의 목적함수다. 작업 전환 시 60초를 넘으면 원인을 조사한다.
감독자 로그는 `runs/queue/supervisor_gpu{0,1}.log`, 이벤트는 `runs/queue/events.jsonl`.

### 감독자가 죽었을 때

`queue_status.py`가 `[STALE]`을 표시하면 감독자 자체가 죽은 것이다. 재기동은 안전하다 —
완료된 작업은 결과로 건너뛰고, 진입 게이트를 처음부터 다시 통과한다.

```bash
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'   # 또는 0
```

### 큐에 작업을 덧붙이려면

감독자는 **작업 사이마다 큐 YAML을 다시 읽는다.** 재시작 없이 `configs/elice/queue_gpu*.yaml`에
작업을 추가하면 반영된다. 이미 결과가 있는 id는 재실행되지 않는다.

## 1. 프로젝트 한 줄 요약

덕트(사각 아크릴 1.2m) 딥러닝 능동소음제어. 학습=Elice 2×A100, 추론=이 Jetson AGX Orin.
모델 HybridANCNet(tiny 1.16M=현행 실시간 / base 5.99M=TRT 목표), digital-ref 모드 우선.
**절대 목표 2가지: ① 저주파+고주파 노이즈 제거 ② 모든 소리 제거(quiet zone)** — AGENTS.md 참조.
상세: docs/00, 물리: docs/01, 구조 지도: docs/10, 목표 측정: docs/07 §0.

## 2. 현재 상태

### 완료 ✅

- 저장소 골격·문서·자동 테스트, GitHub `Roka-jsj/Deep-ANC` 공개 운영
- 19-에이전트 리뷰(결함 15건) + 5-에이전트 구조 감사(이슈 35건) 반영
- 물리 정합 학습 목표(digital-ref lead — **현행 실측 116**, 사전학습 artifact 는 109 —,
  P(z) resolver, trusted-band NMSE — **현행 150–1600Hz**, 사전학습 당시 150–600Hz)
- 원샷 부트스트랩 `scripts/elice/bootstrap_all.sh` (환경+데이터 6종+RIR+QA+테스트+2GPU 학습)
- 데이터: DNS 16,000 / speech 8,065 / music 7,997 / MIMII 3,600 / ESC-50 2,000 / DEMAND 96
  (약 154.9시간). 손상 FMA MP3 3개는 manifest에서 제외
- recorded group-aware manifest·전수 QA·독립 evaluator·파인튜닝 fail-fast **구현 완료**
- **tiny 100k 완주·로컬 회수**: 최종 val trusted **−19.47** / full −18.27dB,
  best step89,500 trusted **−19.5372dB**. 원격 SHA-256 일치
- **tiny_long 20k 완주·로컬 회수**: best step13,500 trusted **−16.6672dB**
- **구조 탐색 종결(잠정)**: 후보 3종 중 어느 것도 20k에서 tiny를 이기지 못했다(§0 표).
  attention 계열은 fullband·held-out 일반화를 1dB 넘게 해쳐 do-no-harm 실격이다.
- **Jetson 실측 (2026-08-04)**:
  - `tiny` best.pt → ONNX, ORT 등가 `8.196e-08`, **P99 1.84ms** (게이트 <3ms 통과)
  - `tiny_long` last.pt → ONNX, ORT 등가 `7.567e-09`, **P99 2.24ms** (통과)
  - 즉 tiny_long은 수용영역 2배에 P99 +0.40ms(+22%). 구조 비교의 비용축 확보
- **GPU 작업 큐 감독자 신규 구현·배포** (`src/deep_anc/ops/job_queue.py`):
  기존 프로세스 불가침 4중 진입 게이트, 실패 격리(작업 하나 실패해도 큐 계속),
  사전 등록 승자 선정, 원자적 상태 JSON, 큐 재로드
- **파인튜닝 진입점 완성**: `--state-dir` 배선, `pipeline.lock`, advisory `status.json`,
  exit code 3/4 분리, 상대 config 경로 fail-open 수정.
  검증: `--check-only` → **exit 1 + `runs/` 미생성** (설계대로 NOT READY)
- **base 100k 완주** (04:48 KST): 최종 val trusted **−19.73** / full −17.37dB,
  best trusted **−19.75dB**. held-out 64아이템 재평가 완료
- **base vs tiny 동일조건 비교 완료** — §0 표. **배포 후보는 tiny 로 확정**

#### 2026-08-04 오후 세션에서 완료한 것

- **G2 통과 — recorded 데이터셋 수집 완료**: **80세션 / 93.3분 / 4계열 각 20개 / 64그룹**,
  분할 train 64 · val 9 · test 7, **형식** QA 80/80 PASS.
  ⚠ **2026-08-06 정정: 정렬 QA 로는 0/80 PASS 이며 80세션 전량 격리됐다** (§0.6 결함 2).
  - `build_recording_sources.py` — 계열별 소스 WAV, tanh 소프트 클리핑으로 크레스트 10dB 제한
  - `record_session_batch.py` — 재개 가능 배치, 세션마다 즉시 QA, 일시적 xrun 1회 재시도
  - `record_duct.py` — settle 1초, **xrun 발생 세션은 저장 자체를 거부**
- **실기 ANC 시연**: 음성+80–800Hz 소음, OFF 10초 → ON 20초 → OFF 5초.
  소스대역 **+4.39dB**(보수적 기준), trusted NMSE −5.66dB. `results/session_20260804_125538/`
- **I²S 기동 트랜지언트 오염 수정** (3개 경로): 첫 0.5초가 −36.3dBFS/peak 0.062라
  실제 바닥(−67.4dBFS/peak 0.002)을 18dB 가리고 있었다. **죽은 마이크도 게이트를 통과하던
  결함**이라 회귀 테스트로 막았다. 재생 진폭도 0.15 → 0.06으로 낮췄다.
- **G4의 기능 2 인코딩 수정**: 소스별 **평균**이 아니라 **최악값**으로 판정한다.
  기존 게이트는 "음성을 6dB 증폭하지만 나머지를 잘 잡는" 모델을 통과시켰다.
- **파인튜닝 진입점 base → tiny 전환** (`train_finetune.yaml`, 테스트, 문서).
  게이트가 열리는 순간 **배포하지 않을 모델**을 학습하게 되어 있었다.
- **동시 인터리브 P/S 측정 도구 신규** — `scripts/data/measure_paths_interleaved.py`,
  `src/deep_anc/dsp/interleaved_probe.py`(자극 설계 + IR 복원 + warp 추적/역보정).
  게이트에 `interleaved_multitone` 방식을 추가하되 **ESS 보다 좁게** 검사한다
  (guard=1, 분석창 ≤2초, 톤 수, 톤 SNR, 그리고 두 파일의 `capture_id` 일치).
- **전체 회귀 테스트 336개 통과** (당시. 2026-08-06 현재 **600개**)
- **README 전면 정리 + 그림 4종** — 덕트 도면(SVG, `duct.yaml`에서 생성), 지연 예산,
  실기 ANC 시연 파형/스펙트럼, 데이터셋 구성. 전부
  `scripts/docs/render_readme_figures.py`가 **실측 산출물에서 재생성**한다.

### 대기 ⬜ (다음 세션이 할 일 순서)

> ⚠ **이 절은 2026-08-04 기준이며 상당 부분이 낡았다. 실행 순서는 §0.5 를 따르라.**
> G1 은 **해결됐다** (2026-08-05 플랜트 복구 — 단, 당시 '통과' 시킨 게이트가 오염된 S(z) 를
> 통과시켰다는 것이 §0.6 결함 1 이다). 실질 블로커는 이제 **결함 2(실측 녹음 시간축 붕괴)** 다.
> 아래 G1 서술의 "클록 도메인 드리프트" 진단도 틀렸다 — 실측 상대 드리프트는 +0.4 ppm 이고
> 실제 원인은 USB 큐 위상 점프다(§0 표).

<details>
<summary>2026-08-04 시점의 기록 (역사적 참고용)</summary>

**파인튜닝 준비 상태: NOT READY. 블로커는 G1 하나뿐이다.**

```
[FAIL] finetune_readiness  (5 PASS / 4 FAIL — FAIL 4개가 전부 같은 뿌리)
  [PASS] config_fail_closed_flags / measured_primary_mode / recorded_mix_ratio
  [PASS] completed_init_checkpoint / recorded_dataset_qa
  [FAIL] official_secondary_path / official_primary_path
  [FAIL] matched_path_measurement_conditions / path_delay_and_lead   ← 위 둘에 종속
```

1. **G1 — 실측 P/S. 이것만 되면 READY다.** 2026-08-04 측정으로 원인이 확정됐다:
   재생(USB DAC)과 녹음(I²S)이 다른 클록 도메인이라 **1초 창 안에서 대응이 100–200샘플
   움직인다.** 신호 자체는 문제가 없다 — 톤 SNR 29.5dB, 대역 내 에너지 97.4%,
   반복 간 `\|H\|` 비 **1.000**. 깨진 것은 시간축 하나다(위상 직선적합 잔차 1.8–3.9 rad).

   | 접근 | 반복 일관성 (요구 0.9) |
   |---|---:|
   | 순차 ESS | 0.08–0.17 |
   | 동시 인터리브, 보정 없음 | 0.05 |
   | 동시 인터리브 + warp 역보정 (창 43ms) | **0.84 / 0.85** |
   | + 궤적 평활 | 0.54 (악화 — 요동이 실재한다는 증거) |

   다음에 시도할 것 (기대순): ① 추적 창을 43ms 아래로 내리면서 상관 첨두가 흐려지는
   지점을 찾기 ② `latency=low`로 재측정(현재 `high`는 버퍼가 커서 warp가 더 실릴 수 있다)
   ③ 재생·녹음을 **한 장치**로 모으는 방안 검토(하드웨어 변경이므로 최후의 수단).
   원자료는 `results/calibration_interleaved/20260804_132812_d1479bae/`에 있다.
   **게이트를 낮추지 않는다.** 성공한 반복만 골라 저장하는 우회도 하지 않는다.
2. G1 통과 후: `duct.yaml`의 `primary_path_npz` / `d_noise_delay_samples` 기입 → `lead` 재계산
   → `check_finetune.py` READY 확인 → Stage-2 open-loop 파인튜닝(tiny) → recorded G4.
3. 남은 감사 항목 (게이트와 무관, 언제든 가능):
   - `finetune_readiness.py`의 완료 판정이 `schedule.total_steps`를 권위로 쓰도록 —
     지금은 20k 파일럿이 "완료"로 잡힐 수 있다
   - `trainer.py`의 `freeze_encoder`를 DDP 래핑 **앞**으로 이동
   - `recorded_qa.py`의 최소 RMS(−80dBFS)가 실제 바닥(−67.4dBFS)보다 낮다 → SNR 여유로 전환
   - `make_recorded_manifest.py`가 `batch_progress.csv`의 판정을 무시한다

</details>

### 승자 연장 작업의 규약 (감독자가 자동 적용 — 참고용)

`configs/elice/queue_gpu1.yaml`의 `extension_template`에 확정돼 있다. 손으로 만들 일이
생기면 다음 세 가지를 반드시 지킨다.

- **`ckpt_dir`은 새 디렉터리** — 같은 곳에 resume 하면 pilot의 20k `best/last`를 덮어써
  구조 비교 근거가 사라진다.
- **`resume`은 `last.pt`** — `best.pt`로 되감으면 optimizer/scheduler가 후퇴해 예산을 낭비한다.
  대신 pilot의 `best.pt`를 새 ckpt 디렉터리로 **복사**해야 `trainer.py:390`의 `best_metric`
  min() 교정이 동작한다(복사하지 않으면 20k best를 넘기 전까지 `best.pt`가 아예 없다).
- **`seed`는 원 seed +100** — worker RNG는 checkpoint에 저장되지 않고 iterator 생성 시
  `seed + split_offset + worker_id*1009`로 재시드된다(`synth_dataset.py:269`). 그대로 두면
  step 20k–40k가 0–20k와 **같은 데이터를 재생**한다. (worker RNG를 checkpoint에 저장하는
  근본 수정은 별도 후속 항목이다.)
- `run_until_step`은 지정하지 않는다 → `resolve_run_until_step()`이 `total_steps` 폴백.

### 데이터/체크포인트 선택 주의

- 내장 val은 고정 16개(현재 seed에는 DEMAND 0개)라 최종 판정용이 아니다.
  `best.pt`만 맹신하지 말고 `last.pt`도 회수한다.
- 공개 데이터의 파일 단위 split은 speech 화자/책, ESC 원본, MIMII 조건, DEMAND 동시녹음
  채널 같은 상관 그룹이 split을 가로지를 수 있다.
- `secondary_surrogate` checkpoint는 표현 사전학습 전용이라 물리 성능 주장에 쓸 수 없다.
- **로컬 Jetson 오프라인 평가는 제한적이다.** `data/manifests/`와 RIR 뱅크가 없어서
  소스별 표에 `synthetic`만 남고 RIR이 즉석 32개로 대체된다. 즉 **기능 2(모든 소리)는
  로컬에서 측정할 수 없다.** 승자 선정과 소스별 평가는 반드시 Elice에서 돌린다.

### 사용자가 직접 해야 하는 것 (권한/자원 소유)

- **I²S 입력 복구**: 전원 OFF에서 공통 GND/SD/LR·pin17 접촉 확인. 이후 §3-C의 무출력
  probe 2개가 clip 0으로 반복 PASS해야 실측 재개 가능. **이것이 풀리기 전에는 실측 P/S,
  recorded 세션, 파인튜닝이 전부 막혀 있다.** Elice 사전학습 완료가 이 게이트를 대체하지 못한다.
- **Elice 인스턴스 중지/삭제**: 큐가 `drained`가 되고 회수가 끝나면 즉시. 인스턴스 켜진
  시간으로 과금되며, 중지해도 스토리지는 과금되고 삭제만 완전 중지다.
- 파인튜닝 현장 준비: AB13X·두 마이크·두 스피커 고정, 같은 출력게인의 S(cancel→ERR),
  P(noise→ERR), THD/IMD 측정. 실제 소스 독립 세션 최소 80개(1.5–2h/3–4GB),
  권장 160개(3–4h/6–8GB)와 10–15GB 여유 공간.
- 덕트 미확정값 확정 시 통보 (에러마이크 X=1.100 잠정 — 확정 시 duct.yaml + RIR 뱅크 재생성)

### 사용자가 확정한 INMP441 물리 배선 (2026-08-03)

- 두 마이크 공통: VDD 빨강→J30 pin1(3.3V), GND 검정→pin6, SCK 주황→pin12,
  WS 노랑→pin35, 두 SD 공통 갈색→pin38
- 레퍼런스 마이크 L/R 초록→pin17(3.3V, right/ch1), 에러 마이크 L/R 파랑→pin39(GND, left/ch0)
- 공식 핀표와 INMP441 L/R 규약 대조 완료. 안전 주의·근거는 docs/02 §1 참조.
- 현재 pinmux/I²S는 의도된 기존 구성이다. **sudo, Jetson-IO, pinmux, device-tree,
  오디오 데몬, 전원모드 변경은 모두 금지.**

### 현재 I²S·출력 경로·P/S 실측 상태

- APE 입력 `hw:1,1`과 AB13X 출력 `hw:2,0`은 장치로 인식되고 스트림 설정도 수락된다.
- pin17 재연결 직후 5초 probe는 ERR **−46.33dBFS**, REF **−46.64dBFS**, clip 0%로 PASS했다.
- **하지만 22:39 이후 다시 FAIL이다.** 무출력 2초 probe에서 ERR −12.84dBFS/clip 2.474%,
  REF −10.67dBFS/clip 5.029%. 10초 재검사도 FAIL. peak/raw가 정확히 ±1.0/INT32 한계까지
  도달하고 0.1초 구간별 간헐 burst가 있어 시작 과도가 아니다. 장치 점유 프로세스는 없었다.
  **스피커 출력은 전혀 시작하지 않았고 전달맵/direct FxLMS 실측은 안전 중단 상태다.**
- peak 0.005 채널 분리 실측: noise ch0와 cancel ch1 모두 ERR/REF에 도달. tone-bin 상승은
  ch0→ERR/REF +25.79/+26.15dB, ch1→ERR/REF +22.83/+28.34dB. REF 기준 ch1이 ch0보다 약
  8.3dB 강해 실제 앰프 게인/물리 매핑 차이는 별도 확인이 필요하다.
- magnitude 진단은 204–210Hz, 348–351Hz, 458–476Hz, 594–613Hz 부근 피크를 반복 검출해
  1D 예측 공진(210/350/489/629Hz)을 부분 지지한다. 이는 **공진 형상 진단**일 뿐 덕트 식별
  완료, 고정 지연, FxLMS 성능 또는 물리 좌표 확정의 근거가 아니다.
- legacy 300Hz 설정의 과거 "약 2dB 감소"는 현재 하드웨어에서 재검증한 결과가 아니다.
  2026-08-03 재현은 실제 `ON/ADAPT`까지 확인했지만 ON 로그 9개에서 중단됐고 감쇠 중앙값
  +0.11dB, 순간 최대 +0.63dB였다. **2dB 성공 증거가 아니다.**
- 19:14에 읽기 전용 감사 에이전트가 legacy 모듈을 import하면서
  `/home/capston/anc_project/__pycache__/main_realtime_anc.cpython-310.pyc` 한 개를 실수로
  생성했다. 그 경로는 이후 건드리지 않았고 **임의 삭제도 하지 말 것.**
  저장소의 안전 실행기는 `python3 -B`와 `PYTHONDONTWRITEBYTECODE=1`로 재발을 막는다.

## 3. 실행 절차

### A. GPU 작업 큐 (현행 운영 방식)

```bash
$SSH 'cd ~/Deep-ANC && python3 scripts/elice/queue_status.py'                    # 상태
$SSH 'cd ~/Deep-ANC && .venv/bin/python scripts/elice/job_queue.py plan \
  --queue configs/elice/queue_gpu1.yaml'                                          # 예정 순서(GPU 무접촉)
$SSH 'cd ~/Deep-ANC && bash scripts/elice/run_job_queue.sh 1'                     # 기동/재기동
```

> **원격 배포는 반드시 신규 경로만 scp한다.** 원격 워크트리는 HEAD가 아니라 워킹트리
> 스냅샷(dirty 37개)이라 `git pull`은 merge abort되거나, stash/checkout으로 우회하는 순간
> corrected physics 코드가 revert되어 **돌고 있는 학습이 조용히 다른 실험이 된다.**
> 또 실행 중인 bash 스크립트를 scp로 덮어쓰면 (scp가 inode를 truncate) bash가 오프셋 기준
> 지연 읽기를 하므로 **watcher가 즉사한다.** `cp -n`/`cp -rn`으로 no-clobber 설치할 것.

### B. 학습 완료 후 → Jetson 배포

```bash
mkdir -p ~/Deep_ANC/runs/pretrain_base_corrected/ckpt
scp -o ControlPath=~/.ssh/cm/%r@%h-%p -P 47863 \
  elicer@central-01.tcp.tunnel.elice.io:~/Deep-ANC/runs/pretrain_base_corrected/ckpt/best.pt \
  ~/Deep_ANC/runs/pretrain_base_corrected/ckpt/
# 회수 목록과 SHA-256 은 감독자가 runs/queue/handoff.json 에 미리 만들어 둔다
# → 이 시점에 사용자에게 "인스턴스 중지/삭제" 안내!

cd ~/Deep_ANC && .venv/bin/python scripts/train/export_onnx.py \
  --ckpt runs/pretrain_base_corrected/ckpt/best.pt --out runs/export/base.onnx
.venv/bin/python scripts/bench/measure_inference_latency.py --config configs/runtime.yaml \
  --set engine.type=ort --set engine.onnx=runs/export/base.onnx
```

`secondary_surrogate` 결과는 오프라인 표현 평가까지만 한다. 실제 스피커 실행 전에는 같은
게인의 실측 P/S로 파인튜닝하고 runtime `digital_reference_lead_samples=109`를 checkpoint
메타와 맞춘다.

### C. 하드웨어 재연결 후 (사용자 입회) — docs/02 §3 + docs/08 §5

```bash
# 1) 스피커를 전혀 열지 않는 입력 게이트. 현재는 FAIL 상태이므로 여기서 중단.
cd /home/capston/Deep_ANC
.venv/bin/python scripts/bench/check_audio_input.py
.venv/bin/python scripts/bench/check_audio_input.py --require-both

# 2) 둘 다 clip 0으로 반복 PASS한 뒤 네 경로 시간-주파수 지도
.venv/bin/python scripts/bench/measure_duct_transfer_map.py --confirm-volume-minimum

# 3) 전달맵 뒤 저음량 direct FxLMS 진단. legacy S라 결과는 diagnostic-only.
.venv/bin/python scripts/demo/evaluate_fxlms_direct.py \
  --amplitude 0.005 --control-limit 0.005 --confirm-user-present-volume-minimum
```

legacy 재현이 필요하면 원본을 기본값으로 실행하지 않는다. 종료 시 weight를 저장하므로
`--weights-output`을 이 저장소의 ignored `results/`로 우회하고 `python3 -B`를 쓴다.

### D. 파인튜닝 (게이트 통과 후)

```bash
.venv/bin/python scripts/train/run_finetune_pipeline.py \
  --config configs/train_finetune.yaml --set data.digital_primary_path_mode=measured
```

현재는 설계대로 **exit 1 (NOT READY)** 이며 `runs/` 아래에 아무것도 만들지 않는다.
exit code 표와 산출물 경로는 README §6.6 참조.

## 4. 조심할 것 (세션에서 배운 것)

- Elice 터널: 로컬 타임아웃이 나도 **원격 작업은 대부분 살아있다** — 재실행 전 반드시 상태 확인.
  원격 장기작업은 `setsid nohup … < /dev/null &` 패턴만.
- Elice는 **인스턴스 켜진 시간 과금** (SSH 연결 여부 무관). 종료(중지)해도 스토리지는 과금,
  삭제만 완전 중지.
- `tail -n 1` (구식 `tail -1`은 다중 파일에서 GNU 오류)
- 원격에서 pytest를 돌릴 때는 `nice -n 19`. 32 vCPU 중 28개를 두 학습의 DataLoader가 쓴다.
- Jetson venv 재구성은 `scripts/jetson/setup_jetson.sh` (lib preload 훅 필수), ORT 1.18.1 고정
- 입력 raw가 `-1`/0 고정이면 장치가 열려도 유효 오디오가 아니다. 무출력 preflight 실패를
  `--force`로 우회하거나 스피커 출력으로 진단하지 않는다.
- S(z)/핸드오프/목표대역은 단일 출처 원칙 (duct.yaml + config.DEFAULT_HANDOFF_SAMPLES)
- 컨테이너에서 `nvidia-smi --query-compute-apps`는 비어 보일 수 있다. GPU 유휴 판정은
  memory.used와 `/proc/*/environ`의 `CUDA_VISIBLE_DEVICES`까지 함께 봐야 한다.
- `/proc/<pid>/stat`의 starttime은 **마지막 `)` 뒤부터 세어 tail[19]** 다. comm 필드에
  공백이 들어갈 수 있어 단순 split은 틀린다. PID 재사용 판별에 이 값이 필요하다.

### 읽기 전용 참고 구현

- `~/anc_project` — legacy FxLMS(Python). block FxNLMS, S(z)는 `calibrate_s_path.py`로
  오프라인 식별(256/low에서 순수지연 1342, coherence 0.40으로 낮음). 종료 시 기본
  `control_filter_last.npy`를 CWD에 저장하므로 원본에서 기본값 실행 금지.
- `~/FxLMS/realtime_fxlms` — C++ 단일 blocking `snd_pcm_readi`→계산→`writei`, block512,
  S 2048tap, W 512tap, 내부 230+370Hz digital tone reference. FxLMS 부호는 Deep_ANC의
  `e=d+S·y`와 일치하지만 REF ch1은 실제 제어에 쓰지 않는다. timestamp가 없고 partial write/
  xrun 뒤에도 계속하므로 **절대 지연·위상 근거로 쓸 수 없다.** (두 클록이 독립이라서가
  아니다 — APE 입력과 USB 출력은 같은 Tegra 발진기를 공유하며 상대 드리프트는 +0.4 ppm 이다.)
  `run_experiment.sh`는 PCM 100%로 바꾸므로 볼륨 최저 규칙과 충돌해 **실행 금지**다.
  기존 `fxlms_run_log.csv`의 ANC control RMS는 거의 0.299로 ±0.3 hard-limit에 포화됐고
  표시 감쇠는 −1.80~+5.69dB로 요동해 **2dB 성공 증거가 아니다.**
- 두 폴더 모두 앞으로도 읽기 전용. python import 금지(`python3 -B`).
