# 125 Hz–8 kHz v3 학습 admission 경계

## [가설]

기존 150–1600 Hz Stage-1 또는 150 Hz 하단 v2 artifact가 남아 있으면, 이를 이름만
바꿔 125 Hz–8 kHz 학습에 사용하려는 실수가 생길 수 있다.

## [근거]

- v3 control-band contract는 8개 physical identification subband, 7개 동등 가중
  octave objective, 4개 Stage-1 저역 guard를 분리한다.
- 현재 `Trainer`, `finetune_readiness`, `eval/recorded`는 이 v3 causal P/S와
  7-octave pass/FxLMS 비교를 소비하지 않는다.
- 현재 RT5640 fullband P/S, canonical public manifest, DNH gradient calibration은
  없다. old local synthetic manifest에는 recorded holdout lineage overlap이 확인됐다.

## [확인 방법]

다음 명령은 장치·GPU·Trainer·run directory를 열지 않고 declared artifact의 실제 file
bytes SHA와 schema/cross-reference를 검사한다.

```bash
PYTHONPATH=src .venv/bin/python scripts/train/check_full_octave_v3_admission.py \
  --config configs/full_octave_v3_admission.yaml --markdown
```

JSON 기록이 필요한 경우에만 미사용 경로를 명시한다. 같은 경로는 덮어쓰지 않는다.

```bash
PYTHONPATH=src .venv/bin/python scripts/train/check_full_octave_v3_admission.py \
  --out results/full_octave_v3_admission/<new-id>.json
```

## [결과]

현재 기본 config의 정상 결과는 `BLOCKED`다. 이는 실패를 숨기는 상태가 아니라 다음
증거가 아직 없음을 파일 단위로 드러내는 상태다.

1. RT5640 S32 fullband raw-first capture와 causal P/S authority
2. external raw에 결속된 population/lineage authority와 zero overlap
3. global-index·component-uniform family-balanced batch receipt
4. actual causal `S*y` batch의 output-y DNH gradient share 0.2–0.4 receipt
5. v3 causal prefix Trainer, readiness, seven-octave/FxLMS evaluator consumer

특히 마지막 항목은 이 branch 코드에 고정된 blocker다. 위 artifact가 나중에 모두
생겨도 기존 trainer에 넣어 학습하는 것은 허용하지 않는다.

로컬 legacy manifest도 이 빈자리를 메우지 못한다. 실제 audit에서는 public candidate
12,298개 중 현존 파일 2,949개, full-target native Nyquist 충족 파일 4개뿐이었고,
canonical population candidate는 0개였다. 또 `speech.jsonl`에는 recorded holdout과
같은 basename 8개, `esc50.jsonl`에는 environment 58개와 machine 24개의 lineage/basename
overlap이 있다. 따라서 `recorded_regrouped.jsonl`의 82 세션은 보존 후보이지만, old
synthetic JSONL/transfer bundle은 full-octave training input이 아니다.

## [판정]

**Confirmed — admission-only boundary.** 이 명령의 실행이나 `BLOCKED` 출력은
P/S, ANC 감쇠, high-frequency cancellation, GPU readiness, checkpoint 또는 배포 적격성을
증명하지 않는다.

## [다음 행동]

1. S32 meter/raw publisher와 electrical witness를 무음 dry-run으로 완성한다.
2. 한 번의 승인된 연결 창에서 fresh fullband P/S와 raw-first authority를 만든다.
3. corrected public raw로 lineage manifest와 population/batch receipt를 발행한다.
4. 그 다음 새 branch에서 v3 causal `S*y` adapter, criterion factory, readiness,
   7-octave + matched physical FxLMS evaluator를 함께 구현하고 해당 code blocker를
   대체한다.
5. 그 모든 gate가 통과한 뒤에만 Elice에서 canonical 100k pretrain과 50k fine-tune을
   시작한다.
