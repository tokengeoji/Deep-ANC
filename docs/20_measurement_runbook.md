# 20. OMAP 16 kHz 실측 전 패킷과 수집 후 검사

이 도구는 **수집 계획·파일 검사·기존 WAV 분리 연결**만 한다. 스피커·ADC 장치·JTAG·CCS를
열지 않고 보드 프로그램을 로드하거나 펌웨어를 바꾸지 않는다. 학습·필터 fitting도 실행하지 않는다.
현재 실제 보드 모델/revision·가용 RAM·연속 raw 수집 방법은 미확정이다. 사용자가 보드 식별을
나중에 제공하기로 했으므로 보드 메모리 배치와 녹음 전용 펌웨어 구현은 명시적으로 보류한다.
패킷의 `hardware_capture_implementation_pending_board_identity=true`는 이 하드웨어 보류를
파일 기반 양식/QA 준비와 구분한다. 보드 식별 전 임의로 메모리·연속 캡처 코드를 만들지 않는다.

## 1. 지금 할 수 있는 준비

Docker 안에서 **존재하지 않는 새 경로**로 통합 준비 패킷을 만든다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/prepare_before_measurement.py \
  --out results/pre_measurement/NEW_PACKET
```

수집 양식은 `NEW_PACKET/capture/` 아래, 학습·기준선 설정은
`sfanc_recipe.template.json`, `baseline_search.template.json`, `validation_cases.template.json`에 있다.
`commands.json`은 후속 명령 목록일 뿐 실행하지 않는다. 모든 미확정 설정은 null/unknown이다.
`report.json`의 `packet_created=true`와 `capture_ready/training_ready=false`를 구분한다.
선택적 `--source-inventory`는 기존 Drive 감사 JSON의 경로·SHA만 연결하며 대용량 자료를 받지 않는다.
공개 음원은 동기 실측 쌍을 대체하지 않는다.

수집 양식만 별도로 필요하면 다음 도구를 쓴다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/prepare_measurement_session.py \
  --out results/measurement_packets/NEW_PACKET
```

`session_plan.json`과 `raw/{train,valid,test}/session_*/capture.json`만 생성된다.
WAV를 생성하거나 녹음 성공을 선언하지 않는다. `unknown`은 실제 확인 전까지 그대로 둔다.
각 split에 독립 세션을 추가하려면 해당 capture 템플릿을 새 세션 폴더에 복사한 뒤
`split`·`session_id`를 정확히 수정한다. 여기서 test는 최종 평가 전용이며 기존 학습 manifest의
train/valid 계약을 확장하거나 바꾸는 항목이 아니다.

## 2. 현장에서 반드시 확인할 항목

- Windows CCS → XDS200(TMDSEMU200-U) → OMAP-L138 연결을 유지한다. 이 문서는 연결
  테스트·CCS 실행·보드 로드 허가가 아니다. 실제 보드 모델/revision과 firmware revision을 기록한다.
- 기존 `ref_mic_buf`/`err_mic_buf` 256샘플 모니터는 16 ms 순환 버퍼다. 임의 주기 덤프나
  정지/재개 조각을 연속 녹음이라고 연결하지 않는다. 가용 RAM·링커·수집 코드는 별도 확정한다.
- 같은 ADC clock, 같은 시작 시점, Left=raw REF, Right=raw ANC-OFF ERR를 확인한다.
  DC 제거된 `ref_x_buf`/`err_e_buf`를 raw라고 표시하지 않는다.
- 사용자 입회·물리 볼륨 최소에서만 실제 소리를 다룬다. ANC OFF뿐 아니라 test tone 출력도
  OFF인지 확인한다. 이 도구는 전원·gain·ANC 상태를 자동 조작하지 않는다.
- 샘플 counter/순서 검증으로 누락·중복을 확인하고 방법과 숫자를 남긴다. WAV 헤더와 frame 수
  일치만으로 실제 연속 수집을 인증할 수 없다.
- ADC REF, ADC ERR, DAC, 출력 증폭기 gain의 **실제 수치와 단위**를 각각 기록한다.
  REF와 ERR gain이 서로 같아야 한다는 뜻은 아니다. 각 채널의 값·단위를 세션 간 유지하며
  `gain_profile_id`와 실측 S 조건을 맞춘다. 출력 gain이 미확정이면 intake를 통과시키지 않는다.
- `secondary_source_sha256`는 원본 `rir.txt` 파일 SHA다. 입력·출력·마이크·gain·배치가 그 S와
  동일한지는 운영자가 확인해야 한다. 다른 Jetson DAC 경로에 OMAP S를 자동 적용하지 않는다.
- train/valid/test의 녹음 세션과 원 소리의 `source_recording_id`를 분리한다. 같은 연속 녹음이나
  같은 음원 일부를 이름만 바꿔 나누지 않는다. 최종 test는 모델·μ·정규화상수·출력한도 선택에 쓰지 않는다.
- `source_recording_ids`에는 모든 부모 원녹음 ID, `source_group_ids`에는 독립성 그룹을 기록한다.
  예를 들어 `speaker:...`, `book:...`, `artist:...`, `recording:...`처럼 근거가 확인된 구분값을 쓴다.
  미확인 그룹에 가짜 ID를 만들지 않는다. 두 목록은 비어 있거나 unknown이면 QA가 실패한다.
  혼합 재생은 `source_family=mixed`, 전체 혼합 ID는 `source_recording_id`, 부모 녹음은 2개 이상
  `source_recording_ids`에 모두 적고 부모의 그룹도 합친다. 혼합 ID가 달라도 부모/그룹 재사용이
  있으면 split 누수로 실패한다. ID 목록의 완전성과 그룹 독립성 자체는 운영자 확인 사항이다.
- 음성·음악·환경음/소음의 출처를 기록한다. MIMII/machine은 현재 정책상 학습 보조 전용이며
  valid/test에는 넣지 않는다. 공개 음원 출처/라이선스 기록과 동기 REF/ERR 실측은 별도다.

`capture.json`의 고정 수집 규격은 16,000 Hz/stereo/PCM16이다. 실제 값이 다르면 파일을
리샘플·정렬·정규화하여 통과시키지 말고 수집 조건을 다시 확인한다. `attestations`와
`recording_loss`를 추정으로 true/0으로 채우지 않는다. `source_corpus`가 일반 현장 소음이면
그 사실을 명시적으로 쓰고 공개 corpus 이름을 임의로 붙이지 않는다.
`regions.onset/transition/steady`는 null 템플릿이다. 관측 구간 규약이 명시되기 전 자동으로
추정하거나 비교 평가에 사용하지 않는다. 한 sidecar를 공유하는 WAV들은 동일한 원천 메타여야
한다. 원천을 바꾸려면 별도 수집 세션으로 계획한다. 같은 연속 수집을 폴더만 나눠 서로 독립적인
세션이라고 주장하거나 서로 다른 split에 넣지 않는다.

## 3. 수집 후 무변환 QA

WAV는 `raw/SPLIT/SESSION/*.wav`에 두고 같은 폴더의 capture를 완성한다.
예: `gains.adc_reference={"value": 12, "unit": "dB"}`는 **형식 예시일 뿐 권장 gain이 아니다.**

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/prepare_measurement_session.py \
  --input results/measurement_packets/NEW_PACKET/raw \
  --out results/measurement_intake/NEW_AUDIT
```

전체 PCM 형식·truncation·원본 채널 hash·session/source 중복·split 간 동일 채널을 검사한다.
정확한 PCM rail 값 **−32768 또는 +32767**이 있으면 실패한다. `abs(PCM)>=32440` 횟수는
별도 near-rail 관측 통계이며 성공/실패 문턱으로 쓰지 않는다. 인위적 gain 보정은 하지 않는다.

성공해도 `physical_synchronization_certified=false`, `anc_off_automatically_verified=false`다.
운영자의 clock/배선/gain/누락/ANC OFF 선언과 파일 수치 검사를 구분한다. 시간 이동·부분 복제·
재인코딩 누출을 완전히 검출하지 못하므로 source/session 기록이 중요하다.
`source_coverage.recordings_by_split_and_family`는 실제 인입 파일의 종류별 개수다. noise만
검사에 통과했다고 speech/music/mixed의 준비가 끝난 것이 아니며, 어떤 개수로도 모든 소리의
감쇠 성능을 인증하지 않는다(`all_sound_readiness_claim_allowed=false`).

실패 시 `report.json`만 기록하고 학습 manifest를 발급하지 않는다. 출력은 덮어쓰지 않으므로
수집/메타 수정 후 다른 새 출력 이름으로 다시 검사한다. 실패 원본도 자동 삭제하지 않는다.

## 4. 기존 train/valid 분리 도구 연결

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/prepare_measurement_session.py \
  --input results/measurement_packets/NEW_PACKET/raw \
  --out results/measurement_intake/NEW_PREPARED --prepare
```

통과한 경우에만 기존 `tools/prepare_recordings.py`의 `prepare_recordings()`를 호출한다.
기존 PCM16 채널 바이트·gain·동기·train/valid manifest 규약을 그대로 보존한다.

```text
NEW_PREPARED/
  train_valid/manifest.jsonl      # 기존 deepanc train/valid 계약 그대로
  train_valid/preparation.json
  train_valid/train/...wav
  train_valid/valid/...wav
  final_test_lock.json            # 원본 test 경로/PCM hash/메타만; 학습 입력 아님
  report.json
```

`final_test_lock.json`의 `records`는 raw root 기준 경로와 session/source 식별자, PCM hash다.
이 단계에서 test의 파일 QA는 수행하지만 학습/튜닝 하네스는 **test의 lock·검수·capture 메타만 읽고
test WAV를 열지 않아야 한다.** 기존 train/valid manifest에는 test가 전혀 들어가지 않는다. 원본 test는 그대로 둔다.
PCM hash는 raw 채널 바이트의 SHA이며 WAV 헤더 hash나 `deepanc.data.pcm_sha256()`의
rate/width 접두어가 붙은 hash와 다른 정의다. 서로 직접 비교하지 않는다.
`report.recordings`에는 `session_id/source_id/source_kind/split` 식별자도 명시된다.
실제 분리된 train/valid에는 `prepared_reference/prepared_disturbance`가 출력 루트 기준 상대
경로로 들어간다. audit-only 및 test 행의 두 경로는 항상 null이며 평가용 pair를 임의 생성하지 않는다.

검사/복사 중 변경이 발견되면 `train_valid.pending`이 남을 수 있다. 최종 report의
`intake_passed=true`와 `prepared_train_valid=true`, 최종 manifest와 test lock을 함께 확인한다.
부분 폴더·manifest 하나만으로 준비 완료를 선언하지 않는다. 입력 원본은 항상 보존한다.

이 단계는 **학습 실행·고역 감쇠·실시간 배포 준비 완료가 아니다.** 자료 확보 뒤 공통 경로·출력
한도·인과성·검증 세션·고정된 최종 평가 규약으로 별도 학습 승인과 비교 절차를 진행한다.
지연·지터·고역 S/피드백·반복 OFF/ON/OFF 실측은 여전히 현장 확인 항목이다.

## 5. 실측 자료를 SFANC 준비 계획으로 연결

수집 검수 후 `configs/sfanc_paired_recipe.template.json`의 사본에 확인한 추가 지연·출력 한도,
창/stride·FFT·탭 수·대역 비용·학습 설정을 명시한다. bank 후보의 `reference`는 준비된
`train_valid/manifest.jsonl`의 **train 상대경로**, start/count는 그 원녹음의 샘플 단위다.
validation/test에서 bank를 계산하거나 다른 녹음을 인위적으로 접합하지 않는다.
`null`은 실행을 막으며 과거 실험 설정을 자동 채우지 않는다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/train/prepare_sfanc_paired.py \
  --packet results/measurement_intake/NEW_PREPARED \
  --recipe PATH_TO_EXPLICIT_RECIPE.json --out results/sfanc_paired/NEW_PLAN
```

이 명령은 원본 S·QA/capture·manifest·train/valid PCM hash·분할을 검증하고
`plan.json`, `ready.json`만 만든다. test WAV·특징 계산·SVD·신경망·optimizer를 실행하지 않는다.
`ready.json`은 계획 파일의 완성 표시이지 학습/배포 허가가 아니다. 입력이 바뀌면 준비를 다시 한다.
후속 `train_paired()`는 기본 거부이며 별도 명시 승인용 코드만 마련했다. 이번 단계에서는 실행하지 않는다.

학습 라벨/필터 목적의 한계와 기존 합성 결과와의 차이는 [docs/19](19_high_frequency_comparison.md)를 따른다.

## 6. FxLMS/FxNLMS 비교 준비

`baseline_search.template.json`에 동일 S·추가 지연·출력 한도와 μ/탭/블록/epsilon/계수 제한 후보를
명시한다. `measurement_packet`은 위 검수·분리가 완료된 `NEW_PREPARED` 디렉터리다.
`validation_case_manifest`는 `validation_cases.template.json`을 채운 파일이며, 각 행은
**검수된 valid 파일과 session/source 식별자**를 그대로 참조한다. train/test 경로로 바꾸지 않는다.
case 경로는 case JSON 부모 기준, packet/case manifest 경로는 plan 부모 기준, S는 저장소 기준이다.
현재 baseline case WAV 상한은 파일당 1,048,576샘플(16 kHz에서 65.536초)이다.
초과하면 자동 자르지 않고 거부하므로 수집/분석 구간을 사전에 계획한다. 경로의 symlink는 허용하지 않는다.
onset/transition/steady는 서로 겹치지 않는 `[start, stop)` 샘플 구간의 목록으로 사전에 정한다.
최종 test를 보고 구간·후보·문턱을 바꾸지 않는다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/eval/tune_classical_baselines.py \
  --plan PATH_TO_BASELINE_PLAN.json --out results/baseline_plan/NEW_CHECK
```

기본 동작은 **WAV를 열지 않는 계획 검사**다. 미확정 설정은 exit 2와 보고서를 남긴다.
`--run-validation-search`는 향후 명시 승인된 validation 탐색 때만 사용하며 지금 실행하지 않는다.
동결 결과는 후보·실패·설정/S/S_hat/입력 근거의 hash를 보존한다. 후보가 모두 부적격이면 실패를 유지한다.
최종 test 실행과 새 DL 학습은 이 준비 명령에 연결하지 않았다.

## 7. 현장 재개 때 남은 순서

1. 나중에 제공할 보드 식별로 가용 메모리·lossless 수집 경로·별도 녹음 프로젝트를 검토한다.
2. 사용자 입회하에 raw REF/ANC-OFF ERR과 gain/clock/누락 증거를 수집하고 위 QA를 통과시킨다.
3. 실제 출력 체인의 고역 S·REF 선행·종단 지연/지터·피드백을 확인한다. 기존 S와 다르면 경로를 분리한다.
4. 미확정 설정/비교 규약을 확정하고 새 학습·validation 탐색을 별도로 승인받아 실행한다.
5. 모델·기준선을 동결한 뒤 독립 최종 세션으로 저역/고역 및 음성·음악·혼합을 비교한다.

현재는 1번의 보드 정보를 기다리는 동안 가능한 파일·수치 준비까지 진행한 상태다.
**고역 우위·실시간 동작·수집 펌웨어 완료를 선언하지 않는다.**
