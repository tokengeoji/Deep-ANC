# 19. 1 kHz 이상 딥러닝 ANC 비교 — 학습 전 준비

## 목표와 중단선

사용자 목표는 **1 kHz 이상에서 딥러닝 ANC가 충분히 튜닝한 FxLMS/FxNLMS보다 좋은 감쇠**를
보이는 것이다. SFANC는 현재 준비 중인 후보이지 우위가 입증된 해답이 아니다.
1000–1600 Hz는 첫 검증 구간이며 상한이 아니다. 저역 동시 감쇠와 음성·음악 quiet zone도 유지한다.

현재 지시는 **새 학습을 시작하지 않고 학습 전까지 준비**하는 것이다.
모델 학습·bank 재설계·하이퍼파라미터 탐색·오디오 출력은 실행하지 않는다.
`prepare_jetson.sh`와 전체 pytest에는 짧은 합성 학습이 포함되므로 이 단계의 점검으로 쓰지 않는다.
기존 실행 결과는 [docs/18](18_sfanc_pretraining.md)에 보존한다.

## 공통 비교 계약

- 기준 S: OMAP `rir.txt`, 16 kHz, 500탭. 원본 SHA·gain·극성·선행 탭을 보존한다.
  `u`는 실제 DAC 명령, `e=d+S*u`이며 추가 지연은 명시한 것만 한 번 적용한다.
- 모든 방법에 같은 원천·P/S·레벨·실제 출력 한도와 동일한 정보 가용 조건을 적용한다.
  S_hat 정확도, ERR 피드백 시점, 참조 관측창, 계산/전송/버퍼 지연 차이를 숨기지 않는다.
  같은 조건의 알고리즘 비교와 장치별 실제 종단 성능 비교는 구분한다.
- 16 kHz 비교의 주 지표는 **[1,8] kHz 감쇠 차이**다. [1,1.6), [1.6,8] kHz와
  저역 [0,1) kHz·전대역도 함께 남긴다. Nyquist 끝점만 포함하며 무신호 대역을 성공 점수로 채우지 않는다.
  8 kHz는 이 샘플레이트의 분석 한계이지 실제 장치 신뢰대역 인증이 아니다.
- 음성·음악·소음과 동시 혼합, 초기 적응·음원/경로 전환·충분한 적응 후 정상상태를 따로 보고한다.
  증폭·제한 전 초과·실제 clip·발산·미평가 결과를 버리지 않는다.
- FxLMS/FxNLMS의 step size·탭 길이·epsilon·적응 시간과 모델/필터 선택은 train/validation에서만 결정한다.
  탐색 후보·예산·선택 지표·동점 규칙·실패 기록을 test 전에 동결한다.
  현재 단일 mu의 cold-start 진단을 강한 기준선으로 부르지 않는다.
- 무제어, 동일 train으로 만든 고정 FIR, 동일 bank의 비학습 PSD 선택을 함께 대조한다.
  test P/S를 알고 재설계한 FIR은 정보 우위가 있는 진단이며 정식 학습 후보와 혼합하지 않는다.
- 최종 비교는 새 독립 세션/화자·작품 그룹 단위 paired 감쇠 차이와 불확실성을 남긴다.
  한 원녹음의 crop 수를 독립 표본 수로 세지 않는다. 이미 관찰한 test는 새 test가 아니다.

최소 실용 개선 dB, 허용 저역 악화, 실제 추가 지연·출력 한도, 최종 학습 분포와 후보 선택은
아직 확정하지 않았다. [준비 계약](../configs/high_frequency_comparison.json)의 `null`을
0이나 과거 기본값으로 채우지 않는다. 합격 기준을 사후에 유리하게 바꾸지 않는다.

## 구현한 준비와 남은 검증

| 항목 | 현재 준비 | 아직 아닌 것 |
|---|---|---|
| 비딥러닝 기준선 | plain FxLMS/FxNLMS와 validation 전용 후보 탐색·실패 기록·선택 동결 코드 | 실제 탐색 실행·강튜닝 완료·원본 펌웨어 재현 |
| SFANC | 실측 QA packet → raw REF/d → 준비 계획 → 별도 승인 후 bank/선택기 학습 연결 | 실측 자료·명시 recipe·새 학습 실행·연속 실시간 검증 |
| 공통 지표 | 고/저/세부대역·초기/전이/정상 구간, clip·증폭·세션 단위 paired bootstrap | 실측 감쇠·독립성·최소 실용 우위 인증 |
| 수집 후 QA | 채널/PCM/출처/세션/gain 선언 검사, train/valid 분리, 최종 test 잠금 | 보드 녹음 구현·물리 동기/ANC OFF 자동 인증 |
| HybridANCNet | 기존 48 kHz 구조·스트리밍 구현 보존, 선택적 비교 후보 | OMAP 16 kHz 동조건 학습 하네스·학습본 |
| `deepanc.CausalController` | 원본 S용 별도 작은 인과 신경망 | HybridANCNet이나 SFANC와 같은 모델 |
| 정적 준비 CLI | 계약·원본 S·원천 경로·분할/이전 test 노출·미확정 사항 검사 | PCM/라이선스/수집 동기성 인증·자동 학습 허가 |

새 기준선은 `run_classical_anc(..., algorithm="fxlms"|"fxnlms")`다.
샘플별 FxLMS는 `w -= mu*e*xf`, FxNLMS는 이를 `sum(xf**2)+epsilon`으로 나눈다.
블록 모드에서는 합산 gradient와 블록 전체 정규화를 쓰므로 같은 mu의 샘플별 갱신과 같지 않다.
FP64 실제 S와 별도 S_hat, 지연, hard clip 후 S 꼬리까지 적응 보류, 계수 norm 제한을 기록한다.
기존 `FxLMSController` 이름의 정규화 동작은 호환성을 위해 변경하지 않았다.

아래는 **학습을 부르지 않는** 준비 명령이다. 출력은 존재하지 않는 새 폴더를 사용한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python scripts/train/prepare_high_frequency.py \
  --config configs/high_frequency_comparison.json \
  --out results/high_frequency_readiness/NEW_RUN
```

선택적 `--inventory`에는 회수한 Drive 감사 JSON, `--capture`에는 수집 메타 파일을 연결할 수 있다.
이는 파일 존재·크기·SHA만 확인하며 자료 자격을 자동 인증하지 않는다.
종료 **2**는 보고서를 만들었으나 학습 준비가 미완료라는 뜻이고, **1**은 입력 오류다.
이 버전은 정적 감사만 제공하므로 `training_ready=false`를 유지한다.
`training_executed/optimizer_created/model_instantiated/forward_backward_executed/bank_fitting_executed`는 모두 false다.
미확정 사항을 채웠다는 이유만으로 실제 데이터 QA·학습 하네스 검증이 끝났다고 하지 않는다.

실측 전 준비 패킷과 이후 파일 흐름의 단일 실행 안내는 [docs/20](20_measurement_runbook.md)다.
SFANC 라벨은 과거 REF만 특징으로 보고, 다음 창의 실제 `d`와 hard clip 후 `S*u`로 비용을 만든다.
저역·[1,1.6) kHz·[1.6,8] kHz 가중치는 명시 설정이며 과거 가중치 3을 자동 사용하지 않는다.
FIR bank 후보 계산은 기존 **전대역 선형 ridge 제안**을 재사용한다. 라벨의 대역 가중/clip 평가와
구분하며 bank 자체가 고역·비선형 제한에 최적화됐다고 하지 않는다. 라벨은 zero-start 창 진단이며
연속 필터 교체 성능을 대신하지 않는다. 미래 승인 학습 연결은 현재 mock 회귀만 검증했다.

기준선 선택 코드는 정상상태 [1,8] kHz 감쇠의 세션별 평균을 최대화하고 정확한 동점은 후보 ID로
결정한다. 탐색 예산·clip·저역 악화 문턱·실제 추가 지연은 명시값이 필요하다. 후보 전부 실패하면
선택을 발급받았다고 간주하지 않는다. 실패/미평가/수치 floor 결과를 보고서에서 제거하지 않는다.
이 목적·구간이 최종 비교에 적합한지 사전 확정해야 하며 실제 탐색은 아직 하지 않았다.

최신 검사 수와 로그는 [HANDOFF](../HANDOFF.md)에만 유지한다. 무학습 실행 목록은 아래와 같다.
모델 학습·SVD fitting을 포함한 전체 pytest는 재실행하지 않는다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q -o addopts= -ra \
  tests/test_classical_anc.py tests/test_high_frequency_readiness.py \
  tests/test_sfanc_stream.py tests/test_sfanc_fxnlms.py tests/test_sfanc_sources.py \
  tests/test_recordings.py tests/test_omap_prepare_secondary_path.py \
  tests/test_metrics.py tests/test_acoustic_readiness.py \
  tests/test_measurement_ready.py tests/test_pre_measurement_packet.py \
  tests/test_sfanc_paired.py tests/test_high_frequency_metrics.py \
  tests/test_baseline_selection.py tests/test_baseline_selection_cli.py \
  tests/test_interleaved_probe.py tests/test_interleaved_time_origin.py \
  tests/test_omap_secondary_path.py::test_authoritative_measurement_matches_successful_controller
```

## 실제 찾은 데이터

2026-09-20 현재 로컬에는 LibriSpeech dev-clean FLAC 2,703개가 있다.
기존 SFANC의 선택 100파일은 존재하며 저장 manifest의 화자·책·원천 그룹 분할은 겹치지 않는다.
다만 기존 test 40crop/20파일과 그 화자·책 그룹은 이미 관찰했다. 새 독립 최종 test로 재사용하지 않는다.
옛 `speech.jsonl`은 화자 누수가 있으므로 경로만 고쳐 새 분할로 쓰지 않는다.

FMA는 로컬 원본 MP3가 없고 CSV·과거 접합 음원만 있다. ESC-50 raw 4개와
DEMAND kitchen 1채널은 전체 corpus가 아니다. machine 접합 WAV는 ESC 유래이며 MIMII 원본이 아니다.

Drive에서는 기존 공개 원본 **13개, 192조각 + 13 manifest, 합계 18,599,035,802 byte**의
ID·이름·크기를 기존 전송 receipt와 대조했다. LibriSpeech train/dev/test, FMA small/metadata,
ESC-50, DEMAND 6환경, MIMII DG fan이 대상이다.
19,663,154 byte 메타데이터 ZIP만 회수했고, 과거 receipt의 ZIP SHA 및 내부 90파일 SHA를 검증했다.
원격 PCM 조각을 복원·해시 검증한 것은 아니다. 과거 문서/48 kHz 합성물을 현재 저장소 위에 풀지 않았다.

- 공개 음원은 합성 학습의 소스 후보이며 OMAP 동기 REF/ERR 녹음이 아니다.
- FMA의 과거 7개 QA 실패·디코더 경고와 개별 라이선스/artist 분할 검토가 남아 있다.
- MIMII DG는 사용자 확정대로 **학습 보조 전용, val/test 제외**다. 원래 공식 split 메타는 보존한다.
- speech는 화자·책, music은 artist·작품, ESC는 원녹음, DEMAND 동시 16채널은 같은 그룹으로 묶는다.
- 여유 공간은 시점마다 다르므로 복원 전에 확인하며 전체 18.6 GB 원본을 무작정 복원하지 않는다.
  필요한 학습 분할을 확정한 뒤 기존 Drive 자료를 검증하며 단계적으로 회수한다.

개인 Drive ID·receipt·검사 산출물은 Git 제외
`results/drive_readiness/20260920_01/`에 보존한다. 기존 Drive 파일/공유/백업을 변경하거나 삭제하지 않았다.
검사한 프로젝트 폴더·이름 검색·대표 세션 메타 범위에서는 적합한 OMAP 녹음을 찾지 못했다.
이는 Drive 전체와 모든 압축파일 안에 없음을 증명한 전수 검색은 아니다.
확인한 과거 `mics.wav` 세션 메타는 48 kHz이며 OMAP 자료로 승격하지 않는다.

## 필요한 실측 — 연결을 재개할 때

1. **16 kHz raw REF + ANC-OFF ERR 동기 연속 녹음.** 동일 ADC clock·공통 디지털 스케일,
   누락/중복 검사, 채널·gain·배치·음원 출처를 함께 기록한다. 별도의 train/valid/최종 test 세션을 둔다.
   소음뿐 아니라 음성·음악·혼합과 여러 레벨을 포함한다. 독립 정규화·미래 정렬을 하지 않는다.
2. **현재 출력 경로의 1 kHz 이상 S 검증.** 원본 OMAP S는 보존하되 실제 사용할 출력→ERR의
   크기·위상·반복 일관성·레벨 선형성을 확인한다. Jetson USB DAC 등 출력 체인이 달라지면
   기존 OMAP S를 그대로 실제 경로로 인증하지 않는다.
3. **REF 선행 시간과 종단 지연/변동.** ADC 수집·처리·전송·DAC·음향 경로와 CS→REF 피드백을
   확인한다. FIR/CNN 단독 계산시간을 종단 지연으로 대신하지 않는다.
4. 학습·안전 게이트를 통과한 뒤 같은 조건에서 FxLMS/FxNLMS와 DL의 반복 OFF/ON/OFF를 비교한다.

이 목록은 지금 스피커 출력·펌웨어 변경을 승인한 것이 아니다.
실제 보드/수집 방식은 현장 단계에서 확인하며 원본 펌웨어를 보존한다.
실기 실행은 사용자 입회·볼륨 최소·ANC OFF 시작 조건을 따른다.
