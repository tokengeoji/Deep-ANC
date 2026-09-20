# 03. 데이터 파이프라인

최우선 준비 경로는 실제 REF 마이크를 사용하는 **acoustic-ref**다. 저역·고역과 음성·음악을
함께 평가하며 Jetson의 우선 개선 대역은 1000–1600 Hz다. 데이터 준비·보관 절차는
[docs/16](16_drive_acoustic_preparation.md), 현재 확보 상태는 [HANDOFF](../HANDOFF.md)를 따른다.
아래 도구는 모두 Docker 안에서 실행한다.

## 1. 신호와 지연 계약

학습 샘플은 입력 `x=(x_ref, err_in)`과 제어 전 ERR 신호 `d`다. 별도의 정답 상쇄 파형 대신
모델 출력 `y`에 미분 가능한 S 플랜트를 적용하여 **`e = d + S·y`**를 최소화한다.
극성은 측정 FIR에 포함되므로 추가 반전을 하지 않는다. open-loop의 지연된 `err_in` 근사는
실제 폐루프 응답과 구분한다.

| 모드 | REF와 d의 구성 | 선행 정보 |
|---|---|---|
| acoustic-ref | 합성에서는 `P_ref·n`과 `P_err·n`, 실측에서는 실제 REF/ERR 채널 | 마이크 기하·실측 경로의 선행 시간; digital lead=0 |
| digital-ref | 자기생성 소스의 연속 조각을 REF로 공급하고 재생은 FIFO로 지연 | artifact·학습·런타임에서 같은 digital lead 사용 |

현재 `configs/data_sim.yaml`의 digital lead 실행값은 **113**이다. 선택된 측정 NPZ의 S 지연은
**1465**, P 지연은 **1608** 샘플이며 `configs/duct.yaml`의 handoff는 **256**이다.
따라서 digital 정렬은 `K=(1465+256)-1608=113`, 총 S 지연은 1721샘플이다.
이는 설정·저장 자산의 계약이며 현재 장치의 재측정 결과를 뜻하지 않는다.

과거 surrogate artifact의 **S=1342, D_noise=1489, K=109**는 별도 이력이다. YAML에 남은
과거 주석보다 실행값과 artifact의 resolved 설정을 확인한다. 기존 checkpoint/ONNX의 메타데이터를
113으로 고쳐 재사용하지 않는다. digital 학습은 source를 `segment+K`만큼 연속 추출하여
`x_ref[t]=source[t+K]`를 구성하고, 런타임은 실제 재생을 K샘플 늦춘다.
acoustic 입력에 이 digital 선행 정보를 공급하지 않는다.

| `digital_primary_path_mode` | d 생성 | 해석 |
|---|---|---|
| `secondary_surrogate` | S의 FIR/gain과 설정된 `D_noise` 지연 | 표현 사전학습; 실제 P 응답을 검증한 것이 아님 |
| `measured` | 설정된 P NPZ의 FIR/gain/delay | 측정 조건·메타데이터가 일치해야 사용 |
| `rir_surrogate` | `p_err` RIR과 음향 onset을 제외한 추가지연 | 과거 호환·진단 경로 |

`data_sim.yaml`은 digital/surrogate 기본 설정이고, acoustic 준비 설정은
`configs/data_acoustic_prepared.yaml`이다. compact FIR·순수지연은 각각 한 번만 적용하며
RIR의 음향 onset을 추가지연과 중복 계산하지 않는다. 세부 물리는 [docs/01](01_physics_limits.md)을 따른다.

## 2. 소스 구성과 acoustic 준비

음성·음악·환경·기계·합성 신호를 포함하되 혼합 비율은 선택한 설정 파일이 단일 출처다.
기존 `data_sim.yaml`의 digital/acoustic 혼합과 strict acoustic 준비 설정을 섞어서 해석하지 않는다.
현재 준비 설정은 음성·음악을 포함하고, 저역 기본 합성과 1000–1600 Hz 목표 합성을 0.5 비율로
혼합한다. 이 비율은 연구 출발점이며 최적 비율·S 신뢰대역·감쇠 성공률이 아니다.

MIMII DG fan은 **train-only 학습 보조**다. `train_only_source_families: [machine]`와
소스 metadata의 `usage_policy: train_only_auxiliary`를 보존하고 val/test에서 제외한다.
평가는 나머지 소스 비율을 재정규화한다. 공식 split/domain/section은 보존하지만 section을
독립 장치나 원녹음 ID로 취급하지 않는다. 독립성을 모르는 녹음에 가짜 그룹 ID를 만들지 않는다.

`require_prepared_data: true` 경로는 필요한 원본·manifest·QA의 결손을 오류로 처리한다.
기존 범용 로더의 합성 폴백이나 파일 수만 확인한 목록을 실제 corpus 준비 완료로 간주하지 않는다.
정확한 staging 구조, PCM 전수 검사와 재사용 조건은 [docs/16 §4](16_drive_acoustic_preparation.md)를 따른다.

음성이나 음악을 acoustic-ref에서 일괄적으로 불가능하다고 분류하지 않는다. 필요한 예측 지평은
실제 REF 선행 시간과 전체 제어 지연으로 결정된다. 그 지평보다 상관시간이 짧은 예측 불가능한
성분은 미래 입력 없이 상쇄할 수 없으며, 주기·준정상 성분의 가능성은 따로 평가한다.
소스 명칭만으로 감쇠 성공을 가정하지 않고 대역·시간 구조·실측 잔차와 증폭 여부를 함께 본다.

## 3. RIR과 증강

`dsp/duct_sim.py`의 1D RIR은 덕트 기하와 반사·감쇠 변형을 사용한다. RIR 변형은 split을
분리하며, 평면파 모델이 고차 모드나 실제 장치 gain을 보증하지 않는다. acoustic 합성의
`P_ref/P_err`와 measured S를 함께 썼다는 사실만으로 절대 P/S 단위가 실측 검증된 것도 아니다.

레벨·마이크 잡음·hum·채널 dropout·피드백 지연은 선택한 설정에 따라 적용한다.
모델이 관측하지 못하는 독립 지연·all-pass·비선형 증강은 손실의 위상 정보를 상쇄할 수 있다.
물리 조건과 평가 근거 없이 증강 범위를 임의 확대하지 않는다. closed-loop 워밍업은
플랜트 적용 후 절단하며 plant/loss 계산은 FP32를 유지한다.

## 4. 실측 녹음과 분할

실제 수집은 [docs/14](14_pc_jetson_workplan.md)의 장치·입력·경로 확인 순서를 따른다.
스피커 출력은 사용자 입회·볼륨 최소에서만 수행하며 ANC OFF로 시작한다. 기본 Docker에는
오디오 장치를 노출하지 않는다. 과거 입력 probe 결과를 새 세션의 입력 상태로 재사용하지 않는다.

세션은 `mics.wav`, 필요할 때 `source.wav`, `session.json`으로 구성한다. ANC OFF 녹음의 ERR가
해당 세션의 d이고, acoustic-ref는 실제 REF 채널을 쓴다. digital-ref는 artifact의 K를 적용한
연속 source 조각을 사용한다. 화자·곡·원본·환경·기계 조건이 같은 반복 녹음은 같은 그룹에 둔다.

| 데이터 | 분할 단위 |
|---|---|
| 기존 공개 노이즈 풀 | 원본 파일 단위; segment별 분할 금지, 상관 누수는 추가 검토 |
| RIR bank | 변형 단위 |
| 실측 녹음 | `group_id` 원자 단위 + `source_family` 층화 |
| strict acoustic corpus | 소스별 공식 metadata·보수적 그룹과 train-only 정책 |

실측 manifest의 `path_base: manifest` 상대경로는 manifest 부모 기준이다. 전송 시 디렉터리
관계를 보존하며, marker가 없는 legacy 경로의 의미를 임의 변경하지 않는다.
`make_recorded_manifest.py`와 `validate_recorded_sessions.py`로 채널·SR·길이·finite·클립·무음·
메타데이터·그룹 누수·family별 평가 coverage를 검사한다. 진단용 게이트 완화는 최종 통과가 아니다.
`evaluate_recorded.py`의 저장 녹음 평가는 소리를 내지 않으며 독립 test와 실제 checkpoint 조건을 사용한다.

## 5. 보관과 이동

**데이터 원본의 최종 보관소는 Google Drive**다. 기존 자료를 우선 재사용하고 공식 checksum과
라이선스를 보존한다. PC Docker에 임시 다운로드·검증할 수 있으며 **Drive 업로드 파일 ID·크기·
완전성을 확인한 뒤 해당 PC 원본만 삭제**한다. 실패·미확인 파일은 남기고 정리 receipt를 기록한다.

학습 환경으로 이동하는 사본과 최종 원본 보관을 구분한다. manifest·원본 식별 정보·QA를 함께
전송하고 원본은 Git에 넣지 않는다. 온더플라이 합성은 학습쌍 저장을 줄이지만 archive·해제본·
변환본이 동시에 필요할 수 있으므로 공간이 충분하다고 가정하지 않는다. 세부 절차는 [docs/16](16_drive_acoustic_preparation.md)에 있다.
