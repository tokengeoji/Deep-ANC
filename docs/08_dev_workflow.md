# 08. 개발 워크플로와 프로젝트 정책

## 1. 저장소 정책

| 정책 | 내용 |
|---|---|
| **anc_project 읽기전용** | `~/anc_project`(기존 FxLMS)는 절대 수정 금지. 검증된 코드(fxlms_core)와 측정 자산(npz)의 **복사만** 허용 — 출처를 주석으로 명기 |
| **Jetson 시스템 불가침** | 핀 설정(pinmux/I2S)·RT 커널·전원모드·오디오 데몬·apt 설치 등 **시스템 변경 금지** (의도된 실험 구성). 모든 도구는 venv/유저 공간에서만 |
| 대용량 산출물 | `data/`, `runs/`, `*.pt`, `*.onnx`, `*.plan` 은 .gitignore — 가중치는 GitHub Release 자산으로 |
| 커밋 자산 예외 | `assets/measured/*.npz` (수십 KB 측정 자산)는 저장소에 포함 |

legacy `main_realtime_anc.py`는 정상 종료 때도 기본 `control_filter_last.npy`를 저장한다.
따라서 과거 300Hz/약 2dB baseline 명령을 원본 디렉터리에서 그대로 실행하면 읽기전용 정책을
위반한다. 재현이 필요할 때만 `--weights-output /home/capston/Deep_ANC/results/legacy_fxlms/control_filter_last.npy`
처럼 저장소의 ignored 경로를 명시하고, `python3 -B`로 `__pycache__` 쓰기도 막는다.
그 외 원본 파일은 읽기만 한다.

## 2. Jetson ↔ Elice 동기화 (git 허브 모델)

```
[Jetson]  코드 개발·실기 검증 → git push
                                   ↓
[GitHub]  단일 동기화 지점 (원격 저장소)
                                   ↓ git pull/clone
[Elice]   학습 실행 → 작은 산출물(설정, metrics, 로그 요약)만 커밋
          가중치/onnx 는 zip 다운로드 또는 GitHub Release 업로드
```

- 브랜치: `main` 은 항상 실행 가능 상태 유지. 실험은 `exp/<이름>` 브랜치.
- 실측 녹음(data/recorded)은 크기가 작으면 zip 으로 Elice 에 직접 업로드,
  크면 `pack_transfer.py` 샤드.

## 3. 재현성

- 모든 실행은 config yaml 이 단일 출처 — CLI 는 `--set key=value` 오버라이드만.
- `train/reproducibility.py` 가 run 디렉토리에 config 스냅샷·git rev·pip freeze 자동 기록.
- 체크포인트는 모델+옵티마이저+스케줄러+step+RNG 를 포함해 `--resume` 완전 재개.
- corrected checkpoint는 resolved model/data/duct 전체와 `physics_status`,
  `trusted_band_hz`, `digital_reference_lead_samples`를 보존한다. ONNX 동반
  JSON에도 lead를 복사하며, 런타임 불일치는 오디오 시작 전에 거부한다.
  해당 키가 없는 legacy artifact는 lead=0으로만 해석한다.
- 시드: 학습 `seed`(+rank), 데이터 분할 고정 시드, val 배치 고정 시드(1234/999).
- 버전 고정: Elice torch 2.5.1+cu121 ↔ Jetson 2.5.0a0(JP6.1) 정렬, ORT 1.18.1(Jetson).

## 4. 코드 품질 게이트

```bash
.venv/bin/python -m pytest -q  # 커밋 전 전체 통과 필수
```

핵심 불변식 (테스트가 강제):
1. 모델 인과성 — 미래 입력 무의존 (비트 단위)
2. 스트리밍 = 오프라인 등가 (≤1e-5), GLSTM nn.LSTM = 수동 셀 등가
3. S(z) torch = scipy 등가, 극성 규약(e = d + S·y, 추가 반전 금지)
4. 덕트 시뮬이 이론 공진(70/210/350Hz) 재현
5. 데이터 분할 무누수 (파일/세션/RIR 변형 단위)
6. digital-ref 연속 source와 런타임 FIFO의 +109 정렬, acoustic-ref의 nonzero lead 거부
7. checkpoint/ONNX lead 메타·legacy0 호환·런타임 mismatch fail-fast
8. trusted-band 목적함수와 fullband 관측 지표 동시 산출, 평가 시 trusted−fullband 간극 저장

## 5. 단계별 진행 체크리스트

### 5.1 corrected Stage-1 — 표현 사전학습

- [x] `P(z)=S(z)` scale-matched surrogate, +109 lead FIFO, 공칭 선형 plant로 영출력 해의 원인 제거
- [x] trusted NMSE(현 **150–1600Hz**)를 학습/체크포인트 선택에 쓰고 fullband NMSE를 동시 로깅
- [x] +109 학습↔실시간 FIFO 정렬, artifact 메타, mismatch fail-fast 자동 테스트
- [ ] base/tiny 완주 후 `best.pt`·`last.pt`·resolved config·로그 회수

> 이 단계의 `secondary_surrogate` 체크포인트는 표현 학습 결과이다.
> 실제 덕트 감쇠, 고역 성능, 음성·음악 quiet-zone 성능으로 표기하지 않는다.

### 5.2 파인튜닝 준비 — 사용자 입회 실측

- [x] APE `hw:1,1`·AB13X `hw:2,0` 장치 인식과 48kHz/2채널 스트림 설정 확인
- [x] `.venv/bin/python scripts/bench/check_audio_input.py`로 ERR ch0 무출력 probe PASS
- [x] `--require-both`로 ERR/REF 모두 PASS(pin17 REF L/R 복구)
- [x] ERR/REF 과클리핑 원인이던 빠진 pin17을 재연결; Jetson pinmux/I²S는 변경하지 않음
- [ ] 동일 앰프·볼륨·오디오 설정에서 `P(z)`(noise→ERR)와 `S(z)`(cancel→ERR) 반복 실측
- [x] 각 반복 일관성 확인 — **요구 대역(150–1600Hz) 모든 부대역 ≥0.9406**, 유지 반복 ≥8,
  **P−S 상대 τ 궤적 상수성**(총계만 보면 오염을 놓친다 — 실제로 놓쳤다). 스피커 THD/IMD 는 미확인
- [x] 실측 순수지연으로 `K=(S delay+256)-P delay`를 재계산 → **K=116** (S 1462 / P 1602).
  학습 설정(`configs/duct.yaml`, `data_sim.yaml`)은 갱신됨. **배포 artifact 는 아직 109**
  (그 ONNX 가 109 로 학습됐기 때문 — 런타임이 불일치를 시작 전에 거부한다)
- [ ] **recorded 80세션 재녹음** — 현 데이터는 재생↔녹음 시간축이 붕괴해 전량 격리됨
  (`data/recorded_broken/`). coh²(source→ERR) 0.021~0.126 vs coh²(REF→ERR) 0.959~0.991
- [ ] 소음·음성·음악·환경·기계음을 source family×대역으로 나눈 독립 세션 수집
- [ ] 화자·곡·환경·기계 조건 그룹을 가로지 않는 8:1:1 train/val/test 생성

`make_recorded_manifest.py`는 같은 화자·곡·원본·환경의 `group_id`를 원자 단위로
보존하고 `source_family`별 8:1:1 층화를 수행한다. `path_base: manifest` 상대경로라
Jetson→Elice 전송 뒤에도 재생성하지 않는다. `validate_recorded_sessions.py`의 파일·클립·
무음·family×split QA가 PASS해야 최종 split로 간주한다. 스피커 출력을 내는 모든 항목은
사용자 입회·볼륨 최저·ANC OFF 상태에서만 실행한다.

2026-08-03 빠져 있던 pin17(REF L/R)을 재연결한 뒤 ERR/REF는 −46dBFS대, clip 0%로
두 채널 probe를 통과했다. 이 PASS는 입력 생존 확인이지 성능 결과가 아니다. legacy FxLMS,
`record_duct`, P/S 보정, `evaluate_session` 직전에 probe를 반복하고 사용자 입회·볼륨 최저를
확인한다. sudo, Jetson-IO, pinmux/device-tree, RT 커널, 전원모드, 오디오 데몬은 변경하지 않는다.

### 5.3 실측 파인튜닝·배포 게이트

- [x] 합성 offline·실기 session 평가에서 S(z) 실측대역∩덕트 목표대역을 trusted로
  산출하고 trusted/fullband/간극을 Markdown+NPZ에 저장(소스별·옥타브 지표 유지)
- [ ] `duct.digital_reference.primary_path_npz`와 실측 `d_noise_delay_samples`를 지정하고
  `digital_primary_path_mode: measured`로 open-loop 파인튜닝
- [ ] 실측 70%+합성 30% 후 closed-loop 20k–50k를 별도 ablation으로 검증
- [x] Trainer의 고정 합성 val 16개와 분리된 recorded val/test 평가기 구현
- [ ] 실제 독립 test를 수집해 trusted/fullband, 소스×대역, 최악 10% G4 PASS
- [ ] ONNX export → lead 메타 정합 → tiny ORT P99<3ms → FxLMS와 동일
  OFF 10s→ON 30s→OFF 5s 세션 실기 비교
- [ ] 덕트 문서 미확정 항목(ERR 위치 등) 확정 시 duct.yaml 갱신 + RIR 뱅크 재생성
- [ ] `trtexec`가 사전 제공된 별도 환경에서만 base TRT를 검증

실측 `P(z)`로 파인튜닝하고 학습에 쓰지 않은 recorded test를 통과하기
전에는 체크포인트와 실기 리포트를 물리 성능 결과로 공개하지 않는다.
