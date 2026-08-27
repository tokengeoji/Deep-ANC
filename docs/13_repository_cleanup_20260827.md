# 저장소 정리·보존 감사 기록

정리일: 2026-08-27 (Asia/Seoul)

## 범위와 원칙

이번 정리는 `/home/capston/Deep_ANC` 안의 명백한 임시 파일과 Python/test 캐시만
대상으로 한다. 다음은 삭제하지 않는다.

- `data/`의 raw/recorded/manifest/RIR 원본
- `results/`의 raw capture, calibration, 평가, legacy/diagnostic 결과
- `runs/`의 checkpoint, ONNX/TensorRT, 학습 로그와 pilot 결과
- `assets/measured/`의 strict 및 legacy 측정 artifact
- `~/anc_project`, `~/DeepANC_CRN_n_codex`와 그 하위 파일
- 개인키 `.pem` 및 기타 비밀정보

legacy artifact는 성능 근거로 사용하지 않더라도 재현·감사에 필요할 수 있으므로
HANDOFF 정책에 따라 보존한다.

## 삭제한 항목

### 임시 readiness 검증 디렉터리

두 디렉터리는 이름 자체가 임시 삭제 검증용이고, strict 승격 전 구형 readiness snapshot만
포함했다. 아래 SHA를 기록한 뒤 삭제했다.

| 경로 | 파일 SHA256 | 삭제 이유 |
|---|---|---|
| `results/_audit_tmp_delete_me/readiness.json` | `239a81c259637a24…` | 임시 readiness snapshot |
| `results/_audit_tmp_delete_me/readiness.md` | `8521effdbbb92572…` | 임시 readiness snapshot |
| `results/_audit_tmp_delete_me2/readiness.json` | `70a97f549e2d0144…` | 임시 readiness snapshot |
| `results/_audit_tmp_delete_me2/readiness.md` | `750826c5683cf8b58…` | 임시 readiness snapshot |

공식 strict 결과와 현재 판정은 `assets/measured/measurement_level_evidence.json`,
`assets/measured/primary_path_il_strict_5dc06fdd.npz`,
`assets/measured/secondary_path_il_strict_5dc06fdd.npz` 및 `HANDOFF.md`에 보존되어 있다.

### stale audio lock

`results/.live_audio_uid_1000.lock`는 내용이 `{"pid":78467,...}`였지만 PID 78467이
존재하지 않았고 `measure_paths_interleaved`, `set_amp_level`, `record_duct`,
`run_realtime`, `evaluate_session` 프로세스도 없었다. 오디오 장치를 점유한 프로세스가
없음을 확인한 뒤 삭제했다. 새 측정은 항상 새 lock을 생성하고 종료 시 해제해야 한다.

### 재생성 가능한 캐시

- 저장소 루트 `.pytest_cache/`
- 저장소 코드·테스트 아래의 `__pycache__/`와 `*.pyc`

`.venv/` 내부 캐시는 환경 자체의 일부이므로 삭제하지 않는다.

## 보존·정규화 규칙

앞으로 생성하는 artifact는 다음 규칙을 따른다.

```text
runs/<stage>_<experiment-contract-sha-prefix>_<seed>/
results/<domain>/<YYYYMMDD_HHMMSS>_<capture-id>/
results/provenance/<content-addressed-name>.<sha256>.json
```

- 공식/legacy/diagnostic을 디렉터리명과 metadata의 `experiment_role`로 구분한다.
- raw → analysis → model → evaluation의 SHA와 commit을 metadata에 결속한다.
- 임시 산출물은 `results/_tmp_*`로 만들고, 승격하지 않으면 같은 작업에서 제거한다.
- 새 측정은 기존 디렉터리를 덮어쓰지 않는다.
- 모델 선택 전에는 test와 Level-5 challenge artifact를 열지 않는다.

## 정리 후 확인

- `git status --short --branch`에서 추적 파일 변경은 이 문서와 HANDOFF 갱신만 남긴다.
- `git diff --check`를 통과시킨다.
- strict P/S, recorded raw, checkpoint와 Elice pilot 디렉터리는 존재·SHA를 재확인한다.
- 오디오 점유와 stale lock을 다시 확인한 뒤에만 현장 측정을 허용한다.

이번 정리는 데이터·체크포인트·legacy 진단 evidence를 삭제하지 않았으며, 삭제된 임시
파일은 위 SHA로 식별할 수 있다.

## 검증 스냅샷 (2026-08-27)

- `.venv/bin/python -m pytest -q`: **0 FAIL** (전체 통과). 로컬에 없는 Elice public
  manifest에 대한 기존 경고만 있었고 테스트 실패는 없었다.
- `bash -n scripts/elice/bootstrap_all.sh scripts/elice/setup_env.sh`: 통과.
- `git diff --check`: 통과.
- 로컬 `scripts/train/check_finetune.py --config configs/train_finetune.yaml`는
  Elice bootstrap receipt SHA가 로컬에 없어서 canonical 학습 config stamp 단계에서
  중단되었다. 이는 readiness 우회나 실패 은폐가 아니라, public corpus가 없는 로컬에서
  의도된 차단이다. Elice의 receipt/manifest를 exact checkout에서 검증한 뒤 다시 실행한다.
- Elice에서는 4개 loss pilot 중 3개가 20k를 완료했고, `alpha=1.0, lambda_frame=0.2`가
  진행 중이다. 완료한 세 run의 기록과 decoder 경고(`libmpg123`)는 HANDOFF에 남겼으며,
  경고 원인 확인 전에는 canonical 학습으로 승격하지 않는다.
- 오디오 장치는 확인 시점에 PulseAudio control 장치만 점유했고 PCM 스트림 점유는 없었다.
  이번 정리에서는 스피커 출력·마이크 녹음·시스템 설정 변경을 수행하지 않았다.

## 브랜치 정리 원칙

현재 작업 브랜치는 `fix/finetune-readiness-repair` 하나로 고정하고, 원격과 동일한
`f299c31ee11a1e34fd7b09c5323f3cc2b2268c65`에 맞춘다. 다른 브랜치는 내용을 섞거나
삭제하지 않고 용도를 보존한다.

| 브랜치 | 현재 역할 | 처리 |
|---|---|---|
| `main` | 공개 기본선 | 변경하지 않음 |
| `fix/finetune-readiness-repair` | 현 감사·readiness 복구·Elice 인수인계 | 현재 작업·push 대상 |
| `work/canonical-training` | canonical 학습 코드·계약 작업선 | pilot 종료 후 필요한 변경만 선별 |
| `work/high-frequency-validation` | 고주파 측정 진단 작업선 | strict 증거가 생길 때까지 독립 보존 |
| `archive/*` | 역사적 분석/사전 rewrite 기록 | 삭제하지 않음 |

브랜치 간 cherry-pick/merge는 pilot이 끝나고 decoder audit 및 계약 SHA를 검토한 뒤
정확한 커밋 단위로 수행한다. 따라서 현재는 브랜치 파일을 임의로 합치거나 과거
결과를 canonical 결과로 재분류하지 않는다.
