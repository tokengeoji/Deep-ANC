# DeepANC → Deep-ANC 저장소 통합

## 정식 위치와 이력

- 최종 저장소: **https://github.com/tokengeoji/Deep-ANC**, 브랜치 **main**.
- 기반: 기존 `Deep-ANC/main`의 `1bbfb49839a05c4d01c9c31d1409ca41838ad89e`.
- 가져온 원본: `DeepANC/main`의 `e1b133c3c44830e6fed1cbaf08581b8a43311b52`.
- 공통 조상이 없는 두 Git 이력을 merge commit으로 연결한다. 강제 push나 이력 재작성은 하지 않는다.
- `Deep-ANC/dev`·`presentation`과 원본 `DeepANC` 원격 저장소는 수정·삭제하지 않는다.
- Git에 없던 녹음·체크포인트·보고서·가상환경은 전송하지 않는다. 기존 로컬 파일은 보존한다.

## 두 경로를 구분해서 사용

| 항목 | 기존 Jetson 경로 | OMAP 기준선에서 가져온 경로 |
|---|---|---|
| 패키지 | `src/deep_anc/` (`deep_anc`) | `deepanc/` (`deepanc`) |
| 기본 샘플레이트 | 48,000 Hz | 16,000 Hz |
| 2차경로 | `assets/measured/*.npz` + 설정의 명시적 지연 | 원본 `rir.txt`, 500탭, 선행 지연 포함 |
| 제어 출력/잔차 | 실제 출력 `y`, `e=d+S*y` | 실제 DAC 명령 `u`, `e=d+S*u` |
| 학습 진입점 | `scripts/train/` | `python -m deepanc.train` |
| 설정 | 기존 YAML 설정 | `configs/anc_train.json` |
| 안내 | `HANDOFF.md`, `docs/01`~`16` | `docs/JETSON_HANDOFF.md`, `DATASET.md`, `TRAINING.md` |

OMAP의 DSP 구현상 `u=-y`는 Python에 추가할 반전이 아니다. 원본 500탭의 이득·부호·지연은
`calibration/secondary_path.json`과 펌웨어 `S_hat` 비교로 검증한다. 리샘플링·정규화·잘라내기·
피크 정렬을 하지 않았으며 기존 48 kHz NPZ나 256샘플 handoff를 이 FIR에 적용하지 않았다.
OMAP에서 성공한 하드웨어 기준선은 Jetson 오디오 출력 경로의 교정값이 아니다.

기존 setuptools 패키징은 `src/deep_anc/`를 유지한다. `deepanc` 도구는 calibration·펌웨어·
원본 FIR을 함께 사용하는 **전체 저장소 checkout의 루트에서 실행**한다. 설치된 wheel만으로
OMAP 도구를 배포했다고 가정하지 않는다.

## 충돌 해결과 보존

- 기존 `Deep-ANC` README·AGENTS·Docker 규칙을 유지하고 OMAP 규칙을 추가했다.
- 기존 `requirements-jetson.txt`와 NVIDIA/ONNX 의존성 계약을 보존했다.
  OMAP 보조 목록은 `requirements-omap.txt`로 분리했으며 torch를 설치하지 않는다.
- 기존 `tests/test_secondary_path.py`를 보존하고 OMAP 검사를 `tests/test_omap_secondary_path.py`로 옮겼다.
  테스트 설정은 기존 `pyproject.toml` 하나를 사용한다. 두 패키지 테스트가 모두 수집된다.
- `tools/prepare_jetson.sh`는 Docker를 재사용한다. `--install`로 호스트 venv를 만들거나 torch를 교체하지 않는다.
- `scripts/train.py`·`scripts/utils/`는 가져온 legacy GCRN 음성 향상 코드다.
  `scripts/train/` 또는 OMAP ANC 목적함수의 대체물이 아니다.
- 원본 DSP 소스·`rir.txt`와 기존 Jetson 런타임·측정 자산은 변경하지 않는다.

## Jetson에서 이어서 진행

이미 `Deep-ANC`를 checkout한 Jetson은 기존 개발 컨테이너를 시작한 뒤 **컨테이너 안에서** 확인한다.

```bash
bash scripts/docker/dev.sh exec git status --short
bash scripts/docker/dev.sh exec git remote -v
# origin이 tokengeoji/Deep-ANC이고 main이 깨끗한지 확인한 다음:
bash scripts/docker/dev.sh exec git pull --ff-only origin main
bash scripts/docker/dev.sh exec bash tools/prepare_jetson.sh
bash scripts/docker/dev.sh exec .venv/bin/python -m pytest -q
```

로컬 커밋 때문에 fast-forward가 불가능하면 `--ff-only`는 중단한다. reset/force로 덮어쓰지 말고
분기와 사용자 변경을 확인한다. 로컬 폴더 이름은 원격 저장소 이름과 달라도 된다.
기존 원격이 `DeepANC`라면 해당 컨테이너에서 old origin을 별도 이름으로 보존하고
`Deep-ANC`를 새 origin으로 지정한 다음, 상태와 분기를 확인해 갱신한다.

기존 Docker가 없는 장치의 설치는 [Docker 안내](../docker/README.md)를 따른다.
x86 PC에서는 준비 명령에 `--allow-cpu`를 붙인다. 실제 Jetson에서는 이를 빼고
CUDA 연산·역전파까지 통과시켜야 한다. 이후 [OMAP 인수인계](JETSON_HANDOFF.md)를 따라
동기 raw REF/ANC-OFF 녹음과 세션별 train/valid manifest부터 확인한다.

## 검증 범위

이번 통합 검증은 x86 CPU Docker(Python 3.10.21 / PyTorch 2.5.1+cpu)에서 수행했다.
전체 회귀는 **1282 passed, 2 skipped, 5 subtests passed (79.50초)**다. skip은 현장 raw 진단
파일과 실측 `metrics.md` 부재 조건이다. `pip check`·원본 FIR 검증·오프라인 smoke 및
checkpoint 저장도 통과했다. GitHub Actions도 같은 Docker 의존성으로 두 경로를 함께 검사한다.
기존 `HANDOFF.md`의 Jetson 1228개 회귀 기록은 통합 전 환경의 이력이다.
통합된 16 kHz 경로의 실제 Jetson CUDA 학습·실시간 지연·음향 감쇠 검증으로 승격하지 않는다.
