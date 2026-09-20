# 08. 개발 워크플로와 프로젝트 정책

작업을 이어받으면 [HANDOFF](../HANDOFF.md)를 먼저 읽고 [AGENTS](../AGENTS.md)의 규칙을 따른다.
진행 상태는 HANDOFF에 남기며 이 문서에 날짜별 완료표·학습 PID·과거 성능을 중복하지 않는다.
설계의 기준은 ERR 한 점에서 **저역·고역과 음성·음악을 모두 감쇠**하는 목표다.
acoustic-ref가 최우선이고 Jetson의 우선 개선 대역은 1000–1600 Hz다.

## 1. 작업 경계

- 코드 읽기·수정·Git·Python·테스트는 **Docker 안에서만** 수행한다. 호스트는 Docker 환경 관리만 한다.
- `~/anc_project`는 읽기 전용이며 필요한 검증 자산의 복사만 허용한다. 원본 스크립트를 실행하지 않는다.
- Jetson의 RT 커널·30W·핀/I²S·클록·오디오 서비스·시스템 권한·호스트 패키지를 변경하지 않는다.
- 기본 컨테이너에는 오디오 장치를 노출하지 않는다. 스피커 출력은 사용자 입회·볼륨 최소·ANC OFF 시작 조건을 지킨다.
- 기존 변경·측정 파일·녹음·checkpoint를 보존한다. 설계를 바꾸는 미확정 조건은 사용자에게 확인한다.

현재 시스템 설정·장치를 유지한 채 가능한 코드·합성 회귀·오프라인 분석을 진행한다.
실제 ARM64/CUDA/TensorRT·I/O·덕트 측정만 Jetson 현장에서 수행하며, x86 CPU 결과를
그 실측으로 보고하지 않는다. 환경 구성은 [Docker 문서](../docker/README.md), 역할과
현장 선행조건은 [docs/14](14_pc_jetson_workplan.md)를 따른다.

## 2. 평소 작업 순서

호스트에서는 Docker 환경을 확인·접속한다. 없으면 Docker 문서에서 실제 아키텍처에 맞는
이미지를 선택하고, 기존 컨테이너가 중지됐으면 `start`로 재사용한다.

```bash
bash scripts/docker/dev.sh status
bash scripts/docker/dev.sh shell
```

접속한 컨테이너 안에서 변경 상태를 확인하고 관련 문서·설정·테스트를 읽는다.
다음은 읽기 전용 확인과 검증 명령이며 테스트는 변경 영향에 맞춰 수행한다.

```bash
git status --short
git diff --check
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

문서에는 실행값·결과·제한을 구분해 적는다. Docker 설치 성공, 합성 시험, corpus QA,
실제 모델 추론, 물리 감쇠를 각각의 근거로 판정한다. 실패·미평가를 통과로 바꾸지 않는다.
검증 후 diff와 산출물을 점검하고 완료 내용·남은 조건·재현 명령을 HANDOFF에 반영한다.

## 3. 재현성과 불변식

설정 YAML과 CLI override로 해결된 실행 설정을 보존한다. run에는 Git revision·의존성 목록·
seed·입력 데이터 및 측정 경로 식별 정보를 함께 남긴다. checkpoint의 모델·optimizer·scheduler·
step·RNG를 사용하는 완전 재개와 다른 실험의 초기 가중치 사용을 구분한다.

- 극성은 `e=d+S·y`이며 S의 지연·핸드오프·RIR onset을 중복 적용하지 않는다.
- 인과적 모델과 오프라인↔스트리밍 수치 등가성을 유지한다.
- digital lead는 설정·checkpoint·ONNX·런타임이 일치해야 한다. acoustic-ref는 lead=0이다.
  현재 digital 설정의 113과 과거 artifact의 109를 혼동하지 않는다([docs/03](03_data_pipeline.md)).
- 손실은 FP32이며 closed-loop 워밍업은 플랜트 적용 후 절단한다.
- ONNX는 opset 17·정적 shape·명시적 상태 입출력, 세그먼트는 256의 배수다.
- SPSC 링버퍼는 생산자가 write_pos만, 소비자가 read_pos만 소유한다.
- 데이터는 원본·그룹 단위로 분리하고 MIMII train-only 정책을 평가에 유지한다.

라이브러리 import나 테스트 개수만으로 실제 고역 경로·모델 성능을 선언하지 않는다.
목표별 판정 기준은 [docs/07](07_evaluation_protocol.md), 학습 계약은 [docs/05](05_training_elice.md)를 따른다.

## 4. 데이터와 Git 반영

데이터 원본의 최종 보관소는 **Google Drive**다. PC Docker의 임시 다운로드는 공식 checksum·
라이선스 검증과 Drive 업로드 확인을 거친 뒤 해당 PC 원본만 정리한다. Drive 파일 ID·크기·
정리 receipt를 남기며 실패·미확인 파일과 불완전한 기존 백업은 보존한다.
정확한 staging·중복 방지·정리 절차는 [docs/16](16_drive_acoustic_preparation.md)을 따른다.

공개 GitHub에는 코드와 검토한 작은 메타데이터만 반영한다. 원본 데이터·대용량 가중치와
API 키·토큰·환경 파일·개인키를 커밋하지 않는다. `.gitignore`의 `/data/` 같은 앵커 패턴을
비앵커로 바꾸지 않는다. 측정 자산을 갱신할 때도 출처·메타데이터와 변경 근거를 확인한다.

Git 작성자는 승인된 저장소 로컬 설정을 쓰고 전역 설정을 바꾸지 않는다.
커밋 메시지에 AI 표기나 AI Co-Authored-By를 넣지 않는다. 현재 원격은
`tokengeoji/Deep-ANC`이며 이전 `Roka-jsj` 주소는 계정명 변경 전 이력이다.
승인된 push는 origin·현재 브랜치·검증한 diff를 확인하여 수행하고 force push하지 않는다.
