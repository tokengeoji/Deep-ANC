# OMAP-L138 CCS 프로젝트

성공한 DSP 소스와 CCS 프로젝트 설정을 프로젝트 단위로 보존한 위치다. 디렉터리 정리 과정에서 알고리즘 소스와 실측 계수는 변경하지 않았다.

| 프로젝트 | 용도 |
| --- | --- |
| [`FxNLMS0`](FxNLMS0) | 실시간 FxNLMS ANC, 16 kHz, 제어 필터 256 taps |
| [`NLMS0`](NLMS0) | 2차경로 `S_hat` 측정, 16 kHz, 500 taps |
| [`NLMS1`](NLMS1) | 1차경로 `P_hat` 측정, 16 kHz, 500 taps |

기존의 `FxNLMS0/FxNLMS0`, `NLMS0/NLMS0`, `NLMS1/NLMS1` 프로젝트 루트는 각각 이 폴더의 바로 아래로 이동했다. 각 폴더의 `.project`, `.cproject`, `.ccsproject`, `app.cfg`, `lib/`를 함께 사용한다. 개별 C 파일만 새 프로젝트에 복사하면 기존 SYS/BIOS·interrupt·codec 설정을 빠뜨릴 수 있다.

## CCS에서 다시 열기

1. CCS의 기존 프로젝트 가져오기에서 이 폴더 아래 원하는 프로젝트 루트를 선택한다. 동일 이름의 예전 경로 프로젝트가 열려 있다면 workspace 등록을 정리하되 기존 파일은 보존한다.
2. 프로젝트의 compiler, XDCtools, SYS/BIOS, PDK 경로를 사용 중인 TI 설치 경로와 맞춘다. 보존된 설정에는 `C:\ti\pdk_omapl138_1_0_11\packages` 같은 Windows 절대 경로와 기존 프로젝트 origin이 남아 있다.
3. CCS에서 Clean 후 전체 rebuild하여 새 위치의 build 파일을 생성한다. 기존 `Debug/`, `Release/`, SYS/BIOS 생성물은 이전 경로를 포함할 수 있다.
4. 실제 보드에서 사용할 설정을 CCS Expressions의 실행 변수와 비교한다. 출력 채널·극성·gain 및 codec 설정은 [하드웨어 기준 문서](../../../docs/HARDWARE_BASELINE.md)에 정리되어 있다.

생성된 build cache와 binary는 로컬에 남을 수 있으나 Git 배포 대상에서 제외한다. Jetson에서 저장소를 받으면 TI 환경에서 다시 빌드할 소스와 프로젝트 설정이 제공된다. 이 디렉터리 정리만으로 CCS rebuild나 실제 보드 동작을 새로 검증한 것은 아니다.

## 계수와 설정 기준

저장소 루트의 [`rir.txt`](../../../rir.txt)가 2차경로 정본이다. `FxNLMS0/ISR.c`의 `S_hat`는 이 파일과 동일한 500개 계수를 포함한다. 학습 파생 파일을 만들 때 정본의 gain, 부호, 선행 지연, tap 수를 보존한다.

`FxNLMS0`는 `ISR.c`에서 실행 변수를 직접 초기화하므로 `define.h` 주석만 보고 설정을 판단하지 않는다. 소스 기본값은 Left 기준 마이크, Right 오류 마이크, 양쪽 동일 스피커 출력, `u=-y`, feedback 제거 활성화다. 실제 성공 실험에서 CCS로 바꾼 값과 외부 amplifier 설정은 별도 실험 기록이 필요하다.

`NLMS0`는 출력 command에서 오류 마이크까지의 경로를 측정한다. `NLMS1`는 두 마이크 사이의 1차경로 측정 코드이며, 계수 dump가 별도로 없으면 측정된 1차경로가 준비된 상태로 간주하지 않는다. `FxNLMS0`의 `F_hat`도 2차경로와 구별되는 feedback 경로다.
