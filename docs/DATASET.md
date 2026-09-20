# OMAP 16 kHz ANC 학습 데이터 준비

이 문서는 `deepanc/`와 `configs/anc_train.json`용 데이터 계약이다. 기존 `src/deep_anc/`의 48 kHz USB 녹음·manifest 계약은 [기존 데이터 안내](03_data_pipeline.md)를 따른다. 서로 다른 장치·샘플레이트의 세션을 변환 없이 교환하지 않는다. Python과 파일 처리는 현재 Deep-ANC의 Docker 내부에서 실행한다.

학습에는 동시에 기록한 기준 마이크 `x[n]`와 ANC OFF 상태의 오류 마이크 disturbance `d[n]`가 필요하다. 취소 스피커를 끈 채 외부 소음을 재생하고 두 ADC 채널을 같은 clock과 같은 시작 시각으로 기록한다. ANC ON 상태의 오류 신호 `e[n]`는 이미 제어 출력의 영향을 받으므로 이 데이터의 `disturbance`로 사용하지 않는다.

이 문서는 이미 확보한 WAV를 준비하는 절차다. 제공 도구는 보드에서 녹음하거나 오디오 장치로 소리를 출력하지 않는다. 기존 OMAP-L138에서 동기화된 raw ADC를 보존하는 녹음 경로를 먼저 확보해야 한다. 별도의 USB 오디오 장치로 녹음하면 codec·gain·지연 조건이 달라질 수 있다.

## 실측 전 수집 경로 확인

현재 `FxNLMS0`의 `ref_mic_buf`·`err_mic_buf`는 256샘플 순환 모니터 버퍼다.
16 kHz에서 16 ms마다 덮어쓰므로 CCS Graph 표시나 주기적인 메모리 덤프만으로
손실 없는 연속 녹음을 확보했다고 볼 수 없다. `prepare_recordings.py`는 보드 수집기가 아니다.

수집 경로를 정하기 전에 다음을 확인한다.

1. 실물 보드 모델·revision, JTAG 디버거 모델, 현재 동작하는 CCS/컴파일러 버전.
   저장된 `.ccsproject`는 LCDKOMAPL138 / CCS 9.3.0 / C6000 compiler 8.3.5 설정이지만
   실제 보드나 현재 설치 버전을 증명하지 않는다. CCS의 기존 `.ccxml`을 열어
   `Connection`과 `Board or Device`를 확인한다. 새 설정·연결 테스트·프로그램 로드는 하지 않는다.
2. 같은 ADC clock의 raw REF/ERR 쌍과 샘플 순서, 누락·중복 검출 방법, 저장할 버퍼 범위.
   두 PCM16 채널은 초당 64,000 byte(헤더 제외), 1분이면 3.84 MB다.
   사용 가능한 RAM·링커 배치·전송 지속 속도는 실물 환경에서 확인하며 추정하지 않는다.
3. 실제 채널·gain·배치와 ANC OFF/테스트 출력 OFF를 확인한다. 원본 DSP의 `anc_enable` 기본값은 1이며,
   `test_tone_mode`는 ANC OFF여도 출력을 낼 수 있다. 사용자 입회·물리 볼륨 최소 조건을 지킨다.

### JTAG 연결과 녹음을 구분

CCS의 기본 연결은 **호스트 PC → 디버그 프로브 → 타깃 보드**다.
Jetson과 OMAP의 JTAG 디버그 단자를 서로 직결하는 보드 간 통신 구조가 아니다.
Jetson GPIO를 OMAP JTAG에 임의 배선하지 않는다.
Jetson AGX Orin 개발 키트의 J502도 JTAG 디버그 커넥터로 문서화되어 있다.
[TI CCS 연결 구조](https://software-dl.ti.com/ccs/esd/documents/users_guide_10.1.0/ccs_debug-main.html#the-basic-elements),
[NVIDIA 커넥터 안내](https://docs.nvidia.com/jetson/agx-orin-devkit/user-guide/hardware_layout.html).

CCS 21.0.1의 공식 호스트 CPU 조건은 x86_64다. Ubuntu 지원이라는 이유로
ARM64 Jetson에서 동일한 CCS/프로브 드라이버를 그대로 쓸 수 있다고 가정하지 않는다.
기존 Windows CCS 환경을 유지하고, 확보·검증한 녹음을 Jetson Docker로 가져와 처리할 수 있다.
[TI 호스트 요구사항](https://software-dl.ti.com/ccs/esd/documents/users_guide_ccs_21.0.1/ccs_overview.html#hardware).

사용 중인 XDS200(TMDSEMU200-U)은 호스트 쪽 USB와 타깃 쪽 디버그 커넥터를 구분한다.
기존 Windows CCS → USB → XDS200 → OMAP JTAG 연결을 유지하며,
이 연결을 Jetson과의 실시간 PCM 전송으로 해석하지 않는다.
[TI XDS200 연결 설명](https://www.ti.com/tool/TMDSEMU200-U).

CCS에는 메모리 저장 기능이 있으나, **먼저 연속 구간을 손실 없이 보관하는 캡처 기능**이 필요하다.
녹음 완료 후 고정된 버퍼를 내보내는 방식과 실행 중 실시간 스트리밍은 다른 설계다.
정지·재개를 반복해 얻은 조각을 연속 녹음처럼 이어 붙이지 않는다.
[TI 메모리 저장 및 디버거 동작](https://software-dl.ti.com/ccs/esd/documents/users_guide_10.1.0/ccs_debug-main.html#load-and-save-memory).
현재 어느 수집 방식도 구현·검증 완료로 선언하지 않는다. 방식 확정과 별도 승인 전에는
원본 펌웨어·GEL·링커·Jetson 시스템을 변경하지 않는다.

## 녹음 조건과 폴더

입력은 16,000 Hz, stereo, PCM16 WAV이며 Left는 기준 마이크, Right는 오류 마이크다. 마이크·스피커 배치와 codec 및 amplifier gain은 [2차경로 실측 기준](HARDWARE_BASELINE.md)과 맞춘다. 두 채널을 개별 정규화하거나, peak를 맞춰 이동하거나, resampling하지 않는다. clipping이 발생한 녹음은 gain 조건을 확인하고 다시 수집한다.

```text
datasets/anc/raw/
├── train/
│   ├── session_001/
│   │   ├── noise_001.wav
│   │   └── capture.json          # 선택: 실제 녹음 조건
│   └── session_002/
│       └── noise_001.wav
└── valid/
    └── session_101/
        └── noise_001.wav
```

같은 연속 녹음을 잘라 train과 valid 양쪽에 넣지 않는다. 녹음 세션 전체를 한 split에 배정하고, 검증에는 별도로 수집한 세션을 사용한다. 도구는 두 split의 세션 ID 중복과 동일 PCM 채널의 복사를 검출하지만, 부분 구간 복사·시간 이동·재인코딩에 의한 모든 누수를 검출하지는 못한다.

세션의 선택적 `capture.json`은 JSON 객체로 작성하며 준비 결과의 `preparation.json`에 복사된다. 녹음 날짜, 보드와 펌웨어 revision, ADC/DAC와 amplifier gain, 채널 연결, 마이크 위치, 소음 종류와 원본 녹음 ID를 기록하면 이후 실험 비교가 쉬워진다. 모르는 실제 값은 `"unknown"`으로 남기고 소스 기본값으로 채우지 않는다. 이 sidecar는 관측 기록이며, 도구가 실제 배선을 검증한 증거가 아니다.

`raw_pcm`은 ADC 값을 그대로 저장했다는 뜻이다. FxNLMS의 모니터 변수 중 `ref_x_buf`와 `err_e_buf`에는 DC 제거 등의 처리가 들어가므로 raw ADC 대신 그대로 사용하지 않는다. `ref_mic_buf`와 `err_mic_buf`도 순환 buffer이므로 안정적인 연속 녹음에는 올바른 순서 복원과 손실 없는 수집이 필요하다. 이미 전처리한 오디오를 `raw_pcm`이라고 표시하지 않는다.

## WAV 분리와 manifest 생성

저장소 루트에서 Docker 실행기를 호출한다. 두 flag는 실제 수집 조건을 확인했다는 명시적 선언이며, 도구가 녹음 상태를 자동 판정한다는 뜻이 아니다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python tools/prepare_recordings.py \
  --input datasets/anc/raw \
  --output datasets/anc/prepared \
  --anc-off --raw-pcm
```

도구는 전체 입력의 형식·길이·split 누수·이름 충돌을 먼저 검사한다. 이후 Left/Right의 PCM16 바이트를 각각 mono WAV로 분리하며 스케일과 시간 정렬을 유지한다. Python 표준 라이브러리만 사용한다. 입력과 출력은 서로 포함되지 않는 디렉터리여야 하며, 기존 출력 폴더는 덮어쓰지 않는다. 다시 준비하려면 새 출력 경로를 사용한다.

```text
datasets/anc/prepared/
├── train/session_001/noise_001_reference.wav
├── train/session_001/noise_001_disturbance.wav
├── ...
├── valid/session_101/noise_001_reference.wav
├── valid/session_101/noise_001_disturbance.wav
├── preparation.json
└── manifest.jsonl
```

`preparation.json`에는 입력 파일별 frame 수, interleaved PCM과 각 채널 PCM의 SHA-256, 수집 조건 선언을 남긴다. `manifest.jsonl`의 경로는 manifest 위치를 기준으로 한다.

```json
{"reference":"train/session_001/noise_001_reference.wav","disturbance":"train/session_001/noise_001_disturbance.wav","split":"train","session":"session_001","anc_enabled":false,"signal_domain":"raw_pcm"}
```

변환 도중 디스크 오류나 원본 변경이 발생하면 일부 출력이 남을 수 있다. manifest는 모든 변환이 끝난 뒤 작성하므로 준비가 완료되지 않은 폴더는 학습에 사용하지 않는다. 원본은 수정하지 않는다.

## 실제 데이터로 학습

Jetson Docker 환경과 CUDA 확인 후 저장소 루트에서 실행한다.

```bash
bash scripts/docker/dev.sh exec .venv/bin/python -m deepanc.train \
  --config configs/anc_train.json \
  --manifest datasets/anc/prepared/manifest.jsonl \
  --device cuda \
  --output runs/omap_anc
```

학습기는 PCM full-scale 변환을 적용해 ADC 디지털 단위를 유지한다. 모델이 직접 DAC command `u`를 출력하므로 잔차 정의는 `e=d+S*u`다. RIR의 gain·극성·선행 지연을 보존해야 데이터와 경로가 같은 단위를 갖는다.

합성 데이터의 smoke test 성공은 학습·미분·저장 절차가 실행된다는 의미다. 실제 녹음의 검증 지표도 오프라인 결과이며, Jetson에서의 실시간 ANC 성능이나 OMAP 성공 실험과 같은 감쇠 성능을 입증하지 않는다.
