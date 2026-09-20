# 부록. 기존 anc_project의 초기 분석 기록

2026-08-02에 기존 FxLMS 환경을 조사한 역사 기록이다.
현재 운영 지침이나 Deep_ANC의 최신 측정 상태로 사용하지 않는다.
`~/anc_project`는 읽기 전용이며 이 작업에서도 수정·정리 대상으로 삼지 않는다.

**현재 채택 S는 `secondary_path_il.npz`, 순수지연 1465샘플, 검증 대역 150–600Hz다.**
P/S와 지연 규약은 [docs/01](01_physics_limits.md),
배선·현재 세션의 입력 점검은 [docs/02](02_hardware_setup.md)가 기준이다.

## 1. 재사용한 코드와 설계

| 당시 파일 | 역할 | Deep_ANC에서의 대응 |
|---|---|---|
| `fxlms_core.py` | block FxNLMS, compact FIR+순수지연 로더, 장치 해석·PCM 변환 | `baselines/fxlms_core.py`의 출처를 명시한 사본, `audio_io.py` |
| `main_realtime_anc.py` | callback 제어·소음 생성·페이드·안전 감시 | `realtime/`의 독립 런타임과 안전장치 |
| `calibrate_s_path.py` | 대역 제한 자극·지연·FIR 추정·품질 검사 | 이 저장소의 별도 경로 측정 도구 |

현재 기준선은 기존 core를 복사한 코드에서 발전시킨다.
원본을 import·실행해 `__pycache__` 같은 파일을 만드는 것도 원본 읽기 전용 규칙에 어긋난다.
기존 저장소의 digital 기본값은 현재 acoustic-ref 우선순위를 결정하지 않는다.

## 2. 당시 캘리브레이션 비교

아래는 **2026-08-02 조사 당시** 기록이며 최신 채택본의 품질 표가 아니다.

| 당시 NPZ | 측정 조건 | delay | fit | coherence |
|---|---|---:|---:|---:|
| `secondary_path_4s.npz` | block256 / low | 1342샘플 | 2.14dB | 0.40 |
| `secondary_path.npz`(4s_512) | block512 / high | 2613샘플 | 1.09dB | 0.27 |
| `secondary_path_before_4s.npz` | block256 / low | 1428샘플 | 0.55dB | 0.10 |

4s본은 당시 비교군 중 더 나았지만 coherence 0.40인 과거 측정이다.
Deep_ANC 초기에는 이를 사용했으나 이후 interleaved S/P로 교체했다.
따라서 1342샘플을 현재 S나 acoustic 제어 지연으로 다시 채택하지 않는다.

당시 I/O 왕복 측정은 block256/low 약 30.6ms, block512/high 약 57.1ms였다.
이 차이를 새 환경의 고정 버퍼 비용이나 현재 I/O 측정값으로 사용하지 않는다.
compact FIR의 순수지연과 I/O 진단 도구의 왕복 측정도 시간 기준을 대조해야 한다.

## 3. 유지할 교훈

- **측정 조건과 런타임 조건을 맞춘다.** 당시에는 512/high 모델과 256/low 런타임이 섞여 있었다.
  Deep_ANC acoustic 경로는 S의 샘플레이트·블록·latency 메타를 대조한다.
- **낮은 품질 기준으로 통과시키지 않는다.** 파일 생성 성공과 경로 식별 성공을 구분하고
  반복 일관성·위상·지연 안정성·대역을 확인한다.
- **입력 상태는 매 세션 재확인한다.** 과거 REF 무신호와 이후 복구·재실패 이력이 모두 있다.
  현재는 ERR/REF 모두의 입력 probe를 기준으로 acoustic 측정을 진행한다.
- **시스템 상태를 임의로 고치지 않는다.** RT 커널·전원모드·오디오 구성은 유지한다.
  Docker 사용자 공간에서 검증하고 현재 측정 증거로 판단한다.
- **합성 self-test나 과거 digital 로그를 실기 성능으로 확대하지 않는다.**
  같은 조건의 독립 세션·소스별·대역별 평가는 [docs/07](07_evaluation_protocol.md)를 따른다.

원본의 파일 수·프로세스·마지막 제어필터 스냅샷은 현행 운영에 필요하지 않아 이 부록에서
반복 관리하지 않는다. 자세한 초기 조사 이력은 Git 기록에 남아 있다.
