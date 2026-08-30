# 부록. 기존 anc_project (FxLMS) 분석 — 2026-08-02 전수 조사

`~/anc_project` 전체(코드 3본 + 로그 8 + 모델 npz 10 + 진단 6 + WAV 5)를 읽고 정리한 기록.
이 분석은 당시 S(z) 선택과 지연 규약·안전장치 계승의 역사적 근거다. 현재 official
interleaved schema의 근거가 아니며 어떤 legacy NPZ도 training-ready로 승격하지 않는다.
**원본은 읽기전용.**

## 1. 소스 코드

| 파일 | 역할 |
|---|---|
| `fxlms_core.py` (22KB) | 블록 FxNLMS 컨트롤러(256탭, μ=0.05, leakage 1e-6, ‖w‖≤20), `SecondaryPathModel` NPZ 로더(compact FIR+순수지연 분리 — CPU 절약 설계), ALSA→PortAudio 매핑, PCM 변환, DCBlocker, 오프라인 self-test(합성 경로 감쇠 ≥6dB 검증) |
| `main_realtime_anc.py` (29KB) | sounddevice 콜백 실시간 ANC. block 256/low, 노이즈 생성기(tone 300Hz/white/band), FadeGate, 안전장치(control limit 0.2, 클립 스트릭 mute, 적응 hold), 키보드 UI, 1초 통계(reduction dB 등). **reference 기본값 digital** |
| `calibrate_s_path.py` (27KB) | S(z) 측정: 대역제한(150–600Hz) 잡음 여기 → FFT 상호상관 지연 → Welch/CSD FIR 추정 → 품질 게이트(fit/coherence/클리핑) → npz 저장 |

Deep_ANC 로의 계승: 장치 해석·PCM 변환·DCBlocker·FadeGate·안전 패턴을 복사 이식
(`src/deep_anc/audio_io.py`, `dsp/filters.py`, `realtime/safety.py`),
FxLMSController 는 `baselines/` 사본으로 베이스라인/폴백에 사용.

## 2. 캘리브레이션 산출물 품질

| npz | 측정 조건 | delay | fit | coherence | 판단 |
|---|---|---|---|---|---|
| `secondary_path_4s.npz` | block 256 / low | 1342 (27.96ms) | **2.14dB** | **0.40** | 당시 후보 중 최상; 현재 legacy diagnostic-only |
| `secondary_path.npz` (=4s_512) | block 512 / high | 2613 (54.4ms) | 1.09dB | 0.27 | 기존 시스템이 사용하던 것 |
| `secondary_path_before_4s.npz` | 256 / low | 1428 | 0.55dB | 0.10 | 실패급 |

측정 왕복지연: 30.6ms(256/low) ↔ 57.1ms(512/high) — 차이 26.5ms 는 순수 버퍼 증가분.

## 3. 발견된 문제점 (기존 시스템)

1. **채택 모델이 최선이 아니었음** — 사용 중인 npz(fit 1.09dB)보다 좋은 4s본(2.14dB)이 방치됨
2. **모델↔런타임 설정 불일치** — 모델은 512/high 측정, 런타임 기본은 256/low → 지연 26ms 어긋남 (실행 시 경고 2개 발생 상태)
3. **품질 게이트 무력화** — `--min-fit-db 0` 으로 실행되어 저품질 모델이 통과
4. **레퍼런스 마이크 ch1 무신호 이력** — mic_stats.txt 기준 RMS −189dBFS (이후 수리 여부 미확인). runtime 기본 reference=digital 인 이유
5. **실시간에 불리한 시스템 상태** — pulseaudio/pipewire 활성, rtprio limit 0, 30W 전원모드, USB full-speed(12Mbps) 오디오 (단, 시스템은 의도된 구성이므로 변경하지 않음 — 정책)

Deep_ANC 반영: ①→4s본 채택, ②→측정/런타임 조건 일치 강제 + `--calibrate` 실효지연 검증,
③→광대역 재보정 스크립트에 일관성 경고, ④→ref 자가진단 + digital-ref 기본, ⑤→정책 준수 하 실측치 기준 설계.

## 4. 기타 인벤토리

- 로그 8종: 성공 3(위 표), 인자 검증 실패 3, 스크립트 구버전 오류 1, 게이트 미달 2
- 진단 캡처 npz 4종(원파형 포함, 각 2.8~4.7MB), WAV 5종(마이크 진단·300Hz 상쇄 테스트)
- `diagnostics/audio_report.txt` — 사운드카드 3장(HDA/APE/AB13X USB) 전체 하드웨어 리포트
- `anc_project_gemini/` — 초기 프로토타입 (샘플 루프 LMS, `np.convolve mode='same'` 비인과 버그,
  검증 없음). 현행 anc_project 의 조상. 참고 가치 낮음
- `control_filter_last.npy` — 마지막 실시간 실행의 제어필터(256탭) 스냅샷
