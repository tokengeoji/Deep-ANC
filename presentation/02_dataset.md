# 2. 학습 데이터셋

설정 파일: [configs/data_sim_acoustic_pilot.yaml](../configs/data_sim_acoustic_pilot.yaml)
(`configs/data_sim.yaml`과 동일하고 `reference_mode: acoustic` 한 줄만 다르다).

## 2.1 데이터 생성 방식 — 온더플라이 시뮬레이션 (실측 녹음 아님)

학습은 미리 녹음된 오디오 세션을 재생하는 것이 아니라, 매 배치마다 다음을
**실시간으로 합성**한다:

```
noise(t)  --[ RIR: p_ref ]-->  x_ref(t)   (레퍼런스 마이크가 듣는 신호)
noise(t)  --[ RIR: p_err ]-->  d(t)       (상쇄 없을 때 에러 마이크가 듣는 신호)
y(t) [모델 출력] --[ RIR: f_fb ]--> s_y(t) (상쇄 스피커→에러 마이크 경로)
e(t) = d(t) + s_y(t)   ← 학습 손실이 최소화하려는 값
```

- `p_ref`, `p_err`, `f_fb`: `data/rir_bank/duct_rirs_v1.npz` (300개 세트, 각 8192
  샘플). **이것은 실측 녹음이 아니라, [configs/duct.yaml](../configs/duct.yaml)의
  실측 기하(치수·반사계수·음속)를 파라미터로 삼아 1D 영상법(image method)으로
  도메인 랜덤화 생성한 임펄스 응답 뱅크**다(`scripts/data/build_rir_bank.py`).
  치수/반사계수 자체는 실측 사양이지만, 각 학습 아이템의 정확한 임펄스응답은
  그 사양 주변을 무작위로 흔든 시뮬레이션값이다.
- 반대로 [configs/duct.yaml](../configs/duct.yaml)의 `secondary_path`/
  `digital_reference` (S(z)/P(z))는 **진짜 실측값**이다 — 실시간 제어(FxLMS,
  digital-reference 모델)와 플랜트 물리 검증에 쓰이며, 학습용 RIR 뱅크와는
  다른 용도다.
- `noise(t)` 자체(합성 사인/멀티톤/논리니어 또는 실제 녹음된 코퍼스 오디오)는
  아래 혼합 비율로 매 아이템마다 하나를 뽑는다.

## 2.2 소스 혼합 비율 (`source_mix_ratio_acoustic`)

2026-09-04 재조정 후 값 — 근거는 [04_training_and_results.md](04_training_and_results.md)의
"학습 정체(flat NMSE) 진단"절 참고:

| 소스 | 비중 | 성격 |
|---|---:|---|
| synthetic | 0.60 | 합성 톤/멀티톤/논리니어 — 주기적, 위 인과 제약 안에서 예측 가능 |
| machine | 0.20 | 회전기계류 실제 녹음 — 준주기적 |
| dns_fullband | 0.10 | DNS 챌린지 광대역 노이즈 코퍼스 — 예측 불가능(비주기) |
| demand | 0.05 | DEMAND 환경음 코퍼스 — 예측 불가능 |
| speech | 0.03 | 음성 — 예측 불가능 |
| esc50 | 0.02 | 환경음 이벤트 코퍼스 — 예측 불가능 |

애초 비율(광대역 40%)은 1.3절의 인과 제약과 부딪혀 학습 objective가 정체되는
원인 중 하나로 의심됐다(최종적으로는 별도의 학습률 버그가 진짜 원인으로
밝혀졌다 — [04](04_training_and_results.md) 참고). 그래도 실제로 상쇄 불가능한
소스 비중을 줄이고 주기성 위주로 재조정한 것 자체는 "달성 가능한 목표에 맞춘
데이터 분포"로서 유효한 방향이라 그대로 유지했다.

## 2.3 증강 (`recorded_augment`)

레벨(-12~+6dB), 극성 반전, 마이크 자체잡음 SNR(12~40dB), EQ tilt/band 왜곡을
아이템마다 무작위로 적용해 모델이 특정 레벨/스펙트럼 형태에 과적합하지 않게 한다.

## 2.4 검증(trusted) 대역

`[150, 1600] Hz`, 4개 서브밴드로 균등 평가:
`[150,300]`, `[300,600]`, `[600,1000]`, `[1000,1600]`. 1.6kHz 초과는 "do-no-harm"
(악화 금지)만 강제하고 개선 목표에서는 제외한다 — 근거는
[configs/duct.yaml](../configs/duct.yaml)의 `acoustics` 절(평면파 컷오프 1633Hz 이후
1D 근사가 깨지고, 넓은 대역 목표가 오히려 고역 증폭을 유발한 실측 근거 포함).
