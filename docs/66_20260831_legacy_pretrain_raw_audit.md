# 2026-08-31 legacy pretrain 현장 raw 재감사

> 역할: diagnostic-only. 현행 strict/canonical 성능으로 승격 금지.

## [가설]

기존 pretrained Tiny가 1.6 kHz에서 약 6 dB 감쇠했고, 2--8 kHz 증폭은 주로
추론 latency 때문일 가능성이 있다고 가정한다.

## [근거]

- checkpoint: `runs/pretrain_tiny_corrected/ckpt/best.pt`
  (`bcfc00869d11ff263430677c61555a92a108f0835154ea5e9ff4fc774b53a0bb`)
- raw: `results/session_20260804_0939/*_dl.npz`
- metrics: `results/session_20260804_0939/metrics.csv`
  (`2e693356b6239dfe50d7dc42f83bb2527e0740f6fa34c6f18c9842e546622b42`)
- checkpoint metadata의 trusted band는 150--600 Hz, lead는 109 samples다.
- 현재 strict P/S로 계산한 lead는 115 samples지만, 이 strict capture는 legacy
  session보다 뒤에 측정됐으므로 역사적 plant가 같았다고 단정할 수 없다.

## [확인 방법]

선행 OFF/ON/후행 OFF에서 시작·끝 guard를 제거하고, source가 실제로 존재하는
대역만 평가했다. 감쇠는 `10 log10(P_OFF/P_ON)`이며 선행/후행 OFF 중 나쁜 값을
보수값으로 사용했다. source-energy gate를 통과하지 못한 octave 수치는 정식 ANC
감쇠에서 제외했다.

## [결과]

| 자극 | source 평가 대역 | 보수 감쇠 |
|---|---:|---:|
| 300 Hz tone | 250--350 Hz | +6.48 dB |
| band | 80--1000 Hz | +4.09 dB |
| multitone | 100--800 Hz | +1.43 dB |
| nonlinear | 150--900 Hz | +5.13 dB |
| 1.2 kHz tone | 1150--1250 Hz | +0.35 dB |
| high band | 800--1600 Hz | -0.05 dB |

source-energy gate를 통과한 1.6 kHz 중심 octave 결과는 자극에 따라
`-0.99--+0.36 dB`였다. 정확한 1.6 kHz tone 및 2/4/8 kHz source-valid 실험은 없다.
따라서 `약 6 dB`는 1.6 kHz가 아니라 300 Hz 부근 결과다.

300 Hz source가 거의 없는 4/8 kHz에서 digital control은 각각 약 -44.7/-40.8 dBFS였고,
ERR ON은 선행 OFF보다 각각 18.0/21.6 dB 커졌다. 이것은 source-valid ANC 감쇠 수치는
아니지만 모델/runtime이 유해 고역 에너지를 직접 주입한 진단 증거다.

legacy session에는 deadline miss, xrun, fallback, queue/backlog, inference wall-time
telemetry가 없어 계산 latency가 원인이었는지는 검증할 수 없다. 반면 checkpoint가
`P=S` surrogate, old S coherence median 0.40, 150--600 Hz objective, 고역 DNH 없음에
결속된 것은 artifact에서 확인된다.

## [판정]

- “1.6 kHz에서 약 6 dB”: **Contradicted**
- “250--500 Hz에서 수 dB 감쇠”: **Confirmed** (legacy diagnostic 범위)
- “1--1.6 kHz가 안정적으로 감쇠”: **Contradicted**
- “2/4/8 kHz 성능”: **Inconclusive**
- “ANC ON이 source 없는 고역 에너지를 주입”: **Confirmed**
- “deadline miss가 주원인”: **Inconclusive**
- “잘못된 surrogate plant·고역 손실 부재가 직접 원인”: **Likely**, 고역 control
  주입 자체는 **Confirmed**

## [다음 행동]

legacy checkpoint를 init/resume하지 않는다. 새 2 kHz Stage-2 P/S와 정확한 lead,
고역 DNH, 네 family의 lineage-clean data로 scratch pretrain/fine-tune하고, source-valid
2 kHz octave raw와 runtime miss/xrun telemetry를 함께 저장한다.
