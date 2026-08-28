#!/bin/bash
# legacy 사전학습 진단 실행기 — canonical tiny에는 사용 금지.
# 이 파일은 historical base/legacy P/S 결과를 재현·진단할 때만 보존한다.
# canonical tiny는 docs/05_training_elice.md의 bootstrap/ledger/alpha/CUBLAS 명령만 쓴다.
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV_PYTHON="$PWD/.venv/bin/python"
VENV_TORCHRUN="$PWD/.venv/bin/torchrun"

if [ "${1:-}" != "--legacy-diagnostic" ] || [ "$#" -ne 2 ]; then
  echo "사용법: DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC=1 bash scripts/elice/run_pretrain.sh --legacy-diagnostic configs/train_pretrain.yaml" >&2
  echo "[중단] canonical tiny는 이 legacy 실행기로 시작할 수 없습니다." >&2
  exit 2
fi
if [ "${DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC:-}" != "1" ]; then
  echo "[중단] legacy 진단 실행은 DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC=1을 명시해야 합니다." >&2
  exit 2
fi
CONFIG="$2"
if [ "$CONFIG" != "configs/train_pretrain.yaml" ]; then
  echo "[중단] 이 실행기는 legacy configs/train_pretrain.yaml만 허용합니다: $CONFIG" >&2
  exit 2
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[오류] $VENV_PYTHON 없음 — setup_env.sh 를 먼저 실행하세요." >&2
  exit 1
fi

NGPU=$("$VENV_PYTHON" -c "import torch; print(torch.cuda.device_count())")
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="runs/train_${STAMP}.log"
mkdir -p runs

if [ "$NGPU" -ge 2 ]; then
  if [ ! -x "$VENV_TORCHRUN" ]; then
    echo "[오류] $VENV_TORCHRUN 없음 — setup_env.sh 를 다시 확인하세요." >&2
    exit 1
  fi
  CMD=("$VENV_TORCHRUN" --nproc_per_node="$NGPU" scripts/train/train.py --config "$CONFIG")
else
  CMD=("$VENV_PYTHON" scripts/train/train.py --config "$CONFIG")
fi

echo "실행: ${CMD[*]}  (GPU ${NGPU}장, 로그 $LOG)"
nohup "${CMD[@]}" >>"$LOG" 2>&1 &
PID=$!
echo "$PID" > runs/train_${STAMP}.pid
sleep 5
if ! kill -0 "$PID" 2>/dev/null; then
  echo "[오류] 학습 프로세스(PID $PID)가 시작 직후 종료했습니다." >&2
  tail -n 20 "$LOG" >&2 || true
  exit 1
fi
echo "백그라운드 시작 (PID $PID). 모니터링:"
echo "  tail -f $LOG"
echo "  .venv/bin/tensorboard --logdir runs --port 6006   # VS Code 포트포워딩으로 접속"
