#!/bin/bash
# historical 2×GPU legacy 모델 비교 실행기. canonical tiny에는 사용 금지.
set -euo pipefail
cd "$(dirname "$0")/../.."
VENV_PYTHON="$PWD/.venv/bin/python"

if [ "${DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC:-}" != "1" ]; then
  echo "[중단] run_parallel_models.sh는 legacy base/tiny diagnostic 전용입니다." >&2
  echo "       canonical tiny는 docs/05_training_elice.md의 ledger-bound 명령을 사용하세요." >&2
  echo "       과거 결과 재현만 필요하면 DEEP_ANC_ALLOW_LEGACY_DIAGNOSTIC=1을 명시하세요." >&2
  exit 2
fi

if [ ! -x "$VENV_PYTHON" ]; then
  echo "[오류] $VENV_PYTHON 없음 — setup_env.sh 를 먼저 실행하세요." >&2
  exit 1
fi

mkdir -p runs
# 두 셸이 프로세스/산출물 사전 점검을 동시에 통과하는 TOCTOU를 막는다.
# lock 파일은 지우지 않으며, 셸 종료 시 커널이 잠금만 자동 해제한다.
exec 9>runs/.run_parallel.lock
if ! flock -n 9; then
  echo "[오류] 다른 run_parallel_models.sh 실행이 시작 절차를 진행 중입니다." >&2
  exit 1
fi

existing_train=$(pgrep -af '[t]rain\.py' || true)
if [ -n "$existing_train" ]; then
  echo "[오류] 기존 train.py 프로세스가 있어 중복 학습을 시작하지 않습니다:" >&2
  echo "$existing_train" >&2
  exit 1
fi

NGPU=$("$VENV_PYTHON" -c 'import torch; print(torch.cuda.device_count())')
if [ "$NGPU" -lt 1 ]; then
  echo "[오류] 사용 가능한 CUDA GPU가 없습니다." >&2
  exit 1
fi

assert_new_run_path() {
  local path
  for path in "$@"; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      echo "[오류] 기존 학습 산출물 $path 을(를) 자동 덮어쓰지 않습니다." >&2
      echo "       보존·이름 변경·명시적 삭제 중 하나를 선택한 뒤 다시 실행하세요." >&2
      return 1
    fi
  done
}

# 어느 한 모델의 사전 점검이 실패해도 다른 모델부터 시작되는 일이 없도록
# 실제 프로세스를 띄우기 전에 이번 실행의 모든 경로를 검사한다.
assert_new_run_path \
  runs/train_base_corrected.log runs/train_base_corrected.pid \
  runs/pretrain_base_corrected
if [ "$NGPU" -ge 2 ]; then
  assert_new_run_path \
    runs/train_tiny_corrected.log runs/train_tiny_corrected.pid \
    runs/pretrain_tiny_corrected
fi

# 사전 점검과 실제 open 사이에 다른 파일이 생겨도 기존 파일을 덮어쓰지 않는다.
set -o noclobber

started_pids=()
startup_committed=0

rollback_startup() {
  local pid path
  local rollback_dir="runs/failed_start_$(date +%Y%m%d_%H%M%S)_$$"

  echo "[복구] 이번 호출이 시작한 학습을 모두 중지하고 산출물을 보존 위치로 옮깁니다." >&2
  # 각 프로세스는 setsid로 시작하므로 leader PID가 곧 PGID다. leader가 TERM으로
  # 먼저 끝나도 DataLoader 자식이 같은 그룹에 남을 수 있어 그룹 자체에 신호를 보낸다.
  for pid in "${started_pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in "${started_pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done

  mkdir -p "$rollback_dir"
  for path in \
    runs/train_base_corrected.log runs/train_base_corrected.pid runs/pretrain_base_corrected \
    runs/train_tiny_corrected.log runs/train_tiny_corrected.pid runs/pretrain_tiny_corrected; do
    if [ -e "$path" ] || [ -L "$path" ]; then
      mv "$path" "$rollback_dir/"
    fi
  done
  echo "[복구] 시작 실패 산출물 보존: $rollback_dir" >&2
}

on_exit() {
  local status=$?
  trap - EXIT
  if [ "$startup_committed" -eq 0 ] && [ "${#started_pids[@]}" -gt 0 ]; then
    rollback_startup
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 기존 base 24×400k, tiny 32×400k의 총 샘플 예산은 유지하면서 A100 처리량을
# 높인다. 32 vCPU 인스턴스에서 모델별 worker 14개(총 28개)를 사용한다.
CUDA_VISIBLE_DEVICES=0 setsid nohup "$VENV_PYTHON" scripts/train/train.py \
  --config configs/train_pretrain.yaml \
  --set ckpt_dir=runs/pretrain_base_corrected \
  --set batch_size=96 --set num_workers=14 --set prefetch_factor=4 \
  --set schedule.total_steps=100000 --set schedule.warmup_steps=1250 \
  --set eval_every=500 --set early_stop_patience=0 \
  9>&- > runs/train_base_corrected.log 2>&1 < /dev/null &
base_pid=$!
started_pids+=("$base_pid")
printf '%s\n' "$base_pid" > runs/train_base_corrected.pid
echo "base  → GPU0 (PID $base_pid, runs/train_base_corrected.log, runs/train_base_corrected.pid)"

if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=1 setsid nohup "$VENV_PYTHON" scripts/train/train.py \
    --config configs/train_pretrain.yaml \
    --set model_config=configs/model_tiny.yaml --set ckpt_dir=runs/pretrain_tiny_corrected \
    --set batch_size=128 --set num_workers=14 --set prefetch_factor=4 \
    --set schedule.total_steps=100000 --set schedule.warmup_steps=1250 \
    --set eval_every=500 --set early_stop_patience=0 \
    9>&- > runs/train_tiny_corrected.log 2>&1 < /dev/null &
  tiny_pid=$!
  started_pids+=("$tiny_pid")
  printf '%s\n' "$tiny_pid" > runs/train_tiny_corrected.pid
  echo "tiny  → GPU1 (PID $tiny_pid, runs/train_tiny_corrected.log, runs/train_tiny_corrected.pid)"
else
  echo "GPU 1장 — tiny 는 base 완료 후 실행하세요"
fi
sleep 5

startup_failed=0
check_started_model() {
  local name=$1
  local pid=$2
  local log=$3
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[오류] $name 학습 프로세스(PID $pid)가 시작 직후 종료했습니다." >&2
    echo "----- $log 마지막 20줄 -----" >&2
    tail -n 20 "$log" >&2 || true
    startup_failed=1
  fi
}

check_started_model base "$base_pid" runs/train_base_corrected.log
if [ "$NGPU" -ge 2 ]; then
  check_started_model tiny "$tiny_pid" runs/train_tiny_corrected.log
fi
if [ "$startup_failed" -ne 0 ]; then
  echo "[오류] 하나 이상의 학습 프로세스가 시작되지 않아 시작 작업을 되돌립니다." >&2
  exit 1
fi

startup_committed=1
pgrep -af '[t]rain\.py' | head -4 || true
