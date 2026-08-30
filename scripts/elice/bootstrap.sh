#!/usr/bin/env bash
#
# 엘리스 인스턴스 부트스트랩 — 코퍼스 확보 → manifest → 검증 → 사전학습 기동.
#
# 왜 스크립트인가
# --------------
# 2026-08-04 의 엘리스 배포는 scp 로 워킹트리 스냅샷을 밀어 넣은 dirty 상태였고,
# 원격 `git status --porcelain` 이 37개였다. 그 상태에서는 "지금 무엇이 돌고 있는가" 를
# 아무도 재현할 수 없다. 이번에는 **git 으로 받고, 검증을 통과해야 학습이 시작된다.**
#
# 사용법:
#   bash scripts/elice/bootstrap.sh            # 검증까지만 (학습 시작 안 함)
#   bash scripts/elice/bootstrap.sh --train    # 검증 통과 시 사전학습까지 기동
#
set -euo pipefail

cd "$(dirname "$0")/../.."
REPO="$(pwd)"
PY="${PY:-python3}"
START_TRAINING=0
[ "${1:-}" = "--train" ] && START_TRAINING=1

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
fail() { printf '\n\033[31m[중단] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 환경
say "1/6  환경"
"$PY" -c "import torch; print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}')" \
  || fail "torch 가 없습니다"
df -h . | tail -1
echo "브랜치: $(git rev-parse --abbrev-ref HEAD)  커밋: $(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠ 워킹트리가 dirty 합니다 — 무엇이 돌고 있는지 재현할 수 없게 됩니다:"
    git status --porcelain | head -20
    fail "커밋하거나 되돌린 뒤 다시 실행하세요"
fi

# ---------------------------------------------------------------- 2. 코퍼스
say "2/6  코퍼스 manifest"
# prepare_noise_pool 은 선언 태그의 원본이 하나라도 없으면 EXIT=1 로 fail-closed 다.
# 그 실패는 정상이며, 무엇을 받아야 하는지 목록으로 알려준다.
if ! "$PY" scripts/data/prepare_noise_pool.py; then
    cat <<'MSG'

위 목록의 원본을 data/raw/<태그이름>/ 아래에 받은 뒤 다시 실행하세요.
  dns_fullband : DNS-Challenge noise_fullband (48kHz)
  demand       : DEMAND 48k
  machine      : MIMII DG fan (16kHz)

⚠ source_mix_ratio 에서 태그를 지우는 것은 **절대목표 2(모든 소리를 제거, 최악값 기준)의
   해당 소스를 포기하는 선언**입니다. 사람이 판단해야 하고, 지웠다면 그 근거를
   configs/data_sim.yaml 에 남기세요.
MSG
    fail "코퍼스가 선언과 다릅니다"
fi

# ---------------------------------------------------------------- 3. 규약 검증
say "3/6  테스트 전수"
"$PY" -m pytest -o addopts="" -q || fail "테스트가 실패했습니다 — 이 상태로 60시간을 태우지 마세요"

# ---------------------------------------------------------------- 4. 손실 예산
say "4/6  손실 그래디언트 예산"
"$PY" -m pytest -o addopts="" -q tests/test_loss_gradient_budget.py \
  || fail "λ 가 설계 예산 밖입니다"

# ---------------------------------------------------------------- 5. 대역 확인
say "5/6  사전학습 대역이 파인튜닝 대역을 덮는가"
"$PY" - <<'PYEOF' || exit 1
import sys, yaml, pathlib
sys.path.insert(0, "src")
from deep_anc.dsp.secondary_path import load_secondary_path

duct = yaml.safe_load(pathlib.Path("configs/duct.yaml").read_text(encoding="utf-8"))
npz = duct["secondary_path"]["npz"]
band = load_secondary_path(npz).trusted_band_hz()
print(f"S(z) 신뢰대역 = {band}  ({npz})")
if band[1] < 1600.0 - 1e-6:
    print("⚠ 신뢰대역 상단이 1600Hz 미만입니다 — 절대목표 1의 고역을 학습할 수 없습니다")
    sys.exit(1)
print("OK — 재사전학습은 이 대역으로 돌아야 하고, 그래야 파인튜닝 게이트를 통과합니다")
PYEOF

# ---------------------------------------------------------------- 6. 기동
say "6/6  사전학습"
if [ "$START_TRAINING" -eq 0 ]; then
    cat <<'MSG'
검증 통과. 학습을 시작하려면 --train 을 붙여 다시 실행하세요.

  bash scripts/elice/bootstrap.sh --train

⚠ 재녹음(로컬, 38.5분)이 끝나 있어야 파인튜닝까지 갈 수 있습니다. 사전학습은 재녹음과
  독립이므로 먼저 시작해도 됩니다.
MSG
    exit 0
fi

mkdir -p runs
LOG="runs/pretrain_$(date +%Y%m%d_%H%M%S).log"
echo "로그: $LOG"
setsid nohup "$PY" scripts/train/train.py \
    --config configs/train_pretrain.yaml \
    > "$LOG" 2>&1 < /dev/null &
echo "PID $! 로 기동했습니다. 진행: tail -f $LOG"
