#!/bin/bash
# Elice Cloud (A100, VSCode CUDA 12.8 환경) 초기 셋업.
# 웹 VS Code 터미널에서:  bash scripts/elice/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# venv 자체를 만드는 최초 한 번만 시스템 python3가 필요하다. 그 이후의 모든
# Python 실행은 인터프리터 혼선을 막기 위해 venv 경로를 명시한다.
python3 -m venv .venv 2>/dev/null || python3 -m venv --without-pip .venv
VENV_PYTHON="$PWD/.venv/bin/python"
SETUP_MARKER="$PWD/.venv/.setup-complete"
ENVIRONMENT_RECEIPT="$PWD/.venv/environment-freeze.txt"
SOURCE_COMMIT=$(GIT_NO_REPLACE_OBJECTS=1 git rev-parse --verify 'HEAD^{commit}')
SOURCE_COMMIT=${SOURCE_COMMIT,,}
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "[오류] environment freeze에 결속할 전체 40자리 source commit을 읽지 못했습니다." >&2
  exit 1
fi
rm -f "$SETUP_MARKER"
rm -f "$ENVIRONMENT_RECEIPT" "${ENVIRONMENT_RECEIPT}.building"

"$VENV_PYTHON" -m pip -q install -U pip 2>/dev/null || {
  curl -fsS https://bootstrap.pypa.io/get-pip.py | "$VENV_PYTHON"
}

"$VENV_PYTHON" -m pip install -r requirements-train.txt
"$VENV_PYTHON" -m pip install -e .

"$VENV_PYTHON" - <<'EOF'
import deep_anc
import h5py
import matplotlib
import numpy
import onnx
import onnxruntime
import pytest
import scipy
import soundfile
import tensorboard
import torch
import tqdm
import yaml

print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if str(torch.__version__) != "2.5.1+cu121":
    raise SystemExit(
        f"torch wheel이 exact 2.5.1+cu121이 아닙니다: {torch.__version__}"
    )
if str(torch.version.cuda) != "12.1":
    raise SystemExit(f"torch CUDA build가 exact 12.1이 아닙니다: {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA를 사용할 수 없어 Elice 학습 환경 셋업을 완료하지 않습니다")
if torch.cuda.device_count() < 1:
    raise SystemExit("CUDA 장치가 없어 Elice 학습 환경 셋업을 완료하지 않습니다")
torch.empty(1, device="cuda")
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}:", torch.cuda.get_device_name(i))
EOF
LC_ALL=C "$VENV_PYTHON" -m pip freeze --all | LC_ALL=C sort > "${ENVIRONMENT_RECEIPT}.building"
if [ ! -s "${ENVIRONMENT_RECEIPT}.building" ] || \
   ! grep -Fxq 'torch==2.5.1+cu121' "${ENVIRONMENT_RECEIPT}.building"; then
  rm -f "${ENVIRONMENT_RECEIPT}.building"
  echo "[오류] freeze receipt에 exact torch wheel이 없어 완료 marker를 쓰지 않습니다." >&2
  exit 1
fi
if ! PYTHONDONTWRITEBYTECODE=1 "$VENV_PYTHON" -B - \
    "${ENVIRONMENT_RECEIPT}.building" "$SOURCE_COMMIT" <<'PY'
import sys
from pathlib import Path

from deep_anc.data.source_trust import (
    SourceTrustError,
    validate_environment_freeze_source_commit,
)

try:
    validate_environment_freeze_source_commit(
        Path(sys.argv[1]).read_bytes(), expected_commit=sys.argv[2]
    )
except (OSError, SourceTrustError) as exc:
    print(f"[오류] environment freeze source 결속 실패: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
then
  rm -f "${ENVIRONMENT_RECEIPT}.building"
  exit 1
fi
mv -f "${ENVIRONMENT_RECEIPT}.building" "$ENVIRONMENT_RECEIPT"
touch "$SETUP_MARKER"
echo "셋업 완료. 환경 receipt: $ENVIRONMENT_RECEIPT"
