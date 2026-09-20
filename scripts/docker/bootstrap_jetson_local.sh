#!/bin/bash
# jetson-local 컨테이너의 전용 venv 볼륨만 초기화한다.
set -euo pipefail
if [[ "${DEEP_ANC_CONTAINER:-}" != jetson || ! -f /.dockerenv || "$(uname -m)" != aarch64 ]]; then
  echo '실제 Jetson Docker 컨테이너 안에서만 실행하세요.' >&2
  exit 1
fi
cd /workspace/Deep-ANC
if ! mountpoint -q .venv; then
  echo '.venv에 별도 Docker 볼륨이 필요합니다. 호스트 venv에는 설치하지 않습니다.' >&2
  exit 1
fi
exec 9>.venv/.bootstrap.lock
flock -n 9 || { echo '다른 venv 설치가 진행 중입니다.' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install 'pip==26.2.1'
.venv/bin/python -m pip install -r requirements-jetson.txt -r docker/requirements-dev.txt
.venv/bin/python -m pip install -r docker/requirements-jetson-runtime.txt
.venv/bin/python docker/preload_jetson.py
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
