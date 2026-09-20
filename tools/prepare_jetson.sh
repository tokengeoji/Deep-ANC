#!/usr/bin/env bash
# 호스트에서는 기존 Docker로 전달하고, 검증과 학습은 컨테이너에서만 실행한다.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install_deps=0
allow_cpu=0
original_args=("$@")
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"

usage() {
    echo "사용: bash tools/prepare_jetson.sh [--install] [--allow-cpu] [--python /path/to/python]"
    echo "  --install    호환 옵션: 기존 Docker 의존성 관리 절차를 안내하며 패키지는 변경하지 않습니다."
    echo "  --allow-cpu  CPU Docker에서 오프라인 준비를 검사합니다. Jetson/CUDA 검증은 아닙니다."
    echo "  --python     컨테이너 내부 Python을 선택합니다. 기본값은 저장소 .venv/bin/python입니다."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) install_deps=1; shift ;;
        --allow-cpu) allow_cpu=1; shift ;;
        --python)
            if [[ $# -lt 2 ]]; then
                echo "오류: --python 뒤에 컨테이너 내부 Python 경로가 필요합니다." >&2
                exit 2
            fi
            python_bin="$2"
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "오류: 알 수 없는 인자: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# 호스트 .venv를 조회하거나 Python을 실행하기 전에 Docker로 전달한다.
if [[ -z "${DEEP_ANC_CONTAINER:-}" && ! -f /.dockerenv ]]; then
    container_running=""
    if command -v docker >/dev/null 2>&1; then
        container_running="$(docker container inspect --format '{{.State.Running}}' deep-anc-dev 2>/dev/null || true)"
    fi
    if [[ "$container_running" != true ]]; then
        echo "오류: 실행 중인 deep-anc-dev Docker 환경이 필요합니다. 호스트 Python은 실행하지 않았습니다." >&2
        echo "기존 환경 확인: bash scripts/docker/dev.sh status" >&2
        echo "중지된 환경 재개: bash scripts/docker/dev.sh start" >&2
        echo "새 x86 CPU 환경: bash scripts/docker/dev.sh build cpu && bash scripts/docker/dev.sh up cpu" >&2
        echo "새 Jetson 환경: bash scripts/docker/dev.sh build jetson-local && bash scripts/docker/dev.sh up jetson-local" >&2
        echo "Jetson 최초 의존성 준비: bash scripts/docker/dev.sh exec bash scripts/docker/bootstrap_jetson_local.sh" >&2
        echo "상세 절차: docker/README.md. 환경을 자동 생성하거나 교체하지 않습니다." >&2
        exit 1
    fi
    exec bash "$repo_root/scripts/docker/dev.sh" exec bash tools/prepare_jetson.sh "${original_args[@]}"
fi

if [[ "$install_deps" -eq 1 ]]; then
    echo "--install: 이 저장소는 Docker 이미지/전용 venv 볼륨으로 의존성을 관리합니다."
    echo "이 명령은 패키지를 설치하지 않고 기존 환경의 검증을 계속합니다."
    echo "최초 설치는 docker/README.md를 따르세요. jetson-local은 기존 bootstrap_jetson_local.sh를 사용합니다."
fi

# 상대 경로를 선택한 경우 호출 위치에서 해석한 뒤 저장소로 이동한다.
if ! python_bin="$(command -v -- "$python_bin")"; then
    echo "오류: 컨테이너 Python이 없습니다. docker/README.md의 환경 준비/복구 절차를 따르세요." >&2
    echo "jetson-local 최초 설치: bash scripts/docker/bootstrap_jetson_local.sh" >&2
    echo "준비된 다른 컨테이너 환경은 --python /path/to/python으로 선택할 수 있습니다." >&2
    exit 1
fi
if [[ "$python_bin" != /* ]]; then
    python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd)/$(basename -- "$python_bin")"
fi

cd -- "$repo_root"
echo "컨테이너 Python: $python_bin"
if [[ "$allow_cpu" -eq 1 ]]; then
    device="cpu"
    "$python_bin" "$repo_root/tools/jetson_preflight.py" --allow-cpu
else
    device="cuda"
    "$python_bin" "$repo_root/tools/jetson_preflight.py" --require-jetson --require-cuda
fi

"$python_bin" "$repo_root/tools/prepare_secondary_path.py" --output-dir "$repo_root/artifacts/secondary_path"
mkdir -p "$repo_root/runs/smoke"
smoke_output="$(mktemp -d "$repo_root/runs/smoke/run-XXXXXXXX")"
"$python_bin" -m deepanc.train \
    --config "$repo_root/configs/anc_train.json" \
    --smoke-test \
    --device "$device" \
    --output "$smoke_output"

if [[ "$allow_cpu" -eq 1 ]]; then
    echo "CPU Docker 준비 검사 통과. Jetson 하드웨어/CUDA와 실제 음향 감쇠는 검증하지 않았습니다."
else
    echo "Jetson Docker 준비 검사 통과: 보정 내보내기와 CUDA 학습 smoke test를 완료했습니다."
fi
echo "Smoke 결과: $smoke_output"
echo "다음: docs/DATASET.md에 따라 동기 녹음과 datasets/anc/prepared/manifest.jsonl을 준비하세요."
echo "합성 smoke test는 실제 ANC 성능 검증이 아닙니다."
printf '컨테이너 내부 학습 명령: %q -m deepanc.train --config %q --manifest %q --device %q --output %q\n' \
    "$python_bin" "$repo_root/configs/anc_train.json" "$repo_root/datasets/anc/prepared/manifest.jsonl" "$device" "$repo_root/runs/omap_anc"
