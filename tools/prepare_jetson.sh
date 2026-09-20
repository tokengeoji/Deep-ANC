#!/usr/bin/env bash
# Run from any directory. Package installation is opt-in and excludes PyTorch.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install_deps=0
allow_cpu=0
python_bin="${PYTHON_BIN:-python3}"
python_explicit=0
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_explicit=1
fi

usage() {
    echo "Usage: bash tools/prepare_jetson.sh [--install] [--allow-cpu] [--python /path/to/python]"
    echo "  --install     Create .venv with system site packages and install non-Torch dependencies."
    echo "  --allow-cpu   Permit local CPU preparation; does not certify Jetson/CUDA readiness."
    echo "  --python     Select a Python interpreter (default: .venv/bin/python if present, else python3)."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install) install_deps=1; shift ;;
        --allow-cpu) allow_cpu=1; shift ;;
        --python)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --python requires a Python executable." >&2
                exit 2
            fi
            python_bin="$2"
            python_explicit=1
            shift 2
            ;;
        --help|-h) usage; exit 0 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Resolve relative interpreter paths before changing into the repository.
if ! python_bin="$(command -v -- "$python_bin")"; then
    echo "ERROR: Python executable not found. Use --python /path/to/python." >&2
    exit 1
fi
if [[ "$python_bin" != /* ]]; then
    python_bin="$(cd -- "$(dirname -- "$python_bin")" && pwd)/$(basename -- "$python_bin")"
fi

cd -- "$repo_root"
venv_dir="$repo_root/.venv"
if [[ "$install_deps" -eq 1 ]]; then
    if [[ ! -e "$venv_dir" ]]; then
        if "$python_bin" -c 'import ensurepip' >/dev/null 2>&1; then
            "$python_bin" -m venv --system-site-packages "$venv_dir"
        elif "$python_bin" -c 'import venv, pip' >/dev/null 2>&1; then
            # Minimal Debian images may omit ensurepip. The environment can use
            # the existing pip module while pip still installs into sys.prefix.
            "$python_bin" -m venv --without-pip --system-site-packages "$venv_dir"
        else
            echo "ERROR: Python venv/pip support is missing; no environment was created." >&2
            echo "Install packages matching the selected Python (on JetPack: sudo apt install python3-venv python3-pip), then retry." >&2
            echo "Or select an existing working environment with --python, without --install." >&2
            exit 1
        fi
    elif [[ ! -x "$venv_dir/bin/python" ]]; then
        echo "ERROR: .venv exists but is not a usable environment; select or repair it manually." >&2
        exit 1
    fi
    if ! "$venv_dir/bin/python" -c 'import pathlib, sys; cfg = pathlib.Path(sys.prefix, "pyvenv.cfg").read_text().lower(); sys.exit(0 if "include-system-site-packages = true" in cfg else 1)'; then
        echo "ERROR: Existing .venv does not expose system site packages and may hide NVIDIA PyTorch." >&2
        echo "Choose a separate environment with --system-site-packages; no existing environment was changed." >&2
        exit 1
    fi
    python_bin="$venv_dir/bin/python"
    if ! "$python_bin" -m pip --version >/dev/null 2>&1; then
        echo "ERROR: The selected .venv cannot import pip; repair its matching Python venv/pip packages." >&2
        exit 1
    fi
    "$python_bin" -m pip install -r "$repo_root/requirements-jetson.txt"
elif [[ "$python_explicit" -eq 0 && -x "$venv_dir/bin/python" ]]; then
    python_bin="$venv_dir/bin/python"
fi

echo "Python: $python_bin"
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
    echo "CPU preparation passed. Jetson hardware/CUDA and live acoustic cancellation are not verified."
else
    echo "Jetson preparation passed: calibration export and CUDA training smoke test completed."
fi
echo "Smoke results: $smoke_output"
echo "Next: prepare aligned recordings and datasets/anc/prepared/manifest.jsonl as described in docs/DATASET.md."
echo "A synthetic smoke test does not establish real-world ANC performance."
printf 'Training command: %q -m deepanc.train --config %q --manifest %q --device %q --output %q\n' \
    "$python_bin" "$repo_root/configs/anc_train.json" "$repo_root/datasets/anc/prepared/manifest.jsonl" "$device" "$repo_root/runs/anc"
