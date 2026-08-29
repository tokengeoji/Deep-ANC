#!/bin/bash
# 새 Elice 인스턴스 부트스트랩 — exact code + canonical holdout + 환경 + 데이터 검증.
# 사용 (새 인스턴스의 홈에서):
#   git clone https://github.com/Roka-jsj/Deep-ANC.git Deep_ANC && cd Deep_ANC
#   bash scripts/elice/bootstrap_all.sh \
#     --expected-commit "$EXPECTED_COMMIT" \
#     --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
#     --expected-transfer-manifest-sha256 "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
#     --no-update
# full-octave 학습을 주장하는 경우에는 일반 Stage-1 bootstrap과 별도로 다음을
# 명시해야 한다. BSD35k fx-m 같은 native >=22628 Hz machine source의 official
# archive/lineage/full-decode/native-PSD evidence가 없으면 public corpus 다운로드나
# 학습으로 넘어가지 않는다.
#   --full-octave \
#   --full-octave-highrate-machine-evidence results/provenance/...json \
#   --expected-full-octave-highrate-machine-evidence-sha256 <64-hex>
# EXPECTED_COMMIT은 실행 전에 신뢰한 출처에서 확인한 **전체 40자리 SHA**여야 한다.
# EXPECTED_HOLDOUT_SHA256도 Jetson에서 확인한 canonical 파일의 64자리 SHA여야 한다.
# 일반 Elice 실행은 canonical provenance/recorded/RIR/strict P·S 전부를 포함한 transfer
# manifest의 외부 전달 SHA-256도 요구한다. venv 전에는 manifest 자체의 immutable SHA
# anchor만 확인하고, 환경이 완성된 직후 공개 raw 다운로드 전에 전체 semantic 검증을 한다.
# --preflight-only는 code+holdout bundle만 본다.
# 이 스크립트는 환경/데이터 준비 전용이며 어떤 학습 프로세스도 시작하지 않는다.
# 이미 실행한 적이 있으면 완전성이 검증된 단계만 건너뛴다 (재실행 안전).
#
# 데이터 구성 (48kHz):
#   DNS noise_fullband 2샤드(실환경 소음 ~11GB) + clean_fullband 음성 1샤드(~4.7GB)
#   + ESC-50(환경음) + FMA-small(음악) + DEMAND(실환경) + MIMII fan(기계소음)
# Azure blob 은 연결당 속도제한이 있어 반드시 pget.py(병렬 range)로 받는다.
set -euo pipefail
export GIT_NO_REPLACE_OBJECTS=1
# 이 실행이 새 cache를 만들지 않게 한다. 구버전 bootstrap/pytest가 이미
# 남긴 protected-root cache는 DNS selector가 exact source 검증 후 import 전에
# repository 밖 no-overwrite quarantine으로 이동하며 자동 삭제하지 않는다.
export PYTHONDONTWRITEBYTECODE=1

# --no-update: 필수. 현재 HEAD가 expected commit과 다르면 즉시 실패한다. 실행 중 pull은
# 하지 않는다. 구버전 스크립트가 자신을 업데이트한 뒤 계속 실행되는 경로를 금지한다.
# --expected-commit: 코드/데이터 계약의 신뢰 경계. 생략·축약 SHA·불일치는 실패한다.
# --expected-holdout-sha256: canonical holdout bytes의 신뢰 경계. 생략·불일치는 실패한다.
EXPECTED_COMMIT=""
EXPECTED_COMMIT_SEEN=0
EXPECTED_HOLDOUT_SHA256=""
EXPECTED_HOLDOUT_SHA256_SEEN=0
EXPECTED_TRANSFER_MANIFEST_SHA256=""
EXPECTED_TRANSFER_MANIFEST_SHA256_SEEN=0
RAW_HASH_WORKERS=1
RAW_HASH_WORKERS_SEEN=0
REUSE_DECODER_AUDIT=0
EXPECTED_DECODER_AUDIT_SHA256=""
EXPECTED_DECODER_AUDIT_SHA256_SEEN=0
EXPECTED_DECODER_AUDIT_FILE_SHA256=""
EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN=0
NO_UPDATE_SEEN=0
PREFLIGHT_ONLY=0
FULL_OCTAVE=0
FULL_OCTAVE_SEEN=0
FULL_OCTAVE_HIGHRATE_EVIDENCE=""
FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN=0
EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256=""
EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-update)
      if [ "$NO_UPDATE_SEEN" -ne 0 ]; then
        echo "[오류] --no-update는 한 번만 지정하세요." >&2
        exit 2
      fi
      NO_UPDATE_SEEN=1
      ;;
    --expected-commit)
      if [ "$EXPECTED_COMMIT_SEEN" -ne 0 ]; then
        echo "[오류] --expected-commit은 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_COMMIT_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-commit 뒤에 전체 40자리 SHA가 필요합니다." >&2
        exit 2
      fi
      EXPECTED_COMMIT=$1
      ;;
    --expected-commit=*)
      if [ "$EXPECTED_COMMIT_SEEN" -ne 0 ]; then
        echo "[오류] --expected-commit은 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_COMMIT_SEEN=1
      EXPECTED_COMMIT=${1#*=}
      ;;
    --expected-holdout-sha256)
      if [ "$EXPECTED_HOLDOUT_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-holdout-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_HOLDOUT_SHA256_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-holdout-sha256 뒤에 64자리 SHA-256이 필요합니다." >&2
        exit 2
      fi
      EXPECTED_HOLDOUT_SHA256=$1
      ;;
    --expected-holdout-sha256=*)
      if [ "$EXPECTED_HOLDOUT_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-holdout-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_HOLDOUT_SHA256_SEEN=1
      EXPECTED_HOLDOUT_SHA256=${1#*=}
      ;;
    --expected-transfer-manifest-sha256)
      if [ "$EXPECTED_TRANSFER_MANIFEST_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-transfer-manifest-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_TRANSFER_MANIFEST_SHA256_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-transfer-manifest-sha256 뒤에 64자리 SHA-256이 필요합니다." >&2
        exit 2
      fi
      EXPECTED_TRANSFER_MANIFEST_SHA256=$1
      ;;
    --expected-transfer-manifest-sha256=*)
      if [ "$EXPECTED_TRANSFER_MANIFEST_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-transfer-manifest-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_TRANSFER_MANIFEST_SHA256_SEEN=1
      EXPECTED_TRANSFER_MANIFEST_SHA256=${1#*=}
      ;;
    --raw-hash-workers)
      if [ "$RAW_HASH_WORKERS_SEEN" -ne 0 ]; then
        echo "[오류] --raw-hash-workers는 한 번만 지정하세요." >&2
        exit 2
      fi
      RAW_HASH_WORKERS_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --raw-hash-workers 뒤에 1~32 정수가 필요합니다." >&2
        exit 2
      fi
      RAW_HASH_WORKERS=$1
      ;;
    --raw-hash-workers=*)
      if [ "$RAW_HASH_WORKERS_SEEN" -ne 0 ]; then
        echo "[오류] --raw-hash-workers는 한 번만 지정하세요." >&2
        exit 2
      fi
      RAW_HASH_WORKERS_SEEN=1
      RAW_HASH_WORKERS=${1#*=}
      ;;
    --reuse-decoder-audit)
      if [ "$REUSE_DECODER_AUDIT" -ne 0 ]; then
        echo "[오류] --reuse-decoder-audit은 한 번만 지정하세요." >&2
        exit 2
      fi
      REUSE_DECODER_AUDIT=1
      ;;
    --expected-decoder-audit-sha256)
      if [ "$EXPECTED_DECODER_AUDIT_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-decoder-audit-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_SHA256_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-decoder-audit-sha256 뒤에 64자리 SHA-256이 필요합니다." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_SHA256=$1
      ;;
    --expected-decoder-audit-sha256=*)
      if [ "$EXPECTED_DECODER_AUDIT_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-decoder-audit-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_SHA256_SEEN=1
      EXPECTED_DECODER_AUDIT_SHA256=${1#*=}
      ;;
    --expected-decoder-audit-file-sha256)
      if [ "$EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-decoder-audit-file-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-decoder-audit-file-sha256 뒤에 64자리 SHA-256이 필요합니다." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_FILE_SHA256=$1
      ;;
    --expected-decoder-audit-file-sha256=*)
      if [ "$EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-decoder-audit-file-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN=1
      EXPECTED_DECODER_AUDIT_FILE_SHA256=${1#*=}
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      ;;
    --full-octave)
      if [ "$FULL_OCTAVE_SEEN" -ne 0 ]; then
        echo "[오류] --full-octave는 한 번만 지정하세요." >&2
        exit 2
      fi
      FULL_OCTAVE_SEEN=1
      FULL_OCTAVE=1
      ;;
    --full-octave-highrate-machine-evidence)
      if [ "$FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN" -ne 0 ]; then
        echo "[오류] --full-octave-highrate-machine-evidence는 한 번만 지정하세요." >&2
        exit 2
      fi
      FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --full-octave-highrate-machine-evidence 뒤에 evidence 상대경로가 필요합니다." >&2
        exit 2
      fi
      FULL_OCTAVE_HIGHRATE_EVIDENCE=$1
      ;;
    --full-octave-highrate-machine-evidence=*)
      if [ "$FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN" -ne 0 ]; then
        echo "[오류] --full-octave-highrate-machine-evidence는 한 번만 지정하세요." >&2
        exit 2
      fi
      FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN=1
      FULL_OCTAVE_HIGHRATE_EVIDENCE=${1#*=}
      ;;
    --expected-full-octave-highrate-machine-evidence-sha256)
      if [ "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-full-octave-highrate-machine-evidence-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN=1
      shift
      if [ "$#" -eq 0 ]; then
        echo "[오류] --expected-full-octave-highrate-machine-evidence-sha256 뒤에 64자리 SHA-256이 필요합니다." >&2
        exit 2
      fi
      EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256=$1
      ;;
    --expected-full-octave-highrate-machine-evidence-sha256=*)
      if [ "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN" -ne 0 ]; then
        echo "[오류] --expected-full-octave-highrate-machine-evidence-sha256는 한 번만 지정하세요." >&2
        exit 2
      fi
      EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN=1
      EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256=${1#*=}
      ;;
    *)
      echo "[오류] 알 수 없는 인자: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "[오류] --expected-commit에 신뢰한 전체 40자리 commit SHA를 지정해야 합니다." >&2
  exit 2
fi
EXPECTED_COMMIT=${EXPECTED_COMMIT,,}
if [[ ! "$EXPECTED_HOLDOUT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "[오류] --expected-holdout-sha256에 신뢰한 64자리 SHA-256을 지정해야 합니다." >&2
  exit 2
fi
EXPECTED_HOLDOUT_SHA256=${EXPECTED_HOLDOUT_SHA256,,}
if [ "$REUSE_DECODER_AUDIT" -eq 1 ]; then
  if [[ ! "$EXPECTED_DECODER_AUDIT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "[오류] audit 재사용에는 --expected-decoder-audit-sha256가 필요합니다." >&2
    exit 2
  fi
  if [[ ! "$EXPECTED_DECODER_AUDIT_FILE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "[오류] audit 재사용에는 --expected-decoder-audit-file-sha256가 필요합니다." >&2
    exit 2
  fi
  EXPECTED_DECODER_AUDIT_SHA256=${EXPECTED_DECODER_AUDIT_SHA256,,}
  EXPECTED_DECODER_AUDIT_FILE_SHA256=${EXPECTED_DECODER_AUDIT_FILE_SHA256,,}
elif [ "$EXPECTED_DECODER_AUDIT_SHA256_SEEN" -ne 0 ] || [ "$EXPECTED_DECODER_AUDIT_FILE_SHA256_SEEN" -ne 0 ]; then
  echo "[오류] decoder audit SHA 인자는 --reuse-decoder-audit과 함께만 지정할 수 있습니다." >&2
  exit 2
fi
if [ "$NO_UPDATE_SEEN" -ne 1 ]; then
  echo "[오류] --no-update는 필수입니다. exact checkout에서 다시 실행하세요." >&2
  exit 2
fi
if [ "$FULL_OCTAVE" -eq 1 ]; then
  if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
    echo "[오류] --full-octave와 --preflight-only는 함께 쓸 수 없습니다. 환경을 열지 않는 source 재검증은 audit_bsd35k_highrate_machine.py verify를 직접 실행하세요." >&2
    exit 2
  fi
  if [ "$FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN" -ne 1 ] || \
     [ -z "$FULL_OCTAVE_HIGHRATE_EVIDENCE" ] || \
     [ "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN" -ne 1 ] || \
     [[ ! "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "[오류] --full-octave에는 high-rate machine evidence 경로와 외부 64자리 SHA-256이 모두 필수입니다." >&2
    exit 2
  fi
  EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256=${EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256,,}
elif [ "$FULL_OCTAVE_HIGHRATE_EVIDENCE_SEEN" -ne 0 ] || \
     [ "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256_SEEN" -ne 0 ]; then
  echo "[오류] full-octave high-rate machine evidence 인자는 --full-octave와 함께만 지정할 수 있습니다." >&2
  exit 2
fi
if [[ ! "$RAW_HASH_WORKERS" =~ ^([1-9]|[12][0-9]|3[0-2])$ ]]; then
  echo "[오류] --raw-hash-workers는 1~32 정수여야 합니다." >&2
  exit 2
fi

# 호출 위치나 clone 디렉터리 이름에 의존하지 않는다. 실제 스크립트 위치에서 저장소
# root를 계산하므로 Elice의 ``~/Deep_ANC``와 사용자의 다른 clone 이름 모두 같은
# exact checkout을 검사한다. 명시 환경변수는 테스트/운영에서만 이 기본값을 덮는다.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DEFAULT_REPO=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd -P)
REPO=${DEEP_ANC_BOOTSTRAP_REPO:-"$DEFAULT_REPO"}
if ! cd "$REPO"; then
  echo "[오류] 저장소에 들어갈 수 없습니다: $REPO" >&2
  exit 1
fi
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null || true)" != "true" ]; then
  echo "[오류] git 작업 트리가 아닙니다: $REPO" >&2
  exit 1
fi

# pget 자체 잠금만으로는 wget .part, unzip 대상, manifest 생성을 보호할 수 없다.
# 별도 lock 파일을 만들면 --preflight-only도 .git에 inode를 남긴다. 이미 존재하는 .git
# 디렉터리 inode를 read-only fd로 열어 같은 bootstrap끼리 advisory lock을 공유한다.
exec 8<"$REPO/.git"
if ! flock -n 8; then
  echo "[오류] 다른 bootstrap_all.sh가 이미 실행 중입니다. 중복 실행하지 않습니다." >&2
  exit 1
fi
active_train=$(pgrep -af '[t]rain\.py' || true)
if [ -n "$active_train" ]; then
  echo "[오류] 기존 train.py 학습이 실행 중이므로 데이터/manifest를 건드리지 않습니다:" >&2
  echo "$active_train" >&2
  exit 1
fi

HOLDOUT_MANIFEST="$REPO/data/manifests/recorded_holdout.json"
HOLDOUT_VALIDATOR="$REPO/src/deep_anc/data/holdout_contract.py"
TRANSFER_MANIFEST="$REPO/data/manifests/elice_transfer_manifest.json"
# Canonical 학습은 raw 전수 decoder audit과 한 세대로 발행한 v4 manifest만 읽는다.
# audit 원본은 결과 증거로 보존하고, prepare transaction이 같은 bytes를 canonical
# directory에 복사해 sidecar와 결속한다.
CANONICAL_MANIFEST_DIR="data/manifests/canonical_v4"
DECODER_AUDIT_REPORT="results/provenance/decoder_audit.json"
RECORDED_QA_JSON="data/manifests/recorded_qa.json"
RECORDED_QA_MD="data/manifests/recorded_qa.md"
RECORDED_SUBBAND_COVERAGE_REPORT_DIR="results/data_audit/recorded_subband_coverage"
RECORDED_MANIFEST=""
RECORDED_MANIFEST_SHA256=""
RECORDED_SESSION_COUNT=""
RECORDED_TRANSFER_SCHEMA=""
RECORDED_GENERATION=""
RECORDED_GENERATION_SHA256=""

verify_exact_checkout() {
  local current_commit hidden_flags replace_refs
  replace_refs=$(git for-each-ref --format='%(refname)' refs/replace)
  if [ -n "$replace_refs" ]; then
    echo "[오류] git replace ref가 있어 object identity를 우회할 수 있습니다:" >&2
    echo "$replace_refs" >&2
    return 1
  fi
  if [ -s "$REPO/.git/info/grafts" ]; then
    echo "[오류] legacy git grafts가 있어 commit ancestry를 우회할 수 있습니다." >&2
    return 1
  fi
  hidden_flags=$(git ls-files -v | awk 'substr($0,1,1) ~ /[a-zS]/ { print }')
  if [ -n "$hidden_flags" ]; then
    echo "[오류] assume-unchanged/skip-worktree index flag가 설정된 tracked 파일이 있습니다:" >&2
    echo "$hidden_flags" >&2
    return 1
  fi
  if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    echo "[오류] 작업 트리에 로컬 변경이 있습니다. 보존·커밋한 뒤 깨끗한 체크아웃에서 다시 실행하세요." >&2
    return 1
  fi
  current_commit=$(git rev-parse --verify 'HEAD^{commit}')
  current_commit=${current_commit,,}
  if [ "$current_commit" != "$EXPECTED_COMMIT" ]; then
    echo "[오류] --no-update 상태에서 HEAD가 expected commit과 다릅니다." >&2
    echo "  expected: $EXPECTED_COMMIT" >&2
    echo "  current:  $current_commit" >&2
    return 1
  fi
  # git status/index flag에 의존하지 않고 expected tree의 모든 blob과 mode를 실제
  # worktree bytes에 대조한다. assume-unchanged로 숨긴 변경도 여기서는 통과할 수 없다.
  if ! PYTHONDONTWRITEBYTECODE=1 python3 -B - "$REPO" "$EXPECTED_COMMIT" <<'PY'
import os
import stat
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
commit = sys.argv[2]
env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1")
tree = subprocess.check_output(
    ["git", "ls-tree", "-r", "-z", "--full-tree", commit], cwd=root, env=env
)
failures = []
for record in tree.split(b"\0"):
    if not record:
        continue
    metadata, raw_path = record.split(b"\t", 1)
    mode, kind, object_id = metadata.decode("ascii").split()
    path_text = os.fsdecode(raw_path)
    path = root / path_text
    if kind == "commit":
        failures.append(f"submodule은 bootstrap exact tree에서 허용하지 않음: {path_text}")
        continue
    if kind != "blob":
        failures.append(f"지원하지 않는 tree object {kind}: {path_text}")
        continue
    try:
        if mode == "120000":
            if not path.is_symlink():
                raise OSError("expected symlink")
            actual = os.fsencode(os.readlink(path))
        else:
            if not path.is_file() or path.is_symlink():
                raise OSError("expected regular file")
            actual = path.read_bytes()
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise OSError(f"executable mode mismatch ({mode})")
        expected = subprocess.check_output(
            ["git", "cat-file", "blob", object_id], cwd=root, env=env
        )
        if actual != expected:
            raise OSError("blob bytes mismatch")
    except OSError as exc:
        failures.append(f"{path_text}: {exc}")
if failures:
    print("[오류] expected commit tree와 실제 tracked bytes가 다릅니다:", file=sys.stderr)
    for item in failures[:20]:
        print(f"  {item}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    return 1
  fi
  echo "[git] no-replace expected commit + tracked blob bytes 일치: $current_commit"
}

verify_canonical_bundle() {
  if [ ! -s "$HOLDOUT_MANIFEST" ]; then
    echo "[오류] 파인튜닝 누수 방지 held-out manifest가 없습니다: $HOLDOUT_MANIFEST" >&2
    echo "Jetson에서 provenance 복구 도구가 만든 canonical holdout을 먼저 전송하세요." >&2
    echo "holdout 없이 공개 코퍼스 manifest를 만들거나 학습을 시작하지 않습니다." >&2
    return 1
  fi
  if [ ! -f "$HOLDOUT_VALIDATOR" ]; then
    echo "[오류] canonical holdout validator가 없습니다: $HOLDOUT_VALIDATOR" >&2
    return 1
  fi
  if ! PYTHONDONTWRITEBYTECODE=1 python3 -B "$HOLDOUT_VALIDATOR" \
      --path "$HOLDOUT_MANIFEST" \
      --repo-root "$REPO" \
      --expected-sha256 "$EXPECTED_HOLDOUT_SHA256"; then
    echo "[오류] canonical recorded holdout hash/schema/provenance 검증 실패." >&2
    return 1
  fi
}

verify_transfer_manifest_anchor() {
  # 새 인스턴스의 system Python에는 NumPy/soundfile이 없다. 여기서는 외부에서 받은
  # exact SHA와 regular-file/symlink 경계만 표준 라이브러리로 확인한다. manifest가
  # 가리키는 모든 파일·generation·DNS 음향 의미 검증은 venv 직후의
  # verify_transfer_bundle에서 반드시 다시 수행하며, 그 전에는 raw 다운로드나
  # manifest/QA/학습을 절대 시작하지 않는다.
  local resolved manifest_sha256 manifest_size extra
  if [ ! -s "$TRANSFER_MANIFEST" ]; then
    echo "[오류] Jetson immutable transfer manifest가 없습니다: $TRANSFER_MANIFEST" >&2
    return 1
  fi
  if ! resolved=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" python3 -B - \
      "$TRANSFER_MANIFEST" "$REPO" "$EXPECTED_TRANSFER_MANIFEST_SHA256" <<'PY'
import os
import sys
from pathlib import Path

from deep_anc.data.holdout_contract import HoldoutContractError, read_regular_file_snapshot

manifest_path = Path(os.path.abspath(sys.argv[1]))
root = Path(os.path.abspath(sys.argv[2]))
expected_sha256 = sys.argv[3]
try:
    snapshot = read_regular_file_snapshot(
        manifest_path,
        root=root,
        label="Jetson immutable transfer manifest anchor",
        capture_bytes=False,
    )
except (OSError, HoldoutContractError) as exc:
    print(f"[오류] {exc}", file=sys.stderr)
    raise SystemExit(1)
if snapshot.sha256 != expected_sha256:
    print(
        "[오류] transfer manifest 외부 SHA-256 anchor 불일치: "
        f"expected={expected_sha256}, actual={snapshot.sha256}",
        file=sys.stderr,
    )
    raise SystemExit(1)
print(f"{snapshot.sha256}\t{snapshot.size}")
PY
  ); then
    echo "[오류] transfer manifest SHA anchor/경로 검증 실패." >&2
    return 1
  fi
  if [[ "$resolved" == *$'\n'* ]]; then
    echo "[오류] transfer manifest anchor 결과가 단일 행이 아닙니다." >&2
    return 1
  fi
  IFS=$'\t' read -r manifest_sha256 manifest_size extra <<<"$resolved"
  if [ "$manifest_sha256" != "$EXPECTED_TRANSFER_MANIFEST_SHA256" ] || \
     [[ ! "$manifest_size" =~ ^[1-9][0-9]*$ ]] || [ -n "$extra" ]; then
    echo "[오류] transfer manifest anchor 출력이 유효하지 않습니다." >&2
    return 1
  fi
  echo "[transfer anchor] immutable manifest SHA 확인: sha256=$manifest_sha256, bytes=$manifest_size"
}

verify_transfer_bundle() {
  # full semantic validator는 NumPy/soundfile까지 포함한 exact venv에서만 실행한다.
  # 이 함수는 setup 전 호출하면 안 된다.
  local verifier_python=${1:-}
  local resolved schema_version selected_manifest selected_manifest_sha256
  local session_count validated_transfer_sha256 file_count generation_path
  local generation_sha256 extra
  if [ -z "$verifier_python" ] || [ ! -x "$verifier_python" ]; then
    echo "[오류] full transfer validator에는 완성된 venv Python이 필요합니다." >&2
    return 1
  fi
  if [ ! -s "$TRANSFER_MANIFEST" ]; then
    echo "[오류] Jetson immutable transfer manifest가 없습니다: $TRANSFER_MANIFEST" >&2
    return 1
  fi
  if ! resolved=$(PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" "$verifier_python" -B - \
      "$TRANSFER_MANIFEST" "$REPO" "$EXPECTED_TRANSFER_MANIFEST_SHA256" <<'PY'
import os
import sys
from pathlib import Path

from deep_anc.data.holdout_contract import FileSnapshot
from deep_anc.data.recorded_generation import COMBINED_SESSION_COUNT
from deep_anc.data.transfer_contract import (
    EXPECTED_RECORDED_SESSIONS,
    TransferContractError,
    validate_transfer_manifest,
)

transfer_path = Path(sys.argv[1])
root = Path(os.path.abspath(sys.argv[2]))
expected_sha256 = sys.argv[3]
try:
    summary = validate_transfer_manifest(
        transfer_path,
        repo_root=root,
        expected_sha256=expected_sha256,
    )
    recorded_manifest = summary.get("_validated_recorded_manifest_snapshot")
    generation = summary.get("_validated_recorded_generation_snapshot")
    if not isinstance(recorded_manifest, FileSnapshot) or recorded_manifest.data is None:
        raise TransferContractError(
            "transfer 검증 결과에 recorded manifest bytes snapshot이 없습니다"
        )
    if generation is None:
        schema_version = 1
        expected_sessions = EXPECTED_RECORDED_SESSIONS
        generation_relative = "-"
        generation_sha256 = "-"
    elif isinstance(generation, FileSnapshot):
        schema_version = 2
        expected_sessions = COMBINED_SESSION_COUNT
        try:
            generation_relative = generation.path.relative_to(root).as_posix()
        except ValueError as exc:
            raise TransferContractError(
                "검증된 recorded generation report가 저장소 밖입니다"
            ) from exc
        generation_sha256 = generation.sha256
    else:
        raise TransferContractError(
            "transfer 검증 결과의 recorded generation snapshot이 유효하지 않습니다"
        )
    session_count = summary.get("recorded_session_count")
    if type(session_count) is not int or session_count != expected_sessions:
        raise TransferContractError(
            "transfer 검증 결과의 recorded schema/session 수가 일치하지 않습니다: "
            f"schema={schema_version}, sessions={session_count!r}, "
            f"expected={expected_sessions}"
        )
    try:
        relative = recorded_manifest.path.relative_to(root).as_posix()
    except ValueError as exc:
        raise TransferContractError(
            "검증된 recorded manifest가 저장소 밖입니다"
        ) from exc
    if not relative or any(character in relative for character in "\t\r\n"):
        raise TransferContractError(
            "검증된 recorded manifest 경로를 shell에 안전하게 전달할 수 없습니다"
        )
    if (
        not generation_relative
        or any(character in generation_relative for character in "\t\r\n")
        or not generation_sha256
        or any(character in generation_sha256 for character in "\t\r\n")
    ):
        raise TransferContractError(
            "검증된 recorded generation path/SHA를 shell에 안전하게 전달할 수 없습니다"
        )
except (OSError, TransferContractError) as exc:
    print(f"[오류] {exc}", file=sys.stderr)
    raise SystemExit(1)

print(
    f"{schema_version}\t{relative}\t{recorded_manifest.sha256}\t"
    f"{session_count}\t{summary['manifest_sha256']}\t{summary['file_count']}\t"
    f"{generation_relative}\t{generation_sha256}"
)
PY
  ); then
    echo "[오류] recorded/RIR/strict P·S immutable transfer bundle 검증 실패." >&2
    return 1
  fi
  if [[ "$resolved" == *$'\n'* ]]; then
    echo "[오류] transfer validator 선택 결과가 단일 행이 아닙니다." >&2
    return 1
  fi
  IFS=$'\t' read -r schema_version selected_manifest selected_manifest_sha256 \
    session_count validated_transfer_sha256 file_count generation_path \
    generation_sha256 extra <<<"$resolved"
  if [[ ! "$schema_version" =~ ^(1|2)$ ]] || [ -z "$selected_manifest" ] || \
     [[ ! "$selected_manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || \
     [[ ! "$session_count" =~ ^[0-9]+$ ]] || \
     [ "$validated_transfer_sha256" != "$EXPECTED_TRANSFER_MANIFEST_SHA256" ] || \
     [[ ! "$file_count" =~ ^[0-9]+$ ]] || [ -n "$extra" ] || \
     { [ "$schema_version" = "1" ] && \
       { [ "$generation_path" != "-" ] || [ "$generation_sha256" != "-" ]; }; } || \
     { [ "$schema_version" = "2" ] && \
       { [ -z "$generation_path" ] || \
         [[ ! "$generation_sha256" =~ ^[0-9a-f]{64}$ ]]; }; }; then
    echo "[오류] transfer validator의 recorded manifest 선택 결과가 유효하지 않습니다." >&2
    return 1
  fi
  if [ -n "$RECORDED_MANIFEST" ] && \
     { [ "$RECORDED_TRANSFER_SCHEMA" != "$schema_version" ] || \
       [ "$RECORDED_MANIFEST" != "$selected_manifest" ] || \
       [ "$RECORDED_MANIFEST_SHA256" != "$selected_manifest_sha256" ] || \
       [ "$RECORDED_SESSION_COUNT" != "$session_count" ] || \
       [ "$RECORDED_GENERATION" != "$generation_path" ] || \
       [ "$RECORDED_GENERATION_SHA256" != "$generation_sha256" ]; }; then
    echo "[오류] 실행 중 검증된 recorded manifest 선택 결과가 바뀌었습니다." >&2
    return 1
  fi
  RECORDED_TRANSFER_SCHEMA=$schema_version
  RECORDED_MANIFEST=$selected_manifest
  RECORDED_MANIFEST_SHA256=$selected_manifest_sha256
  RECORDED_SESSION_COUNT=$session_count
  RECORDED_GENERATION=$generation_path
  RECORDED_GENERATION_SHA256=$generation_sha256
  echo "[transfer] immutable bundle 확인: manifest_sha256=$validated_transfer_sha256, files=$file_count, recorded_schema=$RECORDED_TRANSFER_SCHEMA, recorded_manifest=$RECORDED_MANIFEST, recorded_manifest_sha256=$RECORDED_MANIFEST_SHA256, recorded_sessions=$RECORDED_SESSION_COUNT, recorded_generation=$RECORDED_GENERATION, recorded_generation_sha256=$RECORDED_GENERATION_SHA256"
}

hardware_storage_preflight() {
  local gpu_inventory filesystem_stats total_bytes available_bytes minimum_total_bytes minimum_available_bytes
  # Elice의 nominal 128GiB overlay는 파일시스템 메타데이터 예약으로
  # df 총량이 최대 128MiB 정도 작게 보일 수 있다. nominal 계약을
  # 유지하되 이 예약분만 허용하고, 실제 작업공간은 별도로 96GiB를
  # 엄격히 요구한다.
  minimum_total_bytes=$((128 * 1024 * 1024 * 1024 - 128 * 1024 * 1024))
  # 시작 시 free 예산: untouched/extracted public corpus 약 58GiB + archive/staging
  # peak 24GiB + transferred inputs 5GiB + venv/checkpoint/headroom 9GiB = 96GiB.
  # volume 용량 계약(128GiB)과 FS overhead 뒤 실제 free 계약을 분리한다.
  minimum_available_bytes=$((96 * 1024 * 1024 * 1024))
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[오류] Elice hardware preflight에 nvidia-smi가 필요합니다." >&2
    return 1
  fi
  if ! gpu_inventory=$(nvidia-smi --query-gpu=name,memory.total \
      --format=csv,noheader,nounits 2>/dev/null); then
    echo "[오류] GPU name/total_memory를 조회할 수 없습니다." >&2
    return 1
  fi
  if ! PYTHONDONTWRITEBYTECODE=1 python3 -B - "$gpu_inventory" <<'PY'
import sys

minimum_mib = 79 * 1024
accepted = []
for line in sys.argv[1].splitlines():
    if not line.strip() or "," not in line:
        continue
    name, raw_memory = [part.strip() for part in line.rsplit(",", 1)]
    try:
        memory_mib = int(raw_memory)
    except ValueError:
        continue
    if name.startswith("NVIDIA A100") and memory_mib >= minimum_mib:
        accepted.append((name, memory_mib))
if not accepted:
    print(
        "[오류] A100 80GB 계약 불충족: 정확한 GPU name이 'NVIDIA A100...'이고 "
        f"total_memory가 {minimum_mib} MiB 이상인 장치가 필요합니다.",
        file=sys.stderr,
    )
    raise SystemExit(1)
for name, memory_mib in accepted:
    print(f"[hardware] GPU 확인: name={name}, total_memory_mib={memory_mib}")
PY
  then
    return 1
  fi
  # GNU df는 --output과 -P를 함께 지정하면 상호 배타적으로 거부하는
  # 배포판이 있다(Elice 이미지 포함). --output이 헤더/행 형식을 고정하므로
  # -P는 불필요하며, -B1만 사용해 바이트 단위를 유지한다.
  filesystem_stats=$(df -B1 --output=size,avail "$REPO" | awk 'NR==2 {print $1, $2}')
  read -r total_bytes available_bytes <<<"$filesystem_stats"
  if [[ ! "$total_bytes" =~ ^[0-9]+$ ]] || [ "$total_bytes" -lt "$minimum_total_bytes" ]; then
    echo "[오류] 학습/코퍼스 대상 filesystem total capacity가 128 GiB 미만입니다: ${total_bytes:-unknown} bytes" >&2
    return 1
  fi
  if [[ ! "$available_bytes" =~ ^[0-9]+$ ]]; then
    echo "[오류] filesystem 가용공간을 읽을 수 없습니다: ${available_bytes:-unknown}" >&2
    return 1
  fi
  if [ "$available_bytes" -lt "$minimum_available_bytes" ]; then
    # 중단 후 재실행에서는 public corpus가 이미 완전히 materialize되어
    # archive/staging 예산이 더 이상 필요하지 않을 수 있다. 수량·metadata를
    # 모두 확인한 경우에만 초기 96GiB 조건을 재적용하지 않는다. 불완전한
    # 첫 실행은 기존과 동일하게 즉시 실패한다.
    local dns_count speech_count esc_count demand_count machine_count fma_count
    dns_count=$(find "$REPO/data/raw/noise/dns_fullband" -type f -iname '*.wav' -print 2>/dev/null | wc -l)
    speech_count=$(find "$REPO/data/raw/noise/speech" -type f -iname '*.wav' -print 2>/dev/null | wc -l)
    esc_count=$(find "$REPO/data/raw/noise/esc50/ESC-50-master/audio" -type f -iname '*.wav' -print 2>/dev/null | wc -l)
    demand_count=$(find "$REPO/data/raw/noise/demand" -type f -iname '*.wav' -print 2>/dev/null | wc -l)
    machine_count=$(find "$REPO/data/raw/noise/machine" -type f -iname '*.wav' -print 2>/dev/null | wc -l)
    fma_count=$(find "$REPO/data/raw/music/fma_small" -type f -iname '*.mp3' -print 2>/dev/null | wc -l)
    if [ "$dns_count" -ne 16000 ] || [ "$speech_count" -ne 8065 ] ||
       [ "$esc_count" -ne 2000 ] || [ "$demand_count" -ne 96 ] ||
       [ "$machine_count" -ne 3600 ] || [ "$fma_count" -ne 8000 ] ||
       [ ! -s "$REPO/data/raw/music/fma_metadata/tracks.csv" ]; then
      echo "[오류] archive+staging+corpus 시작 가용공간이 96 GiB 미만입니다: ${available_bytes:-unknown} bytes" >&2
      return 1
    fi
    echo "[hardware] public corpus가 이미 완전하므로 재개 시 96GiB staging 예산 검사를 건너뜁니다 (available_bytes=$available_bytes)"
  fi
  echo "[hardware] filesystem total_bytes=$total_bytes (minimum=$minimum_total_bytes), available_bytes=$available_bytes (minimum=$minimum_available_bytes)"
}

if ! command -v python3 >/dev/null 2>&1; then
  echo "[오류] venv 생성 전 exact tree/bundle을 검증할 system python3가 없습니다." >&2
  exit 1
fi
if ! verify_exact_checkout || ! verify_canonical_bundle; then
  echo "환경 설치나 데이터 다운로드를 시작하지 않습니다." >&2
  exit 1
fi

if [ "$PREFLIGHT_ONLY" -eq 1 ]; then
  echo "[preflight] exact code + canonical bundle 검증 완료. 환경/데이터는 변경하지 않았습니다. 별도 lock 파일도 만들지 않았습니다."
  exit 0
fi

# 로컬 Jetson에서도 쓸 수 있는 --preflight-only 경계 뒤에서만 Elice 자원/대용량
# transfer manifest를 요구한다. system Python에서는 external SHA anchor만 확인한다.
# 전체 파일/계보/음향 의미 검증은 venv 완료 직후, 공개 raw 다운로드보다 먼저 실행된다.
if [[ ! "$EXPECTED_TRANSFER_MANIFEST_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "[오류] 일반 Elice 실행에는 --expected-transfer-manifest-sha256 64자리가 필수입니다." >&2
  exit 2
fi
EXPECTED_TRANSFER_MANIFEST_SHA256=${EXPECTED_TRANSFER_MANIFEST_SHA256,,}
if ! verify_transfer_manifest_anchor || ! hardware_storage_preflight; then
  echo "[오류] Elice transfer SHA anchor/hardware/storage preflight 실패. setup/download를 시작하지 않습니다." >&2
  exit 1
fi

VENV_PYTHON="$REPO/.venv/bin/python"
SETUP_MARKER="$REPO/.venv/.setup-complete"
ENVIRONMENT_RECEIPT="$REPO/.venv/environment-freeze.txt"

environment_probe() {
  [ -x "$VENV_PYTHON" ] && "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
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

if str(torch.__version__) != "2.5.1+cu121" or str(torch.version.cuda) != "12.1":
    raise SystemExit(1)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit(1)
properties = [torch.cuda.get_device_properties(index) for index in range(torch.cuda.device_count())]
if not any(
    item.name.startswith("NVIDIA A100") and item.total_memory >= 79 * 1024**3
    for item in properties
):
    raise SystemExit(1)
torch.empty(1, device="cuda")
PY
}

write_environment_receipt() {
  local building="${ENVIRONMENT_RECEIPT}.building"
  rm -f "$building"
  if ! LC_ALL=C "$VENV_PYTHON" -m pip freeze --all | LC_ALL=C sort > "$building"; then
    rm -f "$building"
    return 1
  fi
  if [ ! -s "$building" ] || ! grep -Fxq 'torch==2.5.1+cu121' "$building"; then
    echo "[오류] 환경 receipt에 exact torch wheel이 기록되지 않았습니다." >&2
    rm -f "$building"
    return 1
  fi
  mv -f "$building" "$ENVIRONMENT_RECEIPT"
}

environment_complete() {
  [ -f "$SETUP_MARKER" ] && [ -s "$ENVIRONMENT_RECEIPT" ] &&
    grep -Fxq 'torch==2.5.1+cu121' "$ENVIRONMENT_RECEIPT" && environment_probe
}

echo "=== [1/6] 환경 (venv + torch cu121 + 패키지) ==="
# 완료 마커 도입 전부터 사용하던 인스턴스는 전체 요구 패키지와 CUDA probe를
# 통과하는 경우에만 마커를 이관한다. 유효한 환경을 불필요하게 재설치하지 않는다.
if [ ! -f "$SETUP_MARKER" ] && environment_probe; then
  if write_environment_receipt; then
    touch "$SETUP_MARKER"
    echo "[setup] 기존 exact 환경에 freeze receipt와 완료 마커를 생성했습니다."
  fi
fi
if ! environment_complete; then
  echo "[setup] 완료 마커/import/CUDA probe 중 하나가 유효하지 않아 환경을 구성합니다."
  bash scripts/elice/setup_env.sh
fi
echo "[setup] reproducibility receipt: $ENVIRONMENT_RECEIPT"
if ! environment_complete; then
  echo "[오류] setup_env.sh 이후에도 Python/CUDA 환경 검증에 실패했습니다." >&2
  exit 1
fi

# 외부 SHA로 고정한 manifest에 열거된 모든 transferred file의 exact SHA, schema,
# recorded generation/DNS receipt와 canonical holdout 결속을 여기서 완전 검증한다.
# 이 gate 전에는 public raw 다운로드·manifest 생성·QA·학습이 시작되지 않는다.
if ! verify_exact_checkout || ! verify_canonical_bundle || \
   ! verify_transfer_bundle "$VENV_PYTHON"; then
  echo "[오류] 환경 완료 뒤 immutable transfer bundle full 검증 실패. public raw 다운로드를 시작하지 않습니다." >&2
  exit 1
fi

# ``--full-octave``는 Stage-1 corpus bootstrap의 별칭이 아니다. high-rate machine
# source가 없으면 16 kHz MIMII를 upsample한 tensor가 마지막 octave를 덮는 것처럼
# 보일 수 있으므로, 환경만 준비한 뒤 public corpus download 전에 raw-bound evidence를
# 먼저 직접 재검증한다. evidence PASS도 causal P/S·population·execution config를
# 대체하지 않으므로 admission-only checker가 READY일 때만 아래 download 단계로 간다.
if [ "$FULL_OCTAVE" -eq 1 ]; then
  echo "=== [full-octave/0] high-rate machine source + admission gate ==="
  if ! "$VENV_PYTHON" scripts/data/audit_bsd35k_highrate_machine.py verify \
      --evidence "$FULL_OCTAVE_HIGHRATE_EVIDENCE" \
      --expected-file-sha256 "$EXPECTED_FULL_OCTAVE_HIGHRATE_EVIDENCE_SHA256"; then
    echo "[오류] full-octave native high-rate machine source/lineage/decode/PSD evidence가 PASS가 아닙니다. public corpus 다운로드·manifest·학습을 시작하지 않습니다." >&2
    exit 1
  fi
  if ! "$VENV_PYTHON" scripts/train/check_full_octave_v3_admission.py \
      --config configs/full_octave_v3_admission.yaml --markdown; then
    echo "[오류] high-rate machine source evidence와 별개로 full-octave P/S·population·execution admission이 BLOCKED입니다. public corpus 다운로드·manifest·학습을 시작하지 않습니다." >&2
    exit 1
  fi
fi

PGET=("$VENV_PYTHON" "$REPO/scripts/elice/pget.py")
DNS_BASE="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"
ZEN="https://zenodo.org/records"
FMA_ROOT="$REPO/data/raw/music"
FMA_METADATA_ARCHIVE_SIZE=358412441
FMA_METADATA_ARCHIVE_SHA256="d9527a5297a65da31c5676484d5047c3e2b8a8060ce72a46e26158be736bf265"
FMA_TRACKS_SIZE=260414445
FMA_TRACKS_SHA256="f73260fd112b8cd42bcd4f7c8918fc66b19d9d4c7b97f4faedce524b59e95d6b"
FMA_TRACK_COUNT=106574

echo "=== [2/6] 데이터 다운로드 (병렬) ==="
mkdir -p data/raw/noise
cd data/raw/noise

declare -A DL=(
  [shard000.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"
  [shard001.tar.bz2]="$DNS_BASE/noise_fullband/datasets_fullband.noise_fullband.audioset_001.tar.bz2"
  [speech000.tar.bz2]="$DNS_BASE/clean_fullband/datasets_fullband.clean_fullband.read_speech_000_0.00_3.75.tar.bz2"
)
declare -A DEST=(
  [shard000.tar.bz2]=dns_fullband
  [shard001.tar.bz2]=dns_fullband
  [speech000.tar.bz2]=speech
)
DEMAND_ENVIRONMENTS=(DKITCHEN DWASHING OOFFICE OHALLWAY TMETRO TCAR)

file_count() {
  local root=$1
  local pattern=$2
  if [ ! -d "$root" ]; then
    echo 0
    return
  fi
  find "$root" -type f -iname "$pattern" -print | wc -l
}

esc50_complete() {
  [ "$(file_count esc50/ESC-50-master/audio '*.wav')" -eq 2000 ] &&
    [ -s esc50/ESC-50-master/meta/esc50.csv ]
}

fma_small_complete() {
  local root=${1:-"$FMA_ROOT/fma_small"}
  [ "$(file_count "$root" '*.mp3')" -eq 8000 ]
}

fma_metadata_complete() {
  local root=${1:-"$FMA_ROOT/fma_metadata"}
  local tracks="$root/tracks.csv"
  [ -s "$tracks" ] &&
    [ "$(stat -c %s "$tracks")" -eq "$FMA_TRACKS_SIZE" ] &&
    [ "$(sha256sum "$tracks" | awk '{print $1}')" = "$FMA_TRACKS_SHA256" ] &&
    "$VENV_PYTHON" - "$tracks" "$FMA_TRACK_COUNT" <<'PY' >/dev/null 2>&1
import csv
import sys
from pathlib import Path

tracks = Path(sys.argv[1])
expected_count = int(sys.argv[2])
with tracks.open(encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    level0 = next(reader)
    level1 = next(reader)
    width = max(len(level0), len(level1))
    level0 += [""] * (width - len(level0))
    level1 += [""] * (width - len(level1))

    def column(group, field):
        hits = [
            index
            for index, (left, right) in enumerate(zip(level0, level1))
            if left.strip().casefold() == group and right.strip().casefold() == field
        ]
        if len(hits) != 1:
            raise SystemExit(1)
        return hits[0]

    artist_col = column("artist", "id")
    album_col = column("album", "id")
    mapped = set()
    for row in reader:
        if not row or not row[0].strip().isdigit():
            continue
        if max(artist_col, album_col) >= len(row):
            raise SystemExit(1)
        if not row[artist_col].strip() or not row[album_col].strip():
            raise SystemExit(1)
        mapped.add(int(row[0]))
if len(mapped) != expected_count:
    raise SystemExit(1)
PY
}

fma_audio_metadata_match() {
  "$VENV_PYTHON" - "$FMA_ROOT/fma_metadata/tracks.csv" "$FMA_ROOT/fma_small" <<'PY' >/dev/null 2>&1
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle)
    next(reader)
    next(reader)
    mapped = {int(row[0]) for row in reader if row and row[0].strip().isdigit()}
audio_ids = {
    int(path.stem)
    for path in Path(sys.argv[2]).rglob("*.mp3")
    if path.stem.isdigit()
}
if len(audio_ids) != 8000 or not audio_ids.issubset(mapped):
    raise SystemExit(1)
PY
}

fma_complete() {
  fma_small_complete && fma_metadata_complete && fma_audio_metadata_match
}

demand_environment_complete() {
  local environment=$1
  [ "$(file_count "demand/$environment" '*.wav')" -eq 16 ]
}

demand_complete() {
  local environment
  for environment in "${DEMAND_ENVIRONMENTS[@]}"; do
    demand_environment_complete "$environment" || return 1
  done
}

mimii_complete() {
  [ "$(file_count machine '*.wav')" -eq 3600 ]
}

file_list_complete() {
  local marker=$1
  local destination=$2
  local relative

  [ -s "$marker" ] || return 1
  while IFS= read -r relative; do
    [ -n "$relative" ] || continue
    [ -s "$destination/$relative" ] || return 1
  done < "$marker"
}

dns_marker_complete() {
  local archive=$1
  local destination=$2
  file_list_complete "${archive}.extracted" "$destination"
}

raw_wav_tree_exact() {
  local root=$1
  local expected=$2
  "$VENV_PYTHON" - "$root" "$expected" <<'PY'
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
if root.is_symlink() or not root.is_dir():
    raise SystemExit(f"raw root is missing/symlink: {root}")
paths = set()
inodes = set()
count = 0
for current_text, directories, files in os.walk(root, followlinks=False):
    current = Path(current_text)
    for name in directories:
        child = current / name
        if child.is_symlink() or not stat.S_ISDIR(child.lstat().st_mode):
            raise SystemExit(f"raw directory symlink/non-directory: {child}")
    for name in files:
        child = current / name
        info = child.lstat()
        if child.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"raw file symlink/non-regular: {child}")
        if child.suffix.casefold() != ".wav":
            continue
        relative = child.relative_to(root).as_posix().casefold()
        inode = (int(info.st_dev), int(info.st_ino))
        if relative in paths or inode in inodes:
            raise SystemExit(f"duplicate raw WAV path/inode: {child}")
        paths.add(relative)
        inodes.add(inode)
        count += 1
if count != expected:
    raise SystemExit(f"raw WAV exact count mismatch: {root}: {count} != {expected}")
print(f"[raw] exact regular WAV count: {root}: {count}")
PY
}

zip_valid() {
  [ -f "$1" ] && unzip -tq "$1" >/dev/null 2>&1
}

# ZIP은 live corpus에 직접 풀지 않는다. entry traversal/symlink를 거부한 뒤 같은
# filesystem staging directory에만 해제하고, completeness gate 뒤 directory rename한다.
safe_extract_zip() {
  local archive=$1
  local staging=$2
  "$VENV_PYTHON" - "$archive" "$staging" <<'PY'
import os
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

archive = Path(sys.argv[1])
root = Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=False)
seen = set()
with zipfile.ZipFile(archive) as source:
    for info in source.infolist():
        raw = info.filename
        name = PurePosixPath(raw)
        if (
            not raw
            or "\\" in raw
            or name.is_absolute()
            or any(part in {"", ".", ".."} for part in name.parts)
        ):
            raise SystemExit(f"unsafe ZIP entry path: {raw!r}")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise SystemExit(f"ZIP symlink entry is forbidden: {raw!r}")
        key = name.as_posix().rstrip("/")
        if key in seen:
            raise SystemExit(f"duplicate ZIP entry is forbidden: {raw!r}")
        seen.add(key)
    source.extractall(root)
directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
PY
}

publish_staged_directory() {
  local staged=$1
  local target=$2
  local backup="${target}.bootstrap-backup.$$"
  if [ -e "$backup" ]; then
    echo "[오류] 이전 bootstrap 복구 directory가 남아 있습니다: $backup" >&2
    return 1
  fi
  if [ -e "$target" ]; then
    mv "$target" "$backup"
  fi
  if ! mv "$staged" "$target"; then
    if [ -e "$backup" ]; then
      mv "$backup" "$target" || true
    fi
    return 1
  fi
  rm -rf "$backup"
}

# wget 결과는 최종 archive와 다른 .part에 받은 뒤 ZIP 검사를 통과한 경우에만
# 최종 이름으로 바꾼다. 중단된 .part는 다음 실행에서 안전하게 덮어쓴다.
ensure_wget_zip() {
  local url=$1
  local archive=$2
  local part="${archive}.part"
  if zip_valid "$archive"; then
    echo "[reuse] 무결성 확인된 $archive"
    return
  fi
  if ! wget -q -O "$part" "$url"; then
    echo "[오류] $archive 다운로드 실패" >&2
    rm -f "$part"
    return 1
  fi
  if ! unzip -tq "$part" >/dev/null 2>&1; then
    echo "[오류] 다운로드한 $archive ZIP 무결성 검사 실패" >&2
    rm -f "$part"
    return 1
  fi
  mv -f "$part" "$archive"
}

# pget.py는 archive.part에 모든 Range를 받은 뒤 바이트 수를 검증하고 원자적으로
# archive로 교체한다. 그 후 ZIP 자체도 검사해 이중으로 완전성을 확인한다.
ensure_pget_zip() {
  local url=$1
  local archive=$2
  local connections=$3
  if zip_valid "$archive"; then
    echo "[reuse] 무결성 확인된 $archive"
    return
  fi
  "${PGET[@]}" "$url" "$archive" "$connections"
  if ! unzip -tq "$archive" >/dev/null 2>&1; then
    echo "[오류] 다운로드한 $archive ZIP 무결성 검사 실패" >&2
    return 1
  fi
}

download_dns_archive() {
  local url=$1
  local archive=$2
  "${PGET[@]}" "$url" "$archive" 12
  if ! bzip2 -t "$archive"; then
    echo "[오류] 다운로드한 $archive bzip2 무결성 검사 실패" >&2
    rm -f "${archive}.done"
    return 1
  fi
  touch "${archive}.done"
}

download_esc50() {
  local archive=esc50.zip stage
  ensure_wget_zip \
    "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master" \
    "$archive"
  mkdir -p esc50
  stage=$(mktemp -d "$PWD/.esc50-stage.XXXXXX")
  rmdir "$stage"
  safe_extract_zip "$archive" "$stage"
  if [ "$(file_count "$stage/ESC-50-master/audio" '*.wav')" -ne 2000 ] ||
      [ ! -s "$stage/ESC-50-master/meta/esc50.csv" ]; then
    echo "[오류] ESC-50 staging 추출 불완전" >&2
    rm -rf "$stage"
    return 1
  fi
  publish_staged_directory "$stage/ESC-50-master" "esc50/ESC-50-master"
  rm -rf "$stage"
  rm -f "$archive"
}

download_fma_small() {
  local archive="$FMA_ROOT/fma_small.zip" stage
  mkdir -p "$FMA_ROOT"
  ensure_pget_zip "https://os.unil.cloud.switch.ch/fma/fma_small.zip" "$archive" 8
  stage=$(mktemp -d "$FMA_ROOT/.fma-small-stage.XXXXXX")
  rmdir "$stage"
  safe_extract_zip "$archive" "$stage"
  if ! fma_small_complete "$stage/fma_small"; then
    echo "[오류] FMA-small staging 추출 불완전" >&2
    rm -rf "$stage"
    return 1
  fi
  publish_staged_directory "$stage/fma_small" "$FMA_ROOT/fma_small"
  rm -rf "$stage"
  rm -f "$archive"
}

download_fma_metadata() {
  local archive="$FMA_ROOT/fma_metadata.zip" stage
  mkdir -p "$FMA_ROOT"
  ensure_pget_zip "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" "$archive" 8
  if [ "$(stat -c %s "$archive")" -ne "$FMA_METADATA_ARCHIVE_SIZE" ] ||
      [ "$(sha256sum "$archive" | awk '{print $1}')" != "$FMA_METADATA_ARCHIVE_SHA256" ]; then
    echo "[오류] FMA metadata archive size/SHA-256 불일치" >&2
    rm -f "$archive"
    return 1
  fi
  stage=$(mktemp -d "$FMA_ROOT/.fma-metadata-stage.XXXXXX")
  rmdir "$stage"
  safe_extract_zip "$archive" "$stage"
  if ! fma_metadata_complete "$stage/fma_metadata"; then
    echo "[오류] FMA metadata 불완전: tracks.csv/artist·album ID/full row 검증 실패" >&2
    rm -rf "$stage"
    return 1
  fi
  publish_staged_directory "$stage/fma_metadata" "$FMA_ROOT/fma_metadata"
  rm -rf "$stage"
  rm -f "$archive"
}

download_demand() {
  local environment archive stage
  mkdir -p demand
  for environment in "${DEMAND_ENVIRONMENTS[@]}"; do
    if demand_environment_complete "$environment"; then
      echo "[skip] DEMAND $environment (WAV 16/16)"
      continue
    fi
    archive="demand/${environment}_48k.zip"
    ensure_wget_zip \
      "$ZEN/1227121/files/${environment}_48k.zip?download=1" \
      "$archive"
    stage=$(mktemp -d "$PWD/.demand-${environment}-stage.XXXXXX")
    rmdir "$stage"
    safe_extract_zip "$archive" "$stage"
    if [ "$(file_count "$stage/$environment" '*.wav')" -ne 16 ]; then
      echo "[오류] DEMAND $environment staging 추출 불완전" >&2
      rm -rf "$stage"
      return 1
    fi
    publish_staged_directory "$stage/$environment" "demand/$environment"
    rm -rf "$stage"
    rm -f "$archive"
  done
}

download_mimii() {
  local archive=mimii_fan.zip stage
  ensure_pget_zip "$ZEN/6529888/files/fan.zip?download=1" "$archive" 8
  stage=$(mktemp -d "$PWD/.mimii-stage.XXXXXX")
  rmdir "$stage"
  safe_extract_zip "$archive" "$stage"
  if [ "$(file_count "$stage" '*.wav')" -ne 3600 ]; then
    echo "[오류] MIMII fan staging 추출 불완전" >&2
    rm -rf "$stage"
    return 1
  fi
  publish_staged_directory "$stage" machine
  rm -f "$archive"
}

pids=()
declare -A download_labels=()
start_download() {
  local label=$1
  shift
  "$@" &
  local pid=$!
  pids+=("$pid")
  download_labels["$pid"]=$label
}

for f in "${!DL[@]}"; do
  d=${DEST[$f]}
  if dns_marker_complete "$f" "$d"; then
    echo "[skip] $f (추출 파일 목록 검증 완료)"
    continue
  fi
  # 두 noise 샤드는 같은 대상 디렉터리를 사용한다. 따라서 대상에 WAV가 하나라도
  # 있다는 이유로 특정 샤드를 완료 처리하면 안 된다. 구버전의 무표식 추출본은
  # 보존한 채 해당 archive만 다시 받아 덮어 추출하고, 이후 파일 목록으로 검증한다.
  if [ -f "$f" ] && [ -f "${f}.done" ]; then
    echo "[skip] $f (다운로드+무결성 검사 완료)"
    continue
  fi
  if [ -f "$f" ] && bzip2 -t "$f"; then
    touch "${f}.done"
    echo "[reuse] $f (기존 archive 무결성 확인)"
    continue
  fi
  rm -f "${f}.done"
  start_download "DNS $f" download_dns_archive "${DL[$f]}" "$f"
done

if esc50_complete; then
  echo "[skip] ESC-50 (WAV 2000/2000 + meta)"
else
  start_download "ESC-50" download_esc50
fi

if fma_small_complete; then
  echo "[skip] FMA-small (MP3 8000/8000)"
else
  start_download "FMA-small" download_fma_small
fi
if fma_metadata_complete; then
  echo "[skip] FMA metadata (tracks.csv full artist/album mapping)"
else
  start_download "FMA metadata" download_fma_metadata
fi

if demand_complete; then
  echo "[skip] DEMAND (6환경 × WAV 16 = 96/96)"
else
  start_download "DEMAND" download_demand
fi

# 기계소음: MIMII DG fan (16kHz, 저역 학습용 — QA 리포트에 표기됨)
if mimii_complete; then
  echo "[skip] MIMII fan (WAV 3600/3600)"
else
  start_download "MIMII fan" download_mimii
fi

download_failed=0
# 빈 배열이면 "${pids[@]}"는 0개 단어로 확장되므로 wait를 한 번도 호출하지 않는다.
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    echo "[오류] ${download_labels[$p]} 다운로드/추출 실패 (PID $p) — 검증된 데이터는 보존됩니다." >&2
    download_failed=1
  fi
done
if [ "$download_failed" -ne 0 ]; then
  echo "[오류] 다운로드 단계가 완료되지 않았습니다. 네트워크 확인 후 bootstrap_all.sh 를 재실행하세요." >&2
  exit 1
fi

# skip 경로와 새 다운로드 경로 모두 같은 완전성 게이트를 마지막에 통과해야 한다.
if ! esc50_complete || ! fma_complete || ! demand_complete || ! mimii_complete; then
  echo "[오류] 데이터셋 완전성 최종 검사 실패" >&2
  exit 1
fi

echo "=== [3/6] DNS 샤드 무결성 검사 + 해제 ==="
for f in "${!DEST[@]}"; do
  if [ -f "$f" ] && ! bzip2 -t "$f"; then
    echo "[오류] $f 손상 — 다음 실행에서 재다운로드합니다." >&2
    rm -f "${f}.done"
    exit 1
  fi
done

# 어떤 archive의 목록 검증이 실패해도 이미 시작된 추출 작업이 남지 않도록
# 모든 목록을 먼저 검증한 뒤 두 번째 loop에서만 background 추출을 시작한다.
extract_archives=()
for f in "${!DEST[@]}"; do
  if [ -f "$f" ]; then
    d=${DEST[$f]}
    mkdir -p "$d"
    marker_building="${f}.extracted.building"
    rm -f "$marker_building"
    if ! "$VENV_PYTHON" - "$f" > "$marker_building" <<'PY'
import sys
import tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], mode="r:bz2") as archive:
    seen = set()
    for member in archive.getmembers():
        raw = member.name
        path = PurePosixPath(raw)
        if (
            not raw
            or "\\" in raw
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise SystemExit(f"unsafe tar member: {raw!r}")
        key = path.as_posix().rstrip("/")
        if key in seen:
            raise SystemExit(f"duplicate tar member: {raw!r}")
        seen.add(key)
        if member.isfile() and raw.casefold().endswith(".wav"):
            print(raw)
PY
    then
      echo "[오류] $f 파일 목록을 읽지 못했습니다." >&2
      for prepared in "${!DEST[@]}"; do
        rm -f "${prepared}.extracted.building"
      done
      rm -f "${f}.done"
      exit 1
    fi
    if [ ! -s "$marker_building" ]; then
      echo "[오류] $f 안에 WAV 파일이 없습니다." >&2
      for prepared in "${!DEST[@]}"; do
        rm -f "${prepared}.extracted.building"
      done
      rm -f "${f}.done"
      exit 1
    fi
    extract_archives+=("$f")
  fi
done

extract_pids=()
declare -A extract_files=()
for f in "${extract_archives[@]}"; do
  d=${DEST[$f]}
  tar --no-same-owner --no-same-permissions -xjf "$f" -C "$d" &
  pid=$!
  extract_pids+=("$pid")
  extract_files["$pid"]=$f
done

extract_failed=0
for p in "${extract_pids[@]}"; do
  f=${extract_files[$p]}
  d=${DEST[$f]}
  marker_building="${f}.extracted.building"
  if wait "$p"; then
    if file_list_complete "$marker_building" "$d"; then
      mv -f "$marker_building" "${f}.extracted"
      rm -f "$f" "${f}.done"
      echo "[완료] $f 해제 및 파일 목록 검증 (PID $p)"
    else
      echo "[오류] $f 해제 후 파일 목록 검증 실패 — archive를 보존합니다." >&2
      rm -f "$marker_building" "${f}.done"
      extract_failed=1
    fi
  else
    echo "[오류] $f 해제 실패 (PID $p) — archive를 보존합니다." >&2
    rm -f "$marker_building" "${f}.done"
    extract_failed=1
  fi
done
if [ "$extract_failed" -ne 0 ]; then
  echo "[오류] DNS 샤드 해제가 완료되지 않았습니다. bootstrap_all.sh 를 재실행하세요." >&2
  exit 1
fi
if ! raw_wav_tree_exact dns_fullband 16000 || ! raw_wav_tree_exact speech 8065; then
  echo "[오류] DNS noise/speech untouched raw exact-count/symlink/duplicate gate 실패" >&2
  exit 1
fi
cd "$REPO"

echo "=== [4/6] manifest + RIR 뱅크 + 데이터셋 QA ==="
# 긴 다운로드 동안 tracked code나 별도 채널로 전달한 canonical bundle이 바뀌지
# 않았는지 manifest 생성 직전에 다시 확인한다. prepare에도 같은 외부 SHA를 전달한다.
if ! verify_exact_checkout || ! verify_canonical_bundle || ! verify_transfer_bundle "$VENV_PYTHON"; then
  echo "[오류] 다운로드 후 manifest 준비 시작 gate에서 exact code/bundle이 바뀌었습니다." >&2
  exit 1
fi
# MP3 등의 header/일부 seek만 정상인 raw를 NoisePool retry가 다른 파일로 조용히
# 대체하지 못하게, 모든 공개 raw를 복수 접근 방식으로 먼저 전수 decode한다. reject는
# forensic evidence로 report에 남기되, --allow-rejections으로 audit 자체는 성공시켜
# prepare가 audit의 accepted inventory만 canonical v4 세대에 넣도록 한다. 이미 끝난
# audit을 재사용하려면 파일/semantic SHA, 현재 decoder fingerprint, raw 전체
# SHA/size를 먼저 fail-closed로 대조해야 하며, 다음 prepare transaction도 같은
# raw 재대조를 다시 수행한다. 재사용 실패 시 새 audit으로 자동 fallback하지 않는다.
if [ "$REUSE_DECODER_AUDIT" -eq 1 ]; then
  echo "[decoder audit] 외부 SHA로 고정한 완료 audit을 재사용 전 전수 검증합니다."
  "$VENV_PYTHON" scripts/data/verify_decoder_audit_reuse.py \
    --root . \
    --audit "$DECODER_AUDIT_REPORT" \
    --scan-root data/raw \
    --expected-audit-sha256 "$EXPECTED_DECODER_AUDIT_SHA256" \
    --expected-file-sha256 "$EXPECTED_DECODER_AUDIT_FILE_SHA256" \
    --hash-workers "$RAW_HASH_WORKERS"
else
  "$VENV_PYTHON" scripts/data/audit_decoder_eligibility.py \
    --root . \
    --scan-root data/raw \
    --out "$DECODER_AUDIT_REPORT" \
    --allow-rejections
fi
recorded_generation_prepare_args=()
if [ "$RECORDED_TRANSFER_SCHEMA" = "2" ]; then
  recorded_generation_prepare_args=(
    --recorded-generation "$RECORDED_GENERATION"
    --expected-recorded-generation-sha256 "$RECORDED_GENERATION_SHA256"
  )
fi
"$VENV_PYTHON" scripts/data/prepare_noise_pool.py \
  --expected-holdout-sha256 "$EXPECTED_HOLDOUT_SHA256" \
  --recorded-source-pool-csv data/source_pool_v2/sources.csv \
  --recorded-source-pool-csv data/source_pool/sources.csv \
  "${recorded_generation_prepare_args[@]}" \
  --out "$CANONICAL_MANIFEST_DIR" \
  --decoder-audit "$DECODER_AUDIT_REPORT" \
  --hash-workers "$RAW_HASH_WORKERS"
if ! verify_exact_checkout || ! verify_canonical_bundle || ! verify_transfer_bundle "$VENV_PYTHON"; then
  echo "[오류] manifest 준비 종료 gate에서 exact code/bundle이 바뀌었습니다." >&2
  exit 1
fi
RIR_BANK=data/rir_bank/duct_rirs_v1.npz
rir_bank_complete() {
  local path=$1
  [ -f "$path" ] && "$VENV_PYTHON" - "$path" <<'PY' >/dev/null 2>&1
import sys

import numpy as np

expected_shape = (300, 8192)
with np.load(sys.argv[1]) as bank:
    for key in ("p_ref", "p_err", "f_fb"):
        value = bank[key]
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise SystemExit(1)
PY
}
if ! rir_bank_complete "$RIR_BANK"; then
  echo "[오류] 신뢰한 transferred RIR bank의 shape/finite 검증 실패. bootstrap은 이를 재생성하거나 덮어쓰지 않습니다." >&2
  exit 1
fi
"$VENV_PYTHON" scripts/data/validate_noise_pool.py \
  --manifest-dir "$CANONICAL_MANIFEST_DIR" \
  --out "$CANONICAL_MANIFEST_DIR/dataset_qa.md"

# transferred recorded bytes의 전수 QA를 먼저 끝낸 뒤, 같은 manifest/timing으로 strict
# 부대역 target coverage를 한 번만 FFT 감사한다. coverage FAIL(종료 1)은 현재 데이터의
# 유효한 진단 증거이므로 report를 보존하고 bootstrap을 계속하되, readiness가 학습을
# 차단한다. 구조·provenance 오류(종료 2)는 bootstrap 자체를 즉시 중단한다.
"$VENV_PYTHON" scripts/data/validate_recorded_sessions.py \
  --manifest "$RECORDED_MANIFEST" \
  --data-config configs/data_sim.yaml \
  --out-md "$RECORDED_QA_MD" \
  --out-json "$RECORDED_QA_JSON"
set +e
"$VENV_PYTHON" scripts/data/audit_recorded_subband_coverage.py \
  --config configs/train_pretrain_tiny.yaml \
  --manifest "$RECORDED_MANIFEST" \
  --canonical-out-dir "$RECORDED_SUBBAND_COVERAGE_REPORT_DIR"
coverage_status=$?
set -e
if [ "$coverage_status" -gt 1 ]; then
  echo "[오류] recorded strict 부대역 coverage report 생성/재검증 실패" >&2
  exit 1
fi
if [ "$coverage_status" -eq 1 ]; then
  echo "[차단 증거] recorded strict 부대역 coverage가 부족합니다. bootstrap은 증거 보존을 위해 계속하지만 readiness/학습은 FAIL이어야 합니다." >&2
fi

echo "=== [5/6] 검증 (pytest) ==="
"$VENV_PYTHON" -B -m pytest -q

echo "=== [6/6] 환경·데이터 준비 완료 (학습은 시작하지 않음) ==="
if ! verify_exact_checkout || ! verify_canonical_bundle || ! environment_complete; then
  echo "[오류] 최종 bootstrap receipt 직전 code/holdout/environment가 바뀌었습니다." >&2
  exit 1
fi
BOOTSTRAP_RECEIPT="$REPO/data/manifests/elice_bootstrap_receipt.json"
if ! PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO/src" "$VENV_PYTHON" -B - \
    "$REPO" "$EXPECTED_COMMIT" "$EXPECTED_HOLDOUT_SHA256" \
    "$TRANSFER_MANIFEST" "$EXPECTED_TRANSFER_MANIFEST_SHA256" \
    "$ENVIRONMENT_RECEIPT" "$BOOTSTRAP_RECEIPT" \
    "$RECORDED_SUBBAND_COVERAGE_REPORT_DIR" "$RECORDED_MANIFEST" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

from deep_anc.config import load_train_config
from deep_anc.data.holdout_contract import FileSnapshot, read_regular_file_snapshot
from deep_anc.data.recorded_subband_coverage import (
    build_recorded_subband_coverage_contract,
    recorded_subband_coverage_report_path,
    validate_recorded_subband_coverage_report,
)
from deep_anc.data.transfer_contract import validate_transfer_manifest

root = Path(os.path.abspath(sys.argv[1]))
expected_commit = sys.argv[2]
expected_holdout_sha = sys.argv[3]
transfer_path = Path(sys.argv[4])
expected_transfer_sha = sys.argv[5]
environment_path = Path(sys.argv[6])
output = Path(sys.argv[7])
coverage_directory = Path(sys.argv[8])
if not coverage_directory.is_absolute():
    coverage_directory = root / coverage_directory
selected_recorded_manifest = sys.argv[9]

# receipt를 쓰기 직전 transfer 전체를 다시 같은 validator로 hash한다. 이 결과의
# recorded aggregate만 resolved config가 신뢰할 수 있다.
summary = validate_transfer_manifest(
    transfer_path,
    repo_root=root,
    expected_sha256=expected_transfer_sha,
)
if summary["canonical_holdout_sha256"] != expected_holdout_sha:
    raise SystemExit("canonical holdout SHA가 bootstrap trust anchor와 다릅니다")
validated_recorded_manifest = summary.get("_validated_recorded_manifest_snapshot")
if (
    not isinstance(validated_recorded_manifest, FileSnapshot)
    or validated_recorded_manifest.data is None
):
    raise SystemExit("receipt 작성 시 validated recorded manifest snapshot이 없습니다")
try:
    validated_recorded_relative = validated_recorded_manifest.path.relative_to(
        root
    ).as_posix()
except ValueError as exc:
    raise SystemExit("receipt 작성 시 validated recorded manifest가 저장소 밖입니다") from exc
if validated_recorded_relative != selected_recorded_manifest:
    raise SystemExit(
        "receipt 직전 transfer 재검증의 recorded manifest가 QA/coverage 선택과 다릅니다: "
        f"selected={selected_recorded_manifest}, validated={validated_recorded_relative}"
    )
environment = read_regular_file_snapshot(
    environment_path,
    root=root,
    label="Elice environment freeze receipt",
    capture_bytes=False,
)
coverage_cfg = load_train_config(
    root / "configs/train_pretrain_tiny.yaml",
    [
        "experiment_role=diagnostic_overfit",
        "init_eligible=false",
        "contract_run_dir=false",
        "run_until_step=500",
        "data.digital_primary_path_mode=measured",
    ],
)
coverage_manifest = validated_recorded_manifest
coverage_contract = build_recorded_subband_coverage_contract(
    manifest_path=coverage_manifest.path,
    manifest_content=coverage_manifest.data,
    data_cfg=coverage_cfg["data"],
    model_hop=int(coverage_cfg["model"]["hop"]),
)
coverage_path = recorded_subband_coverage_report_path(
    coverage_directory, coverage_contract
)
coverage_summary = validate_recorded_subband_coverage_report(
    coverage_path,
    manifest_path=coverage_manifest.path,
    data_cfg=coverage_cfg["data"],
    model_hop=int(coverage_cfg["model"]["hop"]),
    required_families=("speech", "music", "environment", "machine"),
    configured_min_groups_per_family=4,
)
coverage_snapshot = read_regular_file_snapshot(
    coverage_path,
    root=root,
    label="recorded subband coverage report",
)
assert coverage_snapshot.data is not None
coverage_payload = json.loads(coverage_snapshot.data.decode("utf-8"))
payload = {
    "schema_version": 2,
    "expected_commit": expected_commit,
    "canonical_holdout": {
        "path": "data/manifests/recorded_holdout.json",
        "sha256": expected_holdout_sha,
    },
    "transfer_manifest": {
        "path": "data/manifests/elice_transfer_manifest.json",
        "sha256": expected_transfer_sha,
    },
    "recorded_aggregate_sha256": summary["recorded_aggregate_sha256"],
    "recorded_subband_coverage": {
        "path": coverage_path.relative_to(root).as_posix(),
        "sha256": coverage_snapshot.sha256,
        "evidence_sha256": coverage_payload["evidence_sha256"],
        "manifest_sha256": coverage_summary["manifest_sha256"],
        "training_timing_contract_sha256": coverage_payload[
            "training_timing_contract_sha256"
        ],
        "coverage_contract_sha256": coverage_payload[
            "coverage_contract_sha256"
        ],
        "all_requested_splits_pass": coverage_summary[
            "all_requested_splits_pass"
        ],
    },
    "environment": {
        "freeze_receipt": ".venv/environment-freeze.txt",
        "freeze_receipt_sha256": environment.sha256,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    },
}
if payload["environment"]["torch_version"] != "2.5.1+cu121" or payload["environment"]["torch_cuda"] != "12.1":
    raise SystemExit("receipt 작성 시 exact torch/CUDA 계약이 바뀌었습니다")
raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
output.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(prefix=".elice-bootstrap-receipt.", dir=output.parent)
temporary = Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    temporary.unlink(missing_ok=True)
print(f"[bootstrap receipt] path={output}")
print(f"[bootstrap receipt] sha256={hashlib.sha256(raw).hexdigest()}")
PY
then
  echo "[오류] final transfer/environment bootstrap receipt 생성 실패" >&2
  exit 1
fi
echo "학습 CLI에 data.bootstrap_receipt와 위 bootstrap receipt SHA-256을 외부 trust anchor로 전달하세요."
echo "readiness와 G0/smoke를 별도 명령으로 통과한 뒤 승인된 tiny campaign만 실행하세요."
echo "이 부트스트랩은 legacy base/tiny 학습을 자동 시작하지 않습니다."
