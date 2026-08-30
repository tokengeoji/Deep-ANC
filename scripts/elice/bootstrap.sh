#!/usr/bin/env bash
#
# 폐기된 Elice 진입점.
#
# 과거 이 파일의 ``--train`` 경로는 exact commit/holdout/transfer manifest,
# bootstrap receipt, readiness와 canonical campaign ledger 없이 legacy pretrain을
# 시작할 수 있었다. 파일명을 기억한 운영자가 실수로 실행하더라도 학습이 열리지
# 않도록 영구 fail-closed stub으로 남긴다.
set -euo pipefail

cat >&2 <<'MSG'
[중단] scripts/elice/bootstrap.sh는 폐기된 legacy 진입점입니다.

환경과 데이터 준비는 반드시 다음 exact-SHA 진입점만 사용하세요.

  bash scripts/elice/bootstrap_all.sh \
    --expected-commit <40자리_SHA> \
    --expected-holdout-sha256 <64자리_SHA256> \
    --expected-transfer-manifest-sha256 <64자리_SHA256> \
    --raw-hash-workers 8 \
    --no-update

bootstrap_all.sh도 학습을 자동 시작하지 않습니다. bootstrap receipt와 단계별
readiness를 검증한 뒤 docs/05_training_elice.md의 canonical campaign 절차를 따르세요.
MSG
exit 2
