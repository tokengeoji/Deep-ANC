import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.ci.check_static_contract_references import (  # noqa: E402
    StaticContractReferenceError,
    audit_static_contract_references,
)


def pytest_sessionstart(session: object) -> None:
    """무거운 collection 전에 운영 코드의 stale pytest node를 차단한다."""

    try:
        audit_static_contract_references(REPO)
    except StaticContractReferenceError as exc:
        # pytest Session이 제공하는 표준 조기중단 예외를 사용한다. 이 conftest 자체는
        # pytest 외 제3자 패키지를 import하지 않으므로 bootstrap 환경에서도 동일하다.
        interrupted = getattr(session, "Interrupted", RuntimeError)
        raise interrupted(str(exc)) from exc
