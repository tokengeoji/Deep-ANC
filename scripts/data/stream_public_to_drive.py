#!/usr/bin/env python3
"""승인된 인증 환경 전용 전송 CLI. 기본은 계획 출력이며 원본 다운로드를 하지 않는다."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from deep_anc.data.drive_transfer import GoogleDriveUploadAPI, stream_archive_to_drive, validate_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--parent-folder-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-source-absent", action="store_true")
    args = parser.parse_args()
    try:
        catalog = json.loads(args.catalog.read_text())
        source = catalog["sources"][args.source]
        validate_source(source)
        if not args.execute:
            print(json.dumps({"action": "plan_only", "source": source, "local_raw_written": False}, ensure_ascii=False))
            return 0
        # 현재 개발 PC에서는 스트리밍을 포함한 원본 다운로드를 금지한다.
        # 미설정 호스트/다른 컨테이너도 기본 거부. 이 표시 자체가 사용자 승인이나
        # 보안 격리 수단은 아니다. 별도 승인된 원격 환경 운영자만 설정해야 한다.
        if os.environ.get("DEEP_ANC_CONTAINER") != "approved-transfer":
            raise ValueError("별도 승인된 원격 전송 환경만 실행할 수 있습니다. 현재 PC/Jetson에서는 금지됩니다")
        if not args.confirm_source_absent or not args.parent_folder_id:
            raise ValueError("기존 Drive 자료의 부재 확인 및 대상 폴더 ID가 필요합니다")
        import google.auth
        from google.auth.transport.requests import Request
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.file"])
        def token():
            if not credentials.valid:
                credentials.refresh(Request())
            return credentials.token
        receipt = stream_archive_to_drive(source, args.parent_folder_id, GoogleDriveUploadAPI(token))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        # 외부 인증/HTTP 예외 원문은 토큰·세션 URL이 섞일 수 있어 출력하지 않는다.
        print(f"[실패] {type(exc).__name__}: 전송 조건/인증/파일 검증 실패. 원본 디스크 저장과 자동 복구는 하지 않습니다.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
